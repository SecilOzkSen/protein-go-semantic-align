from typing import Dict, Any, List, Tuple, Optional
import copy
import torch
import torch.nn.functional as F
import time
import math

from src.models.alignment_model import ProteinGoAligner
from src.loss.attribution import attribution_loss, windowed_attr_loss
from src.configs.data_classes import TrainerConfig, AttrConfig
from src.miners.queue_miner import MoCoQueue
from contextlib import nullcontext

# Wandb
import wandb
from collections import defaultdict
from src.utils.wandb_logger import WabLogger

from src.metrics.cafa import compute_fmax, compute_term_aupr


# ------------- Helpers -------------
def to_f32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float32 else x.float()

def norm_f32(x: torch.Tensor, p: int = 2, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(to_f32(x), p=p, dim=dim, eps=eps)

def clone_as_target(module: torch.nn.Module) -> torch.nn.Module:
    k = copy.deepcopy(module).eval()
    for p in k.parameters():
        p.requires_grad_(False)
    return k

@torch.no_grad()
def ema_update(q: torch.nn.Module, k: torch.nn.Module, m: float):
    for p_q, p_k in zip(q.parameters(), k.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)

def entropy_regularizer(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = alpha.clamp_min(eps)
    ent = -(a * a.log()).sum(dim=-1)
    return ent.mean()

def multi_positive_infonce_from_candidates(scores: torch.Tensor, pos_mask: torch.Tensor, tau: float) -> torch.Tensor:
    """
    scores: (B, K)
    pos_mask: (B, K) boolean; True at positives
    """
    logits = scores / max(1e-8, tau)                          # (B, K)
    denom = torch.logsumexp(logits, dim=-1)                   # (B,)
    pos_logits = logits.masked_fill(~pos_mask, float('-inf')) # (B, K)
    pos_any = pos_mask.any(dim=1)                             # (B,)
    if (~pos_any).any():
        pos_logits = pos_logits.clone()
        pos_logits[~pos_any] = -1e9
    num = torch.logsumexp(pos_logits, dim=-1)                 # (B,)
    loss = -(num - denom)                                     # (B,)
    return loss[pos_any].mean() if pos_any.any() else denom.mean() * 0.0

def surrogate_delta_y_from_mask_grad(H, G, model, pad_mask=None) -> Tuple[torch.Tensor, dict]:
    """
    Quick surrogate for attribution: use ||dy/dH|| as importance proxy.
    Returns (proxy_deltas, alpha_info).
    """
    H = H.clone().detach().requires_grad_(True)
    out = model(H=H, G=G, mask=pad_mask, return_alpha=True)
    if isinstance(out, tuple) and len(out) >= 2:
        scores, alpha_info = out[0], (out[1] or {})
    else:
        scores, alpha_info = out, {}
    y = scores.mean()
    y.backward(retain_graph=True)
    with torch.no_grad():
        dy_dH = H.grad.norm(dim=-1)  # (B, L)
    B, T = scores.shape
    proxy = dy_dH.unsqueeze(1).repeat(1, T, 1)
    proxy = proxy / (proxy.amax(dim=-1, keepdim=True) + 1e-8)
    return proxy, alpha_info

def dag_consistency_loss_pos(scores_pos: torch.Tensor,
                             pos_local: List[torch.Tensor],
                             uniq_go_ids: torch.Tensor,
                             dag_parents: Optional[dict],
                             margin: float = 0.0,
                             scale: float = 1.0) -> torch.Tensor:
    """
    DAG consistency on POSITIVES only.
    Enforce: score(parent) >= score(child) + margin
    """
    if dag_parents is None or len(pos_local) == 0:
        return torch.zeros((), device=scores_pos.device)

    B, T_max = scores_pos.shape
    losses: List[torch.Tensor] = []
    for b in range(B):
        loc = pos_local[b]
        t = int(loc.numel())
        if t <= 1:
            continue
        sp = scores_pos[b, :t]  # (t,)
        glob = uniq_go_ids.index_select(0, loc.to(uniq_go_ids.device))  # (t,)
        id2t = {int(glob[i].item()): i for i in range(t)}
        for child_gid, child_t in list(id2t.items()):
            parents = dag_parents.get(child_gid, [])
            for pg in parents:
                if pg in id2t:
                    p_t = id2t[pg]
                    diff = sp[child_t] - sp[p_t] + margin
                    losses.append(F.softplus(scale * diff))
    if len(losses) == 0:
        return torch.zeros((), device=scores_pos.device)
    return torch.stack(losses).mean()

def topk_maskout_full(H, G, alpha_full, k, model, pad_mask=None):
    """
    Eval-time mask-out for full-length case.
    """
    B, T, L = alpha_full.shape
    device = H.device
    delta = torch.zeros_like(alpha_full)
    out = model(H=H, G=G, mask=pad_mask, return_alpha=True)  # (scores, alpha_info)
    base_scores = out[0] if isinstance(out, tuple) else out   # (B, T)

    for b in range(B):
        for t in range(T):
            topk = min(k, L)
            _, idx = torch.topk(alpha_full[b, t], k=topk, dim=-1)
            for i in idx.tolist():
                Hminus = H.clone()
                Hminus[b, i, :] = 0.0
                out_m = model(
                    H=Hminus,
                    G=G[b:b+1],
                    mask=pad_mask[b:b+1] if pad_mask is not None else None,
                    return_alpha=True
                )
                y_minus = out_m[0] if isinstance(out_m, tuple) else out_m  # (1, T)
                delta[b, t, i] = (base_scores[b, t] - y_minus.squeeze(0)[t]).clamp_min(0.0)
            m = delta[b, t].amax()
            if m > 0:
                delta[b, t] = delta[b, t] / m
    return delta


# ------------- Trainer -------------
class OppTrainer:
    def __init__(self, cfg: TrainerConfig, attr: AttrConfig, ctx, go_encoder, wandb_run=None, wlogger=None):
        self.cfg, self.attr, self.ctx = cfg, attr, ctx
        self.device = torch.device(self.cfg.device)

        # === TF32 ===
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # tek tip normalizer (fp16 olsa da olmasa da aynı imza)
        self.normalizer = lambda x, dim: norm_f32(x, p=2, dim=dim)
        self.to_f32 = to_f32 if ctx.fp16_enabled else None

        self.model = ProteinGoAligner(
            d_h=cfg.d_h,
            d_g=None,
            d_z=cfg.d_z,          # e.g., 768
            go_encoder=go_encoder,
            normalize=True,
        ).to(self.device)

        # --- EMA (momentum) key encoder: GO tarafı ---
        self.m_ema = float(getattr(cfg, "m_ema", 0.999))
        self.go_encoder_k = None
        if getattr(self.model, "go_encoder", None) is not None:
            self.go_encoder_k = clone_as_target(self.model.go_encoder).to(self.device)

        # --- OPT ---
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self._global_step = 0

        # projector for teacher alignment (Dh -> Dz)
        self.index_projector = copy.deepcopy(self.model.proj_p).eval().requires_grad_(False)
        self.ctx.vres.set_align_dim(self.cfg.d_z)
        self.ctx.vres.set_query_projector(self.index_projector)

        # --- Queue config ---
        self.use_moco_miner: bool = True
        self.queue_K: int = int(getattr(cfg, "queue_K", getattr(ctx, "queue_K", 4096)))
        self.k_hard_queue: int = int(getattr(cfg, "k_hard_queue", getattr(ctx, "k_hard", 32)))
        self.queue_miner: Optional[MoCoQueue] = None

        # === Learnable logit scale (CLIP) ===
        init_ln = math.log(1.0 / 0.07)  # ~2.659
        self.logit_scale = torch.nn.Parameter(torch.tensor(init_ln, dtype=torch.float32, device=self.device))

        # W&B
        self._wandb_configs(wandb_run=wandb_run, wlogger=wlogger)
        self._phase_acc = defaultdict(lambda: defaultdict(list))
        self._current_phase_id = None

    # ---------- robust mask helper ----------
    def _valid_and_pad_masks(self, batch):
        """
        Returns:
          attn_valid: bool[B,L]  (True = VALID token)
          pad_mask:   bool[B,L]  (True = PAD)
        """
        m = batch.get("prot_attn_mask", None)
        if m is None:
            raise KeyError("batch['prot_attn_mask'] missing")
        m = m.to(self.device)
        if m.dim() == 3 and m.size(-1) == 1:
            m = m.squeeze(-1)
        if m.dtype is not torch.bool:
            m = m != 0
        attn_valid = m
        pad_mask = ~attn_valid
        return attn_valid, pad_mask

    def _wandb_configs(self, wandb_run=None, wlogger=None):
        if wandb_run is not None:
            self.wandb_run = wandb_run
        else:
            self.wandb_run = wandb.init(
                project="protein-go-semantic-align",
                name=self.ctx.run_name,
                config=self.ctx.to_dict(),
                settings=wandb.Settings(code_dir=".", _disable_stats=True),
                reinit=False
            )
            wandb.define_metric("*", summary="none")
        if wlogger is not None:
            self.wlogger = wlogger
        else:
            self.wlogger = WabLogger(self.wandb_run, project="protein-go-semantic-align",
                                     config=self.ctx.to_dict())
    #    try:
    #        wandb.watch(self.model, log="gradients", log_freq=max(100, getattr(self.cfg, "log_every", 50)))
    #    except Exception:
    #        pass

    # --------- Wandb helpers (phase bookkeeping) ---------
    def _on_phase_change(self, new_phase_id: int, new_phase_name: str, step: int):
        if self._current_phase_id is not None:
            self._flush_phase_table(self._current_phase_id, step)
        self._current_phase_id = new_phase_id
        self._phase_acc[new_phase_id].clear()
        wandb.log({"phase/change_to": new_phase_id, "phase/name": new_phase_name}, step=step)

    def on_epoch_finish(self):
        ema_update(self.model.go_encoder, self.go_encoder_k, m=self.m_ema)

    def _to_device(self, x):
        import torch
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_device(v) for v in x)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        return x

    def _cpu_bank_gather(self, bank_cpu: torch.Tensor, idx: torch.Tensor, dev: torch.device) -> torch.Tensor:
        """
        CPU'daki vektör bankasından (bank_cpu) idx ile satır toplayıp
        sonucu compute cihazına (dev) taşır.
        """
        if idx.device.type != "cpu":
            idx = idx.to("cpu", non_blocking=True)
        if idx.dtype != torch.long:
            idx = idx.long()

        out_cpu = bank_cpu.index_select(0, idx)          # CPU seçimi
        return out_cpu.to(dev, non_blocking=True)        # GPU'ya taşı


    def _flush_phase_table(self, phase_id: int, step: int):
        import numpy as np
        rows = []
        for k, vals in self._phase_acc[phase_id].items():
            if vals:
                arr = np.array(vals, dtype=float)
                rows.append([phase_id, k, float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()), len(arr)])
        if rows:
            table = wandb.Table(columns=["phase_id", "metric", "mean", "std", "min", "max", "n"], data=rows)
            wandb.log({f"phase_summary/phase_{phase_id}": table}, step=step)

    def _maybe_init_queue_miner(self, Dg: int):
        if self.queue_miner is None and self.use_moco_miner:
            self.queue_miner = MoCoQueue(dim=int(Dg), K=int(self.queue_K), device=str(self.device))
            print(f"[Trainer] QueueMiner enabled (K={self.queue_K}, k_hard={self.k_hard_queue}, Dg={Dg}).")

    def _build_candidates_with_queue(self,
                                     uniq_go_embs: torch.Tensor,
                                     pos_go_local: List[torch.Tensor],
                                     neg_from_queue: Optional[torch.Tensor]
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Positives from batch, negatives from QueueMiner.
        Returns:
          G_cand:  [B, K, Dg]
          pos_mask:[B, K]
        """
        B = len(pos_go_local)
        device = uniq_go_embs.device
        Dg = uniq_go_embs.size(1)

        # positives
        pos_embs, pos_lens = [], []
        for b in range(B):
            loc = pos_go_local[b]
            if loc.numel() == 0:
                pos_embs.append(torch.zeros(0, Dg, device=device, dtype=uniq_go_embs.dtype))
                pos_lens.append(0)
            else:
                pos_embs.append(uniq_go_embs.index_select(0, loc.to(device)))
                pos_lens.append(int(loc.numel()))

        kq = 0 if (neg_from_queue is None) else neg_from_queue.size(1)
        K = max((p + kq) for p in pos_lens) if B > 0 else (kq or 1)
        G_cand = torch.zeros(B, K, Dg, device=device, dtype=uniq_go_embs.dtype)
        pos_mask = torch.zeros(B, K, dtype=torch.bool, device=device)

        for b in range(B):
            p = pos_lens[b]
            if p > 0:
                G_cand[b, :p] = pos_embs[b]
                pos_mask[b, :p] = True
            if kq > 0:
                G_cand[b, p:p+kq] = neg_from_queue[b]
        return G_cand, pos_mask

    # --------- Eval space builder (for val/test) ---------
    def _build_eval_space(self, batch):
        """
        Returns:
          G_eval:   (B, Geval, Dg)
          y_true:   (B, Geval)
          eval_ids: (Geval,)
        """
        device = self.device
        B = batch["prot_emb_pad"].size(0)

        if hasattr(self.ctx, "eval_id_list") and self.ctx.eval_id_list:
            eval_ids = torch.as_tensor(self.ctx.eval_id_list, dtype=torch.long, device=device)
            rows = torch.as_tensor([self.ctx.go_cache.id2row[int(g)] for g in eval_ids.tolist()],
                                   dtype=torch.long, device=device)
            bank = self.ctx.go_cache.embs.to(self.device)
            G_eval_once = bank.index_select(0, rows)  # (Geval, Dg)
        else:
            eval_ids = batch["uniq_go_ids"].to(device)  # (Geval,)
            if ("pos_go_tokens" in batch) and (hasattr(self.model, "go_encoder") and self.model.go_encoder is not None):
                toks = batch["pos_go_tokens"]
                G_eval_once = self.model.go_encoder(
                    input_ids=toks["input_ids"].to(device),
                    attention_mask=toks["attention_mask"].to(device),
                )
                G_eval_once = self.normalizer(G_eval_once, dim=1)
            else:
                G_eval_once = batch["uniq_go_embs"].to(device)

        Geval, Dg = G_eval_once.size(0), G_eval_once.size(1)
        G_eval = G_eval_once.unsqueeze(0).expand(B, Geval, Dg).contiguous()

        # y_true multi-hot
        y_true = torch.zeros(B, Geval, dtype=torch.float32, device=device)
        pos_local = batch["pos_go_local"]

        same_space = torch.equal(eval_ids, batch["uniq_go_ids"].to(device))
        if same_space:
            for b, loc in enumerate(pos_local):
                if loc.numel() > 0:
                    y_true[b, loc.to(device)] = 1.0
        else:
            uniq_go_ids = batch["uniq_go_ids"].to(device)
            id2col = {int(eval_ids[i].item()): i for i in range(Geval)}
            for b, loc in enumerate(pos_local):
                if loc.numel() == 0:
                    continue
                glob = uniq_go_ids.index_select(0, loc.to(uniq_go_ids.device))
                for g in glob.tolist():
                    j = id2col.get(int(g), None)
                    if j is not None:
                        y_true[b, j] = 1.0

        return G_eval, y_true, eval_ids

    # --------- Scoring (tensor path) ---------
    def forward_scores(self, H, G, pad_mask, return_alpha=False,
                       cand_chunk_k: int = 32,
                       pos_chunk_t: int = 256,
                       **kwargs):
        """
        Safe VRAM streaming forward.
        """
        device = H.device
        cand_chunk_k = int(getattr(self.cfg, "cand_chunk_k", cand_chunk_k))
        pos_chunk_t = int(getattr(self.cfg, "pos_chunk_t", pos_chunk_t))

        def _unpack(out):
            if isinstance(out, tuple):
                if len(out) >= 2: return out[0], (out[1] or {})
                if len(out) == 1: return out[0], {}
                return out, {}
            return out, {}

        # --- Candidate path ---
        if G.dim() == 4:  # [B,K,T,Dg]
            B, K, T, Dg = G.shape
            if G.is_cuda:  # keep CPU!
                G = G.to("cpu", non_blocking=True)

            scores_all = []
            with torch.no_grad():
                for ks in range(0, K, cand_chunk_k):
                    ke = min(K, ks + cand_chunk_k)
                    G_chunk = G[:, ks:ke].contiguous()  # [B,k,T,Dg]
                    G_chunk = G_chunk.to(device, non_blocking=True)

                    out = self.model(
                        H=H, G=G_chunk, mask=pad_mask,
                        return_alpha=False,
                        cand_chunk_k=None,
                        pos_chunk_t=pos_chunk_t,
                        **kwargs
                    )
                    sc, _ = _unpack(out)
                    scores_all.append(sc.detach().cpu())

                    del G_chunk, out, sc
                    torch.cuda.empty_cache()

            scores = torch.cat(scores_all, dim=1).to(device)
            return scores

        # --- Positive path ---
        out = self.model(
            H=H, G=G, mask=pad_mask,
            return_alpha=return_alpha,
            cand_chunk_k=0,
            pos_chunk_t=pos_chunk_t,
            **kwargs
        )
        sc, alpha = _unpack(out)
        return (sc, alpha) if return_alpha else sc

    # --------- Training step (losses + logging) ---------
    def step_losses(self, batch: Dict[str, Any], epoch_idx: int) -> Dict[str, torch.Tensor]:

        t0 = time.time()
        self.model.train()
        device = self.device

        # ----------- Protein & mask -----------
        H = batch["prot_emb_pad"].to(device, non_blocking=True)  # (B, Lmax, Dh)
        attn_valid, pad_mask = self._valid_and_pad_masks(batch)
        if self.to_f32 is not None:
            H = self.to_f32(H)
        pos_local: List[torch.Tensor] = batch["pos_go_local"]
        B = H.size(0)

        # ---- curriculum / attribution flags ----
        use_attr = (epoch_idx < self.attr.curriculum_epochs and self.attr.lambda_attr > 0.0)
        try:
            self.wlogger.log_scalar({"debug/use_attr": int(use_attr), "epoch": epoch_idx}, step=self._global_step)
        except Exception:
            pass

        # ========== POSITIVES: uniq_go_embs ==========
        # GO encoder'ı sadece gerektiğinde ve no_grad ile kullan.
        if ("pos_go_tokens" in batch) and (hasattr(self.model, "go_encoder") and self.model.go_encoder is not None):
            toks = batch["pos_go_tokens"]
            with torch.no_grad():
                uniq_go_embs = self.model.go_encoder(
                    input_ids=toks["input_ids"].to(device, non_blocking=True),
                    attention_mask=toks["attention_mask"].to(device, non_blocking=True),
                )  # [G, Dg]  (GPU)
            uniq_go_embs = self.normalizer(uniq_go_embs, dim=1)  # [G, Dg] (GPU)
        else:
            # Zaten hazır embedding verilmişse kopyalamadan GPU'ya al, normalize et.
            uniq_go_embs = batch["uniq_go_embs"].to(device, non_blocking=True)  # [G, Dg]
            uniq_go_embs = self.normalizer(uniq_go_embs, dim=1)

        uniq_go_ids = batch["uniq_go_ids"].to(device, non_blocking=True)  # [G]
        Dg_batch = int(uniq_go_embs.size(1))
        self._maybe_init_queue_miner(Dg_batch)

        # ========== TEACHER PROT QUERY ==========
        # Teacher vektörü (fused bank varsa oradan), yoksa residue-based teacher.
        with torch.no_grad():
            prot_ids = batch.get("protein_ids", None)
            vt_for_miner = None
            if prot_ids is not None and getattr(self.ctx, "fused_bank", None) is not None:
                id2row = self.ctx.fused_bank["id2row"]  # CPU dict
                rows = [id2row.get(pid, None) for pid in prot_ids]
                if all(r is not None for r in rows) and len(rows) > 0:
                    vecs_cpu = self.ctx.fused_bank["vecs"]  # CPU tensor [N, D?]
                    idx_cpu = torch.as_tensor(rows, dtype=torch.long, device=vecs_cpu.device)
                    vt_for_miner = vecs_cpu.index_select(0, idx_cpu).to(device, non_blocking=True)
                    if vt_for_miner.size(1) != Dg_batch and hasattr(self.ctx.vres, "project_queries_to_index"):
                        try:
                            vt_for_miner = self.ctx.vres.project_queries_to_index(vt_for_miner)
                        except Exception:
                            pass
                    vt_for_miner = self.normalizer(vt_for_miner, dim=1)

            if vt_for_miner is None:
                v_true_early = self.ctx.vres.true_prot_vecs(H, attn_valid)  # [B, Dh] GPU
                vt_for_miner = v_true_early
                if hasattr(self.ctx.vres, "project_queries_to_index"):
                    try:
                        vt_for_miner = self.ctx.vres.project_queries_to_index(v_true_early)  # [B, Dg]
                    except Exception:
                        pass
                if vt_for_miner.size(1) != Dg_batch:
                    raise RuntimeError(f"prot_query dim mismatch: got {vt_for_miner.size(1)}, expected {Dg_batch}.")
                vt_for_miner = self.normalizer(vt_for_miner, dim=1)
        prot_query = vt_for_miner  # [B, Dg] (GPU)

        # ========== QUEUE MINER: hard negatives (chunklı ve no_grad) ==========
        neg_from_queue = None
        neg_ids_from_queue = None
        if self.queue_miner is not None:
            with torch.no_grad():
                res = self.queue_miner.get_all_neg()  # -> (vecs[K,Dg], ids[K]) veya None
                if res is not None:
                    all_neg_vecs_cpu, all_neg_ids = res  # CPU bekliyoruz
                    if all_neg_vecs_cpu is not None and all_neg_vecs_cpu.numel() > 0:
                        if all_neg_vecs_cpu.size(1) != Dg_batch:
                            raise RuntimeError(f"Queue D mismatch: {all_neg_vecs_cpu.size(1)} vs expected {Dg_batch}")

                        # Benzerlikleri K çok büyükken GPU’ya kısım kısım taşı.
                        K_all = all_neg_vecs_cpu.shape[0]
                        sim_chunks = []
                        # chunk büyüklüğü: VRAM güvenli (ör. 8192)
                        qk = int(getattr(self.cfg, "queue_sim_chunk", 8192))
                        Q = self.normalizer(prot_query.detach(), dim=1)  # [B,Dg] GPU
                        for ks in range(0, K_all, qk):
                            ke = min(K_all, ks + qk)
                            K_part = all_neg_vecs_cpu[ks:ke].to(device, non_blocking=True)  # [k,Dg] GPU
                            K_part = self.normalizer(K_part, dim=1)
                            sim_chunks.append(Q @ K_part.T)  # [B,k] GPU
                            del K_part
                        sims = torch.cat(sim_chunks, dim=1)  # [B,K_all] GPU

                        # False-negative mask
                        if all_neg_ids is not None:
                            pos_gid_sets = []
                            for b, loc in enumerate(pos_local):
                                if loc.numel() > 0:
                                    gids = uniq_go_ids.index_select(0, loc.to(uniq_go_ids.device))
                                    pos_gid_sets.append(set(map(int, gids.tolist())))
                                else:
                                    pos_gid_sets.append(set())
                            mask = torch.zeros_like(sims, dtype=torch.bool)  # GPU
                            ids_list = all_neg_ids.tolist()
                            for b in range(B):
                                if pos_gid_sets[b]:
                                    for j, gid in enumerate(ids_list):
                                        if gid in pos_gid_sets[b]:
                                            mask[b, j] = True
                            sims = sims.masked_fill(mask, float('-inf'))

                        k = min(int(self.k_hard_queue), K_all)
                        if k > 0:
                            _, idx = sims.topk(k, dim=1)  # [B,k] GPU
                            # --- güvenli ve cihaz tutarlı seçim ---
                            neg_from_queue = []
                            neg_ids_from_queue = []
                            for b in range(B):
                                neg_idx_b = idx[b]  # [k] (GPU)
                                # CPU bank’tan güvenli seçim (idx → CPU long, sonra seç, sonra CPU'da bırak)
                                take_vecs_b = self._cpu_bank_gather(all_neg_vecs_cpu, neg_idx_b,
                                                                    dev="cpu")  # [k,Dg] CPU

                                # id’leri CPU long yap
                                if neg_idx_b.device.type != "cpu":
                                    neg_idx_cpu = neg_idx_b.to("cpu", non_blocking=True)
                                else:
                                    neg_idx_cpu = neg_idx_b
                                if neg_idx_cpu.dtype != torch.long:
                                    neg_idx_cpu = neg_idx_cpu.long()

                                take_ids_b = all_neg_ids.index_select(0, neg_idx_cpu)  # [k] CPU long

                                neg_from_queue.append(take_vecs_b)
                                neg_ids_from_queue.append(take_ids_b)

                            # [B,k,Dg] CPU / [B,k] CPU
                            neg_from_queue = torch.stack(neg_from_queue, dim=0)
                            neg_ids_from_queue = torch.stack(neg_ids_from_queue, dim=0)
                        del sims, sim_chunks
                        torch.cuda.empty_cache()

        # ========== Easy negatives (batch içinden) ==========
        easy_ids = None
        easy_vecs = None
        hard_frac = None
        if getattr(self, "curriculum_params", None) is not None:
            src = getattr(self.curriculum_params, "cfg", self.curriculum_params)
            hard_frac = getattr(src, "hard_frac", None)

        if (hard_frac is not None) and (neg_from_queue is not None):
            with torch.no_grad():
                all_neg_pool = self.normalizer(uniq_go_embs.detach(), dim=1)  # [G,Dg] GPU
                sims_all = self.normalizer(prot_query, dim=1) @ all_neg_pool.T  # [B,G] GPU
                for b, loc in enumerate(pos_local):
                    if loc.numel() > 0:
                        sims_all[b, loc.to(sims_all.device)] = -1e9
                k_easy = max(0, int(neg_from_queue.size(1) * (1 - float(hard_frac))))
                if k_easy > 0:
                    _, idx_easy = sims_all.topk(k_easy, dim=1)  # [B,k_easy] GPU
                    # CPU'ya çek
                    easy_vecs = []
                    easy_ids = []
                    idx_easy_list = idx_easy.tolist()
                    UG_cpu = uniq_go_embs.detach().to("cpu")
                    UID_cpu = uniq_go_ids.detach().to("cpu")
                    for b in range(B):
                        take_idx = torch.tensor(idx_easy_list[b], dtype=torch.long)  # CPU long
                        easy_vecs.append(UG_cpu.index_select(0, take_idx))
                        easy_ids.append(UID_cpu.index_select(0, take_idx))
                    easy_vecs = torch.stack(easy_vecs, dim=0)  # [B,k_easy,Dg] CPU
                    easy_ids = torch.stack(easy_ids, dim=0)  # [B,k_easy]    CPU

                del sims_all

        # ========== GLOBAL CANDIDATE POOL (tamamen CPU) ==========
        # 1) Batch pozitfileri (CPU'ya)
        batch_pos_ids_cpu = uniq_go_ids.detach().to("cpu")  # [G]
        batch_pos_vecs_cpu = self.normalizer(uniq_go_embs, dim=1).detach().to("cpu")  # [G,Dg]

        # 2) Seçilmiş negatifler (queue + easy) → unique id'ye göre tekilleştir
        id2vec = {}
        if neg_ids_from_queue is not None and neg_from_queue is not None:
            flat_ids = neg_ids_from_queue.reshape(-1).tolist()
            flat_vecs = neg_from_queue.reshape(-1, neg_from_queue.size(-1))
            for gid, v in zip(flat_ids, flat_vecs):
                id2vec[int(gid)] = v  # CPU tensor

        if (easy_ids is not None) and (easy_vecs is not None):
            flat_ids_e = easy_ids.reshape(-1).tolist()
            flat_vecs_e = easy_vecs.reshape(-1, easy_vecs.size(-1))
            for gid, v in zip(flat_ids_e, flat_vecs_e):
                id2vec[int(gid)] = v  # CPU tensor

        if len(id2vec) > 0:
            all_neg_ids_selected_cpu = torch.tensor(sorted(id2vec.keys()), dtype=torch.long)
            neg_vecs_unique_cpu = torch.stack([id2vec[k] for k in all_neg_ids_selected_cpu.tolist()], dim=0)
        else:
            all_neg_ids_selected_cpu = torch.empty(0, dtype=torch.long)
            neg_vecs_unique_cpu = torch.empty(0, batch_pos_vecs_cpu.size(1))

        # 3) Global havuz (CPU)
        G_ids_global_cpu = torch.cat([batch_pos_ids_cpu, all_neg_ids_selected_cpu], dim=0)  # [K]
        G_vecs_global_cpu = torch.cat([batch_pos_vecs_cpu, neg_vecs_unique_cpu], dim=0)  # [K,Dg]
        K_global = int(G_vecs_global_cpu.size(0))

        # 4) Per-row multi-pozitif maskesi: [B,K] (GPU'da lazım)
        id2col = {int(G_ids_global_cpu[i].item()): i for i in range(K_global)}
        pos_mask = torch.zeros(B, K_global, dtype=torch.bool, device=device)
        for b, loc in enumerate(pos_local):
            if loc.numel() == 0:
                continue
            glob = batch["uniq_go_ids"].index_select(0, loc.to(batch["uniq_go_ids"].device)).tolist()
            for g in glob:
                j = id2col.get(int(g), None)
                if j is not None:
                    pos_mask[b, j] = True

        # 5) Skorlar (CANDIDATES) — G_cand CPU, forward_scores chunk’lı
        # CPU'da [B,K,Dg] görünümü oluşturalım; bu "expand" kopyasız view’dur; GPU’ya chunk ile taşınacak.
        G_cand_cpu = G_vecs_global_cpu.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B,K,Dg] CPU

        amp_ctx = torch.amp.autocast(device_type='cuda',
                                     enabled=(torch.cuda.is_available() and self.ctx.fp16_enabled)) \
            if torch.cuda.is_available() else nullcontext()

        with amp_ctx:
            scores_cand = self.forward_scores(
                H, G_cand_cpu, pad_mask,
                return_alpha=False,
                cand_chunk_k=int(getattr(self.cfg, "cand_chunk_k", 32)),
                pos_chunk_t=int(getattr(self.cfg, "pos_chunk_t", 128)),
            )  # (B,K) GPU
            scale = self.logit_scale.exp().clamp(max=100.0)
            scores_cand = scores_cand * scale

            # InfoNCE (multi-positive)
            l_con = multi_positive_infonce_from_candidates(scores_cand, pos_mask, tau=1.0)

            # ---- KL distillation (teacher → student) ----
            lambda_v = float(getattr(self.attr, "lambda_vtrue", 0.0))
            if lambda_v > 0.0:
                with torch.no_grad():
                    Gn_cpu = self.normalizer(G_vecs_global_cpu, dim=1)  # [K,Dg] CPU
                # Öğrenci logits (scores_cand) zaten üstte var; teacher p_t'yi üret:
                vt = self.normalizer(prot_query, dim=1)  # [B,Dg] GPU
                # Gn'yi GPU'ya parça parça taşı ve teacher skorlarını oluştur
                tau = float(self.attr.tau_distill)
                bk = int(getattr(self.cfg, "teacher_chunk_k", 8192))
                scores_teacher_parts = []
                with torch.no_grad():
                    for s in range(0, K_global, bk):
                        e = min(K_global, s + bk)
                        Gn_part = Gn_cpu[s:e].to(device, non_blocking=True)  # [k,Dg] GPU
                        scores_teacher_parts.append(vt @ Gn_part.T)  # [B,k]
                        del Gn_part
                    scores_teacher = torch.cat(scores_teacher_parts, dim=1)  # [B,K]
                    p_t = F.softmax(scores_teacher / tau, dim=1)
                log_p_s = F.log_softmax(scores_cand / tau, dim=1)
                l_con_teacher = F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)
                del scores_teacher, scores_teacher_parts, p_t
            else:
                l_con_teacher = torch.zeros((), device=device)

        # ========== POSITIVES-ONLY (küçük T) ==========
        # Attribution veya DAG için gereken küçük pozitif matrisi
        T_max = max((int(x.numel()) for x in pos_local), default=1)
        Dg = uniq_go_embs.size(1)
        G_pos = torch.zeros(B, T_max, Dg, device=device, dtype=uniq_go_embs.dtype)
        for b, loc in enumerate(pos_local):
            t = int(loc.numel())
            if t > 0:
                G_pos[b, :t] = uniq_go_embs.index_select(0, loc.to(uniq_go_embs.device))
        if self.to_f32 is not None:
            G_pos = self.to_f32(G_pos)

        if use_attr:
            scores_pos, alpha_info = self.forward_scores(H, G_pos, pad_mask, return_alpha=True)
        else:
            scores_pos = self.forward_scores(H, G_pos, pad_mask, return_alpha=False)
            alpha_info = {}

        # ========== Attribution & Entropy ==========
        if alpha_info is not None and "alpha_full" in alpha_info:
            alpha = alpha_info["alpha_full"]
            if use_attr:
                delta, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, pad_mask)
                l_attr = attribution_loss(alpha, delta, mask=None, reduce="mean")
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(alpha)
            else:
                l_attr = torch.zeros((), device=device);
                l_ent = torch.zeros((), device=device)
        elif alpha_info is not None and all(k in alpha_info for k in ["alpha_windows", "win_weights", "spans"]):
            AW = alpha_info["alpha_windows"];
            Ww = alpha_info["win_weights"];
            spans = alpha_info["spans"]
            if use_attr:
                delta_full, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, pad_mask)
                delta_win = torch.zeros_like(AW)
                for wi, (s, e) in enumerate(spans):
                    delta_win[:, :, wi, :e - s] = delta_full[:, :, s:e]
                l_attr = windowed_attr_loss(AW, Ww, spans, delta_win)
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(AW)
            else:
                l_attr = torch.zeros((), device=device);
                l_ent = torch.zeros((), device=device)
        else:
            try:
                self.wlogger.log_scalar({"warn/no_alpha_info": 1}, step=self._global_step)
            except Exception:
                pass
            l_attr = torch.zeros((), device=device);
            l_ent = torch.zeros((), device=device)

        # ========== DAG consistency ==========
        l_dag = dag_consistency_loss_pos(scores_pos, pos_local, uniq_go_ids, self.ctx.dag_parents,
                                         margin=0.0, scale=1.0)

        total = (l_con + float(getattr(self.attr, "lambda_vtrue", 0.0)) * l_con_teacher) \
                + 0.3 * l_dag + self.attr.lambda_attr * l_attr + l_ent

        if not torch.isfinite(total):
            raise RuntimeError(f"NaN/Inf loss at step {self._global_step}")

        # ========== EMA update ==========
        self._global_step += 1
        if self.go_encoder_k is not None:
            with torch.no_grad():
                ema_update(self.model.go_encoder, self.go_encoder_k, m=self.m_ema)

        # ========== QUEUE ENQUEUE (CPU uyumlu) ==========
        if self.queue_miner is not None:
            with torch.no_grad():
                if ("pos_go_tokens" in batch) and (self.go_encoder_k is not None):
                    toks = batch["pos_go_tokens"]
                    go_pos_all = self.go_encoder_k(
                        input_ids=toks["input_ids"].to(device, non_blocking=True),
                        attention_mask=toks["attention_mask"].to(device, non_blocking=True),
                    )
                    go_pos_all = self.normalizer(go_pos_all, dim=1)
                else:
                    go_pos_all = self.normalizer(uniq_go_embs, dim=1)
                # Batch'teki tüm pozitif local index'leri tekilleştir
                local_idx_list = [loc.to(device) for loc in pos_local if loc.numel() > 0]
                if local_idx_list:
                    local_cat = torch.unique(torch.cat(local_idx_list, dim=0))
                    pos_vecs = go_pos_all.index_select(0, local_cat).detach().to("cpu")  # CPU
                    pos_ids = uniq_go_ids.index_select(0, local_cat).detach().to("cpu")
                    self.queue_miner.enqueue(pos_vecs, pos_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0

        # ====== Quick retrieval metrics (GPU'da hafif) ======
        with torch.no_grad():
            if 'scores_cand' in locals() and scores_cand.numel() > 1:
                topk = min(5, scores_cand.size(1))
                topk_idx = scores_cand.topk(topk, dim=1).indices  # (B, topk)
                hit1 = pos_mask.gather(1, topk_idx[:, :1]).any(dim=1).float().mean().item()
                hit5 = pos_mask.gather(1, topk_idx).any(dim=1).float().mean().item()
                ranks = torch.argsort(torch.argsort(-scores_cand, dim=1), dim=1)  # 0=best
                pos_ranks = ranks[pos_mask].float()
                mean_pos_rank = pos_ranks.mean().item() if pos_ranks.numel() else float('nan')
                try:
                    pos_scores = scores_cand[pos_mask]
                    neg_scores = scores_cand[~pos_mask]
                    hard_gap = (
                                pos_scores.mean() - neg_scores.mean()).item() if pos_scores.numel() and neg_scores.numel() else float(
                        'nan')
                except Exception:
                    hard_gap = float('nan')
            else:
                hit1 = hit5 = mean_pos_rank = hard_gap = float('nan')

        # ====== Logging (hafif) ======
        losses_log = {
            "total": float(total.detach().item()),
            "contrastive": float(l_con.detach().item()),
            "kl_teacher": float(l_con_teacher.detach().item()),
            "hierarchy": float(l_dag.detach().item()),
            "attr": float(l_attr.detach().item()),
            "entropy": float(l_ent.detach().item()),
            "retrieval/recall@1": hit1,
            "retrieval/recall@5": hit5,
            "retrieval/mean_pos_rank": mean_pos_rank,
            "retrieval/hard_gap": hard_gap,
            "perf/step_sec": dt,
            "perf/samples_per_sec": H.size(0) / dt
        }
        sched = None
        if getattr(self, "curriculum_params", None) is not None:
            cp = self.curriculum_params
            source = getattr(cp, "cfg", cp)
            sched = dict(
                hard_frac=float(getattr(source, "hard_frac", getattr(source, "hard_frac_start", 0.0))) if hasattr(
                    source, "hard_frac") or hasattr(source, "hard_frac_start") else None,
                k_hard=int(getattr(source, "k_hard", 0)) if hasattr(source, "k_hard") else None,
            )
        phase = dict(
            id=getattr(self, "phase_id", getattr(self.ctx, "phase_id", -1)),
            name=getattr(self, "phase_name", getattr(self.ctx, "phase_name", f"phase{getattr(self, 'phase_id', -1)}"))
        )
        self.wlogger.log_losses(losses_log, step=self._global_step, epoch=epoch_idx, phase=phase, sched=sched)

        pid = int(phase["id"]) if isinstance(phase.get("id", -1), (int,)) else -1
        for k, v in losses_log.items():
            self._phase_acc[pid][f"loss/{k}"].append(v)
        if sched:
            for k, v in sched.items():
                if v is not None:
                    self._phase_acc[pid][f"sched/{k}"].append(float(v))

        # ====== Return ======
        return {
            "total": total,
            "contrastive": l_con,
            "contrastive_teacher": l_con_teacher,
            "dag": l_dag,
            "attr": l_attr,
            "entropy": l_ent
        }
    @torch.no_grad()
    def eval_epoch(self, loader, epoch_idx: int):
        self.model.eval()
        device = self.device
        logs = {"total": 0.0, "contrastive": 0.0, "dag": 0.0, "attr": 0.0, "entropy": 0.0}
        n = 0

        all_pred_blocks: List[torch.Tensor] = []
        all_true_blocks: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                # --- inputs & masks ---
                H = batch["prot_emb_pad"].to(device)  # (B, Lmax, Dh)
                if self.to_f32 is not None:
                    H = self.to_f32(H)
                attn_valid, pad_mask = self._valid_and_pad_masks(batch)
                pos_local: List[torch.Tensor] = batch["pos_go_local"]

                # --- POSITIVE GO embs ---
                if ("pos_go_tokens" in batch) and (hasattr(self.model, "go_encoder") and self.model.go_encoder is not None):
                    toks = batch["pos_go_tokens"]
                    uniq_go_embs = self.model.go_encoder(
                        input_ids=toks["input_ids"].to(device),
                        attention_mask=toks["attention_mask"].to(device),
                    )  # [G,Dg]
                    uniq_go_embs = self.normalizer(uniq_go_embs, dim=1)
                else:
                    uniq_go_embs = batch["uniq_go_embs"].to(device)  # [G,Dg]
                    uniq_go_embs = self.normalizer(uniq_go_embs, dim=1)

                uniq_go_ids = batch["uniq_go_ids"].to(device)  # [G]
                Dg_batch = int(uniq_go_embs.size(1))

                # --- Queue negatives in eval (no enqueue) ---
                neg_from_queue = None
                if self.queue_miner is not None:
                    v_true_early = self.ctx.vres.true_prot_vecs(H, attn_valid)  # [B, Dh]
                    vt_for_miner = v_true_early
                    if hasattr(self.ctx.vres, "project_queries_to_index"):
                        try:
                            vt_for_miner = self.ctx.vres.project_queries_to_index(v_true_early)  # [B, Dg]
                        except Exception:
                            pass
                    if vt_for_miner.size(1) != Dg_batch:
                        raise RuntimeError(f"[eval] prot_query dim mismatch: {vt_for_miner.size(1)} vs {Dg_batch}")
                    vt_for_miner = self.normalizer(vt_for_miner, dim=1)

                    res = self.queue_miner.get_all_neg()
                    if res is not None:
                        all_neg_vecs, all_neg_ids = res
                        if all_neg_vecs is not None and all_neg_vecs.numel() > 0:
                            if all_neg_vecs.size(1) != Dg_batch:
                                raise RuntimeError(f"[eval] Queue D mismatch: {all_neg_vecs.size(1)} vs {Dg_batch}")
                            Q = self.normalizer(vt_for_miner, dim=1)                 # [B,Dg]
                            Kmat = self.normalizer(all_neg_vecs.to(device), dim=1)   # [K,Dg]
                            sims = Q @ Kmat.T                                         # [B,K]

                            if all_neg_ids is not None:
                                pos_gid_sets = []
                                for b, loc in enumerate(pos_local):
                                    if loc.numel() > 0:
                                        gids = uniq_go_ids.index_select(0, loc.to(uniq_go_ids.device))
                                        pos_gid_sets.append(set(map(int, gids.tolist())))
                                    else:
                                        pos_gid_sets.append(set())
                                mask = torch.zeros_like(sims, dtype=torch.bool)
                                ids_list = all_neg_ids.tolist()
                                for b in range(H.size(0)):
                                    if pos_gid_sets[b]:
                                        for j, gid in enumerate(ids_list):
                                            if gid in pos_gid_sets[b]:
                                                mask[b, j] = True
                                sims = sims.masked_fill(mask, float('-inf'))

                            k = min(int(self.k_hard_queue), Kmat.size(0))
                            if k > 0:
                                _, idx = sims.topk(k, dim=1)
                                neg_from_queue = Kmat.index_select(0, idx.reshape(-1)) \
                                                   .reshape(idx.size(0), k, -1).contiguous()

                # --- Candidates ---
                G_cand, pos_mask = self._build_candidates_with_queue(
                    uniq_go_embs=uniq_go_embs,
                    pos_go_local=pos_local,
                    neg_from_queue=neg_from_queue
                )

                # --- Scores & InfoNCE (eval) ---
                scores_cand = self.forward_scores(H, G_cand, pad_mask, return_alpha=False)  # (B,K)
                l_con = multi_positive_infonce_from_candidates(scores_cand, pos_mask, tau=float(self.attr.temperature))

                # --- Positives-only for attribution & DAG ---
                B = H.size(0)
                T_max = max((int(x.numel()) for x in pos_local), default=1)
                Dg = uniq_go_embs.size(1)
                G_pos = torch.zeros(B, T_max, Dg, device=device, dtype=uniq_go_embs.dtype)
                for b, loc in enumerate(pos_local):
                    t = int(loc.numel())
                    if t > 0:
                        G_pos[b, :t] = uniq_go_embs.index_select(0, loc.to(uniq_go_embs.device))

                use_attr_eval = (self.attr.lambda_attr > 0.0)
                if use_attr_eval:
                    scores_pos, alpha_info = self.forward_scores(H, G_pos, pad_mask, return_alpha=False)
                else:
                    scores_pos = self.forward_scores(H, G_pos, pad_mask, return_alpha=False)
                    alpha_info = {}

                if alpha_info is not None and ("alpha_full" in alpha_info):
                    alpha = alpha_info["alpha_full"]
                    delta = topk_maskout_full(H, G_pos, alpha, k=self.attr.topk_per_window,
                                              model=self.model, pad_mask=pad_mask)
                    l_attr = attribution_loss(alpha, delta, mask=None, reduce="mean")
                    l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(alpha)
                elif (alpha_info is not None) and all(k in alpha_info for k in ["alpha_windows", "win_weights", "spans"]):
                    AW = alpha_info["alpha_windows"]; Ww = alpha_info["win_weights"]; spans = alpha_info["spans"]
                    delta_full, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, pad_mask)
                    delta_win = torch.zeros_like(AW)
                    for wi, (s, e) in enumerate(spans):
                        delta_win[:, :, wi, :e - s] = delta_full[:, :, s:e]
                    l_attr = windowed_attr_loss(AW, Ww, spans, delta_win)
                    l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(AW)
                else:
                    l_attr = torch.zeros((), device=device); l_ent = torch.zeros((), device=device)

                l_dag = dag_consistency_loss_pos(scores_pos, pos_local, uniq_go_ids, self.ctx.dag_parents,
                                                 margin=0.0, scale=1.0)

                total = l_con + 0.3 * l_dag + self.attr.lambda_attr * l_attr + l_ent

                # ---- CAFA eval space & predictions ----
                G_eval, y_true_b, _ = self._build_eval_space(batch)  # (B,Keval,Dg)
                G_eval = G_eval.to(device)
                scores_eval = self.forward_scores(H, G_eval, pad_mask, return_alpha=False)
                probs_eval = torch.sigmoid(scores_eval)

                all_pred_blocks.append(probs_eval.detach().cpu())
                all_true_blocks.append(y_true_b.detach().cpu())

                logs["total"] += float(total.detach().item())
                logs["contrastive"] += float(l_con.detach().item())
                logs["dag"] += float(l_dag.detach().item())
                logs["attr"] += float(l_attr.detach().item())
                logs["entropy"] += float(l_ent.detach().item())
                n += 1

        # ---- average logs ----
        for k in logs:
            logs[k] /= max(1, n)

        # ---- CAFA metrics ----
        if all_pred_blocks:
            y_pred = torch.cat(all_pred_blocks, dim=0).numpy()
            y_true = torch.cat(all_true_blocks, dim=0).numpy()
            fmax, _ = compute_fmax(y_true, y_pred)
            aupr = compute_term_aupr(y_true, y_pred)
            logs["cafa_fmax"] = float(fmax)
            logs["cafa_aupr"] = float(aupr)
        else:
            logs["cafa_fmax"] = 0.0
            logs["cafa_aupr"] = 0.0

        return logs