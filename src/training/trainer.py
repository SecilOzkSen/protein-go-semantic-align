# src/training/trainer.py
from typing import Dict, Any, List, Tuple, Optional
import copy
import torch
import torch.nn.functional as F
import time

from src.models.alignment_model import ProteinGoAligner
from src.loss.attribution import attribution_loss, windowed_attr_loss
from src.configs.data_classes import TrainerConfig, AttrConfig
from src.miners.queue_miner import QueueMiner

# Wandb
import wandb
from collections import defaultdict
from src.utils.wandb_logger import WabLogger

from src.metrics.cafa import compute_fmax, compute_term_aupr


# ------------- Helpers -------------
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

def mean_pool(pad: torch.Tensor, mask_valid: torch.Tensor) -> torch.Tensor:
    m = mask_valid.float()
    return (pad * m.unsqueeze(-1)).sum(1) / m.sum(1).clamp_min(1.0).unsqueeze(-1)

def entropy_regularizer(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = alpha.clamp_min(eps)
    ent = -(a * a.log()).sum(dim=-1)
    return ent.mean()

def multi_positive_infonce_from_candidates(scores: torch.Tensor, pos_mask: torch.Tensor, tau: float) -> torch.Tensor:
    """
    scores: (B, K)  (positives + negatives)
    pos_mask: (B, K) boolean; True at positives
    NaN-safe when a row has 0 positives.
    """
    logits = scores / max(1e-8, tau)                    # (B, K)
    denom = torch.logsumexp(logits, dim=-1)             # (B,)
    pos_logits = logits.masked_fill(~pos_mask, float('-inf'))
    pos_any = pos_mask.any(dim=1)
    # rows with no positives: use a very small finite value to avoid -inf
    if (~pos_any).any():
        pos_logits = pos_logits.clone()
        pos_logits[~pos_any] = -1e9
    num = torch.logsumexp(pos_logits, dim=-1)           # (B,)
    loss = -(num - denom)                               # (B,)
    return loss[pos_any].mean() if pos_any.any() else denom.mean()*0.0

def surrogate_delta_y_from_mask_grad(H, G, model, pad_mask=None) -> torch.Tensor:
    """
    Quick surrogate for attribution: use ||dy/dH|| as importance proxy.
    Returns proxy deltas (broadcast per target) and alpha_info from the forward pass.
    """
    H = H.clone().detach().requires_grad_(True)
    scores, alpha_info = model(H, G, mask=pad_mask, return_alpha=True)
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
    base_scores, _ = model(H, G, mask=pad_mask, return_alpha=True)  # (B, T)

    for b in range(B):
        for t in range(T):
            topk = min(k, L)
            _, idx = torch.topk(alpha_full[b, t], k=topk, dim=-1)
            for i in idx.tolist():
                Hminus = H.clone()
                Hminus[b, i, :] = 0.0
                y_minus, _ = model(
                    Hminus,
                    G[b:b+1],
                    mask=pad_mask[b:b+1] if pad_mask is not None else None,
                    return_alpha=True
                )
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

        self.model = ProteinGoAligner(
            d_h=cfg.d_h,
            d_g=None,
            d_z=cfg.d_z,          # e.g., 768
            go_encoder=go_encoder,
            normalize=True,
        ).to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self._global_step = 0

        # projector for teacher alignment
        self.index_projector = copy.deepcopy(self.model.proj_p).eval().requires_grad_(False)
        self.ctx.vres.align_dim = self.cfg.d_z
        self.ctx.vres.query_projector = self.index_projector

        # --- Queue miner (FAISS tamamen kaldırıldı) ---
        self.use_queue_miner: bool = True
        self.queue_K: int = int(getattr(cfg, "queue_K", getattr(ctx, "queue_K", 8192)))
        self.k_hard_queue: int = int(getattr(cfg, "k_hard", getattr(ctx, "k_hard", 32)))
        self.queue_miner: Optional[QueueMiner] = None

        # W&B
        self._wandb_configs(wandb_run=wandb_run, wlogger=wlogger)
        self._phase_acc = defaultdict(lambda: defaultdict(list))
        self._current_phase_id = None

        self._mb_embs = getattr(self.ctx, "memory_bank", None)
        if self._mb_embs is not None:
            self._mb_embs = self.ctx.memory_bank.embs.to(self.device)

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
                reinit=False
            )
        if wlogger is not None:
            self.wlogger = wlogger
        else:
            self.wlogger = WabLogger(self.wandb_run, project="protein-go-semantic-align",
                                     config=self.ctx.to_dict())
        try:
            wandb.watch(self.model, log="gradients", log_freq=max(100, getattr(self.cfg, "log_every", 50)))
        except Exception:
            pass

    # --------- Wandb helpers (phase bookkeeping) ---------
    def _on_phase_change(self, new_phase_id: int, new_phase_name: str, step: int):
        if self._current_phase_id is not None:
            self._flush_phase_table(self._current_phase_id, step)
        self._current_phase_id = new_phase_id
        self._phase_acc[new_phase_id].clear()
        wandb.log({"phase/change_to": new_phase_id, "phase/name": new_phase_name}, step=step)

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
        if self.queue_miner is None:
            self.queue_miner = QueueMiner(dim=int(Dg), K=int(self.queue_K), device=str(self.device))
            print(f"[Trainer] QueueMiner enabled (K={self.queue_K}, k_hard={self.k_hard_queue}, Dg={Dg}).")

    def _build_candidates_with_queue(self,
                                     uniq_go_embs: torch.Tensor,
                                     pos_go_local: List[torch.Tensor],
                                     neg_from_queue: Optional[torch.Tensor]
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Positives from batch, negatives from QueueMiner.
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
            mb = self._mb_embs if self._mb_embs is not None else self.ctx.memory_bank.embs.to(device)
            G_eval_once = mb.index_select(0, rows)  # (Geval, Dg)
        else:
            eval_ids = batch["uniq_go_ids"].to(device)  # (Geval,)
            if ("pos_go_tokens" in batch) and (hasattr(self.model, "go_encoder") and self.model.go_encoder is not None):
                toks = batch["pos_go_tokens"]
                G_eval_once = self.model.go_encoder(
                    input_ids=toks["input_ids"].to(device),
                    attention_mask=toks["attention_mask"].to(device),
                )
                G_eval_once = F.normalize(G_eval_once, p=2, dim=1)
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

    # --------- Projection layer updates ---------
    @torch.no_grad()
    def update_ema(self, dst, src, m=0.995):
        for pd, ps in zip(dst.parameters(), src.parameters()):
            pd.mul_(m).add_(ps, alpha=1 - m)

    # --------- Scoring (tensor path) ---------
    def forward_scores(self, H, G, pad_mask, **kwargs):
        return_alpha = kwargs.pop("return_alpha", False)
        return self.model(H=H, G=G, mask=pad_mask, return_alpha=return_alpha, **kwargs)

    # --------- Training step (losses + logging) ---------
    def step_losses(self, batch: Dict[str, Any], epoch_idx: int) -> Dict[str, torch.Tensor]:
        t0 = time.time()
        self.model.train()
        device = self.device

        # --- Protein & mask ---
        H = batch["prot_emb_pad"].to(device)               # (B, Lmax, Dh)
        attn_valid, pad_mask = self._valid_and_pad_masks(batch)
        pos_local: List[torch.Tensor] = batch["pos_go_local"]
        B = H.size(0)

        # Use attribution loss only during early curriculum epochs
        use_attr = (epoch_idx < self.attr.curriculum_epochs and self.attr.lambda_attr > 0.0)
        print("[Trainer] use_attr =", use_attr, f"(epoch {epoch_idx} < {self.attr.curriculum_epochs})")

        # --- POSITIVES (GoEncoder → uniq_go_embs) ---
        if ("pos_go_tokens" in batch) and (hasattr(self.model, "go_encoder") and self.model.go_encoder is not None):
            toks = batch["pos_go_tokens"]
            uniq_go_embs = self.model.go_encoder(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            )  # [G, Dg]
            try:
                uniq_go_embs = F.normalize(uniq_go_embs, p=2, dim=1)
            except Exception:
                pass
        else:
            uniq_go_embs = batch["uniq_go_embs"].to(device)

        # --- Queue miner init (once we know Dg) ---
        self._maybe_init_queue_miner(uniq_go_embs.size(1))

        # --- Curriculum knobs (k_hard & optional hard_frac) ---
        hard_frac = None
        if getattr(self.ctx, "scheduler", None) is not None:
            curr = self.ctx.scheduler(self._global_step)
            src = getattr(curr, "cfg", curr)
            kh = getattr(src, "k_hard", None)
            if kh is not None:
                self.k_hard_queue = int(kh)
            hard_frac = getattr(src, "hard_frac", None)

            # Phase logging
            new_phase_id = getattr(src, "phase_id", getattr(src, "phase", None))
            new_phase_name = getattr(src, "phase_name", getattr(src, "name", None))
            if new_phase_id is not None and new_phase_id != self._current_phase_id:
                if new_phase_name is None:
                    new_phase_name = f"phase{int(new_phase_id)}"
                self._on_phase_change(int(new_phase_id), str(new_phase_name), step=self._global_step)
            self.curriculum_params = curr

        # --- prot_query = vt (teacher) projected to GO space ---
        v_true_early = self.ctx.vres.true_prot_vecs(H, attn_valid)        # [B, Dh]
        vt_for_miner = v_true_early
        try:
            vt_for_miner = self.ctx.vres.project_queries_to_index(v_true_early)  # [B, Dg]
        except Exception:
            pass
        prot_query = vt_for_miner

        # --- get hard negatives from queue ---
        neg_from_queue = self.queue_miner.get_negatives(
            prot_query.detach(), k_hard=self.k_hard_queue
        ) if (self.queue_miner is not None) else None

        # --- (optional) mix easy in-batch negatives by hard_frac ---
        if neg_from_queue is not None and hard_frac is not None:
            all_neg = F.normalize(uniq_go_embs.detach(), dim=1)              # [G, Dg]
            sims = F.normalize(prot_query, dim=1) @ all_neg.T                # [B, G]
            # mask positives
            for b, loc in enumerate(pos_local):
                if loc.numel() > 0:
                    sims[b, loc.to(sims.device)] = -1e9
            k_easy = max(0, int(neg_from_queue.size(1) * (1 - float(hard_frac))))
            if k_easy > 0:
                _, idx_easy = sims.topk(k_easy, dim=1)
                easy_embs = all_neg.index_select(0, idx_easy.reshape(-1)).reshape(idx_easy.size(0), idx_easy.size(1), -1)
                neg_from_queue = torch.cat([neg_from_queue, easy_embs], dim=1)

        # --- Candidate set (positives + queue negatives) ---
        G_cand, pos_mask = self._build_candidates_with_queue(
            uniq_go_embs=uniq_go_embs,
            pos_go_local=pos_local,
            neg_from_queue=neg_from_queue
        )

        # ====================== Teacher (v_true) → KL DISTILLATION ======================
        v_true = v_true_early
        vt = vt_for_miner
        vt = F.normalize(vt, dim=-1)                 # [B, Dg]
        Gn = F.normalize(G_cand, dim=-1)             # [B, K, Dg]

        # ---- Student scores + InfoNCE ----
        scores_cand = self.forward_scores(H, G_cand, pad_mask, return_alpha=False)  # (B, K)
        l_con = multi_positive_infonce_from_candidates(scores_cand, pos_mask, self.attr.temperature)

        # ---- KL distillation (teacher → student) ----
        lambda_v = float(getattr(self.attr, "lambda_vtrue", 0.0))
        if lambda_v > 0.0: #KL Distillation active
            scores_teacher = torch.einsum("bd,bkd->bk", vt, Gn)
            tau = float(self.attr.tau_distill)
            with torch.no_grad():
                p_t = F.softmax(scores_teacher / tau, dim=1)
            log_p_s = F.log_softmax(scores_cand / tau, dim=1)
            l_con_teacher = F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)
        else:
            l_con_teacher = torch.zeros((), device=device) # TODO: düzelt

        # ---- Positives-only for attribution / DAG ----
        T_max = max((int(x.numel()) for x in pos_local), default=1)
        Dg = uniq_go_embs.size(1)
        G_pos = torch.zeros(B, T_max, Dg, device=device, dtype=uniq_go_embs.dtype)
        for b, loc in enumerate(pos_local):
            t = int(loc.numel())
            if t > 0:
                G_pos[b, :t] = uniq_go_embs.index_select(0, loc.to(uniq_go_embs.device))

        scores_pos, alpha_info = self.forward_scores(H, G_pos, pad_mask, return_alpha=use_attr)

        if alpha_info is not None and "alpha_full" in alpha_info:
            alpha = alpha_info["alpha_full"]
            if use_attr:
                delta, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, pad_mask)
                l_attr = attribution_loss(alpha, delta, mask=None, reduce="mean")
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(alpha)
            else:
                l_attr = torch.zeros((), device=device); l_ent = torch.zeros((), device=device)
        elif alpha_info is not None and all(k in alpha_info for k in ["alpha_windows", "win_weights", "spans"]):
            AW = alpha_info["alpha_windows"]; Ww = alpha_info["win_weights"]; spans = alpha_info["spans"]
            if use_attr:
                delta_full, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, pad_mask)
                B_, T_, W, win = AW.shape
                delta_win = torch.zeros_like(AW)
                for wi, (s, e) in enumerate(spans):
                    delta_win[:, :, wi, :e - s] = delta_full[:, :, s:e]
                l_attr = windowed_attr_loss(AW, Ww, spans, delta_win)
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(AW)
            else:
                l_attr = torch.zeros((), device=device); l_ent = torch.zeros((), device=device)
        else:
            print("[Trainer] Warning: no alpha info for attribution loss.")
            l_attr = torch.zeros((), device=device); l_ent = torch.zeros((), device=device)

        uniq_go_ids = batch['uniq_go_ids'].to(device)
        l_dag = dag_consistency_loss_pos(scores_pos, pos_local, uniq_go_ids, self.ctx.dag_parents, margin=0.0, scale=1.0)

        total = (l_con + lambda_v * l_con_teacher) + 0.3 * l_dag + self.attr.lambda_attr * l_attr + l_ent

        # === Step / logging ===
        self._global_step += 1

        # --- Update the queue: enqueue all positive GO embeddings from the batch ---
        if self.queue_miner is not None:
            with torch.no_grad():
                pos_list = []
                for loc in pos_local:
                    if loc.numel() > 0:
                        pos_list.append(uniq_go_embs.index_select(0, loc.to(uniq_go_embs.device)))
                if len(pos_list) > 0:
                    go_pos_all = torch.cat(pos_list, dim=0)   # [sum_t, Dg]
                    self.queue_miner.enqueue(go_pos_all.detach())
        torch.cuda.synchronize()  # doğru süre için
        dt = time.time() - t0
        # TODO: Quick eval - to be erased!

        # --- Quick on-batch retrieval metrics (cheap) ---
        with torch.no_grad():
            # scores_cand: (B, K)  -- zaten hesaplandı (InfoNCE/Distill açıkken)
            # pos_mask: (B, K)
            if scores_cand.numel() > 1:
                topk = min(5, scores_cand.size(1))
                topk_idx = scores_cand.topk(topk, dim=1).indices  # (B, topk)
                # recall@1/@5
                hit1 = pos_mask.gather(1, topk_idx[:, :1]).any(dim=1).float().mean().item()
                hit5 = pos_mask.gather(1, topk_idx).any(dim=1).float().mean().item()
                # mean rank of positives
                ranks = torch.argsort(torch.argsort(-scores_cand, dim=1), dim=1)  # 0=best
                pos_ranks = ranks[pos_mask].float()
                mean_pos_rank = pos_ranks.mean().item() if pos_ranks.numel() else float('nan')
            else:
                hit1 = hit5 = mean_pos_rank = float('nan')

            # queue hardness (opsiyonel)
            try:
                # pozitif skorlar (sadece pozitif kolonlarda)
                pos_scores = scores_cand[pos_mask]
                # negatif skorlar
                neg_mask = ~pos_mask
                neg_scores = scores_cand[neg_mask]
                if pos_scores.numel() > 0 and neg_scores.numel() > 0:
                    hard_gap = (pos_scores.mean() - neg_scores.mean()).item()
                else:
                    hard_gap = float('nan')
            except:
                hard_gap = float('nan')

        # --- Logging ---
        losses_log = {
            "total": float(total.detach().item()),
            "contrastive": float(l_con.detach().item()),
            "kl_teacher": float(l_con_teacher.detach().item()),
            "hierarchy": float(l_dag.detach().item()),
            "attr": float(l_attr.detach().item()),
            "entropy": float(l_ent.detach().item()),
            # TODO: to be deleted
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
                hard_frac=float(getattr(source, "hard_frac", getattr(source, "hard_frac_start", 0.0))) if hasattr(source, "hard_frac") or hasattr(source, "hard_frac_start") else None,
                k_hard=int(getattr(source, "k_hard", 0)) if hasattr(source, "k_hard") else None,
            )

        phase = dict(
            id=getattr(self, "phase_id", getattr(self.ctx, "phase_id", -1)),
            name=getattr(self, "phase_name", getattr(self.ctx, "phase_name", f"phase{getattr(self,'phase_id',-1)}"))
        )

        self.wlogger.log_losses(losses_log, step=self._global_step, epoch=epoch_idx, phase=phase, sched=sched)

        pid = int(phase["id"]) if isinstance(phase.get("id", -1), (int,)) else -1
        for k, v in losses_log.items():
            self._phase_acc[pid][f"loss/{k}"].append(v)
        if sched:
            for k, v in sched.items():
                if v is not None:
                    self._phase_acc[pid][f"sched/{k}"].append(float(v))

        log_every = int(getattr(self.cfg, "log_every", 50))
        if (self._global_step % log_every) == 0:
            self.wlogger.log_lora_summaries(self.model, step=self._global_step)
            if bool(getattr(self.cfg, "log_lora_hist", False)):
                self.wlogger.log_lora_histograms(self.model, step=self._global_step)

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

        for batch in loader:
            H = batch["prot_emb_pad"].to(device)
            attn_valid, pad_mask = self._valid_and_pad_masks(batch)
            pos_local: List[torch.Tensor] = batch["pos_go_local"]

            # POSITIVES
            if ("pos_go_tokens" in batch) and (hasattr(self.model, "go_encoder") and self.model.go_encoder is not None):
                toks = batch["pos_go_tokens"]
                uniq_go_embs = self.model.go_encoder(
                    input_ids=toks["input_ids"].to(device),
                    attention_mask=toks["attention_mask"].to(device),
                )
                try:
                    uniq_go_embs = F.normalize(uniq_go_embs, p=2, dim=1)
                except Exception:
                    pass
            else:
                uniq_go_embs = batch["uniq_go_embs"].to(device)

            # Queue negatives in eval (no enqueue)
            if self.queue_miner is not None:
                v_true_early = self.ctx.vres.true_prot_vecs(H, attn_valid)  # [B, Dh]
                vt_for_miner = v_true_early
                try:
                    vt_for_miner = self.ctx.vres.project_queries_to_index(v_true_early)  # [B, Dg]
                except Exception:
                    pass
                prot_query = vt_for_miner
                neg_from_queue = self.queue_miner.get_negatives(prot_query.detach(), k_hard=self.k_hard_queue)
                G_cand, pos_mask = self._build_candidates_with_queue(
                    uniq_go_embs=uniq_go_embs,
                    pos_go_local=pos_local,
                    neg_from_queue=neg_from_queue
                )
            else:
                # Fallback: only positives if queue empty
                G_cand, pos_mask = self._build_candidates_with_queue(
                    uniq_go_embs=uniq_go_embs,
                    pos_go_local=pos_local,
                    neg_from_queue=None
                )

            scores_cand = self.forward_scores(H, G_cand, pad_mask, return_alpha=False)
            l_con = multi_positive_infonce_from_candidates(scores_cand, pos_mask, self.attr.temperature)

            # Positives-only for attribution
            B = H.size(0)
            T_max = max((int(x.numel()) for x in pos_local), default=1)
            Dg = uniq_go_embs.size(1)
            G_pos = torch.zeros(B, T_max, Dg, device=device, dtype=uniq_go_embs.dtype)
            for b, loc in enumerate(pos_local):
                t = int(loc.numel())
                if t > 0:
                    G_pos[b, :t] = uniq_go_embs.index_select(0, loc.to(uniq_go_embs.device))

            scores_pos, alpha_info = self.forward_scores(H, G_pos, pad_mask)

            if "alpha_full" in alpha_info:
                alpha = alpha_info["alpha_full"]
                delta = topk_maskout_full(H, G_pos, alpha, k=self.attr.topk_per_window, model=self.model, pad_mask=pad_mask)
                l_attr = attribution_loss(alpha, delta, mask=None, reduce="mean")
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(alpha)
            elif all(k in alpha_info for k in ["alpha_windows", "win_weights", "spans"]):
                AW = alpha_info["alpha_windows"]; Ww = alpha_info["win_weights"]; spans = alpha_info["spans"]
                delta_full, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, pad_mask)
                delta_win = torch.zeros_like(AW)
                for wi, (s, e) in enumerate(spans):
                    delta_win[:, :, wi, :e - s] = delta_full[:, :, s:e]
                l_attr = windowed_attr_loss(AW, Ww, spans, delta_win)
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(AW)
            else:
                l_attr = torch.zeros((), device=device); l_ent = torch.zeros((), device=device)

            uniq_go_ids = batch['uniq_go_ids'].to(device)
            l_dag = dag_consistency_loss_pos(scores_pos, pos_local, uniq_go_ids, self.ctx.dag_parents, margin=0.0, scale=1.0)
            total = l_con + 0.3 * l_dag + self.attr.lambda_attr * l_attr + l_ent

            # ---- CAFA eval space & predictions ----
            G_eval, y_true_b, _ = self._build_eval_space(batch)
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

        for k in logs:
            logs[k] /= max(1, n)

        if all_pred_blocks:
            y_pred = torch.cat(all_pred_blocks, dim=0).numpy()
            y_true = torch.cat(all_true_blocks, dim=0).numpy()
            fmax, _ = compute_fmax(y_true, y_pred)
            aupr    = compute_term_aupr(y_true, y_pred)
            logs["cafa_fmax"] = float(fmax)
            logs["cafa_aupr"] = float(aupr)
        else:
            logs["cafa_fmax"] = 0.0
            logs["cafa_aupr"] = 0.0

        return logs
