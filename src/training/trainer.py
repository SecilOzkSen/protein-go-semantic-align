from typing import Dict, Any, List, Tuple, Optional
import copy
import torch
import torch.nn.functional as F
import time
import math
import re

from src.models.alignment_model import ProteinGoAligner
from src.loss.attribution import attribution_loss, windowed_attr_loss
from src.configs.data_classes import TrainerConfig, AttrConfig
from src.miners.queue_miner import MoCoQueue
from src.metrics.cafa import compute_fmax, compute_term_aupr

# ------------- Helpers -------------
def to_f32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float32 else x.float()

def norm_f32(x: torch.Tensor, p: int = 2, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    norm = F.normalize(to_f32(x), p=p, dim=dim, eps=eps)
    norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    return norm

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

def multi_positive_infonce_from_candidates(
    scores: torch.Tensor,
    pos_mask: torch.Tensor,
    tau: float,
    cand_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    scores: (B, K)
    pos_mask: (B, K) boolean; True at positives
    cand_valid_mask: (B, K) boolean; True only where a real candidate exists (not padding)
    """
    logits = scores / max(1e-8, tau)  # (B, K)

    if cand_valid_mask is not None:
        cand_valid_mask = cand_valid_mask.to(dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(~cand_valid_mask, float("-inf"))
        pos_mask = pos_mask & cand_valid_mask

    denom = torch.logsumexp(logits, dim=-1)  # (B,)

    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))  # (B, K)
    pos_any = pos_mask.any(dim=1)  # (B,)

    if (~pos_any).any():
        pos_logits = pos_logits.clone()
        pos_logits[~pos_any] = -1e9

    num = torch.logsumexp(pos_logits, dim=-1)  # (B,)
    loss = -(num - denom)  # (B,)

    return loss[pos_any].mean() if pos_any.any() else denom.mean() * 0.0


def surrogate_delta_y_from_mask_grad(H, G, model, mask=None, return_alpha=False):
    """
    Quick surrogate for attribution: use ||dy/dH|| as importance proxy.
    IMPORTANT: do NOT accumulate grads into model params.
    Returns (proxy_deltas [B,T,L], alpha_info).
    """
    H_req = H.detach().clone().requires_grad_(True)

    # Make sure this proxy is stable: no AMP
    with torch.autocast(device_type="cuda", enabled=False):
        out = model(H=H_req, G=G, mask=mask, return_alpha=return_alpha)

        if isinstance(out, tuple) and len(out) >= 2:
            scores, alpha_info = out[0], (out[1] or {})
        else:
            scores, alpha_info = out, {}

        # scores expected [B,T]
        y = scores.mean()

    # gradient only w.r.t H_req, does NOT write into model parameters .grad
    (dy_dH,) = torch.autograd.grad(
        y, H_req,
        retain_graph=False,
        create_graph=False,
        allow_unused=False
    )

    with torch.no_grad():
        dy_norm = dy_dH.norm(dim=-1)  # [B,L]
        B, T = scores.shape
        proxy = dy_norm.unsqueeze(1).expand(B, T, dy_norm.size(1)).contiguous()
        proxy = proxy / (proxy.amax(dim=-1, keepdim=True) + 1e-8)

    return proxy, alpha_info


def dag_consistency_loss_pos_ids(
    scores_pos: torch.Tensor,          # [B,Pmax]
    pos_go_ids: torch.Tensor,          # [B,Pmax] long, pad=-1
    dag_parents: Optional[dict],
    margin: float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    if dag_parents is None:
        return torch.zeros((), device=scores_pos.device)
    if pos_go_ids is None or pos_go_ids.numel() == 0:
        return torch.zeros((), device=scores_pos.device)

    B, Pmax = scores_pos.shape
    losses = []
    for b in range(B):
        ids_b = pos_go_ids[b]
        valid = ids_b >= 0
        if valid.sum().item() <= 1:
            continue

        ids_list = ids_b[valid].tolist()
        sp = scores_pos[b, valid]  # [P]
        id2i = {int(g): i for i, g in enumerate(ids_list)}

        for child_gid in ids_list:
            parents = dag_parents.get(int(child_gid), [])
            c_i = id2i[int(child_gid)]
            for pg in parents:
                if int(pg) in id2i:
                    p_i = id2i[int(pg)]
                    diff = sp[c_i] - sp[p_i] + margin
                    losses.append(F.softplus(scale * diff))

    if not losses:
        return torch.zeros((), device=scores_pos.device)
    return torch.stack(losses).mean()


def topk_maskout_full(H, G, alpha_full, k, model, mask=None, return_alpha = False):
    """
    Eval-time mask-out for full-length case.
    """
    B, T, L = alpha_full.shape
    device = H.device
    delta = torch.zeros_like(alpha_full)
    out = model(H=H, G=G, mask=mask, return_alpha=return_alpha)  # (scores, alpha_info)
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
                    mask=mask[b:b+1] if mask is not None else None,
                    return_alpha=return_alpha
                )
                y_minus = out_m[0] if isinstance(out_m, tuple) else out_m  # (1, T)
                delta[b, t, i] = (base_scores[b, t] - y_minus.squeeze(0)[t]).clamp_min(0.0)
            m = delta[b, t].amax()
            if m > 0:
                delta[b, t] = delta[b, t] / m
    return delta

# ------------- Optim helpers -------------
def _split_lora_params(module: torch.nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Returns: (lora_params, other_trainable_params) for the given module.
    Heuristic based on parameter names to be framework-agnostic.
    """
    lora_params: List[torch.nn.Parameter] = []
    other: List[torch.nn.Parameter] = []
    if module is None:
        return lora_params, other

    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()
        if ("lora" in n) or ("adapter" in n) or re.search(r"\blora\b", n) is not None:
            print(f"[Optimizer] LoRA param: {name}, shape={p.shape}")
            lora_params.append(p)
        else:
            other.append(p)
    return lora_params, other

def _freeze_all_but_lora(module: torch.nn.Module):
    """
    Make sure we train LoRA only: freeze everything then unfreeze LoRA-like params.
    Safe even if module already configured.
    """
    if module is None:
        return
    for _, p in module.named_parameters():
        p.requires_grad_(False)
    for name, p in module.named_parameters():
        n = name.lower()
        if ("lora" in n) or ("adapter" in n):
            p.requires_grad_(True)

# ------------- Trainer -------------
class OppTrainer:
    def __init__(self, cfg: TrainerConfig, attr: AttrConfig, ctx, go_encoder, wandb_run=None):
        self.cfg, self.attr, self.ctx = cfg, attr, ctx
        self.device = torch.device(cfg.device)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.normalizer = lambda x, dim: norm_f32(x, p=2, dim=dim)
        self.to_f32 = to_f32 if ctx.fp16_enabled else None
        self.return_alpha = ctx.return_alpha

        self.model = ProteinGoAligner(
            d_h=cfg.d_h,
            d_g=ctx.go_cache.embs.size(1) if go_encoder is None else None,
            d_z=cfg.d_z,
            go_encoder=go_encoder,
            normalize=True,
            mean_pool=(ctx.pooling_strategy == "mean"),
        ).to(self.device)

        self.m_ema = float(getattr(cfg, "m_ema", 0.999))
        self.go_encoder_k = None
        if getattr(self.model, "go_encoder", None) is not None:
            _freeze_all_but_lora(self.model.go_encoder)
            self.go_encoder_k = clone_as_target(self.model.go_encoder).to(self.device)

        init_ln = math.log(1.0 / 0.07)
        self.logit_scale = torch.nn.Parameter(torch.tensor(init_ln, dtype=torch.float32, device=self.device))

       # self.opt = torch.optim.AdamW(list(self.model.parameters()) + [self.logit_scale], lr=cfg.lr)
        # -------- Optimizer: split GO encoder LoRA params --------
        wd = float(getattr(cfg, "weight_decay", 0.01))
        lr_main = float(cfg.lr)
        lr_lora = float(getattr(cfg, "lr_lora", lr_main * 0.1))
        # Main params: everything trainable except GO encoder (we will add LoRA separately)
        main_params: List[torch.nn.Parameter] = []
        go_encoder = getattr(self.model, "go_encoder", None)
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # if it belongs to go_encoder, skip here (LoRA group will handle)
            if (go_encoder is not None) and name.startswith("go_encoder."):
                continue
            main_params.append(p)

        lora_params, go_other_trainable = _split_lora_params(go_encoder) if go_encoder is not None else ([], [])
        if len(go_other_trainable) > 0:
            # This means base encoder params are still trainable, which we do NOT want in LoRA-only mode.
            # Make it explicit to catch silent misconfig.
            raise RuntimeError(f"GO encoder has non-LoRA trainable params ({len(go_other_trainable)}). "
             "Freeze base encoder, only LoRA should be trainable.")

        param_groups = [{"params": main_params, "lr": lr_main, "weight_decay": wd},
                        {"params": [self.logit_scale], "lr": lr_main, "weight_decay": 0.0}]
        if len(lora_params) > 0:
            param_groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": 0.0})

        self.opt = torch.optim.AdamW(param_groups)
        self._global_step = 0

        self.use_moco_miner = bool(ctx.use_queue_miner)
        self.queue_K = int(getattr(cfg, "queue_K", getattr(ctx, "queue_K", 4096)))
        self.k_hard_queue = int(getattr(cfg, "k_hard_queue", getattr(ctx, "k_hard", 32)))
        self.queue_miner = None

        if wandb_run is None:
            raise RuntimeError("wandb_run must be passed explicitly")
        self.wandb_run = wandb_run

        self._eval_cache_ready = False

    # ----------------- basic helpers -----------------
    def _build_pos_go_ids(self, pos_local: List[torch.Tensor], uniq_go_ids: torch.Tensor) -> torch.Tensor:
        B = len(pos_local)
        Pmax = max((int(x.numel()) for x in pos_local), default=1)
        out = torch.full((B, Pmax), -1, dtype=torch.long, device=self.device)
        for b, loc in enumerate(pos_local):
            t = int(loc.numel())
            if t <= 0:
                continue
            gids = uniq_go_ids.index_select(0, loc.to(uniq_go_ids.device)).to(self.device)
            out[b, :t] = gids
        return out

    def _valid_and_pad_masks(self, batch):
        m = batch["prot_attn_mask"].to(self.device)
        if m.dim() == 3 and m.size(-1) == 1:
            m = m.squeeze(-1)
        if m.dtype is not torch.bool:
            m = m != 0
        attn_valid = m
        pad_mask = ~attn_valid
        return attn_valid, pad_mask

    def _maybe_init_queue(self, Dg: int):
        if self.queue_miner is None and self.use_moco_miner:
            self.queue_miner = MoCoQueue(dim=int(Dg), K=int(self.queue_K), device=str(self.device))
            print(f"[Trainer] MoCo Queue enabled (K={self.queue_K}, k_hard={self.k_hard_queue}, Dg={Dg}).")

    def _get_uniq_go_embs(self, batch):
        device = self.device
        if ("pos_go_tokens" in batch) and (getattr(self.model, "go_encoder", None) is not None):
            toks = batch["pos_go_tokens"]
            embs = self.model.go_encoder(
                    input_ids=toks["input_ids"].to(device, non_blocking=True),
                    attention_mask=toks["attention_mask"].to(device, non_blocking=True),
                )
        else:
            embs = batch["uniq_go_embs"].to(device, non_blocking=True)
        embs = self.normalizer(embs, dim=1)
        ids = batch["uniq_go_ids"].to(device, non_blocking=True)
        return embs, ids  # [G,Dg], [G]

    @torch.no_grad()
    def _get_prot_query(self, H, attn_valid, Dg_batch: int):
        v = self.ctx.vres.true_prot_vecs(H, attn_valid)  # [B, Dh]
        if hasattr(self.ctx.vres, "project_queries_to_index"):
            v = self.ctx.vres.project_queries_to_index(v)  # [B, Dg]
        if v.size(1) != Dg_batch:
            raise RuntimeError(f"prot_query dim mismatch: {v.size(1)} vs {Dg_batch}")
        return self.normalizer(v, dim=1)

    @torch.no_grad()
    def _mine_queue_hard_negs(self, prot_query, pos_local, uniq_go_ids, Dg_batch: int):
        if self.queue_miner is None:
            return None

        res = self.queue_miner.get_all_neg()
        if res is None:
            return None
        all_neg_vecs, all_neg_ids = res
        if all_neg_vecs is None or all_neg_vecs.numel() == 0:
            return None
        if all_neg_vecs.size(1) != Dg_batch:
            raise RuntimeError(f"Queue D mismatch: {all_neg_vecs.size(1)} vs {Dg_batch}")

        Kmat = self.normalizer(all_neg_vecs.to(self.device), dim=1)  # [K,Dg]
        sims = prot_query @ Kmat.T  # [B,K]

        # false-negative mask
        if all_neg_ids is not None:
            ids_list = all_neg_ids.tolist()
            mask = torch.zeros_like(sims, dtype=torch.bool)
            for b, loc in enumerate(pos_local):
                if loc.numel() == 0:
                    continue
                gids = uniq_go_ids.index_select(0, loc.to(uniq_go_ids.device)).tolist()
                s = set(map(int, gids))
                for j, gid in enumerate(ids_list):
                    if gid in s:
                        mask[b, j] = True
            sims = sims.masked_fill(mask, float("-inf"))

        k = min(int(self.k_hard_queue), Kmat.size(0))
        if k <= 0:
            return None
        idx = sims.topk(k, dim=1).indices  # [B,k]
        neg = Kmat.index_select(0, idx.reshape(-1)).reshape(idx.size(0), k, -1).contiguous()
        return neg  # [B,k,Dg]

    def _build_candidates(self, uniq_go_embs, pos_local, neg_from_queue):
        B = len(pos_local)
        Dg = uniq_go_embs.size(1)
        kq = 0 if neg_from_queue is None else neg_from_queue.size(1)

        pos_lens = [int(loc.numel()) for loc in pos_local]
        # K: batch içindeki max (p + kq)
        K = max((p + kq) for p in pos_lens) if B > 0 else max(1, kq)

        G_cand = torch.zeros(B, K, Dg, device=self.device, dtype=uniq_go_embs.dtype)
        pos_mask = torch.zeros(B, K, dtype=torch.bool, device=self.device)

        # NEW: candidate validity mask
        cand_valid_mask = torch.zeros(B, K, dtype=torch.bool, device=self.device)

        for b, loc in enumerate(pos_local):
            p = int(loc.numel())

            # positives
            if p > 0:
                G_cand[b, :p] = uniq_go_embs.index_select(0, loc.to(uniq_go_embs.device))
                pos_mask[b, :p] = True
                cand_valid_mask[b, :p] = True

            # negatives from queue
            if kq > 0:
                # neg_from_queue[b] shape: [kq, Dg]
                G_cand[b, p:p + kq] = neg_from_queue[b]
                cand_valid_mask[b, p:p + kq] = True

        return G_cand, pos_mask, cand_valid_mask

    # ----------------- forward scoring -----------------
    def forward_scores(self, H, G, mask, return_alpha=False, cand_chunk_k=32, pos_chunk_t=256, **kwargs):
        cand_chunk_k = int(getattr(self.cfg, "cand_chunk_k", cand_chunk_k))
        pos_chunk_t = int(getattr(self.cfg, "pos_chunk_t", pos_chunk_t))

        def _unpack(out):
            if isinstance(out, tuple):
                return out[0], (out[1] or {})
            return out, {}

        if G.dim() == 4:  # [B,K,T,Dg] (kept for backward compat)
            if return_alpha:
                raise RuntimeError("return_alpha=True not supported for 4D G chunking path")
            scores_all = []
            B, K, T, Dg = G.shape
            for ks in range(0, K, cand_chunk_k):
                ke = min(K, ks + cand_chunk_k)
                out = self.model(H=H, G=G[:, ks:ke].contiguous(), mask=mask,
                                 return_alpha=return_alpha, cand_chunk_k=None, pos_chunk_t=pos_chunk_t, **kwargs)
                sc, _ = _unpack(out)
                scores_all.append(sc)
            scores = torch.cat(scores_all, dim=1)
            return scores
        out = self.model(H=H, G=G, mask=mask, return_alpha=return_alpha,
                         cand_chunk_k=cand_chunk_k, pos_chunk_t=pos_chunk_t, **kwargs)
        sc, alpha = _unpack(out)

        return (sc, alpha) if return_alpha else sc

    # ----------------- eval space cache -----------------
    def _ensure_eval_cache(self):
        if self._eval_cache_ready:
            return
        if not (hasattr(self.ctx, "eval_id_list") and self.ctx.eval_id_list):
            raise RuntimeError("ctx.eval_id_list missing")

        eval_ids = torch.as_tensor(self.ctx.eval_id_list, dtype=torch.long)  # CPU
        rows = torch.as_tensor([self.ctx.go_cache.id2row[int(g)] for g in eval_ids.tolist()],
                               dtype=torch.long, device=self.ctx.go_cache.embs.device)
        G_once = self.ctx.go_cache.embs.index_select(0, rows).contiguous()  # bank device
        # cache as attributes (avoid rebuilding every batch)
        self.ctx._eval_ids_cpu = eval_ids
        self.ctx._eval_G_once_bank = G_once
        self.ctx._eval_id2col = {int(eval_ids[i].item()): i for i in range(eval_ids.numel())}
        self._eval_cache_ready = True

    def _build_eval_space(self, batch):
        self._ensure_eval_cache()
        device = self.device
        B = batch["prot_emb_pad"].size(0)

        eval_ids_cpu = self.ctx._eval_ids_cpu
        id2col = self.ctx._eval_id2col

        G_once = self.ctx._eval_G_once_bank.to(device, non_blocking=True)  # [Geval,Dg]
        Geval, Dg = G_once.size()
        G_eval = G_once.unsqueeze(0).expand(B, Geval, Dg).contiguous()

        y_true = torch.zeros(B, Geval, dtype=torch.float32, device=device)
        uniq_go_ids = batch["uniq_go_ids"].to(device, non_blocking=True)
        pos_local = batch["pos_go_local"]

        for b, loc in enumerate(pos_local):
            if loc.numel() == 0:
                continue
            gids = uniq_go_ids.index_select(0, loc.to(uniq_go_ids.device)).tolist()
            for g in gids:
                j = id2col.get(int(g), None)
                if j is not None:
                    y_true[b, j] = 1.0

        return G_eval, y_true

    # ----------------- training step -----------------
    def step_losses(self, batch, epoch_idx: int):
        self.model.train()
        device = self.device

        H = batch["prot_emb_pad"].to(device, non_blocking=True)
        attn_valid, pad_mask = self._valid_and_pad_masks(batch)
        if self.to_f32 is not None:
            H = self.to_f32(H)

        pos_local = batch["pos_go_local"]
        uniq_go_embs, uniq_go_ids = self._get_uniq_go_embs(batch)
        if getattr(self.model, "go_encoder", None) is not None:
            assert "pos_go_tokens" in batch, "GO encoder present but pos_go_tokens missing, LoRA won't train"
        Dg_batch = int(uniq_go_embs.size(1))
        self._maybe_init_queue(Dg_batch)

        with torch.no_grad():
            prot_query = self._get_prot_query(H, attn_valid, Dg_batch)
            neg_from_queue = self._mine_queue_hard_negs(prot_query, pos_local, uniq_go_ids, Dg_batch)

        G_cand, pos_mask, cand_valid_mask = self._build_candidates(uniq_go_embs, pos_local, neg_from_queue)

        amp_ctx = torch.amp.autocast(
            device_type="cuda",
            enabled=(torch.cuda.is_available() and self.ctx.fp16_enabled),
        )

        with amp_ctx:
            # 2) scores
            scores_cand = self.forward_scores(H, G_cand, attn_valid, return_alpha=False)
            assert scores_cand.requires_grad, "scores_cand grad not enabled"

            # 3) scale
            scores_cand = scores_cand * self.logit_scale.exp().clamp(max=100.0)

            # 4) loss (pad candidate'ları mask’le)
            l_con = multi_positive_infonce_from_candidates(
                scores_cand,
                pos_mask,
                tau=1.0,
                cand_valid_mask=cand_valid_mask,
            )

            if not torch.isfinite(l_con):
                raise RuntimeError("contrastive loss NaN, batch protein_ids=" + str(batch.get("protein_ids", "")[:5]))

            # positives-only
            B = H.size(0)
            T_max = max((int(x.numel()) for x in pos_local), default=1)
            G_pos = torch.zeros(B, T_max, Dg_batch, device=device, dtype=uniq_go_embs.dtype)
            for b, loc in enumerate(pos_local):
                t = int(loc.numel())
                if t > 0:
                    G_pos[b, :t] = uniq_go_embs.index_select(0, loc.to(uniq_go_embs.device))

            # attr train: keep as you had, but use return_alpha correctly
            use_attr = self.ctx.attribute_loss_enabled and (epoch_idx < self.attr.curriculum_epochs and self.attr.lambda_attr > 0.0)
            if use_attr:
                scores_pos, alpha_info = self.forward_scores(H, G_pos, attn_valid, return_alpha=True)
            else:
                scores_pos = self.forward_scores(H, G_pos, attn_valid, return_alpha=False)
                alpha_info = {}

        # attr + entropy
        if use_attr and alpha_info and ("alpha_full" in alpha_info):
            alpha = alpha_info["alpha_full"]
            delta, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, mask=attn_valid, return_alpha=False)
            l_attr = attribution_loss(alpha, delta, mask=None, reduce="mean")
            l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(alpha)
        else:
            l_attr = torch.zeros((), device=device)
            l_ent = torch.zeros((), device=device)

        #DAG: build explicit [B, Pmax] GO-id tensor, then apply pos_ids DAG loss.
        l_dag = torch.zeros((), device=device)
        if self.attr.lambda_dag > 0:
            # DAG: GO-only (protein path detached)
            pos_go_ids = self._build_pos_go_ids(pos_local, uniq_go_ids)
            with amp_ctx:
                scores_pos_dag = self.forward_scores(H.detach(), G_pos, attn_valid, return_alpha=False)
            scores_pos_dag_f32 = scores_pos_dag.float()
            l_dag = dag_consistency_loss_pos_ids(scores_pos_dag_f32, pos_go_ids, self.ctx.dag_parents, margin=0.0,
                                                 scale=1.0)
        total = l_con + self.attr.lambda_dag * l_dag + self.attr.lambda_attr * l_attr + l_ent

        # EMA
        self._global_step += 1
        if self.go_encoder_k is not None:
            # q = online encoder (trainable LoRA), k = EMA target encoder
            ema_update(self.model.go_encoder, self.go_encoder_k, m=self.m_ema)

        # enqueue (unchanged logic, but keep it minimal)
        if self.queue_miner is not None:
            with torch.no_grad():
                local_idx_list = [loc.to(device) for loc in pos_local if loc.numel() > 0]
                if local_idx_list:
                    local_cat = torch.unique(torch.cat(local_idx_list, dim=0))
                 #   pos_vecs = uniq_go_embs.index_select(0, local_cat).detach()
                    pos_vecs = None
                    if("pos_go_tokens" in batch) and (self.go_encoder_k is not None):
                        toks = batch["pos_go_tokens"]
                        assert toks["input_ids"].size(0) == uniq_go_ids.size(
                            0), "pos_go_tokens must align with uniq_go_ids order"
                        embs_k = self.go_encoder_k(input_ids = toks["input_ids"].to(device, non_blocking=True),
                            attention_mask = toks["attention_mask"].to(device, non_blocking=True))
                        embs_k = self.normalizer(embs_k, dim=1)
                        pos_vecs = embs_k.index_select(0, local_cat).detach()
                    else:
                        pos_vecs = uniq_go_embs.index_select(0, local_cat).detach()
                    pos_ids = uniq_go_ids.index_select(0, local_cat).detach()
                    self.queue_miner.enqueue(pos_vecs, pos_ids)

        return {"total": total, "contrastive": l_con, "dag": l_dag, "attr": l_attr, "entropy": l_ent}

    @torch.no_grad()
    def eval_epoch(self, loader, epoch_idx: int):
        self.model.eval()
        device = self.device

        logs = {"cafa_fmax": 0.0, "cafa_aupr": 0.0}
        n = 0

        preds, trues = [], []

        for batch in loader:
            H = batch["prot_emb_pad"].to(device, non_blocking=True)
            if self.to_f32 is not None:
                H = self.to_f32(H)
            attn_valid, pad_mask = self._valid_and_pad_masks(batch)

            # build global eval space from cache
            G_eval, y_true = self._build_eval_space(batch)
            scores = self.forward_scores(H, G_eval, attn_valid, return_alpha=False)
            logit_scale = self.logit_scale.clamp(min=-10.0, max=10.0)
            scale = logit_scale.exp()
            scores = scores * scale
            probs = torch.sigmoid(scores)

            preds.append(probs.cpu())
            trues.append(y_true.cpu())
            n += 1

        if preds:
            y_pred = torch.cat(preds, dim=0).numpy()
            y_true = torch.cat(trues, dim=0).numpy()

            # CAFA-style protein-centric Fmax
            fmax, _ = compute_fmax(
                y_true=y_true,
                y_pred=y_pred,
                num_thresholds=101
            )

            aupr = compute_term_aupr(y_true, y_pred)
        else:
            fmax, aupr = 0.0, 0.0

        logs["cafa_fmax"] = float(fmax)
        logs["cafa_aupr"] = float(aupr)
        return logs
