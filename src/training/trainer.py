from typing import List, Optional
import copy
import torch
import torch.nn.functional as F
import math

from go import load_go_parents
from src.models.alignment_model import ProteinGoAligner
from src.loss.attribution import attribution_loss
from src.configs.data_classes import TrainerConfig, AttrConfig, QueueConfig
from src.miners.queue_miner import MoCoQueue
from src.metrics.cafa import compute_fmax, compute_term_aupr
from src.metrics.retrieval import retrieval_metrics_from_scores
from src.utils.helpers import go_str_to_int_any
import numpy as np

'''
	•	queue is open by default
	•	in token_align mode, the queue is active
	•	queue mining and shortlist are performed in the pooled-projection space
	•	in the token_align forward pass, negatives coming from the queue will be re-encoded as tokens via their IDs
	•	_get_uniq_go_embs will now return a dict
	•	the candidate builder will now also return metadata
	•	then, depending on the mode:
	•	pooled candidate tensor
	•	token candidate tensor
'''
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


def entropy_regularizer(alpha: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    a = alpha.clamp_min(eps)
    ent = -(a * a.log()).sum(dim=-1)  # [...]
    if mask is not None:
        # mask: [B,L] -> expand to alpha dims and compute log(L_valid)
        m = mask.to(a.dtype)
        while m.dim() < a.dim():
            m = m.unsqueeze(1)
        L_valid = m.sum(dim=-1).clamp_min(1.0)
        ent = ent / (L_valid.log() + eps)
    return ent.mean()


def multi_positive_infonce_from_candidates_v2(
        scores: torch.Tensor,
        pos_mask: torch.Tensor,
        tau: float,
        cand_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    scores: (B, K)
    pos_mask: (B, K) bool
    cand_valid_mask: (B, K) bool, True=valid candidate
    """
    # safer math
    tau = float(tau)
    tau = tau if tau > 1e-8 else 1e-8

    logits = (scores / tau).float()  # logsumexp stability

    if cand_valid_mask is not None:
        cand_valid_mask = cand_valid_mask.bool()
        pos_mask = pos_mask.bool() & cand_valid_mask
        logits = logits.masked_fill(~cand_valid_mask, -1e9)
        valid_any = cand_valid_mask.any(dim=1)
    else:
        pos_mask = pos_mask.bool()
        valid_any = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)

    # denom
    denom = torch.logsumexp(logits, dim=-1)  # (B,)

    # numerator: log-mean-exp over positives
    pos_any = pos_mask.any(dim=1) & valid_any
    pos_count = pos_mask.sum(dim=1).clamp_min(1)  # avoid log(0)

    pos_logits = logits.masked_fill(~pos_mask, -1e9)
    num = torch.logsumexp(pos_logits, dim=-1) - pos_count.float().log()  # (B,)

    # loss only where we have positives and at least one valid candidate
    keep = pos_any
    if keep.any():
        loss = -(num - denom)
        return loss[keep].mean()

    # return 0 with correct dtype/device
    return denom.mean() * 0.0


@torch.no_grad()
def delta_y_from_occlusion_windows(
        H: torch.Tensor,  # [B, L, Dh]
        G_pos: torch.Tensor,  # [B, P, Dg]
        model,  # ProteinGoAligner
        valid_mask: torch.Tensor | None = None,  # [B, L] bool
        window: int = 32,
        stride: int = 16,
        mask_value: float = 0.0,
        chunk_windows: int = 32,  # memory control
) -> torch.Tensor:
    """
    Returns:
      delta: [B, P, L]  (label-specific importance proxy)
    Meaning:
      delta[b,p,l] is proportional to drop in score for label p when masking windows covering residue l.
    """

    device = H.device
    B, L, Dh = H.shape
    _, P, _ = G_pos.shape

    if valid_mask is not None and valid_mask.dtype != torch.bool:
        valid_mask = valid_mask != 0

    # 1) base scores: [B, P]
    with torch.autocast(device_type="cuda", enabled=False):
        base_scores = model(H=H, G=G_pos, mask=valid_mask, return_alpha=False)  # [B, P]
        base_scores = base_scores.float()

    # 2) define window starts
    if L <= window:
        starts = torch.tensor([0], device=device, dtype=torch.long)
    else:
        starts = torch.arange(0, L - window + 1, stride, device=device, dtype=torch.long)
        if starts.numel() == 0:
            starts = torch.tensor([0], device=device, dtype=torch.long)

    # 3) accumulate deltas into [B, P, L]
    delta = torch.zeros((B, P, L), device=device, dtype=torch.float32)
    counts = torch.zeros((L,), device=device, dtype=torch.float32)  # how many windows cover each residue

    # precompute coverage counts once (for normalization)
    for s in starts.tolist():
        e = min(L, s + window)
        counts[s:e] += 1.0
    counts = counts.clamp_min(1.0)  # avoid divide-by-zero

    # 4) iterate windows in chunks
    for ws in range(0, starts.numel(), chunk_windows):
        we = min(starts.numel(), ws + chunk_windows)
        cur_starts = starts[ws:we]  # [W]

        W = int(cur_starts.numel())

        # Build masked H batch: [B, W, L, Dh]
        # We keep it explicit for clarity, then reshape to [B*W, L, Dh]
        H_rep = H.unsqueeze(1).expand(B, W, L, Dh).contiguous()
        if valid_mask is not None:
            mask_rep = valid_mask.unsqueeze(1).expand(B, W, L).contiguous()
        else:
            mask_rep = None

        # Apply masking windows
        for i, s in enumerate(cur_starts.tolist()):
            e = min(L, s + window)
            H_rep[:, i, s:e, :] = mask_value
            if mask_rep is not None:
                # masked residues are effectively invalid for pooling
                mask_rep[:, i, s:e] = False

        H_flat = H_rep.view(B * W, L, Dh)
        if mask_rep is not None:
            mask_flat = mask_rep.view(B * W, L)
        else:
            mask_flat = None

        # Repeat G_pos for each window: [B*W, P, Dg]
        G_flat = G_pos.unsqueeze(1).expand(B, W, P, G_pos.size(-1)).contiguous().view(B * W, P, G_pos.size(-1))

        # 5) masked scores
        with torch.autocast(device_type="cuda", enabled=False):
            masked_scores = model(H=H_flat, G=G_flat, mask=mask_flat, return_alpha=False)  # [B*W, P]
            masked_scores = masked_scores.view(B, W, P).float()  # [B, W, P]

        # 6) score drop: relu(base - masked) to keep only evidence of importance
        drop = (base_scores.unsqueeze(1) - masked_scores).clamp_min(0.0)  # [B, W, P]
        drop = drop.permute(0, 2, 1).contiguous()  # [B, P, W]

        # 7) distribute each window’s drop over residues it covers
        for i, s in enumerate(cur_starts.tolist()):
            e = min(L, s + window)
            # add contribution to all residues in window
            delta[:, :, s:e] += drop[:, :, i].unsqueeze(-1)

    # 8) normalize by coverage count and per-label max
    delta = delta / counts.view(1, 1, L)
    delta = delta / (delta.amax(dim=-1, keepdim=True) + 1e-8)

    # Optionally zero out padding residues
    if valid_mask is not None:
        delta = delta * valid_mask.to(delta.dtype).unsqueeze(1)

    return delta  # [B, P, L]


def build_dag_ancestors(dag_parents: dict[int, list[int]]) -> dict[int, list[int]]:
    # child -> all ancestors (including itself)
    memo: dict[int, list[int]] = {}

    def dfs(x: int) -> list[int]:
        if x in memo:
            return memo[x]
        out = {x}
        for p in dag_parents.get(x, []):
            out.update(dfs(int(p[0])))
        memo[x] = list(out)
        return memo[x]

    # materialize for all keys (and parents that might not be keys)
    nodes = set(dag_parents.keys())
    for ps in dag_parents.values():
        nodes.update(int(p[0]) for p in ps)
    for n in nodes:
        dfs(int(n))
    return memo


def dag_consistency_loss_pos_ids(
        scores_pos: torch.Tensor,  # [B,Pmax]
        pos_go_ids: torch.Tensor,  # [B,Pmax] long, pad=-1
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


def _tstats(x: torch.Tensor, name: str):
    if x is None:
        print(f"[DBG] {name}=None")
        return
    xf = x.detach()
    if xf.numel() == 0:
        print(f"[DBG] {name}: empty shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
        return
    xf32 = xf.float()
    nan = torch.isnan(xf32).any().item()
    inf = torch.isinf(xf32).any().item()
    mn = float(xf32.min().item())
    mx = float(xf32.max().item())
    mean = float(xf32.mean().item())
    std = float(xf32.std(unbiased=False).item())
    nrm = float(torch.linalg.vector_norm(xf32, dim=-1).mean().item()) if xf32.dim() >= 2 else float(
        torch.linalg.vector_norm(xf32).item())
    print(
        f"[DBG] {name}: shape={tuple(x.shape)} dtype={x.dtype} dev={x.device} nan={nan} inf={inf} min={mn:.4g} max={mx:.4g} mean={mean:.4g} std={std:.4g} mean_norm={nrm:.4g}")


# ------------- Trainer -------------
class OppTrainer:
    def __init__(self, cfg: TrainerConfig, attr: AttrConfig, queue_cfg: QueueConfig, ctx, go_encoder, wandb_run=None):
        self.cfg, self.attr, self.ctx, self.queue_cfg = cfg, attr, ctx, queue_cfg
        self.device = torch.device(cfg.device)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self._eval_cols_seen = None
        self._eval_cols_rare = None
        self._eval_cols_unseen = None
        self._current_uniq_go_ids_for_shortlist = None

        self.normalizer = lambda x, dim: norm_f32(x, p=2, dim=dim)
        self.to_f32 = to_f32 if ctx.fp16_enabled else None
        self.return_alpha = ctx.return_alpha
        self.dag_ancestors = build_dag_ancestors(self.ctx.dag_parents) if getattr(ctx, "dag_parents") else None
        self.dag_anc = load_go_parents()
        self.model = ProteinGoAligner(
            d_h=cfg.d_h,
            d_g=ctx.go_cache.embs.size(1) if go_encoder is None else None,
            d_z=cfg.d_z,
            go_encoder=go_encoder,
            normalize=True,
            protein_pool_type=ctx.protein_pooling_strategy,
            protein_n_slots=ctx.protein_n_slots,
            go_pool_type=ctx.go_pool_type,
        ).to(self.device)

        self.m_ema = float(getattr(cfg, "m_ema", 0.999))
        self.go_encoder_k = None
        if getattr(self.model, "go_encoder", None) is not None:
            self.go_encoder_k = clone_as_target(self.model.go_encoder).to(self.device)

        init_ln = math.log(10)  # 1.0 / 0.07
        self.logit_scale = torch.nn.Parameter(torch.tensor(init_ln, dtype=torch.float32, device=self.device))
        if getattr(cfg, "is_logit_scale_constant", None) is not None and cfg.is_logit_scale_constant is True:
            with torch.no_grad():
                self.logit_scale.fill_(init_ln)
            self.logit_scale.requires_grad_(False)
            assert not self.logit_scale.requires_grad
        else:
            self.logit_scale.requires_grad_(True)

        # self.opt = torch.optim.AdamW(list(self.model.parameters()) + [self.logit_scale], lr=cfg.lr)
        # -------- Optimizer: main + GO encoder param groups --------
        wd = float(getattr(cfg, "weight_decay", 0.01))
        lr_main = float(cfg.lr)

        # Main params: everything trainable except go_encoder.*
        main_params: List[torch.nn.Parameter] = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # IMPORTANT: skip GO encoder here, it will be added via ge.param_groups_for_optimizer()
            if (self.model.go_encoder is not None) and name.startswith("go_encoder."):
                continue
            main_params.append(p)

        param_groups = [
            {"params": main_params, "lr": lr_main, "weight_decay": wd},
            {"params": [self.logit_scale], "lr": 1e-3, "weight_decay": 0.0},
        ]

        # Add GO encoder groups (LoRA + embeddings + attn head), if present
        if self.model.go_encoder is not None:
            lr_lora = float(getattr(cfg, "lr_lora", lr_main * 0.1))
            #   lr_emb = float(getattr(cfg, "lr_go_emb", lr_main * 0.5))
            #   lr_attn = float(getattr(cfg, "lr_go_attn", lr_main * 0.1))

            # BioMedBERTEncoder implements this
            ge_groups = self.model.go_encoder.param_groups_for_optimizer(
                lr_emb=None,
                lr_lora=lr_lora,
                wd_lora=0.01,
                lr_attn=None,
                wd_attn=0.0,
            )
            for g in ge_groups:
                g["name"] = "go_lora"
            param_groups.extend(ge_groups)

        if self.model.go_encoder is not None:
            lr_lora = float(getattr(cfg, "lr_lora", lr_main * 0.1))
            self._lora_lr_target = lr_lora
        else:
            lr_lora = 0.0
            self._lora_lr_target = 0.0
        self._lora_warmup_steps = int(getattr(cfg, "lora_warmup_steps", 2000))
        self._lora_lr_start = float(getattr(cfg, "lora_lr_start", self._lora_lr_target * 0.2))

        # Optional: one-time sanity check that GO encoder is LoRA-only
        if self.model.go_encoder is not None:
            trainable = [n for n, p in self.model.go_encoder.named_parameters() if p.requires_grad]
            if trainable:
                bad = [n for n in trainable if "lora_" not in n]
                if bad:
                    print("[WARN] GO encoder has non-LoRA trainables (expected LoRA-only):", bad[:10])

        self.opt = torch.optim.AdamW(param_groups)
        self._global_step = 0

        self.use_moco_miner = bool(ctx.use_queue_miner)
        self.queue_miner = None
        self._queue_was_active = False

        if wandb_run is None:
            raise RuntimeError("wandb_run must be passed explicitly")
        self.wandb_run = wandb_run
        self.eval_id_list = ctx.eval_id_list
        self._eval_cache_ready = False
        self._eval_ids_cpu = None
        self._eval_G_once_cpu = None
        self._eval_id2col = None
        self._use_token_align = (
                getattr(self.model, "protein_pool_type", None) == "slots"
                and getattr(self.model, "go_pool_type", None) == "token_align"
        )

        if self.ctx.eval_seen_go_ids is not None:
            self._eval_seen_go_ids_cpu = torch.as_tensor(sorted(set(int(x) for x in self.ctx.eval_seen_go_ids)), dtype=torch.long)
        else:
            self._eval_seen_go_ids_cpu = None
        # init sonrası bir kere
        opt_params = set()
        for g in self.opt.param_groups:
            for p in g["params"]:
                opt_params.add(id(p))

        for name, p in self.model.named_parameters():
            if ("pooler" in name or "proj_p" in name) and p.requires_grad:
                in_opt = (id(p) in opt_params)
                print("[OPTCHK]", name, "in_opt=", in_opt, "shape=", tuple(p.shape))
    def _queue_active(self) -> bool:
        return self.use_moco_miner and (self._global_step >= self.queue_cfg.queue_start_step)

    @torch.no_grad()
    def _maybe_activate_queue(self):
        """
        Activate queue only after warmup.
        Reset once at activation time so old / partial contents do not leak in.
        """
        if not self._queue_active():
            return

        if not self._queue_was_active:
            if self.queue_miner is None:
                # projected queue uses Dz
                Dz = self._get_Dz()
                self.queue_miner = MoCoQueue(dim=int(Dz), K=int(self.queue_cfg.queue_K), device=str(self.device))
                print(
                    f"[Trainer] MoCo Queue activated "
                    f"(start_step={self.queue_cfg.queue_start_step}, K={self.queue_cfg.queue_K}, k_hard={self._k_hard_queue_schedule(self._global_step)}, Dz={Dz})."
                )
            else:
                self.queue_miner.reset()
                print(f"[Trainer] MoCo Queue reset at activation step {self._global_step}.")

            self._queue_was_active = True
    # ----------------- debug -----------------
    @torch.no_grad()
    def _debug_pos_neg_cosines(
            self,
            H: torch.Tensor,
            attn_valid: torch.Tensor,
            y_true: torch.Tensor,
            max_neg: int = 32,
    ):
        """
        H: [B, L, Dh]
        attn_valid: [B, L]
        y_true: [B, G] over observed eval space
        """
        device = H.device

        Dz = self._get_Dz()
        prot_query = self._get_prot_query(H, attn_valid, Dz)
        prot_query = F.normalize(prot_query.float(), dim=-1)

        G_once = self._eval_G_once_cpu.to(device, non_blocking=True)
        G_proj = self.model.go_ln(G_once)
        G_proj = self.model.proj_g(G_proj)
        G_proj = F.normalize(G_proj.float(), dim=-1)

        pos_vals = []
        neg_vals = []

        B, G = y_true.shape
        for b in range(B):
            pos_idx = torch.nonzero(y_true[b] > 0, as_tuple=False).flatten()
            if pos_idx.numel() == 0:
                continue

            neg_idx = torch.nonzero(y_true[b] <= 0, as_tuple=False).flatten()
            if neg_idx.numel() == 0:
                continue

            if neg_idx.numel() > max_neg:
                perm = torch.randperm(neg_idx.numel(), device=device)[:max_neg]
                neg_idx = neg_idx[perm]

            q = prot_query[b:b + 1]

            pos_cos = (q * G_proj.index_select(0, pos_idx)).sum(dim=-1)
            neg_cos = (q * G_proj.index_select(0, neg_idx)).sum(dim=-1)

            pos_vals.append(pos_cos.mean())
            neg_vals.append(neg_cos.mean())

        if len(pos_vals) == 0 or len(neg_vals) == 0:
            return {
                "pos_cos_mean": 0.0,
                "neg_cos_mean": 0.0,
                "margin": 0.0,
                "num_debug_samples": 0,
            }

        pos_mean = torch.stack(pos_vals).mean().item()
        neg_mean = torch.stack(neg_vals).mean().item()

        return {
            "pos_cos_mean": float(pos_mean),
            "neg_cos_mean": float(neg_mean),
            "margin": float(pos_mean - neg_mean),
            "num_debug_samples": int(len(pos_vals)),
        }
    @torch.no_grad()
    def _dbg_norms(self, *, prot_query=None, pos_vecs=None, uniq_go_embs=None, tag=""):
        def _mean_norm(x):
            if x is None or x.numel() == 0:
                return None
            x = x.detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.size(-1))
            return float(x.float().norm(dim=-1).mean().item())

        pq = _mean_norm(prot_query)
        pv = _mean_norm(pos_vecs)
        ug = _mean_norm(uniq_go_embs)

        msg = f"[DBG-NORM]{tag} prot_query={pq} pos_vecs={pv} uniq_go_embs={ug}"
        print(msg)

        # queue stats
        if self.queue_miner is not None:
            q = self.queue_miner.queue_proj
            qmn = _mean_norm(q)
            valid_gpu = self.queue_miner.valid.to(q.device, non_blocking=True)
            qnz = float(((q.float().norm(dim=-1) > 1e-6) & valid_gpu).float().mean().item())
            ptr = int(self.queue_miner._ptr) if hasattr(self.queue_miner, "_ptr") else -1
            print(
                f"[DBG-NORM]{tag} queue_mean_norm={qmn} queue_nz_frac={qnz:.3f} "
                f"ptr={ptr} K={q.size(0)} D={q.size(1)}"
            )

    # ----------------- queue helpers -----------------

    def _linear_schedule(self, step: int, start: float, end: float, warmup_steps: int):
        if warmup_steps <= 0:
            return end
        if step <= 0:
            return start
        if step >= warmup_steps:
            return end
        t = step / warmup_steps
        return start + t * (end - start)

    def _queue_hard_frac_schedule(self, step: int):
        return float(self._linear_schedule(
            step=step,
            start=self.queue_cfg.queue_hard_frac_start,
            end=self.queue_cfg.queue_hard_frac_end,
            warmup_steps=self.queue_cfg.queue_hard_frac_warmup_steps,
        ))

    def _queue_weight_schedule(self, step: int):
        return float(self._linear_schedule(
            step=step,
            start=self._queue_weight_start,
            end=self._queue_weight_end,
            warmup_steps=self._queue_weight_warmup_steps,
        ))

    def _k_hard_queue_schedule(self, step: int):
        v = self._linear_schedule(
            step=step,
            start=float(self.queue_cfg.k_hard_queue_start),
            end=float(self.queue_cfg.k_hard_queue_end),
            warmup_steps=self.queue_cfg.k_hard_queue_warmup_steps,
        )
        return max(1, int(round(v)))

    # ----------------- basic helpers -----------------
    def _set_group_lr(self, name: str, lr: float):
        for g in self.opt.param_groups:
            if g.get("name") == name:
                g["lr"] = float(lr)

    def _lora_lr_schedule(self, step: int):
        """
        Linear warmup. lr_start -> lr_target over warmup_steps, then constant lr_target
        :param step:
        :return:
        """
        if self._lora_warmup_steps <= 0:
            return self._lora_lr_target
        if step <= 0:
            return self._lora_lr_start
        if step < self._lora_warmup_steps:
            t = step / self._lora_warmup_steps
            return self._lora_lr_start + t * (self._lora_lr_target - self._lora_lr_start)
        return self._lora_lr_target

    def _build_pos_go_ids(self, pos_local: List[torch.Tensor], uniq_go_ids: torch.Tensor) -> torch.Tensor:
        device = self.device
        B = len(pos_local)
        Pmax = max((int(x.numel()) for x in pos_local), default=0)
        out = torch.full((B, Pmax), -1, dtype=torch.long, device=device)
        if Pmax == 0:
            return out
        uniq_go_ids = uniq_go_ids.to(device, non_blocking=True)
        for b, loc in enumerate(pos_local):
            t = int(loc.numel())
            if t <= 0:
                continue
            loc = loc.to(device, non_blocking=True)
            out[b, :t] = uniq_go_ids.index_select(0, loc)
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

    def _maybe_init_queue(self, Dz: int):
        """
        Optional eager init. Queue will not be USED before queue_start_step.
        You can keep this, or remove calls to it entirely.
        """
        if self.queue_miner is None and self.use_moco_miner:
            self.queue_miner = MoCoQueue(dim=int(Dz), K=int(self.queue_cfg.queue_K), device=str(self.device))
            print(
                f"[Trainer] MoCo Queue initialized "
                f"(K={self.queue_cfg.queue_K}, k_hard={self._k_hard_queue_schedule(self._global_step)}, Dz={Dz}, start_step={self.queue_cfg.queue_start_step})."
            )


    def _get_Dz(self) -> int:
        pp = self.model.proj_p
        if hasattr(pp, "fc1"):
            return int(pp.fc1.out_features)
        if hasattr(pp, "weight"):
            return int(pp.weight.size(0))
        raise RuntimeError("Cannot infer Dz from proj_p")


    @torch.no_grad()
    def _fetch_raw_go_embs_from_queue(self, neg_idx_2d: torch.Tensor, dtype: torch.dtype):
        """
        neg_idx_2d: [B, Kq] indices into current queue snapshot
        returns:
          raw_go_embs: [B, Kq, Dg] on self.device
          valid_mask:  [B, Kq] bool on self.device
        """
        if neg_idx_2d is None:
            return None, None
        if self.queue_miner is None:
            return None, None

        res = self.queue_miner.get_all_neg()
        if res is None:
            return None, None

        _all_neg_proj, all_neg_raw, _all_neg_ids = res
        if all_neg_raw is None:
            return None, None

        device = self.device
        B, Kq = neg_idx_2d.shape

        flat_idx_cpu = neg_idx_2d.reshape(-1).to("cpu")
        raw_sel = all_neg_raw.index_select(0, flat_idx_cpu)  # [B*Kq, Dg] CPU
        raw_sel = raw_sel.view(B, Kq, -1).contiguous()

        raw_sel = raw_sel.to(device=device, dtype=dtype, non_blocking=True)
        valid = torch.ones(B, Kq, device=device, dtype=torch.bool)
        return raw_sel, valid

    def _collect_structured_negatives(
            self,
            batch_global_pos: set[int],
            need: int,
    ) -> list[int]:
        """
        Build structured negatives from:
          1) siblings via parent->children
          2) same-namespace GO terms

        Returns:
          list of global GO ids
        """
        if need <= 0:
            return []

        dag_parents = getattr(self.ctx, "dag_parents", {}) or {}
        dag_children = getattr(self.ctx, "dag_children", {}) or {}
        go_ns = getattr(self.ctx, "go_namespace_map", {}) or {}

        structured = []
        structured_set = set()

        # ---- 1) sibling negatives ----
        for g in batch_global_pos:
            parents = dag_parents.get(int(g), [])
            for p_item in parents:
                # p_item can be (parent_id, rel) or just parent_id
                if isinstance(p_item, (list, tuple)):
                    parent_id = int(p_item[0])
                else:
                    parent_id = int(p_item)

                sibs = dag_children.get(parent_id, [])
                for sib_item in sibs:
                    # sib_item can be (child_id, rel) or just child_id
                    if isinstance(sib_item, (list, tuple)):
                        sib = int(sib_item[0])
                    else:
                        sib = int(sib_item)

                    if sib in batch_global_pos:
                        continue
                    if sib in structured_set:
                        continue
                    structured.append(sib)
                    structured_set.add(sib)
                    if len(structured) >= need:
                        return structured

        # ---- 2) same-namespace negatives ----
        pos_namespaces = set()
        for g in batch_global_pos:
            ns = go_ns.get(int(g), None)
            if ns is not None:
                pos_namespaces.add(ns)

        if len(structured) < need and len(pos_namespaces) > 0:
            seen_go_ids = getattr(self, "_eval_seen_go_ids_cpu", None)
            if seen_go_ids is not None:
                if torch.is_tensor(seen_go_ids):
                    pool_ids = [int(x) for x in seen_go_ids.tolist()]
                else:
                    pool_ids = [int(x) for x in seen_go_ids]

                ns_pool = [
                    gid for gid in pool_ids
                    if gid not in batch_global_pos
                       and gid not in structured_set
                       and go_ns.get(int(gid), None) in pos_namespaces
                ]

                if len(ns_pool) > 0:
                    # randomize a bit
                    perm = torch.randperm(len(ns_pool)).tolist()
                    for idx in perm:
                        gid = int(ns_pool[idx])
                        structured.append(gid)
                        structured_set.add(gid)
                        if len(structured) >= need:
                            break

        return structured

    @torch.no_grad()
    def _mine_queue_hard_neg_ids(self, prot_query, pos_local, uniq_go_ids):
        """
        Returns:
          neg_idx: [B, k] long  -> indices into current queue snapshot
          neg_ids: [B, k] long  -> global GO ids
        Queue is used for mining only.
        Final scorer must use RAW GO embeddings from queue_raw[selected_idx].
        """
        if self.queue_miner is None:
            return None, None

        res = self.queue_miner.get_all_neg()
        if res is None:
            return None, None

        all_neg_proj, _all_neg_raw, all_neg_ids = res
        if all_neg_proj is None or all_neg_proj.numel() == 0:
            return None, None

        if all_neg_proj.dim() != 2:
            raise RuntimeError(f"Queue projected vecs must be [Kq,D], got {tuple(all_neg_proj.shape)}")

        device = self.device
        B = int(prot_query.size(0))

        Kmat = all_neg_proj.to(device, non_blocking=True).to(prot_query.dtype)  # [Kq, Dz]
        Kq = int(Kmat.size(0))

        if all_neg_ids is None:
            raise RuntimeError("Queue ids missing, cannot mine negative GO ids.")
        if not torch.is_tensor(all_neg_ids):
            all_neg_ids = torch.as_tensor(all_neg_ids, dtype=torch.long)
        all_neg_ids = all_neg_ids.to(device, non_blocking=True).long()  # [Kq]

        sims = prot_query @ Kmat.T  # [B, Kq]

        if getattr(self, "_global_step", 0) % 200 == 0:
            qn = float(prot_query.norm(dim=1).mean().item())
            kn = float(Kmat.norm(dim=1).mean().item())
            frac_finite = float(torch.isfinite(sims).float().mean().item())
            print(f"[DBG] queue sims: finite={frac_finite:.3f} Kq={Kq} norms q={qn:.3f} k={kn:.3f}")

        # false-negative filtering
        if pos_local is not None and len(pos_local) > 0:
            uniq_go_ids_dev = uniq_go_ids.to(device, non_blocking=True).long()
            dag_anc = getattr(self, "dag_ancestors", None)

            for b in range(B):
                loc = pos_local[b]
                if loc is None or int(loc.numel()) == 0:
                    continue
                loc = loc.to(device, non_blocking=True).long()
                pos_ids = uniq_go_ids_dev.index_select(0, loc)

                if dag_anc is not None:
                    ids_list = [int(x) for x in pos_ids.detach().cpu().tolist()]
                    s = set()
                    for gid in ids_list:
                        anc = dag_anc.get(int(gid), None)
                        if anc is None:
                            s.add(int(gid))
                        else:
                            for a in anc:
                                s.add(int(a))
                    excl = torch.as_tensor(list(s), device=device, dtype=torch.long)
                else:
                    excl = pos_ids.unique()

                if excl.numel() > 0:
                    mask = torch.isin(all_neg_ids, excl)
                    sims[b].masked_fill_(mask, float("-inf"))

        k_total = int(self._k_hard_queue_schedule(self._global_step))
        if k_total <= 0:
            return None, None
        k_total = min(k_total, Kq)

        hard_frac = float(self._queue_hard_frac_schedule(self._global_step))
        k_hard = int(round(k_total * hard_frac))
        k_hard = max(0, min(k_hard, k_total))
        k_rand = k_total - k_hard

        neg_idx = torch.empty((B, k_total), device=device, dtype=torch.long)

        for b in range(B):
            row = sims[b]
            finite_mask = torch.isfinite(row)

            if int(finite_mask.sum().item()) == 0:
                perm = torch.randperm(Kq, device=device)[:k_total]
                neg_idx[b] = perm
                continue

            if k_hard > 0:
                hard = torch.topk(row, k=min(k_hard, Kq), dim=0).indices
            else:
                hard = torch.empty((0,), device=device, dtype=torch.long)

            if k_rand > 0:
                pool = torch.nonzero(finite_mask, as_tuple=False).squeeze(1)
                if hard.numel() > 0:
                    hard_set_mask = torch.isin(pool, hard)
                    pool = pool[~hard_set_mask]

                if pool.numel() == 0:
                    pool = hard if hard.numel() > 0 else torch.nonzero(finite_mask, as_tuple=False).squeeze(1)

                if pool.numel() <= k_rand:
                    rand = pool
                    if rand.numel() < k_rand and pool.numel() > 0:
                        extra = pool[torch.randperm(pool.numel(), device=device)[: (k_rand - rand.numel())]]
                        rand = torch.cat([rand, extra], dim=0)
                else:
                    rand = pool[torch.randperm(pool.numel(), device=device)[:k_rand]]
            else:
                rand = torch.empty((0,), device=device, dtype=torch.long)

            sel = torch.cat([hard, rand], dim=0)
            if int(sel.numel()) < k_total:
                need = k_total - int(sel.numel())
                extra = torch.randperm(Kq, device=device)[:need]
                sel = torch.cat([sel, extra], dim=0)

            neg_idx[b] = sel[:k_total]

        neg_ids = all_neg_ids.index_select(0, neg_idx.reshape(-1)).view(B, k_total).contiguous()
        return neg_idx, neg_ids

    def _build_candidates_obsolete(
            self,
            uniq_go_embs,
            pos_local,
            neg_raw_from_queue=None,
            neg_valid_from_queue=None,
            *,
            max_inbatch: int | None = None,
    ):
        """
        Returns:
          G_cand: [B, K, Dg]        RAW GO embeddings only
          pos_mask: [B, K]
          cand_valid_mask: [B, K]
          U: number of in-batch uniq GO candidates kept
          kq: number of queue negatives appended
        """
        device = self.device
        B = len(pos_local)
        U = int(uniq_go_embs.size(0))
        Dg = int(uniq_go_embs.size(1))

        # --------------------------------------------------
        # 1) If U > max_inbatch, shrink in-batch candidates
        #    BUT always keep all positive locals.
        # --------------------------------------------------
        if max_inbatch is not None and U > max_inbatch:
            keep = set()
            for loc in pos_local:
                keep.update(int(x) for x in loc.tolist())
            keep = sorted(keep)

            keep_t = (
                torch.tensor(keep, device=device, dtype=torch.long)
                if len(keep) > 0
                else torch.empty(0, device=device, dtype=torch.long)
            )

            all_idx = torch.arange(U, device=device, dtype=torch.long)
            mask = torch.ones(U, device=device, dtype=torch.bool)
            if keep_t.numel() > 0:
                mask[keep_t] = False
            rest = all_idx[mask]

            need = max(0, int(max_inbatch) - int(keep_t.numel()))
            if need > 0 and rest.numel() > 0:
                perm = torch.randperm(rest.numel(), device=device)[:need]
                extra = rest[perm]
                cand_idx = torch.cat([keep_t, extra], dim=0) if keep_t.numel() > 0 else extra
            else:
                cand_idx = keep_t

            cand_idx = cand_idx.unique(sorted=False)

            uniq_go_embs = uniq_go_embs.index_select(0, cand_idx)
            old2new = {int(old): i for i, old in enumerate(cand_idx.tolist())}

            new_pos_local = []
            for loc in pos_local:
                new_loc = [old2new[int(x)] for x in loc.tolist() if int(x) in old2new]
                new_pos_local.append(torch.tensor(new_loc, device=device, dtype=torch.long))

            pos_local = new_pos_local
            U = int(uniq_go_embs.size(0))

        # --------------------------------------------------
        # 2) If U < max_inbatch, fill with extra negatives
        #    sampled from SEEN GO pool (global ids seen in training),
        #    excluding current batch positives.
        # --------------------------------------------------
        extra_raw_negs = None
        extra_n = 0

        if max_inbatch is not None and U < max_inbatch:
            need = int(max_inbatch) - U

            # batch global positive ids
            batch_global_pos = set()
            if hasattr(self, "_current_uniq_go_ids_for_shortlist") and self._current_uniq_go_ids_for_shortlist is not None:
                uniq_go_ids = self._current_uniq_go_ids_for_shortlist
                if torch.is_tensor(uniq_go_ids):
                    batch_global_pos.update(int(x) for x in uniq_go_ids.detach().cpu().tolist())
                else:
                    batch_global_pos.update(int(x) for x in uniq_go_ids)

            chosen = []

            # ---- 1) structured negatives: siblings + same namespace ----
            structured = self._collect_structured_negatives(
                batch_global_pos=batch_global_pos,
                need=need,
            )
            chosen.extend(structured)

            # ---- 2) fallback random seen negatives ----
            remain = need - len(chosen)
            if remain > 0:
                seen_go_ids = getattr(self, "_eval_seen_go_ids_cpu", None)
                if seen_go_ids is not None:
                    if torch.is_tensor(seen_go_ids):
                        pool_ids = [int(x) for x in seen_go_ids.tolist()]
                    else:
                        pool_ids = [int(x) for x in seen_go_ids]

                    chosen_set = set(chosen)
                    pool_ids = [
                        g for g in pool_ids
                        if g not in batch_global_pos and g not in chosen_set
                    ]

                    if len(pool_ids) > 0:
                        if len(pool_ids) > remain:
                            perm = torch.randperm(len(pool_ids))[:remain]
                            extra = [pool_ids[i] for i in perm.tolist()]
                        else:
                            extra = pool_ids
                        chosen.extend(extra)

            # global GO ids -> go_cache rows
            rows = []
            for gid in chosen:
                row = self.ctx.go_cache.id2row.get(int(gid), None)
                if row is not None:
                    rows.append(int(row))

            if len(rows) > 0:
                rows_t = torch.tensor(rows, dtype=torch.long, device=self.ctx.go_cache.embs.device)
                extra_raw_negs = self.ctx.go_cache.embs.index_select(0, rows_t).to(
                    device=device,
                    dtype=uniq_go_embs.dtype,
                    non_blocking=True
                ).contiguous()
                extra_n = int(extra_raw_negs.size(0))

        # --------------------------------------------------
        # 3) Queue negatives
        # --------------------------------------------------
        kq = 0 if neg_raw_from_queue is None else int(neg_raw_from_queue.size(1))

        # candidate layout:
        # [0:U)                  -> in-batch uniq GO
        # [U:U+extra_n)          -> extra seen negatives
        # [U+extra_n:U+extra_n+kq) -> queue negatives
        K = U + extra_n + kq

        G_cand = torch.zeros(B, K, Dg, device=device, dtype=uniq_go_embs.dtype)
        pos_mask = torch.zeros(B, K, device=device, dtype=torch.bool)
        cand_valid_mask = torch.zeros(B, K, device=device, dtype=torch.bool)

        # --------------------------------------------------
        # 4) in-batch candidates
        # --------------------------------------------------
        G_cand[:, :U] = uniq_go_embs.unsqueeze(0).expand(B, U, Dg)
        cand_valid_mask[:, :U] = True

        for b, loc in enumerate(pos_local):
            if loc.numel() > 0:
                pos_mask[b, loc.to(device)] = True

        # --------------------------------------------------
        # 5) extra seen negatives
        # --------------------------------------------------
        if extra_n > 0:
            s = U
            e = U + extra_n
            G_cand[:, s:e] = extra_raw_negs.unsqueeze(0).expand(B, extra_n, Dg)
            cand_valid_mask[:, s:e] = True
            # pos_mask stays False there

        # --------------------------------------------------
        # 6) queue negatives
        # --------------------------------------------------
        if kq > 0:
            s = U + extra_n
            e = s + kq
            G_cand[:, s:e] = neg_raw_from_queue.to(device)
            if neg_valid_from_queue is None:
                cand_valid_mask[:, s:e] = True
            else:
                cand_valid_mask[:, s:e] = neg_valid_from_queue.to(device)
            if neg_raw_from_queue is not None:
                U = s

        if getattr(self, "_global_step", 0) % 200 == 0:
            print(f"[DBG-NEG] U={U} extra_n={extra_n} kq={kq} K={K}")

        return G_cand, pos_mask, cand_valid_mask, U, kq


    # ----------------- eval space cache -----------------
    @torch.no_grad()
    def _refresh_eval_go_cache(self, chunk: int = 256):
        if self.model.go_encoder is None:
            return  # cache already has embeddings, nothing to refresh via encoder

        if not hasattr(self.ctx, "go_text_store") or self.ctx.go_text_store is None:
            raise RuntimeError("ctx.go_text_store is required for GO refresh")

        eval_ids = [int(x) for x in self.eval_id_list]
        device = self.device

        # pick which encoder to use for cache refresh
        enc = self.model.go_encoder
        was_training = enc.training
        enc.eval()

        toks = self.ctx.go_text_store.batch(eval_ids)
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]

        out_cpu = []
        for s in range(0, input_ids.size(0), chunk):
            e = min(input_ids.size(0), s + chunk)
            embs = enc(
                input_ids=input_ids[s:e].to(device, non_blocking=True),
                attention_mask=attention_mask[s:e].to(device, non_blocking=True),
                output_mode="pooled", #pooled olmalı!
            )

            # unwrap just in case
            if isinstance(embs, tuple):
                embs = embs[0]
            elif isinstance(embs, dict):
                embs = embs["pooled"]
            if embs.dim() != 2:
                raise RuntimeError(f"go_encoder must return [G,D], got {tuple(embs.shape)}")

            # IMPORTANT: go_cache should store RAW encoder space, not projected
            # normalize here only if you decided go_cache is normalized-space
            embs = torch.nan_to_num(embs).float().cpu().contiguous()
            out_cpu.append(embs)

        new_embs_cpu = torch.cat(out_cpu, dim=0).contiguous()
        self.ctx.go_cache.update(eval_ids, new_embs_cpu)

        if was_training:
            enc.train()
    @torch.no_grad()
    def _ensure_eval_cache_v2(self, chunk=1024):

        def to_cols(go_ids):
            cols = [id2col[int(g)] for g in go_ids if int(g) in id2col]
            cols = sorted(set(cols))
            return torch.tensor(cols, dtype=torch.long)  # CPU tensor,

        if getattr(self, "_eval_cache_ready", False):
            return
        print("Eval cache preparation...")
        if not hasattr(self, "eval_id_list") or not self.eval_id_list:
            raise RuntimeError("trainer.eval_id_list missing. Set trainer.eval_id_list = eval_ids in main.")
        if self.ctx is None or not hasattr(self.ctx, "go_cache"):
            raise RuntimeError("trainer.ctx.go_cache missing")

        eval_ids = [int(x) for x in self.eval_id_list]

        # 1) go_cache'de var mı kontrol (id list consistency)
        missing = [g for g in eval_ids if int(g) not in self.ctx.go_cache.id2row]
        if missing:
            raise RuntimeError(f"Eval ids not in go_cache. Missing {len(missing)}. Example: {missing[:10]}")

        # 2) mapping (CPU)
        eval_ids_cpu = torch.as_tensor(eval_ids, dtype=torch.long)  # CPU
        id2col = {int(eval_ids_cpu[i].item()): i for i in range(eval_ids_cpu.numel())}

        # 3) Build eval GO embedding matrix ONCE
        device = self.device

        rows = torch.as_tensor(
            [self.ctx.go_cache.id2row[int(g)] for g in eval_ids],
            dtype=torch.long,
            device=self.ctx.go_cache.embs.device
        )
        G_bank = self.ctx.go_cache.embs.index_select(0, rows).contiguous()
        G_once_cpu = G_bank.detach().float().cpu().contiguous()

        self._eval_cols_seen = to_cols(getattr(self.ctx, "eval_seen_go_ids", []))
        self._eval_cols_rare = to_cols(getattr(self.ctx, "eval_rare_go_ids", []))
        self._eval_cols_unseen = to_cols(getattr(self.ctx, "eval_unseen_ids", []))

        # 4) store on trainer
        self._eval_ids_cpu = eval_ids_cpu
        self._eval_G_once_cpu = G_once_cpu
        self._eval_id2col = id2col
        self._eval_cache_ready = True

    @torch.no_grad()
    def _ensure_eval_cache(self):
        if getattr(self, "_eval_cache_ready", False):
            return
        print("Eval cache preparation...")
        if not hasattr(self, "eval_id_list") or not self.eval_id_list:
            raise RuntimeError("trainer.eval_id_list missing. Set trainer.eval_id_list = eval_ids in main.")
        if self.ctx is None or not hasattr(self.ctx, "go_cache"):
            raise RuntimeError("trainer.ctx.go_cache missing")

        eval_ids = [int(x) for x in self.eval_id_list]

        # 1) go_cache'de var mı kontrol
        missing = [g for g in eval_ids if int(g) not in self.ctx.go_cache.id2row]
        if missing:
            raise RuntimeError(f"Eval ids not in go_cache. Missing {len(missing)}. Example: {missing[:10]}")

        # 2) bank'ten rows çek, ama GPU'ya taşıma: CPU cache oluştur
        rows = torch.as_tensor(
            [self.ctx.go_cache.id2row[int(g)] for g in eval_ids],
            dtype=torch.long,
            device=self.ctx.go_cache.embs.device
        )
        G_bank = self.ctx.go_cache.embs.index_select(0, rows).contiguous()  # bank device
        G_once_cpu = G_bank.detach().to("cpu", non_blocking=False).contiguous()  # CPU cache

        # 3) mapping
        eval_ids_cpu = torch.as_tensor(eval_ids, dtype=torch.long)  # CPU
        id2col = {int(eval_ids_cpu[i].item()): i for i in range(eval_ids_cpu.numel())}

        # 4) store on trainer
        self._eval_ids_cpu = eval_ids_cpu
        self._eval_G_once_cpu = G_once_cpu
        self._eval_id2col = id2col
        self._eval_cache_ready = True

    def _build_eval_space(self, batch):
        self._ensure_eval_cache_v2(chunk=256)
        device = self.device
        B = batch["prot_emb_pad"].size(0)

        # global eval GO matrix (Geval, Dg)
        G_once = self._eval_G_once_cpu.to(device, non_blocking=True)
        Geval, Dg = G_once.size()
        G_eval = G_once.unsqueeze(0).expand(B, Geval, Dg).contiguous()

        y_true = torch.zeros(B, Geval, dtype=torch.float32, device=device)
        id2col = self._eval_id2col

        # use global ids directly (safe)
        for b, gids in enumerate(batch["pos_go_global"]):
            if gids.numel() == 0:
                continue
            for g in gids.tolist():
                j = id2col.get(int(g), None)
                if j is not None:
                    y_true[b, j] = 1.0

        return G_eval, y_true

    def logit_scale_tensor(self):
        if not self.cfg.is_logit_scale_constant:
            return self.logit_scale.clamp(min=-10.0, max=4.6).exp()
        else:
            return self.logit_scale.exp()

    def debug_queue(self):
        q = self.queue_miner
        if q is None:
            print("[QDBG] queue_miner=None")
            return

        print("[QDBG] queue_proj shape:", tuple(q.queue_proj.shape))
        print("[QDBG] queue_proj dtype:", q.queue_proj.dtype)
        print("[QDBG] queue_proj device:", q.queue_proj.device)

        if q.queue_raw is not None:
            print("[QDBG] queue_raw shape:", tuple(q.queue_raw.shape))
            print("[QDBG] queue_raw dtype:", q.queue_raw.dtype)
            print("[QDBG] queue_raw device:", q.queue_raw.device)

        print("[QDBG] ids shape:", tuple(q.ids.shape))
        print("[QDBG] ids device:", q.ids.device)
        print("[QDBG] valid shape:", tuple(q.valid.shape))
        print("[QDBG] valid device:", q.valid.device)

        if hasattr(q, "_ptr"):
            print("[QDBG] ptr:", q._ptr)

        # check if queue has anything
        with torch.no_grad():
            norms = q.queue_proj.float().norm(dim=1)
            valid_gpu = q.valid.to(q.queue_proj.device, non_blocking=True)
            nonzero = ((norms > 1e-6) & valid_gpu).sum().item()
            print(f"[QDBG] nonzero valid proj rows: {nonzero}/{q.queue_proj.size(0)}")

    def _get_uniq_go_embs(self, batch):
        device = self.device

        if self.model.go_encoder is None:
            raise RuntimeError("Training requires go_encoder, but model.go_encoder is None.")

        if "pos_go_tokens" not in batch:
            raise RuntimeError("GO encoder present but pos_go_tokens missing. Fix collator to emit pos_go_tokens.")

        toks = batch["pos_go_tokens"]
        input_ids = toks["input_ids"].to(device, non_blocking=True)
        attn = toks["attention_mask"].to(device, non_blocking=True)

        mode = self.ctx.go_encoder_output_mode

        out = self.model.go_encoder(
            input_ids=input_ids,
            attention_mask=attn,
            output_mode=mode,
        )

        pooled_embs = None
        token_embs = None
        token_mask = None

        # ------------------------------------------
        # 1) Plain tensor output
        # ------------------------------------------
        if isinstance(out, torch.Tensor):
            if mode == "tokens":
                token_embs = out
                token_mask = attn
            else:
                pooled_embs = out

        # ------------------------------------------
        # 2) Tuple output, usually (pooled, attn_weights)
        # ------------------------------------------
        elif isinstance(out, tuple):
            if mode == "tokens":
                token_embs = out[0]
                token_mask = attn
            else:
                pooled_embs = out[0]

        # ------------------------------------------
        # 3) Dict output
        # ------------------------------------------
        elif isinstance(out, dict):
            pooled_embs = out.get("pooled", None)
            token_embs = out.get("tokens", None)
            token_mask = out.get("attention_mask", None)

            # if tokens exist but mask is not explicitly returned, use batch mask
            if token_embs is not None and token_mask is None:
                token_mask = attn

            # backward compatibility fallback
            if pooled_embs is None and token_embs is None:
                hidden = out.get("last_hidden_state", None)
                pooled = out.get("pooler_output", None)

                if mode == "tokens":
                    if hidden is not None:
                        token_embs = hidden
                        token_mask = attn
                    else:
                        raise RuntimeError(
                            "go_encoder output_mode='tokens' but dict output has no token embeddings."
                        )
                else:
                    if pooled is not None:
                        pooled_embs = pooled
                    elif hidden is not None:
                        pooled_embs = hidden[:, 0]
                    else:
                        raise RuntimeError(
                            "go_encoder dict output missing supported keys: "
                            "'pooled', 'tokens', 'attention_mask', "
                            "or legacy 'last_hidden_state'/'pooler_output'"
                        )
        else:
            raise RuntimeError(f"Unsupported go_encoder output type: {type(out)}")

        # ------------------------------------------
        # sanitize
        # ------------------------------------------
        if pooled_embs is not None:
            pooled_embs = torch.nan_to_num(pooled_embs)

        if token_embs is not None:
            token_embs = torch.nan_to_num(token_embs)

        if token_mask is not None:
            token_mask = token_mask.to(device, non_blocking=True)
            if token_mask.dtype != torch.bool:
                token_mask = token_mask != 0

        #TODO: Erase later
        if token_embs is not None:
            with torch.no_grad():
                print("\n[DBG-GO-MASK]")
                print("token_embs:", tuple(token_embs.shape), token_embs.dtype, token_embs.device)

                if token_mask is None:
                    print("WARNING: token_mask is None")
                else:
                    print("token_mask:", tuple(token_mask.shape), token_mask.dtype, token_mask.device)
                    print("valid tokens first rows:", token_mask.sum(dim=-1).detach().cpu().tolist()[:8])

                    pad_frac = (~token_mask.bool()).float().mean().item()
                    print("pad_frac:", pad_frac)

                    assert token_mask.shape[-1] == token_embs.shape[-2], (
                        f"token_mask length {token_mask.shape[-1]} vs token_embs length {token_embs.shape[-2]}"
                    )

        ids = batch["uniq_go_ids"].to(device, non_blocking=True).long()

        # ------------------------------------------
        # validate by requested mode
        # ------------------------------------------
        if mode == "pooled":
            if pooled_embs is None:
                raise RuntimeError("go_encoder_output_mode='pooled' but pooled_embs is None.")
            if pooled_embs.dim() != 2:
                raise RuntimeError(f"Expected pooled_embs [G,D], got {tuple(pooled_embs.shape)}")

        elif mode == "tokens":
            if token_embs is None:
                raise RuntimeError("go_encoder_output_mode='tokens' but token_embs is None.")
            if token_embs.dim() != 3:
                raise RuntimeError(f"Expected token_embs [G,L,D], got {tuple(token_embs.shape)}")
            if token_mask is None:
                raise RuntimeError("go_encoder_output_mode='tokens' but token_mask is None.")
            if token_mask.dim() != 2:
                raise RuntimeError(f"Expected token_mask [G,L], got {tuple(token_mask.shape)}")

        elif mode == "both":
            if pooled_embs is None:
                raise RuntimeError("go_encoder_output_mode='both' but pooled_embs is None.")
            if pooled_embs.dim() != 2:
                raise RuntimeError(f"Expected pooled_embs [G,D], got {tuple(pooled_embs.shape)}")
            if token_embs is None:
                raise RuntimeError("go_encoder_output_mode='both' but token_embs is None.")
            if token_embs.dim() != 3:
                raise RuntimeError(f"Expected token_embs [G,L,D], got {tuple(token_embs.shape)}")
            if token_mask is None:
                raise RuntimeError("go_encoder_output_mode='both' but token_mask is None.")
            if token_mask.dim() != 2:
                raise RuntimeError(f"Expected token_mask [G,L], got {tuple(token_mask.shape)}")

        else:
            raise RuntimeError(f"Unsupported go_encoder_output_mode: {mode}")

        return {
            "pooled_embs": pooled_embs,   # [G,D] or None
            "token_embs": token_embs,     # [G,L,D] or None
            "token_mask": token_mask,     # [G,L] or None
            "ids": ids,                   # [G]
        }

    @torch.no_grad()
    def _encode_go_ids_as_tokens(self, go_ids_2d: torch.Tensor):
        """
        go_ids_2d: [B, K] global GO ids
        returns:
          tok_embs:  [B, K, L, D]
          tok_mask:  [B, K, L]
        """
        if not hasattr(self.ctx, "go_text_store") or self.ctx.go_text_store is None:
            raise RuntimeError("ctx.go_text_store is required for token-align queue negatives.")

        device = self.device
        if go_ids_2d.dtype != torch.long:
            go_ids_2d = go_ids_2d.long()

        B, K = go_ids_2d.shape
        flat_ids = go_ids_2d.reshape(-1).detach().cpu().tolist()

        toks = self.ctx.go_text_store.batch(flat_ids)
        input_ids = toks["input_ids"].to(device, non_blocking=True)
        attention_mask = toks["attention_mask"].to(device, non_blocking=True)

        out = self.model.go_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_mode="tokens",
        )

        if isinstance(out, dict):
            tok = out["tokens"]
            msk = out["attention_mask"]
        elif torch.is_tensor(out):
            tok = out
            msk = attention_mask
        else:
            raise RuntimeError(f"Unsupported token output from go_encoder: {type(out)}")

        if tok.dim() != 3:
            raise RuntimeError(f"Expected token embeddings [N,L,D], got {tuple(tok.shape)}")
        if msk.dim() != 2:
            raise RuntimeError(f"Expected token mask [N,L], got {tuple(msk.shape)}")

        N, L, D = tok.shape
        if N != B * K:
            raise RuntimeError(f"Token encode size mismatch: got N={N}, expected {B*K}")

        tok = torch.nan_to_num(tok).view(B, K, L, D).contiguous()
        if msk.dtype != torch.bool:
            msk = msk != 0
        msk = msk.view(B, K, L).contiguous()

        return tok, msk

    @torch.no_grad()
    def _get_prot_query(self, H: torch.Tensor, attn_valid: torch.Tensor, Dz: int) -> torch.Tensor:
        """
        GO-independent protein query for queue mining.
        Returns [B, Dz]
        """
        if attn_valid is not None and attn_valid.dtype != torch.bool:
            attn_valid = attn_valid != 0

        if self.model.protein_pool_type in {"mean", "go_align"}:
            if attn_valid is not None:
                w = attn_valid.to(H.dtype).unsqueeze(-1)
                denom = w.sum(dim=1).clamp_min(1.0)
                h_pool = (H * w).sum(dim=1) / denom
            else:
                h_pool = H.mean(dim=1)

        elif self.model.protein_pool_type == "attn":
            h_pool, _ = self.model.protein_attn_pool(H, attn_valid)

        elif self.model.protein_pool_type == "slots":
            slots, _ = self.model.slot_extractor(H, attn_valid)   # [B,S,Dh]
            h_pool = slots.mean(dim=1)                            # GO-independent slot summary

        else:
            raise ValueError(f"Unsupported protein_pool_type: {self.model.protein_pool_type}")

        h_pool = self.model.protein_ln(h_pool)
        q = self.model.proj_p(h_pool)

        if getattr(self.model, "normalize", False):
            q = self.model._norm(q, dim=-1)

        if q.size(1) != int(Dz):
            raise RuntimeError(f"prot_query dim mismatch: got {q.size(1)} expected {Dz}")

        return q

    def _build_candidates_meta(
            self,
            uniq_go_ids: torch.Tensor,              # [U] global GO ids for current batch uniqs
            pos_local,
            neg_ids_from_queue: torch.Tensor | None = None,   # [B,kq] global ids
            *,
            max_inbatch: int | None = None,
    ):
        """
        Returns metadata, not candidate tensors.

        Layout:
          [0:U_kept)        -> in-batch uniq GO locals
          [U_kept:U_kept+extra_n) -> extra seen negatives (global ids)
          [.. + kq)         -> queue negatives (global ids)
        """
        device = self.device
        U0 = int(uniq_go_ids.numel())
        B = len(pos_local)

        # -----------------------------
        # 1) shrink in-batch if needed
        # -----------------------------
        if max_inbatch is not None and U0 > max_inbatch:
            keep = set()
            for loc in pos_local:
                keep.update(int(x) for x in loc.tolist())
            keep = sorted(keep)

            keep_t = (
                torch.tensor(keep, device=device, dtype=torch.long)
                if len(keep) > 0
                else torch.empty(0, device=device, dtype=torch.long)
            )

            all_idx = torch.arange(U0, device=device, dtype=torch.long)
            mask = torch.ones(U0, device=device, dtype=torch.bool)
            if keep_t.numel() > 0:
                mask[keep_t] = False
            rest = all_idx[mask]

            need = max(0, int(max_inbatch) - int(keep_t.numel()))
            if need > 0 and rest.numel() > 0:
                perm = torch.randperm(rest.numel(), device=device)[:need]
                extra = rest[perm]
                inbatch_local_idx = torch.cat([keep_t, extra], dim=0) if keep_t.numel() > 0 else extra
            else:
                inbatch_local_idx = keep_t

            inbatch_local_idx = inbatch_local_idx.unique(sorted=False)

            old2new = {int(old): i for i, old in enumerate(inbatch_local_idx.tolist())}
            new_pos_local = []
            for loc in pos_local:
                new_loc = [old2new[int(x)] for x in loc.tolist() if int(x) in old2new]
                new_pos_local.append(torch.tensor(new_loc, device=device, dtype=torch.long))
            pos_local = new_pos_local
        else:
            inbatch_local_idx = torch.arange(U0, device=device, dtype=torch.long)

        U = int(inbatch_local_idx.numel())
        inbatch_global_ids = uniq_go_ids.index_select(0, inbatch_local_idx)   # [U]

        # -----------------------------
        # 2) extra seen negatives
        # -----------------------------
        extra_global_ids = torch.empty(0, device=device, dtype=torch.long)
        if max_inbatch is not None and U < max_inbatch:
            need = int(max_inbatch) - U

            batch_global_pos = set(int(x) for x in inbatch_global_ids.detach().cpu().tolist())

            chosen = []
            structured = self._collect_structured_negatives(
                batch_global_pos=batch_global_pos,
                need=need,
            )
            chosen.extend(structured)

            remain = need - len(chosen)
            if remain > 0:
                seen_go_ids = getattr(self, "_eval_seen_go_ids_cpu", None)
                if seen_go_ids is not None:
                    if torch.is_tensor(seen_go_ids):
                        pool_ids = [int(x) for x in seen_go_ids.tolist()]
                    else:
                        pool_ids = [int(x) for x in seen_go_ids]

                    chosen_set = set(chosen)
                    pool_ids = [
                        g for g in pool_ids
                        if g not in batch_global_pos and g not in chosen_set
                    ]

                    if len(pool_ids) > 0:
                        if len(pool_ids) > remain:
                            perm = torch.randperm(len(pool_ids))[:remain]
                            extra = [pool_ids[i] for i in perm.tolist()]
                        else:
                            extra = pool_ids
                        chosen.extend(extra)

            if len(chosen) > 0:
                extra_global_ids = torch.tensor(chosen, dtype=torch.long, device=device)

        extra_n = int(extra_global_ids.numel())
        kq = 0 if neg_ids_from_queue is None else int(neg_ids_from_queue.size(1))
        K = U + extra_n + kq

        pos_mask = torch.zeros(B, K, device=device, dtype=torch.bool)
        cand_valid_mask = torch.zeros(B, K, device=device, dtype=torch.bool)

        # in-batch always valid
        cand_valid_mask[:, :U] = True
        for b, loc in enumerate(pos_local):
            if loc.numel() > 0:
                pos_mask[b, loc.to(device)] = True

        # extra negatives valid
        if extra_n > 0:
            cand_valid_mask[:, U:U + extra_n] = True

        # queue negatives valid
        if kq > 0:
            cand_valid_mask[:, U + extra_n:U + extra_n + kq] = True

        return {
            "inbatch_local_idx": inbatch_local_idx,      # [U]
            "inbatch_global_ids": inbatch_global_ids,    # [U]
            "extra_global_ids": extra_global_ids,        # [extra_n]
            "queue_global_ids": neg_ids_from_queue,      # [B,kq] or None
            "pos_mask": pos_mask,                        # [B,K]
            "cand_valid_mask": cand_valid_mask,          # [B,K]
            "U": U,
            "extra_n": extra_n,
            "kq": kq,
            "K": K,
            "pos_local_kept": pos_local,
        }

    def _build_candidate_tensors_pooled(
            self,
            meta: dict,
            pooled_go: torch.Tensor,                     # [G,D]
            uniq_go_ids: torch.Tensor,                   # [G] global
            neg_raw_from_queue: torch.Tensor | None = None,   # [B,kq,D]
    ):
        device = self.device
        B = len(meta["pos_local_kept"])
        Dg = int(pooled_go.size(1))
        K = int(meta["K"])
        U = int(meta["U"])
        extra_n = int(meta["extra_n"])
        kq = int(meta["kq"])

        G_cand = torch.zeros(B, K, Dg, device=device, dtype=pooled_go.dtype)

        # in-batch
        inbatch_local_idx = meta["inbatch_local_idx"]
        inbatch_embs = pooled_go.index_select(0, inbatch_local_idx)  # [U,D]
        G_cand[:, :U] = inbatch_embs.unsqueeze(0).expand(B, U, Dg)

        # extra negatives from go_cache / go_cache current bank
        if extra_n > 0:
            extra_global_ids = meta["extra_global_ids"]
            rows = []
            for gid in extra_global_ids.detach().cpu().tolist():
                row = self.ctx.go_cache.id2row.get(int(gid), None)
                if row is not None:
                    rows.append(int(row))
            if len(rows) != extra_n:
                raise RuntimeError("Some extra_global_ids are missing in go_cache.id2row.")
            rows_t = torch.tensor(rows, dtype=torch.long, device=self.ctx.go_cache.embs.device)
            extra_raw = self.ctx.go_cache.embs.index_select(0, rows_t).to(
                device=device,
                dtype=pooled_go.dtype,
                non_blocking=True,
            ).contiguous()
            G_cand[:, U:U + extra_n] = extra_raw.unsqueeze(0).expand(B, extra_n, Dg)

        # queue negatives
        if kq > 0:
            if neg_raw_from_queue is None:
                raise RuntimeError("kq>0 but neg_raw_from_queue is None.")
            G_cand[:, U + extra_n:U + extra_n + kq] = neg_raw_from_queue.to(device)

        return G_cand

    def _gather_token_candidates(
            self,
            token_embs: torch.Tensor,      # [G,L,D]
            token_mask: torch.Tensor,      # [G,L]
            local_idx_1d: torch.Tensor,    # [U]
            B: int,
    ):
        gathered_tok = token_embs.index_select(0, local_idx_1d)   # [U,L,D]
        gathered_msk = token_mask.index_select(0, local_idx_1d)   # [U,L]

        U, L, D = gathered_tok.shape
        G_tok = gathered_tok.unsqueeze(0).expand(B, U, L, D).contiguous()
        G_msk = gathered_msk.unsqueeze(0).expand(B, U, L).contiguous()
        return G_tok, G_msk

    def _pad_token_block(self, tok: torch.Tensor, msk: torch.Tensor, target_L: int):
        """
        tok: [B, K, L, D]
        msk: [B, K, L]
        returns:
          tok_padded: [B, K, target_L, D]
          msk_padded: [B, K, target_L]
        """
        if tok.dim() != 4:
            raise RuntimeError(f"tok must be [B,K,L,D], got {tuple(tok.shape)}")
        if msk.dim() != 3:
            raise RuntimeError(f"msk must be [B,K,L], got {tuple(msk.shape)}")

        B, K, L, D = tok.shape

        if L == target_L:
            return tok, msk

        if L > target_L:
            return tok[:, :, :target_L, :].contiguous(), msk[:, :, :target_L].contiguous()

        pad_len = target_L - L
        pad_tok = torch.zeros(B, K, pad_len, D, device=tok.device, dtype=tok.dtype)
        pad_msk = torch.zeros(B, K, pad_len, device=msk.device, dtype=msk.dtype)

        tok = torch.cat([tok, pad_tok], dim=2)
        msk = torch.cat([msk, pad_msk], dim=2)
        return tok.contiguous(), msk.contiguous()

    def _build_candidate_tensors_token(
            self,
            meta: dict,
            token_go: torch.Tensor,                 # [G,L,D]
            token_go_mask: torch.Tensor,            # [G,L]
            queue_global_ids: torch.Tensor | None = None,  # [B,kq]
    ):
        device = self.device
        B = len(meta["pos_local_kept"])
        U = int(meta["U"])
        extra_n = int(meta["extra_n"])
        kq = int(meta["kq"])
        K = int(meta["K"])

        blocks_tok = []
        blocks_msk = []
        block_sizes = []

        # -----------------------------------
        # 1) in-batch token candidates
        # -----------------------------------
        G_in_tok, G_in_msk = self._gather_token_candidates(
            token_embs=token_go,
            token_mask=token_go_mask,
            local_idx_1d=meta["inbatch_local_idx"],
            B=B,
        )  # [B,U,L,D], [B,U,L]

        blocks_tok.append(G_in_tok)
        blocks_msk.append(G_in_msk)
        block_sizes.append(U)

        # -----------------------------------
        # 2) extra negatives
        # -----------------------------------
        if extra_n > 0:
            extra_ids = meta["extra_global_ids"].view(1, extra_n).expand(B, extra_n).contiguous()
            extra_tok, extra_msk = self._encode_go_ids_as_tokens(extra_ids)  # [B,extra_n,Lx,D], [B,extra_n,Lx]

            blocks_tok.append(extra_tok)
            blocks_msk.append(extra_msk)
            block_sizes.append(extra_n)

        # -----------------------------------
        # 3) queue negatives
        # -----------------------------------
        if kq > 0:
            if queue_global_ids is None:
                raise RuntimeError("kq>0 but queue_global_ids is None in token mode.")

            q_tok, q_msk = self._encode_go_ids_as_tokens(queue_global_ids)  # [B,kq,Lq,D], [B,kq,Lq]
            blocks_tok.append(q_tok)
            blocks_msk.append(q_msk)
            block_sizes.append(kq)

        # -----------------------------------
        # 4) unify length by padding to Lmax
        # -----------------------------------
        Lmax = max(int(x.size(2)) for x in blocks_tok)
        Dg = int(blocks_tok[0].size(3))

        padded_tok = []
        padded_msk = []
        for tok, msk in zip(blocks_tok, blocks_msk):
            tok_p, msk_p = self._pad_token_block(tok, msk, target_L=Lmax)
            padded_tok.append(tok_p)
            padded_msk.append(msk_p)

        # -----------------------------------
        # 5) concatenate on candidate dim
        # -----------------------------------
        G_cand = torch.cat(padded_tok, dim=1).contiguous()       # [B,K,Lmax,D]
        G_cand_mask = torch.cat(padded_msk, dim=1).contiguous()  # [B,K,Lmax]

        if G_cand.size(1) != K:
            raise RuntimeError(
                f"Token candidate size mismatch after concat: got K={G_cand.size(1)}, expected {K}"
            )

        if G_cand.size(3) != Dg:
            raise RuntimeError(
                f"Token candidate hidden dim mismatch: got D={G_cand.size(3)}, expected {Dg}"
            )

        return G_cand, G_cand_mask

    def forward_scores(self, H, G, mask, return_alpha=False, cand_chunk_k=32, pos_chunk_t=256, go_mask=None, **kwargs):
        cand_chunk_k = int(getattr(self.cfg, "cand_chunk_k", cand_chunk_k))
        pos_chunk_t = int(getattr(self.cfg, "pos_chunk_t", pos_chunk_t))

        def _unpack(out):
            if torch.is_tensor(out):
                return out, None, {}
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                return out[0], None, (out[1] or {})
            if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[1]):
                return out[0], out[1], {}
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple):
                scores, logits = out[0]
                return scores, logits, (out[1] or {})
            raise RuntimeError(f"Unsupported model output type/shape: {type(out)}")

        # token candidates [B,K,L,D]
        if G.dim() == 4:
            scores_all = []
            alpha_all = [] if return_alpha else None
            B, K, L, Dg = G.shape
            for ks in range(0, K, cand_chunk_k):
                ke = min(K, ks + cand_chunk_k)
                g_chunk = G[:, ks:ke].contiguous()
                gm_chunk = None if go_mask is None else go_mask[:, ks:ke].contiguous()

                out = self.model(
                    H=H,
                    G=g_chunk,
                    mask=mask,
                    go_mask=gm_chunk,
                    return_alpha=return_alpha,
                    cand_chunk_k=None,
                    pos_chunk_t=pos_chunk_t,
                    **kwargs,
                )
                sc, _, alpha = _unpack(out)
                scores_all.append(sc)
                if return_alpha:
                    alpha_all.append(alpha)

            scores = torch.cat(scores_all, dim=1)

            if return_alpha:
                merged = {}
                if len(alpha_all) > 0:
                    # keep only keys that can be concatenated on candidate dim
                    keys = set()
                    for a in alpha_all:
                        keys.update(a.keys())
                    for k in keys:
                        vals = [a[k] for a in alpha_all if k in a]
                        if len(vals) == 0:
                            continue
                        try:
                            # candidate dim assumed dim=2 for slot_token_sim [B,S,K,L]
                            # or dim=2 for best_token_per_slot [B,S,K]
                            merged[k] = torch.cat(vals, dim=2)
                        except Exception:
                            try:
                                merged[k] = torch.cat(vals, dim=1)
                            except Exception:
                                merged[k] = vals[0]
                return scores, merged

            return scores

        out = self.model(
            H=H,
            G=G,
            mask=mask,
            go_mask=go_mask,
            return_alpha=return_alpha,
            cand_chunk_k=cand_chunk_k,
            pos_chunk_t=pos_chunk_t,
            **kwargs,
        )
        sc, logits, alpha = _unpack(out)

        if G.dim() == 3:
            assert sc.dim() == 2 and sc.size(0) == H.size(0), "forward_scores: bad score shape"

        if return_alpha:
            return sc, alpha

        return sc

    def step_losses(self, batch, epoch_idx: int, debug: bool = False):
        self.model.train()
        device = self.device

        if self.model.go_encoder is not None:
            self._set_group_lr("go_lora", self._lora_lr_schedule(self._global_step))

        if debug:
            if ("pos_go_tokens" in batch) and ("uniq_go_ids" in batch):
                assert batch["pos_go_tokens"]["input_ids"].size(0) == batch["uniq_go_ids"].numel(), \
                    "pos_go_tokens and uniq_go_ids size mismatch"

        H = batch["prot_emb_pad"].to(device, non_blocking=True)
        attn_valid, pad_mask = self._valid_and_pad_masks(batch)

        if self.to_f32 is not None:
            H = self.to_f32(H)

        pos_local = batch["pos_go_local"]

        go_pack = self._get_uniq_go_embs(batch)
        uniq_go_ids = go_pack["ids"]                     # [G]
        pooled_go = go_pack["pooled_embs"]              # [G,D] or None
        token_go = go_pack["token_embs"]                # [G,L,D] or None
        token_go_mask = go_pack["token_mask"]           # [G,L] or None

        if self._global_step % 200 == 0:
            if pooled_go is not None:
                _tstats(pooled_go, "uniq_go_pooled(raw)")
            if token_go is not None:
                _tstats(token_go, "uniq_go_tokens(raw)")
            print("[DBG] uniq_go_ids:", tuple(uniq_go_ids.shape), "uniq=", int(torch.unique(uniq_go_ids).numel()))

        if getattr(self.model, "go_encoder", None) is not None:
            assert "pos_go_tokens" in batch, "GO encoder present but pos_go_tokens missing, LoRA won't train"

        Dz = self._get_Dz()
        self._maybe_init_queue(Dz)
        self._maybe_activate_queue()

        amp_ctx = torch.amp.autocast(
            device_type="cuda",
            enabled=(torch.cuda.is_available() and self.ctx.fp16_enabled),
        )

        # -----------------------------
        # queue mining always in pooled/proj space
        # -----------------------------
        with torch.no_grad():
            prot_query = self._get_prot_query(H, attn_valid, Dz)

            neg_idx_from_queue = None
            neg_ids_from_queue = None
            neg_raw_from_queue = None

            if self._queue_active() and self.queue_miner is not None:
                neg_idx_from_queue, neg_ids_from_queue = self._mine_queue_hard_neg_ids(
                    prot_query, pos_local, uniq_go_ids
                )

                # pooled raw negatives only needed in pooled mode
                if (not self._use_token_align) and neg_idx_from_queue is not None:
                    if pooled_go is None:
                        raise RuntimeError("pooled_go is required for pooled queue negatives.")
                    neg_raw_from_queue, _ = self._fetch_raw_go_embs_from_queue(
                        neg_idx_from_queue,
                        dtype=pooled_go.dtype,
                    )

            if self._global_step % 200 == 0:
                dbg_go = pooled_go if pooled_go is not None else None
                self._dbg_norms(prot_query=prot_query, uniq_go_embs=dbg_go, tag="[pre-enq]")
                if neg_ids_from_queue is not None:
                    print("[DBG] queue neg ids shape:", tuple(neg_ids_from_queue.shape))
                if neg_raw_from_queue is not None:
                    print("[DBG] queue neg raw shape:", tuple(neg_raw_from_queue.shape))

        max_inbatch = None
        if getattr(self.cfg, "max_inbatch", None) is not None:
            try:
                max_inbatch = int(self.cfg.max_inbatch)
            except Exception:
                max_inbatch = None

        self._current_uniq_go_ids_for_shortlist = uniq_go_ids.detach().cpu()

        meta = self._build_candidates_meta(
            uniq_go_ids=uniq_go_ids,
            pos_local=pos_local,
            neg_ids_from_queue=neg_ids_from_queue,
            max_inbatch=max_inbatch,
        )

        pos_local_kept = meta["pos_local_kept"]
        pos_mask = meta["pos_mask"]
        cand_valid_mask = meta["cand_valid_mask"]
        U = int(meta["U"])
        extra_n = int(meta["extra_n"])
        kq = int(meta["kq"])

        if self._use_token_align:
            if token_go is None or token_go_mask is None:
                raise RuntimeError("token_align mode requires token_go and token_go_mask.")
            G_cand, G_cand_mask = self._build_candidate_tensors_token(
                meta=meta,
                token_go=token_go,
                token_go_mask=token_go_mask,
                queue_global_ids=meta["queue_global_ids"],
            )
        else:
            if pooled_go is None:
                raise RuntimeError("pooled mode requires pooled_go.")
            G_cand = self._build_candidate_tensors_pooled(
                meta=meta,
                pooled_go=pooled_go,
                uniq_go_ids=uniq_go_ids,
                neg_raw_from_queue=neg_raw_from_queue,
            )
            G_cand_mask = None

        with amp_ctx:
            scores_cand = self.forward_scores(
                H,
                G_cand,
                attn_valid,
                go_mask=G_cand_mask,
                return_alpha=False,
            )

            if self._global_step % 200 == 0:
                _tstats(scores_cand, "scores_cand(pre_scale)")
                grad = self.logit_scale.grad.item() if self.logit_scale.grad is not None else None
                print(
                    f"[DBG] logit_scale_raw={self.logit_scale.item():.8f} "
                    f"logit_scale_exp={self.logit_scale.exp().item():.8f} "
                    f"logit_scale_clamped_exp={self.logit_scale.clamp(min=-10.0, max=4.6).exp().item():.8f} "
                    f"logit_scale_grad={grad}"
                )

            assert scores_cand.requires_grad, "scores_cand grad not enabled"

            scale = self.logit_scale_tensor()
            scores_cand = scores_cand * scale

            # queue weighting only on queue tail
            if kq > 0:
                q_start = U + extra_n
                queue_w = self._queue_weight_schedule(self._global_step)
                scores_cand[:, q_start:q_start + kq] *= queue_w

            #TODO: Erase debug
            with torch.no_grad():
                print("\n[DBG-TARGET]")
                print("scores_cand:", tuple(scores_cand.shape), scores_cand.dtype)

                if "cand_ids" in batch:
                    cand_ids = batch["cand_ids"]
                    print("cand_ids:", tuple(cand_ids.shape))
                    print("cand_ids[0][:20]:", cand_ids[0, :20].detach().cpu().tolist())

                if "labels" in batch:
                    labels = batch["labels"]
                    print("labels:", tuple(labels.shape), labels.dtype)
                    print("labels unique:", torch.unique(labels.detach().cpu()).tolist())
                    print("labels positives per row:", labels.sum(dim=1).detach().cpu().tolist())
                    print("labels[0][:20]:", labels[0, :20].detach().cpu().tolist())

                    pos_mask = labels.bool()
                    assert scores_cand.shape == pos_mask.shape, (
                        f"scores_cand {scores_cand.shape} vs pos_mask {pos_mask.shape}"
                    )

                    pos_per_row = pos_mask.sum(dim=1)
                    assert (pos_per_row > 0).all(), f"Some rows have no positives: {pos_per_row.tolist()}"
                    assert (pos_per_row < pos_mask.size(1)).all(), f"All-positive rows: {pos_per_row.tolist()}"

            with torch.no_grad():
                sc = scores_cand.detach().float()
                print("\n[DBG-SCORES]")
                print(
                    "min/max/mean/std:",
                    sc.min().item(),
                    sc.max().item(),
                    sc.mean().item(),
                    sc.std().item(),
                )

                if "labels" in batch:
                    pos_mask = batch["labels"].bool().to(sc.device)
                    pos_scores = sc[pos_mask]
                    neg_scores = sc[~pos_mask]

                    print(
                        "pos mean/std/n:",
                        pos_scores.mean().item() if pos_scores.numel() else None,
                        pos_scores.std().item() if pos_scores.numel() > 1 else None,
                        pos_scores.numel(),
                    )
                    print(
                        "neg mean/std/n:",
                        neg_scores.mean().item() if neg_scores.numel() else None,
                        neg_scores.std().item() if neg_scores.numel() > 1 else None,
                        neg_scores.numel(),
                    )

            l_con = multi_positive_infonce_from_candidates_v2(
                scores_cand,
                pos_mask,
                tau=1.0,
                cand_valid_mask=cand_valid_mask,
            )

            if not torch.isfinite(l_con):
                raise RuntimeError("contrastive loss NaN, batch protein_ids=" + str(batch.get("protein_ids", "")[:5]))

            # positives-only tensors for attr / dag
            B = H.size(0)
            T_max = max((int(x.numel()) for x in pos_local_kept), default=1)

            if self._use_token_align:
                L = token_go.size(1)
                Dg_batch = token_go.size(2)
                G_pos = torch.zeros(B, T_max, L, Dg_batch, device=device, dtype=token_go.dtype)
                G_pos_mask = torch.zeros(B, T_max, L, device=device, dtype=torch.bool)
                for b, loc in enumerate(pos_local_kept):
                    t = int(loc.numel())
                    if t > 0:
                        G_pos[b, :t] = token_go.index_select(0, loc.to(token_go.device))
                        G_pos_mask[b, :t] = token_go_mask.index_select(0, loc.to(token_go_mask.device))
            else:
                Dg_batch = int(pooled_go.size(1))
                G_pos = torch.zeros(B, T_max, Dg_batch, device=device, dtype=pooled_go.dtype)
                G_pos_mask = None
                for b, loc in enumerate(pos_local_kept):
                    t = int(loc.numel())
                    if t > 0:
                        G_pos[b, :t] = pooled_go.index_select(0, loc.to(pooled_go.device))

            use_attr = (not self._use_token_align) and self.ctx.attribute_loss_enabled and (
                epoch_idx < self.attr.curriculum_epochs and self.attr.lambda_attr > 0.0
            )
            if use_attr:
                scores_pos, alpha_info = self.forward_scores(
                    H, G_pos, attn_valid, return_alpha=True
                )
            else:
                alpha_info = {}

        # attr + entropy
        if use_attr and alpha_info and ("alpha_full" in alpha_info):
            alpha = alpha_info["alpha_full"]
            delta = delta_y_from_occlusion_windows(
                H=H.detach(),
                G_pos=G_pos.detach(),
                model=self.model,
                valid_mask=attn_valid,
                window=32,
                stride=16,
                mask_value=0.0,
                chunk_windows=32,
            )
            l_attr = attribution_loss(alpha, delta, mask=None, reduce="mean")
            l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(alpha)
        else:
            l_attr = torch.zeros((), device=device)
            l_ent = torch.zeros((), device=device)

        # DAG only on pooled positive path for now
        l_dag = torch.zeros((), device=device)
        if (not self._use_token_align) and self.attr.lambda_dag > 0:
            pos_go_ids = self._build_pos_go_ids(pos_local_kept, uniq_go_ids)
            with amp_ctx:
                scores_pos_dag = self.forward_scores(H.detach(), G_pos, attn_valid, return_alpha=False)
            scores_pos_dag_f32 = scores_pos_dag.float()
            l_dag = dag_consistency_loss_pos_ids(
                scores_pos_dag_f32,
                pos_go_ids,
                self.ctx.dag_parents,
                margin=0.0,
                scale=1.0
            )


        total = l_con + self.attr.lambda_dag * l_dag + self.attr.lambda_attr * l_attr + l_ent

        self._global_step += 1

        if self.go_encoder_k is not None:
            ema_update(self.model.go_encoder, self.go_encoder_k, m=self.m_ema)

        # enqueue always pooled/raw representations
        if self._queue_active() and self.queue_miner is not None:
            with torch.no_grad():
                local_idx_list = [loc.to(device) for loc in pos_local_kept if loc.numel() > 0]
                if local_idx_list:
                    local_cat = torch.unique(torch.cat(local_idx_list, dim=0))

                    if ("pos_go_tokens" in batch) and (self.go_encoder_k is not None):
                        toks = batch["pos_go_tokens"]
                        assert toks["input_ids"].size(0) == uniq_go_ids.size(0)

                        pos_vecs_raw = self.go_encoder_k(
                            input_ids=toks["input_ids"].to(device, non_blocking=True),
                            attention_mask=toks["attention_mask"].to(device, non_blocking=True),
                            output_mode="pooled",
                        )
                        if isinstance(pos_vecs_raw, tuple):
                            pos_vecs_raw = pos_vecs_raw[0]
                        elif isinstance(pos_vecs_raw, dict):
                            pos_vecs_raw = pos_vecs_raw["pooled"]

                        pos_vecs_raw = pos_vecs_raw.index_select(0, local_cat)
                    else:
                        if pooled_go is None:
                            raise RuntimeError("Queue enqueue requires pooled_go or EMA pooled encoding.")
                        pos_vecs_raw = pooled_go.index_select(0, local_cat)

                    if self._global_step % 1000 == 0:
                        self._dbg_norms(pos_vecs=pos_vecs_raw, tag="[enq-raw]")

                    pos_vecs_proj = self.model.go_ln(pos_vecs_raw)
                    pos_vecs_proj = self.model.proj_g(pos_vecs_proj)
                    pos_vecs_proj = self.normalizer(pos_vecs_proj, dim=1)

                    pos_ids = uniq_go_ids.index_select(0, local_cat).detach()
                    self.queue_miner.enqueue(
                        proj_vecs=pos_vecs_proj.detach(),
                        raw_vecs=pos_vecs_raw.detach(),
                        ids=pos_ids
                    )

                    if self._global_step % 1000 == 0:
                        self.debug_queue()
                        self._dbg_norms(pos_vecs=pos_vecs_proj, tag="[enq-proj]")

        try:
            self.wandb_run.log(
                {
                    "trainer_step": self._global_step,
                    "train/total": float(total.detach().item()),
                    "train/contrastive": float(l_con.detach().item()),
                    "train/dag": float(l_dag.detach().item()),
                    "train/attr": float(l_attr.detach().item()),
                    "train/entropy": float(l_ent.detach().item()),
                    "train/logit_scale": float(self.logit_scale.detach().exp().item()),
                    "train/lr_go_lora": float(
                        self._lora_lr_schedule(self._global_step - 1)
                    ) if self.model.go_encoder is not None else 0.0,
                    "train/queue_hard_frac": float(self._queue_hard_frac_schedule(self._global_step)),
                    "train/queue_weight_eff": float(self._queue_weight_schedule(self._global_step)),
                    "train/k_hard_queue_eff": int(self._k_hard_queue_schedule(self._global_step)),
                },
                step=int(self._global_step),
            )
        except Exception:
            pass

        return {
            "total": total,
            "contrastive": l_con,
            "dag": l_dag,
            "attr": l_attr,
            "entropy": l_ent
        }


    @torch.no_grad()
    def _subset_cols_from_ids(self, ids: list[int] | set[int] | torch.Tensor | None):
        """
        ids: global GO ids (int), subset of eval_id_list
        returns CPU LongTensor of column indices into OBSERVED eval matrix
        """
        if ids is None:
            return torch.empty(0, dtype=torch.long)
        if torch.is_tensor(ids):
            ids_list = [int(x) for x in ids.detach().cpu().long().flatten().tolist()]
        else:
            ids_list = [int(x) for x in ids]
        id2col = self._eval_id2col or {}
        cols = [id2col[g] for g in ids_list if g in id2col]
        cols = sorted(set(cols))
        return torch.tensor(cols, dtype=torch.long)

    @torch.no_grad()
    def _ancestor_recall_at_k(self, scores_full: torch.Tensor, y_true_full: torch.Tensor, k: int):
        """
        No Post-processing.
        Ranking observed (scores_full).
        Ground truth = Ancestors(positives) ∩ observed_go
        Returns: (mean_recall, num_valid_proteins)
        """
        device = scores_full.device
        B, G = y_true_full.shape
        k = min(int(k), int(G))
        if k <= 0 or B == 0:
            return 0.0, 0

        # mapping: col -> global gid
        eval_ids_cpu = self._eval_ids_cpu  # CPU LongTensor [G]
        id2col = self._eval_id2col  # dict: gid -> col
        if self.dag_anc is None:
            return 0.0, 0

        # topk over observed
        topk = torch.topk(scores_full, k=k, dim=1).indices  # [B,k]
        topk_cpu = topk.detach().cpu()

        y_cpu = y_true_full.detach().cpu()

        sum_rec = 0.0
        num = 0

        for b in range(B):
            # positives in observed cols
            pos_cols = torch.nonzero(y_cpu[b] > 0, as_tuple=False).flatten().tolist()
            if not pos_cols:
                continue

            # positives -> global GO ids
            pos_gids = eval_ids_cpu.index_select(0, torch.tensor(pos_cols, dtype=torch.long)).tolist()

            # ancestor closure in global ids
            anc_set = set()
            for gid in pos_gids:
                # dag_ancestors includes itself in your build_dag_ancestors
                anc = self.dag_anc.get(go_str_to_int_any(gid), None)
                if anc is None:
                    anc_set.add(go_str_to_int_any(gid))
                else:
                    for a in anc:
                        anc_set.add(go_str_to_int_any(a[0]))

            # restrict to observed by mapping to columns
            anc_cols = set()
            for a in anc_set:
                j = id2col.get(int(a), None)
                if j is not None:
                    anc_cols.add(int(j))

            denom = len(anc_cols)
            if denom == 0:
                continue

            # hits among topk
            pred_cols = topk_cpu[b].tolist()
            hits = sum((int(c) in anc_cols) for c in pred_cols)
            sum_rec += hits / denom
            num += 1

        if num == 0:
            return 0.0, 0
        return float(sum_rec / num), int(num)

    @torch.no_grad()
    def _recall_at_k_on_subset(self, scores_full: torch.Tensor, y_true_full: torch.Tensor,
                               subset_cols_cpu: torch.Tensor, k: int):
        """
        Ranking is over FULL observed space (scores_full).
        Hits counted only if the predicted GO is in subset_cols AND is a true positive.
        Returns: (recall_mean, num_valid_proteins)
        """
        device = scores_full.device
        B, G = y_true_full.shape
        k = min(int(k), int(G))
        if k <= 0:
            return 0.0, 0

        subset_cols = subset_cols_cpu.to(device, non_blocking=True)
        if subset_cols.numel() == 0:
            return 0.0, 0

        # subset ground truth in full space
        y_sub = torch.zeros((B, G), device=device, dtype=y_true_full.dtype)
        y_sub[:, subset_cols] = y_true_full[:, subset_cols]

        topk = torch.topk(scores_full, k=k, dim=1).indices  # [B,k] over OBSERVED
        hits = torch.gather(y_sub, 1, topk)  # [B,k] 1s only for subset true positives

        denom = y_sub.sum(dim=1)  # [B]
        valid = denom > 0
        if valid.any():
            rec = (hits.sum(dim=1) / denom.clamp_min(1.0))
            return float(rec[valid].mean().item()), int(valid.sum().item())
        return 0.0, 0

    @torch.no_grad()
    def _token_score_full_chunked(
            self,
            H: torch.Tensor,                 # [B,T,Dh]
            attn_valid: torch.Tensor,        # [B,T]
            chunk_k: int = 128,
            return_cpu: bool = True,
    ):
        """
        Exhaustive token-level scoring over the full eval GO space.

        Returns:
          scores_full: [B, G] on CPU if return_cpu=True, else on device
        """
        if not hasattr(self, "_eval_tok_ids_cpu"):
            raise RuntimeError("Token eval cache not prepared. Call _refresh_eval_go_token_cache first.")
        if not hasattr(self, "_eval_tok_embs_cpu"):
            raise RuntimeError("Token eval token embeddings cache missing.")
        if not hasattr(self, "_eval_tok_mask_cpu"):
            raise RuntimeError("Token eval token mask cache missing.")

        device = self.device
        B = H.size(0)

        go_tok_cpu = self._eval_tok_embs_cpu   # [G,L,D]
        go_msk_cpu = self._eval_tok_mask_cpu   # [G,L]
        G_total = int(go_tok_cpu.size(0))

        out_chunks = []

        for s in range(0, G_total, chunk_k):
            e = min(G_total, s + chunk_k)

            tok_chunk = go_tok_cpu[s:e].to(device, non_blocking=True)   # [C,L,D]
            msk_chunk = go_msk_cpu[s:e].to(device, non_blocking=True)   # [C,L]
            C = int(tok_chunk.size(0))

            G_chunk = tok_chunk.unsqueeze(0).expand(B, C, tok_chunk.size(1), tok_chunk.size(2)).contiguous()
            M_chunk = msk_chunk.unsqueeze(0).expand(B, C, msk_chunk.size(1)).contiguous()

            scores = self.forward_scores(
                H=H,
                G=G_chunk,
                mask=attn_valid,
                go_mask=M_chunk,
                return_alpha=False,
            )  # [B,C]

            if isinstance(scores, tuple):
                scores = scores[0]

            scores = scores.float()

            if return_cpu:
                out_chunks.append(scores.cpu())
            else:
                out_chunks.append(scores)

        scores_full = torch.cat(out_chunks, dim=1).contiguous()  # [B,G]

        if scores_full.size(1) != G_total:
            raise RuntimeError(
                f"Full token score matrix width mismatch: got {scores_full.size(1)}, expected {G_total}"
            )

        return scores_full

    @torch.no_grad()
    def _topk_from_full_scores(
            self,
            scores_full: torch.Tensor,   # [B,G] CPU or GPU
            topk: int = 200,
    ):
        """
        Convert full score matrix into top-k ids and scores.

        Returns:
          top_ids_cpu    : [B,topk]
          top_scores_cpu : [B,topk]
        """
        if self._eval_ids_cpu is None:
            raise RuntimeError("_eval_ids_cpu is missing. Call _ensure_eval_cache_v2 first.")

        if scores_full.dim() != 2:
            raise RuntimeError(f"scores_full must be [B,G], got {tuple(scores_full.shape)}")

        B, G = scores_full.shape
        topk = min(int(topk), int(G))

        vals, idxs = torch.topk(scores_full, k=topk, dim=1)
        idxs_cpu = idxs.cpu() if idxs.device.type != "cpu" else idxs
        vals_cpu = vals.cpu() if vals.device.type != "cpu" else vals

        top_ids_cpu = self._eval_ids_cpu.index_select(0, idxs_cpu.reshape(-1)).view(B, topk)
        return top_ids_cpu, vals_cpu

    @torch.no_grad()
    def eval_epoch_token_align_exhaustive(self, loader, epoch_idx: int):
        self.model.eval()
        device = self.device

        self._refresh_eval_go_token_cache(chunk=getattr(self.cfg, "eval_go_bs", 128))
        self._ensure_eval_cache_v2(chunk=getattr(self.cfg, "eval_go_bs", 256))

        logs = {
            "obs_fmax": 0.0,
            "obs_aupr": 0.0,
            "seen_fmax": 0.0,
            "seen_aupr": 0.0,
            "rare_fmax": 0.0,
            "rare_aupr": 0.0,
            "unseen_R@10": 0.0,
            "unseen_R@50": 0.0,
            "unseen_num": 0,
            "align_R@1": 0.0,
            "align_R@5": 0.0,
            "align_R@10": 0.0,
            "align_R@50": 0.0,
            "align_R@100": 0.0,
            "align_R@200": 0.0,
            "align_MRR": 0.0,
            "align_nDCG@10": 0.0,
            "anc_R@10": 0.0,
            "anc_R@50": 0.0,
            "anc_num": 0,
        }

        preds_obs, trues_obs = [], []
        preds_seen, trues_seen = [], []
        preds_rare, trues_rare = [], []

        sum_num = 0
        sum_R1 = sum_R5 = sum_R10 = sum_R50 = sum_R100 = sum_R200 = 0.0
        sum_MRR = sum_nDCG = 0.0

        sum_unseen_R10 = 0.0
        sum_unseen_R50 = 0.0
        sum_unseen_num = 0

        sum_anc_R10 = 0.0
        sum_anc_R50 = 0.0
        sum_anc_num = 0

        seen_cols_cpu = self._eval_cols_seen if self._eval_cols_seen is not None else torch.empty(0, dtype=torch.long)
        rare_cols_cpu = self._eval_cols_rare if self._eval_cols_rare is not None else torch.empty(0, dtype=torch.long)
        unseen_cols_cpu = self._eval_cols_unseen if self._eval_cols_unseen is not None else torch.empty(0, dtype=torch.long)

        for batch in loader:
            H = batch["prot_emb_pad"].to(device, non_blocking=True)
            if self.to_f32 is not None:
                H = self.to_f32(H)

            attn_valid, _ = self._valid_and_pad_masks(batch)

            # full observed eval-space truth
            _, y_true = self._build_eval_space(batch)   # [B,Gobs]
            y_true_cpu = y_true.detach().cpu()

            # exhaustive token-level full score matrix over observed eval IDs
            scores_full_cpu = self._token_score_full_chunked(
                H=H,
                attn_valid=attn_valid,
                chunk_k=getattr(self.cfg, "eval_cand_chunk_k", 128),
                return_cpu=True,
            )  # [B,Gobs] on CPU
            scale = float(self.logit_scale_tensor().detach().cpu().item())
            scores_full_cpu = scores_full_cpu * scale

            # top-k metrics from full matrix
            top_ids_cpu, top_scores_cpu = self._topk_from_full_scores(
                scores_full=scores_full_cpu,
                topk=200,
            )

            B = scores_full_cpu.size(0)
            topk = top_ids_cpu.size(1)

            pred_cols = torch.full((B, topk), -1, dtype=torch.long)
            for b in range(B):
                for j in range(topk):
                    gid = int(top_ids_cpu[b, j].item())
                    col = self._eval_id2col.get(gid, None)
                    if col is not None:
                        pred_cols[b, j] = int(col)

            valid_num = 0
            sum_r1 = sum_r5 = sum_r10 = sum_r50 = sum_r100 = sum_r200 = 0.0
            sum_mrr = sum_ndcg = 0.0

            for b in range(B):
                pos_cols = set(torch.nonzero(y_true_cpu[b] > 0, as_tuple=False).flatten().tolist())
                if len(pos_cols) == 0:
                    continue

                preds = [int(x) for x in pred_cols[b].tolist() if int(x) >= 0]
                if len(preds) == 0:
                    continue

                valid_num += 1

                def recall_at_k(k):
                    kk = min(k, len(preds))
                    hit = sum(1 for x in preds[:kk] if x in pos_cols)
                    return hit / max(1, len(pos_cols))

                sum_r1 += recall_at_k(1)
                sum_r5 += recall_at_k(5)
                sum_r10 += recall_at_k(10)
                sum_r50 += recall_at_k(50)
                sum_r100 += recall_at_k(100)
                sum_r200 += recall_at_k(200)

                rr = 0.0
                for rank, x in enumerate(preds, start=1):
                    if x in pos_cols:
                        rr = 1.0 / rank
                        break
                sum_mrr += rr

                dcg = 0.0
                for rank, x in enumerate(preds[:10], start=1):
                    if x in pos_cols:
                        dcg += 1.0 / math.log2(rank + 1.0)

                ideal_hits = min(len(pos_cols), 10)
                idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, ideal_hits + 1))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                sum_ndcg += ndcg

            if valid_num > 0:
                sum_num += valid_num
                sum_R1 += sum_r1
                sum_R5 += sum_r5
                sum_R10 += sum_r10
                sum_R50 += sum_r50
                sum_R100 += sum_r100
                sum_R200 += sum_r200
                sum_MRR += sum_mrr
                sum_nDCG += sum_ndcg

            # classification metrics on full observed score matrix
            preds_obs.append(scores_full_cpu)
            trues_obs.append(y_true_cpu)

            if seen_cols_cpu.numel() > 0:
                preds_seen.append(scores_full_cpu.index_select(1, seen_cols_cpu))
                trues_seen.append(y_true_cpu.index_select(1, seen_cols_cpu))

            if rare_cols_cpu.numel() > 0:
                preds_rare.append(scores_full_cpu.index_select(1, rare_cols_cpu))
                trues_rare.append(y_true_cpu.index_select(1, rare_cols_cpu))

            # subset recalls and ancestor recalls use full score matrix
            scores_full_dev = scores_full_cpu.to(device, non_blocking=True)
            r10, n10 = self._recall_at_k_on_subset(scores_full_dev, y_true, unseen_cols_cpu, k=10)
            r50, n50 = self._recall_at_k_on_subset(scores_full_dev, y_true, unseen_cols_cpu, k=50)
            n_u = max(n10, n50)
            if n_u > 0:
                sum_unseen_R10 += r10 * n_u
                sum_unseen_R50 += r50 * n_u
                sum_unseen_num += n_u

            a10, an10 = self._ancestor_recall_at_k(scores_full_dev, y_true, k=10)
            a50, an50 = self._ancestor_recall_at_k(scores_full_dev, y_true, k=50)
            an = max(an10, an50)
            if an > 0:
                sum_anc_R10 += a10 * an
                sum_anc_R50 += a50 * an
                sum_anc_num += an

        def _finish_fmax_aupr(pred_list, true_list):
            if not pred_list:
                return 0.0, 0.0

            y_pred = torch.cat(pred_list, dim=0).numpy().astype(np.float32)
            y_true_np = torch.cat(true_list, dim=0).numpy().astype(np.int32)

            t_min = float(np.min(y_pred))
            t_max = float(np.max(y_pred))
            if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min == t_max:
                return 0.0, 0.0

            thresholds = np.linspace(t_min, t_max, 101, dtype=np.float32)

            best_f = 0.0
            for t in thresholds:
                y_hat = (y_pred >= t).astype(np.int32)
                tp = (y_hat & y_true_np).sum()
                fp = (y_hat & (1 - y_true_np)).sum()
                fn = ((1 - y_hat) & y_true_np).sum()

                prec = tp / (tp + fp + 1e-12)
                rec = tp / (tp + fn + 1e-12)
                f = (2 * prec * rec) / (prec + rec + 1e-12)
                if f > best_f:
                    best_f = float(f)

            aupr = compute_term_aupr(y_true_np, y_pred)
            return float(best_f), float(aupr)

        logs["obs_fmax"], logs["obs_aupr"] = _finish_fmax_aupr(preds_obs, trues_obs)
        logs["seen_fmax"], logs["seen_aupr"] = _finish_fmax_aupr(preds_seen, trues_seen)
        logs["rare_fmax"], logs["rare_aupr"] = _finish_fmax_aupr(preds_rare, trues_rare)

        if sum_num > 0:
            logs["align_R@1"] = sum_R1 / sum_num
            logs["align_R@5"] = sum_R5 / sum_num
            logs["align_R@10"] = sum_R10 / sum_num
            logs["align_R@50"] = sum_R50 / sum_num
            logs["align_R@100"] = sum_R100 / sum_num
            logs["align_R@200"] = sum_R200 / sum_num
            logs["align_MRR"] = sum_MRR / sum_num
            logs["align_nDCG@10"] = sum_nDCG / sum_num

        if sum_unseen_num > 0:
            logs["unseen_R@10"] = sum_unseen_R10 / sum_unseen_num
            logs["unseen_R@50"] = sum_unseen_R50 / sum_unseen_num
            logs["unseen_num"] = int(sum_unseen_num)

        if sum_anc_num > 0:
            logs["anc_R@10"] = sum_anc_R10 / sum_anc_num
            logs["anc_R@50"] = sum_anc_R50 / sum_anc_num
            logs["anc_num"] = int(sum_anc_num)

        return logs

    @torch.no_grad()
    def _refresh_eval_go_token_cache(self, chunk: int = 128):
        """
        Build token-level GO cache for exhaustive token-align evaluation.

        Stores:
          self._eval_tok_ids_cpu   : [G]      CPU long
          self._eval_tok_embs_cpu  : [G,L,D]  CPU float
          self._eval_tok_mask_cpu  : [G,L]    CPU bool
        """
        if self.model.go_encoder is None:
            raise RuntimeError("Token eval cache requires go_encoder.")

        if not hasattr(self.ctx, "go_text_store") or self.ctx.go_text_store is None:
            raise RuntimeError("ctx.go_text_store is required for token eval cache.")

        eval_ids = [int(x) for x in self.eval_id_list]
        device = self.device

        enc = self.model.go_encoder
        was_training = enc.training
        enc.eval()

        toks = self.ctx.go_text_store.batch(eval_ids)
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]

        tok_cpu = []
        msk_cpu = []

        for s in range(0, input_ids.size(0), chunk):
            e = min(input_ids.size(0), s + chunk)

            out = enc(
                input_ids=input_ids[s:e].to(device, non_blocking=True),
                attention_mask=attention_mask[s:e].to(device, non_blocking=True),
                output_mode="tokens",
            )

            if isinstance(out, dict):
                tok = out["tokens"]
                msk = out["attention_mask"]
            elif torch.is_tensor(out):
                tok = out
                msk = attention_mask[s:e].to(device, non_blocking=True)
            else:
                raise RuntimeError(f"Unsupported token cache output type: {type(out)}")

            if tok.dim() != 3:
                raise RuntimeError(f"Expected token cache tensor [G,L,D], got {tuple(tok.shape)}")
            if msk.dim() != 2:
                raise RuntimeError(f"Expected token cache mask [G,L], got {tuple(msk.shape)}")

            tok = torch.nan_to_num(tok).float().cpu().contiguous()
            if msk.dtype != torch.bool:
                msk = (msk != 0)
            msk = msk.cpu().contiguous()

            tok_cpu.append(tok)
            msk_cpu.append(msk)

        self._eval_tok_ids_cpu = torch.as_tensor(eval_ids, dtype=torch.long)
        self._eval_tok_embs_cpu = torch.cat(tok_cpu, dim=0).contiguous()
        self._eval_tok_mask_cpu = torch.cat(msk_cpu, dim=0).contiguous()

        if was_training:
            enc.train()

    @torch.no_grad()
    def eval_epoch(self, loader, epoch_idx: int):
        if self._use_token_align:
            return self.eval_epoch_token_align_exhaustive(loader, epoch_idx)
        self.model.eval()
        device = self.device
        self._refresh_eval_go_cache(chunk=self.cfg.eval_go_bs)
        self._eval_cache_ready = False
        self._ensure_eval_cache_v2(chunk=self.cfg.eval_go_bs)

        logs = {
            "obs_fmax": 0.0,
            "obs_aupr": 0.0,
            "seen_fmax": 0.0,
            "seen_aupr": 0.0,
            "rare_fmax": 0.0,
            "rare_aupr": 0.0,
            "unseen_R@10": 0.0,
            "unseen_R@50": 0.0,
            "unseen_num": 0,
            "align_R@1": 0.0,
            "align_R@5": 0.0,
            "align_R@10": 0.0,
            "align_R@50": 0.0,
            "align_R@100": 0.0,
            "align_R@200": 0.0,
            "align_MRR": 0.0,
            "align_nDCG@10": 0.0,
            "anc_R@10": 0.0,
            "anc_R@50": 0.0,
            "anc_num": 0,
            "debug_pos_cos": 0.0,
            "debug_neg_cos": 0.0,
            "debug_margin": 0.0,
            "debug_num": 0,
        }

        preds_obs, trues_obs = [], []
        preds_seen, trues_seen = [], []
        preds_rare, trues_rare = [], []

        sum_num = 0
        sum_R1 = sum_R5 = sum_R10 = sum_R50 = sum_R100 = sum_R200 = 0.0
        sum_MRR = sum_nDCG = 0.0

        sum_unseen_R10 = 0.0
        sum_unseen_R50 = 0.0
        sum_unseen_num = 0

        sum_anc_R10 = 0.0
        sum_anc_R50 = 0.0
        sum_anc_num = 0

        scale = self.logit_scale_tensor()

        seen_cols_cpu = self._eval_cols_seen if self._eval_cols_seen is not None else torch.empty(0, dtype=torch.long)
        rare_cols_cpu = self._eval_cols_rare if self._eval_cols_rare is not None else torch.empty(0, dtype=torch.long)
        unseen_cols_cpu = self._eval_cols_unseen if self._eval_cols_unseen is not None else torch.empty(0,
                                                                                                        dtype=torch.long)

        for batch in loader:
            H = batch["prot_emb_pad"].to(device, non_blocking=True)
            if self.to_f32 is not None:
                H = self.to_f32(H)

            attn_valid, _ = self._valid_and_pad_masks(batch)

            # observed eval space
            G_eval, y_true = self._build_eval_space(batch)

            scores_raw = self.forward_scores(H, G_eval, attn_valid, return_alpha=False)
            scores_rank = scores_raw * scale

            # retrieval over observed space
            m = retrieval_metrics_from_scores(scores_rank, y_true, ks=(1, 5, 10, 50, 100, 200))
            if m["num"] > 0:
                sum_num += m["num"]
                sum_R1 += m["R@1"] * m["num"]
                sum_R5 += m["R@5"] * m["num"]
                sum_R10 += m["R@10"] * m["num"]
                sum_R50 += m["R@50"] * m["num"]
                sum_R100 += m["R@100"] * m["num"]
                sum_R200 += m["R@200"] * m["num"]
                sum_MRR += m["MRR"] * m["num"]
                sum_nDCG += m["nDCG@10"] * m["num"]

            preds_obs.append(scores_rank.detach().cpu())
            trues_obs.append(y_true.detach().cpu())

            if seen_cols_cpu.numel() > 0:
                cols = seen_cols_cpu.to(device, non_blocking=True)
                preds_seen.append(scores_rank.index_select(1, cols).detach().cpu())
                trues_seen.append(y_true.index_select(1, cols).detach().cpu())

            if rare_cols_cpu.numel() > 0:
                cols = rare_cols_cpu.to(device, non_blocking=True)
                preds_rare.append(scores_rank.index_select(1, cols).detach().cpu())
                trues_rare.append(y_true.index_select(1, cols).detach().cpu())

            # unseen recall on observed ranking
            r10, n10 = self._recall_at_k_on_subset(scores_rank, y_true, unseen_cols_cpu, k=10)
            r50, n50 = self._recall_at_k_on_subset(scores_rank, y_true, unseen_cols_cpu, k=50)
            n_u = max(n10, n50)
            if n_u > 0:
                sum_unseen_R10 += r10 * n_u
                sum_unseen_R50 += r50 * n_u
                sum_unseen_num += n_u

            # ancestor recall
            a10, an10 = self._ancestor_recall_at_k(scores_rank, y_true, k=10)
            a50, an50 = self._ancestor_recall_at_k(scores_rank, y_true, k=50)
            an = max(an10, an50)
            if an > 0:
                sum_anc_R10 += a10 * an
                sum_anc_R50 += a50 * an
                sum_anc_num += an

            # debug cosine separation
            dbg = self._debug_pos_neg_cosines(H, attn_valid, y_true, max_neg=32)
            if dbg["num_debug_samples"] > 0:
                logs["debug_pos_cos"] += dbg["pos_cos_mean"] * dbg["num_debug_samples"]
                logs["debug_neg_cos"] += dbg["neg_cos_mean"] * dbg["num_debug_samples"]
                logs["debug_margin"] += dbg["margin"] * dbg["num_debug_samples"]
                logs["debug_num"] += dbg["num_debug_samples"]

        def _finish_fmax_aupr(pred_list, true_list):
            if not pred_list:
                return 0.0, 0.0

            y_pred = torch.cat(pred_list, dim=0).numpy().astype(np.float32)
            y_true_np = torch.cat(true_list, dim=0).numpy().astype(np.int32)

            t_min = float(np.min(y_pred))
            t_max = float(np.max(y_pred))
            if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min == t_max:
                return 0.0, 0.0

            thresholds = np.linspace(t_min, t_max, 101, dtype=np.float32)

            best_f = 0.0
            for t in thresholds:
                y_hat = (y_pred >= t).astype(np.int32)
                tp = (y_hat & y_true_np).sum()
                fp = (y_hat & (1 - y_true_np)).sum()
                fn = ((1 - y_hat) & y_true_np).sum()

                prec = tp / (tp + fp + 1e-12)
                rec = tp / (tp + fn + 1e-12)
                f = (2 * prec * rec) / (prec + rec + 1e-12)
                if f > best_f:
                    best_f = float(f)

            aupr = compute_term_aupr(y_true_np, y_pred)
            return float(best_f), float(aupr)

        logs["obs_fmax"], logs["obs_aupr"] = _finish_fmax_aupr(preds_obs, trues_obs)
        logs["seen_fmax"], logs["seen_aupr"] = _finish_fmax_aupr(preds_seen, trues_seen)
        logs["rare_fmax"], logs["rare_aupr"] = _finish_fmax_aupr(preds_rare, trues_rare)

        if sum_num > 0:
            logs["align_R@1"] = sum_R1 / sum_num
            logs["align_R@5"] = sum_R5 / sum_num
            logs["align_R@10"] = sum_R10 / sum_num
            logs["align_R@50"] = sum_R50 / sum_num
            logs["align_R@100"] = sum_R100 / sum_num
            logs["align_R@200"] = sum_R200 / sum_num
            logs["align_MRR"] = sum_MRR / sum_num
            logs["align_nDCG@10"] = sum_nDCG / sum_num

        if sum_unseen_num > 0:
            logs["unseen_R@10"] = sum_unseen_R10 / sum_unseen_num
            logs["unseen_R@50"] = sum_unseen_R50 / sum_unseen_num
            logs["unseen_num"] = int(sum_unseen_num)
        else:
            logs["unseen_R@10"] = 0.0
            logs["unseen_R@50"] = 0.0
            logs["unseen_num"] = 0

        if sum_anc_num > 0:
            logs["anc_R@10"] = sum_anc_R10 / sum_anc_num
            logs["anc_R@50"] = sum_anc_R50 / sum_anc_num
            logs["anc_num"] = int(sum_anc_num)
        else:
            logs["anc_R@10"] = 0.0
            logs["anc_R@50"] = 0.0
            logs["anc_num"] = 0

        if logs["debug_num"] > 0:
            logs["debug_pos_cos"] /= logs["debug_num"]
            logs["debug_neg_cos"] /= logs["debug_num"]
            logs["debug_margin"] /= logs["debug_num"]
        else:
            logs["debug_pos_cos"] = 0.0
            logs["debug_neg_cos"] = 0.0
            logs["debug_margin"] = 0.0

        return logs