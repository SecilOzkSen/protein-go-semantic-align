from typing import List, Tuple, Optional
import copy
import torch
import torch.nn.functional as F
import math
import re

from src.models.alignment_model import ProteinGoAligner
from src.loss.attribution import attribution_loss, windowed_attr_loss
from src.configs.data_classes import TrainerConfig, AttrConfig
from src.miners.queue_miner import MoCoQueue
from src.metrics.cafa import compute_fmax, compute_term_aupr
from src.metrics.retrieval import retrieval_metrics_from_scores

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


def multi_positive_infonce_from_candidates(scores: torch.Tensor,
                                           pos_mask: torch.Tensor,
                                           tau: float,
                                           cand_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    scores: (B, K)
    pos_mask: (B, K) boolean
    cand_valid_mask: (B, K) boolean, True=valid candidate
    """
    logits = scores / max(1e-8, tau)

    if cand_valid_mask is not None:
        logits = logits.masked_fill(~cand_valid_mask, float("-inf"))
        pos_mask = pos_mask & cand_valid_mask

    denom = torch.logsumexp(logits, dim=-1)  # (B,)

    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    pos_any = pos_mask.any(dim=1)
    if (~pos_any).any():
        pos_logits = pos_logits.clone()
        pos_logits[~pos_any] = -1e9

    num = torch.logsumexp(pos_logits, dim=-1)  # (B,)
    loss = -(num - denom)
    return loss[pos_any].mean() if pos_any.any() else denom.mean() * 0.0

@torch.no_grad()
def delta_y_from_occlusion_windows(
    H: torch.Tensor,                  # [B, L, Dh]
    G_pos: torch.Tensor,              # [B, P, Dg]
    model,                            # ProteinGoAligner
    valid_mask: torch.Tensor | None = None,  # [B, L] bool
    window: int = 32,
    stride: int = 16,
    mask_value: float = 0.0,
    chunk_windows: int = 32,          # memory control
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
    nrm = float(torch.linalg.vector_norm(xf32, dim=-1).mean().item()) if xf32.dim() >= 2 else float(torch.linalg.vector_norm(xf32).item())
    print(f"[DBG] {name}: shape={tuple(x.shape)} dtype={x.dtype} dev={x.device} nan={nan} inf={inf} min={mn:.4g} max={mx:.4g} mean={mean:.4g} std={std:.4g} mean_norm={nrm:.4g}")

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
        self.dag_ancestors = build_dag_ancestors(self.ctx.dag_parents)
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
            self.go_encoder_k = clone_as_target(self.model.go_encoder).to(self.device)

        init_ln = math.log(1.0 / 0.07)
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
            {"params": [self.logit_scale], "lr": lr_main*0.05, "weight_decay": 0.0},
        ]

        # Add GO encoder groups (LoRA + embeddings + attn head), if present
        if self.model.go_encoder is not None:
            lr_lora = float(getattr(cfg, "lr_lora", lr_main * 0.1))
            lr_emb = float(getattr(cfg, "lr_go_emb", lr_main * 0.5))
            lr_attn = float(getattr(cfg, "lr_go_attn", lr_main * 0.1))

            # BioMedBERTEncoder implements this
            ge_groups = self.model.go_encoder.param_groups_for_optimizer(
                lr_emb=lr_emb,
                lr_lora=lr_lora,
                wd_lora=0.01,
                lr_attn=lr_attn,
                wd_attn=0.0,
            )
            param_groups.extend(ge_groups)

        self.opt = torch.optim.AdamW(param_groups)
        # TODO - Erase later
        go = getattr(self.model, "go_encoder", None)
        if go is not None:
            # kaç trainable param var
            trainable = [(n, p) for n, p in go.named_parameters() if p.requires_grad]
            self.ctx.logger.info(f"[debug] go_trainable_params={len(trainable)}")

            # trainable isimlerinden ilk 10 tanesini yaz
            for n, _ in trainable[:10]:
                self.ctx.logger.info(f"[debug] go_trainable: {n}")

            # optimizer param group’larında LoRA var mı?
            nopt = 0
            for gi, g in enumerate(self.opt.param_groups):
                for p in g["params"]:
                    if any(p is tp for _, tp in trainable):
                        nopt += 1
                        break
            self.ctx.logger.info(f"[debug] opt_groups_touching_go_trainable={nopt} (should be >=1)")
        # TODO: Erase Later
        self._global_step = 0

        self.use_moco_miner = bool(ctx.use_queue_miner)
        self.queue_K = int(getattr(cfg, "queue_K", getattr(ctx, "queue_K", 4096)))
        self.k_hard_queue = int(getattr(cfg, "k_hard_queue", getattr(ctx, "k_hard", 32)))
        self.queue_miner = None

        if wandb_run is None:
            raise RuntimeError("wandb_run must be passed explicitly")
        self.wandb_run = wandb_run
        self.eval_id_list = ctx.eval_id_list
        self._eval_cache_ready = False
        self._eval_ids_cpu = None
        self._eval_G_once_cpu = None
        self._eval_id2col = None
        # init sonrası bir kere
        opt_params = set()
        for g in self.opt.param_groups:
            for p in g["params"]:
                opt_params.add(id(p))

        for name, p in self.model.named_parameters():
            if ("pooler" in name or "proj_p" in name) and p.requires_grad:
                in_opt = (id(p) in opt_params)
                print("[OPTCHK]", name, "in_opt=", in_opt, "shape=", tuple(p.shape))

    # ----------------- basic helpers -----------------
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

    def _maybe_init_queue(self, Dg: int):
        if self.queue_miner is None and self.use_moco_miner:
            self.queue_miner = MoCoQueue(dim=int(Dg), K=int(self.queue_K), device=str(self.device))
            print(f"[Trainer] MoCo Queue enabled (K={self.queue_K}, k_hard={self.k_hard_queue}, Dg={Dg}).")

    def _get_uniq_go_embs(self, batch):
        device = self.device

        if self.model.go_encoder is None or ("pos_go_tokens" not in batch):
            embs = batch["uniq_go_embs"].to(device, non_blocking=True)
            ids = batch["uniq_go_ids"].to(device, non_blocking=True).long()
            return embs, ids

        toks = batch["pos_go_tokens"]
        input_ids = toks["input_ids"].to(device, non_blocking=True)
        attn = toks["attention_mask"].to(device, non_blocking=True)

        out = self.model.go_encoder(input_ids=input_ids, attention_mask=attn)

        # BioMedBERTEncoder returns [G,D] already
        if isinstance(out, torch.Tensor):
            if out.dim() != 2:
                raise RuntimeError(f"go_enc returned tensor with shape {tuple(out.shape)}, expected [G,D]")
            embs = out
        elif isinstance(out, tuple):
            # if you ever return (pooled, attn) etc
            embs = out[0]
            if embs.dim() != 2:
                raise RuntimeError(f"go_enc returned tuple[0] shape {tuple(embs.shape)}, expected [G,D]")
        elif isinstance(out, dict):
            # only if you ever swap in raw HF model
            hidden = out.get("last_hidden_state", None)
            pooled = out.get("pooler_output", None)
            if pooled is not None:
                embs = pooled
            elif hidden is not None:
                # if raw HF: choose pooling strategy here
                pooling = getattr(self.cfg, "go_pooling", "cls")
                if pooling == "cls":
                    embs = hidden[:, 0]
                else:
                    m = attn.to(hidden.dtype).unsqueeze(-1)
                    embs = (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)
            else:
                raise RuntimeError("go_encoder dict output missing last_hidden_state/pooler_output")

        else:
            raise RuntimeError(f"Unsupported go_encoder output type: {type(out)}")

        #embs = self.normalizer(embs, dim=1)
        embs = torch.nan_to_num(embs)
        ids = batch["uniq_go_ids"].to(device, non_blocking=True).long()
        return embs, ids

    def _get_Dz(self) -> int:
        pp = self.model.proj_p
        if hasattr(pp, "fc1"):
            return int(pp.fc1.out_features)
        if hasattr(pp, "weight"):
            return int(pp.weight.size(0))
        raise RuntimeError("Cannot infer Dz from proj_p")

    @torch.no_grad()
    def _get_prot_query(self, H: torch.Tensor, attn_valid: torch.Tensor, Dz: int) -> torch.Tensor:
        """
        H: [B, T, Dh]
        attn_valid: [B, T] bool
        Return: prot_query [B, Dz] in the SAME space as queued vectors (Dz).
        """
        device = H.device
        B, T, Dh = H.shape

        # 1) masked mean pool -> [B, Dh]
        if attn_valid is not None:
            w = attn_valid.to(H.dtype).unsqueeze(-1)  # [B,T,1]
            denom = w.sum(dim=1).clamp_min(1.0)  # [B,1]
            h_pool = (H * w).sum(dim=1) / denom  # [B,Dh]
        else:
            h_pool = H.mean(dim=1)

        # 2) project to Dz using the SAME projection head used in scoring
        # model.proj_p: Dh -> Dz
        q = self.model.proj_p(h_pool)  # [B,Dz]

        # 3) normalize (cosine space)
        q = self.normalizer(q, dim=1)  # [B,Dz]

        # 4) hard safety
        if q.size(1) != int(Dz):
            raise RuntimeError(f"prot_query dim mismatch: got {q.size(1)} expected {Dz}")

        return q

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

        device = self.device
        # [K, Dg]
        Kmat = all_neg_vecs
        # [B, K]
        sims = prot_query @ Kmat.T
        if getattr(self, "_global_step", 0) % 200 == 0:
            frac_neginf = float(torch.isneginf(sims).float().mean().item())
            frac_finite = float(torch.isfinite(sims).float().mean().item())
            print(
                f"[DBG] queue sims: frac_neginf={frac_neginf:.3f} frac_finite={frac_finite:.3f} K={int(sims.size(1))}")
        # ---- DAG-aware false-negative mask ----
        if all_neg_ids is not None:
            if not torch.is_tensor(all_neg_ids):
                all_neg_ids = torch.as_tensor(all_neg_ids, dtype=torch.long)
            all_neg_ids = all_neg_ids.to(device, non_blocking=True).long()  # [K]

            B = len(pos_local)
            if B > 0:
                Pmax = max((int(loc.numel()) for loc in pos_local), default=0)
            else:
                Pmax = 0

            if Pmax > 0:
                # Build base pos ids: [B, Pmax] (pad=-1)
                pos_go_ids = torch.full((B, Pmax), -1, device=device, dtype=torch.long)
                uniq_go_ids_dev = uniq_go_ids.to(device, non_blocking=True).long()

                for b, loc in enumerate(pos_local):
                    if loc.numel() == 0:
                        continue
                    loc = loc.to(device, non_blocking=True)
                    gids = uniq_go_ids_dev.index_select(0, loc)
                    t = int(gids.numel())
                    if t > 0:
                        pos_go_ids[b, :t] = gids

                # Optionally expand with ancestors (guard even if you do propagation)
                dag_anc = self.dag_ancestors  # dict[int, list[int]] or None
                if dag_anc is not None:
                    # Build expanded exclude ids per sample (small loop over B, OK)
                    expanded: list[torch.Tensor] = []
                    Emax = 0
                    for b in range(B):
                        ids = pos_go_ids[b]
                        ids = ids[ids >= 0].tolist()
                        if not ids:
                            expanded.append(torch.empty(0, device=device, dtype=torch.long))
                            continue
                        s = set()
                        for gid in ids:
                            s.update(dag_anc.get(int(gid), [int(gid)]))
                        ex = torch.as_tensor(list(s), device=device, dtype=torch.long)
                        expanded.append(ex)
                        Emax = max(Emax, int(ex.numel()))

                    if Emax > 0:
                        excl = torch.full((B, Emax), -1, device=device, dtype=torch.long)
                        for b, ex in enumerate(expanded):
                            if ex.numel() > 0:
                                excl[b, : ex.numel()] = ex
                        # vectorized compare: [B, Emax, 1] == [1, 1, K] -> [B, Emax, K] -> any -> [B, K]
                        fn_mask = (excl.unsqueeze(-1) == all_neg_ids.view(1, 1, -1)).any(dim=1)
                    else:
                        fn_mask = None
                else:
                    # propagation-only mode: exclude just the pos ids
                    fn_mask = (pos_go_ids.unsqueeze(-1) == all_neg_ids.view(1, 1, -1)).any(dim=1)

                if fn_mask is not None:
                    sims = sims.masked_fill(fn_mask, float("-inf"))

        k = min(int(self.k_hard_queue), int(Kmat.size(0)))
        if k <= 0:
            return None

        idx = sims.topk(k, dim=1).indices  # [B, k]
        neg = Kmat.index_select(0, idx.reshape(-1)).reshape(idx.size(0), k, -1).contiguous()
        return neg  # [B, k, Dg]

    def _build_candidates(self, uniq_go_embs, pos_local, neg_from_queue=None, *, max_inbatch: int | None = None):
        """
        Returns:
          G_cand: [B, K, Dg]
          pos_mask: [B, K]  (True sadece pozitifler için)
          cand_valid_mask: [B, K] (True valid slot)
        """
        device = self.device
        B = len(pos_local)
        U = int(uniq_go_embs.size(0))  # in-batch uniq candidate count
        Dg = int(uniq_go_embs.size(1))

        # (opsiyonel) in-batch candidate sayısını kıs
        if max_inbatch is not None and U > max_inbatch:
            # Pozitifleri koru, geri kalanı random kıs
            keep = set()
            for loc in pos_local:
                keep.update([int(x) for x in loc.tolist()])
            keep = sorted(keep)

            # kalanlardan sample
            all_idx = torch.arange(U, device=device)
            keep_t = torch.tensor(keep, device=device, dtype=torch.long) if keep else torch.empty(0, device=device,
                                                                                                  dtype=torch.long)
            mask = torch.ones(U, device=device, dtype=torch.bool)
            if keep_t.numel() > 0:
                mask[keep_t] = False
            rest = all_idx[mask]
            need = max(0, max_inbatch - int(keep_t.numel()))
            if need > 0 and rest.numel() > 0:
                perm = torch.randperm(rest.numel(), device=device)[:need]
                extra = rest[perm]
                cand_idx = torch.cat([keep_t, extra], dim=0) if keep_t.numel() > 0 else extra
            else:
                cand_idx = keep_t
            cand_idx = cand_idx.unique(sorted=False)
            uniq_sub = uniq_go_embs.index_select(0, cand_idx.to(uniq_go_embs.device))
            # pos_local’ları yeni index’e map et
            old2new = {int(old): i for i, old in enumerate(cand_idx.tolist())}
            new_pos_local = []
            for loc in pos_local:
                new_loc = [old2new[int(x)] for x in loc.tolist() if int(x) in old2new]
                new_pos_local.append(torch.tensor(new_loc, device=device, dtype=torch.long))
            uniq_go_embs = uniq_sub
            pos_local = new_pos_local
            U = int(uniq_go_embs.size(0))

        kq = 0 if neg_from_queue is None else int(neg_from_queue.size(1))
        K = U + kq

        G_cand = torch.zeros(B, K, Dg, device=device, dtype=uniq_go_embs.dtype)
        pos_mask = torch.zeros(B, K, device=device, dtype=torch.bool)
        cand_valid_mask = torch.zeros(B, K, device=device, dtype=torch.bool)

        # 1) in-batch candidates: hepsi valid
        G_cand[:, :U] = uniq_go_embs.unsqueeze(0).expand(B, U, Dg).to(device)
        cand_valid_mask[:, :U] = True

        # 2) pozitif mask: pos_local indexleri in-batch segmentine işaret ediyor
        for b, loc in enumerate(pos_local):
            if loc.numel() > 0:
                pos_mask[b, loc.to(device)] = True

        # 3) queue negleri append
        if kq > 0:
            G_cand[:, U:U + kq] = neg_from_queue.to(device)
            cand_valid_mask[:, U:U + kq] = True
            # pos_mask queue kısmında False kalmalı

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
        if G.dim() == 3:
            assert sc.dim() == 2 and sc.size(0) == H.size(0), "forward_scores: bad score shape"
        #TODO: Erase later
        if self._global_step % 500 == 0 and getattr(self.model, "go_encoder", None) is not None:
            ge = self.model.go_encoder
            g = None
            for n, p in ge.named_parameters():
                if p.requires_grad and p.grad is not None:
                    g = p.grad.detach().float().abs().mean().item()
                    print("[DEBUG] grad example:", n, g)
                    break
            if g is None:
                print("[DEBUG] NO GRAD flowing into go_encoder trainables")
        #TODO: Erase later

        return (sc, alpha) if return_alpha else sc

    # ----------------- eval space cache -----------------
    @torch.no_grad()
    def _ensure_eval_cache_v2(self, chunk=1024):
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

        if self.model.go_encoder is not None:
            # We want deterministic eval embeddings
            was_training = self.model.go_encoder.training
            self.model.go_encoder.eval()

            # Need go_text_store for tokenization
            if not hasattr(self.ctx, "go_text_store") or self.ctx.go_text_store is None:
                raise RuntimeError("ctx.go_text_store is required to build eval cache with go_encoder")

            toks = self.ctx.go_text_store.batch(eval_ids)  # dict: input_ids [Geval,L], attention_mask [Geval,L]

            input_ids = toks["input_ids"]
            attention_mask = toks["attention_mask"]

            # Optional safety: ensure no dropout in eval tokens at collator level.
            # Here we assume toks are clean.

            # Encode in chunks to avoid OOM
            embs_list = []
            for s in range(0, input_ids.size(0), chunk):
                e = min(input_ids.size(0), s + chunk)
                out = self.model.go_encoder(
                    input_ids=input_ids[s:e].to(device, non_blocking=True),
                    attention_mask=attention_mask[s:e].to(device, non_blocking=True),
                )

                # unwrap outputs
                if isinstance(out, tuple):
                    embs = out[0]
                elif isinstance(out, dict):
                    embs = out.get("pooler_output", None)
                    if embs is None:
                        embs = out["last_hidden_state"][:, 0]
                else:
                    embs = out

                if embs.dim() == 3:
                    embs = embs[:, 0]
                if embs.dim() != 2:
                    raise RuntimeError(f"go_encoder must return [G,D], got {tuple(embs.shape)}")

             #   embs = self.normalizer(embs, dim=1)  # cosine space
                embs_list.append(embs.detach().to("cpu", non_blocking=False).contiguous())

            G_once_cpu = torch.cat(embs_list, dim=0).contiguous()

            # restore
            if was_training:
                self.model.go_encoder.train()

        else:
            # Fallback: use precomputed cache embeddings (old behavior)
            rows = torch.as_tensor(
                [self.ctx.go_cache.id2row[int(g)] for g in eval_ids],
                dtype=torch.long,
                device=self.ctx.go_cache.embs.device
            )
            G_bank = self.ctx.go_cache.embs.index_select(0, rows).contiguous()
            G_once_cpu = G_bank.detach().to("cpu", non_blocking=False).contiguous()

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
        print("Eval G_once std:", float(self._eval_G_once_cpu.float().std().item()))
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

    def logit_scale_value(self) -> float:
        if not self.cfg.is_logit_scale_constant:
            return float(self.logit_scale.clamp(min=-10.0, max=4.6).exp())
        else:
            return float(self.logit_scale.exp())

    # ----------------- training step -----------------
    def step_losses(self, batch, epoch_idx: int, debug: bool = False):
        self.model.train()
        device = self.device

        # === SANITY CHECK ===
        if debug:
            if ("pos_go_tokens" in batch) and ("uniq_go_ids" in batch):
                assert batch["pos_go_tokens"]["input_ids"].size(0) == batch["uniq_go_ids"].numel(), \
                    "pos_go_tokens and uniq_go_ids size mismatch"
            if "uniq_go_embs" in batch:
                assert batch["uniq_go_embs"].size(0) == batch["uniq_go_ids"].numel(), \
                    "uniq_go_embs and uniq_go_ids size mismatch"

        H = batch["prot_emb_pad"].to(device, non_blocking=True)
        attn_valid, pad_mask = self._valid_and_pad_masks(batch)
        if self.to_f32 is not None:
            H = self.to_f32(H)

        pos_local = batch["pos_go_local"]
        uniq_go_embs, uniq_go_ids = self._get_uniq_go_embs(batch)
        if self._global_step % 200 == 0:
            _tstats(uniq_go_embs, "uniq_go_embs(raw)")
            print("[DBG] uniq_go_ids:", tuple(uniq_go_ids.shape), "uniq=", int(torch.unique(uniq_go_ids).numel()))
        if getattr(self.model, "go_encoder", None) is not None:
            assert "pos_go_tokens" in batch, "GO encoder present but pos_go_tokens missing, LoRA won't train"
        Dg_batch = int(uniq_go_embs.size(1))
        Dz = self._get_Dz()
        self._maybe_init_queue(Dz)

        amp_ctx = torch.amp.autocast(
            device_type="cuda",
            enabled=(torch.cuda.is_available() and self.ctx.fp16_enabled),
        )

        # queue miner varsa hard neg çıkar, yoksa None
        with torch.no_grad():
            prot_query = self._get_prot_query(H, attn_valid, Dz)
            neg_from_queue = None
            if self.queue_miner is not None:
                neg_from_queue = self._mine_queue_hard_negs(prot_query, pos_local, uniq_go_ids, Dz)
                if self._global_step % 200 == 0:
                    if neg_from_queue is None:
                        print("[DBG] neg_from_queue=None (queue empty or disabled)")
                    else:
                        _tstats(neg_from_queue, "neg_from_queue(Dz)")

        # candidates: in-batch + optional queue
        max_inbatch = None
        if getattr(self.ctx, "scheduler", None) is not None:
            try:
                max_inbatch = int(self.ctx.scheduler(self._global_step)["shortlist_M"])
            except Exception:
                max_inbatch = None

        G_cand, pos_mask, cand_valid_mask = self._build_candidates(
            uniq_go_embs, pos_local, neg_from_queue, max_inbatch=max_inbatch
        )

        with amp_ctx:
            # 2) scores
            scores_cand = self.forward_scores(H, G_cand, attn_valid, return_alpha=False)
            if self._global_step % 200 == 0:
                _tstats(scores_cand, "scores_cand(pre_scale)")
                scale = self.logit_scale_value()
                print(
                    f"[DBG] logit_scale_exp={scale:.4g} requires_grad={bool(getattr(self.logit_scale, 'requires_grad', False))}")
            assert scores_cand.requires_grad, "scores_cand grad not enabled"

            # 3) scale
            scale = self.logit_scale_value()
            scores_cand = scores_cand * scale
            if self._global_step % 200 == 0:
                _tstats(scores_cand, "scores_cand(post_scale)")
                # pos var mı
                pos_any = bool(pos_mask.any(dim=1).all().item()) if pos_mask.numel() else False
                print(
                    f"[DBG] pos_mask any-per-sample? {pos_mask.any(dim=1).detach().cpu().tolist()} overall_all_have_pos={pos_any}")

            # 4) loss (pad candidate'ları mask’le)
            l_con = multi_positive_infonce_from_candidates_v2(
                scores_cand,
                pos_mask,
                tau=1.0,
                cand_valid_mask=cand_valid_mask,
            )
            if self._global_step % 200 == 0:
                print(f"[DBG] l_con={float(l_con.detach().item()):.4f}")

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

                    if ("pos_go_tokens" in batch) and (self.go_encoder_k is not None):
                        toks = batch["pos_go_tokens"]
                        assert toks["input_ids"].size(0) == uniq_go_ids.size(0)

                        pos_vecs = self.go_encoder_k(
                            input_ids=toks["input_ids"].to(device, non_blocking=True),
                            attention_mask=toks["attention_mask"].to(device, non_blocking=True),
                        )
                        pos_vecs = pos_vecs.index_select(0, local_cat)  # raw encoder space
                    else:
                        pos_vecs = uniq_go_embs.index_select(0,
                                                             local_cat)  # raw encoder space (trainer'da normalize yok artık)

                    # enqueue in projected+normalized space
                    pos_vecs = self.model.proj_g(pos_vecs)
                    pos_vecs = self.normalizer(pos_vecs, dim=1)

                    pos_ids = uniq_go_ids.index_select(0, local_cat).detach()
                    self.queue_miner.enqueue(pos_vecs.detach(), pos_ids)
                    if self._global_step % 200 == 0:
                        try:
                            qn = int(self.queue_miner.ptr) if hasattr(self.queue_miner, "ptr") else -1
                            print(f"[DBG] enqueue done. queue_ptr={qn}")
                        except Exception as e:
                            print("[DBG] enqueue done (ptr read failed):", repr(e))

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
                },
                step=int(self._global_step),
            )
        except Exception:
            pass

        return {"total": total, "contrastive": l_con, "dag": l_dag, "attr": l_attr, "entropy": l_ent}

    @torch.no_grad()
    def eval_epoch(self, loader, epoch_idx: int):
        self.model.eval()
        device = self.device

        logs = {"cafa_fmax": 0.0, "cafa_aupr": 0.0, "align_R@1": 0.0, "align_R@5": 0.0,
                "align_R@10": 0.0, "align_MRR": 0.0, "align_nDCG@10": 0.0}
        n = 0

        preds, trues = [], []
        sum_num = 0
        sum_R1 = sum_R5 = sum_R10 = 0.0
        sum_MRR = sum_nDCG = 0.0

        for batch in loader:
            H = batch["prot_emb_pad"].to(device, non_blocking=True)
            if self.to_f32 is not None:
                H = self.to_f32(H)
            attn_valid, pad_mask = self._valid_and_pad_masks(batch)

            # build global eval space from cache
            G_eval, y_true = self._build_eval_space(batch)
            scores = self.forward_scores(H, G_eval, attn_valid, return_alpha=False)
            scale = self.logit_scale_value()
            scores = scores * scale
            #Alignment metrics on raw scores
            m = retrieval_metrics_from_scores(scores, y_true, ks=(1, 5, 10))
            if m["num"] > 0:
                sum_num += m["num"]
                sum_R1 += m["R@1"] * m["num"]
                sum_R5 += m["R@5"] * m["num"]
                sum_R10 += m["R@10"] * m["num"]
                sum_MRR += m["MRR"] * m["num"]
                sum_nDCG += m["nDCG@10"] * m["num"]

            probs = torch.sigmoid(scores)

            preds.append(probs.cpu())
            trues.append(y_true.cpu())
            n += H.size(0)

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

        if sum_num > 0:
            logs["align_R@1"] = sum_R1 / sum_num
            logs["align_R@5"] = sum_R5 / sum_num
            logs["align_R@10"] = sum_R10 / sum_num
            logs["align_MRR"] = sum_MRR / sum_num
            logs["align_nDCG@10"] = sum_nDCG / sum_num
        else:
            logs["align_R@1"] = logs["align_R@5"] = logs["align_R@10"] = 0.0
            logs["align_MRR"] = logs["align_nDCG@10"] = 0.0

        logs["cafa_fmax"] = float(fmax)
        logs["cafa_aupr"] = float(aupr)
        return logs
