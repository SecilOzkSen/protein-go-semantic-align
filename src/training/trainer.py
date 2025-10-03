from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F

from src.models.alignment_model import ProteinGoAligner
from src.loss.attribution import attribution_loss, windowed_attr_loss
from src.configs.data_classes import TrainerConfig, AttrConfig, CurriculumConfig

from src.training.batch_builder import BatchBuilder

# Wandb
import wandb
from collections import defaultdict
from src.utils.wandb_logger import WabLogger


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
    scores: (B, K) similarity over candidate set (positives + mined negatives)
    pos_mask: (B, K) boolean; True where candidate is a positive for that item
    """
    logits = scores / max(1e-8, tau)          # (B, K)
    denom = torch.logsumexp(logits, dim=-1)   # (B,)
    # Avoid -inf when no positives (shouldn't happen); we mask with -inf then logsumexp.
    pos_logits = logits.masked_fill(~pos_mask, float('-inf'))
    num = torch.logsumexp(pos_logits, dim=-1) # (B,)
    return -(num - denom).mean()

def gather_go_rows(go_ids: torch.Tensor, id2row: dict) -> torch.Tensor:
    # Map GO integer ids (shape: (N,)) to row indices in cache
    idx = [id2row[int(g)] for g in go_ids.tolist()]
    return torch.as_tensor(idx, dtype=torch.long, device=go_ids.device)

def build_candidates_for_batch(
    uniq_go_embs: torch.Tensor,                  # (G, Dg) for this batch (positives universe)
    pos_go_local: List[torch.Tensor],            # list of LongTensor (indices into uniq_go_embs)
    neg_go_ids: torch.Tensor,                    # (B, k_hard) global GO ids from BatchBuilder (-1 padded)
    go_cache_embs: torch.Tensor,                 # (N_GO, Dg) global cache embs
    id2row: dict,                                # mapping id->row
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Returns:
      G_cand:  (B, K, Dg) candidates per item (positives first, then negatives)
      pos_mask:(B, K) booleans for positives
      slices:  list of (pos_len, neg_len) per item (for debugging)
    """
    B = len(pos_go_local)
    device = uniq_go_embs.device
    Dg = uniq_go_embs.size(1)

    # Build per-item positives (from uniq_go_embs)
    pos_embs, pos_lens = [], []
    for b in range(B):
        if pos_go_local[b].numel() == 0:
            pos_embs.append(torch.zeros(0, Dg, device=device, dtype=uniq_go_embs.dtype))
            pos_lens.append(0)
        else:
            pos_embs.append(uniq_go_embs.index_select(0, pos_go_local[b].to(device)))
            pos_lens.append(int(pos_go_local[b].numel()))

    # Build per-item negatives (from global cache via id2row)
    neg_embs, neg_lens = [], []
    for b in range(B):
        valid = neg_go_ids[b] >= 0
        gids = neg_go_ids[b, valid]
        if gids.numel() == 0:
            neg_embs.append(torch.zeros(0, Dg, device=device, dtype=go_cache_embs.dtype))
            neg_lens.append(0)
        else:
            rows = gather_go_rows(gids.to(device), id2row)  # (k_v,)
            neg_embs.append(go_cache_embs.index_select(0, rows))
            neg_lens.append(int(rows.numel()))

    # Pad to common K (positives first, then negatives)
    K = max((p + n) for p, n in zip(pos_lens, neg_lens)) if B > 0 else 1
    G_cand = torch.zeros(B, K, Dg, device=device, dtype=uniq_go_embs.dtype)
    pos_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
    slices = []
    for b in range(B):
        p, n = pos_lens[b], neg_lens[b]
        if p > 0:
            G_cand[b, :p] = pos_embs[b]
            pos_mask[b, :p] = True
        if n > 0:
            G_cand[b, p:p+n] = neg_embs[b]
        slices.append((p, n))
    return G_cand, pos_mask, slices

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
    scores_pos: (B, T_max) scores for positives G_pos (padded with zeros beyond t_b)
    pos_local: list of LongTensor (indices into uniq_go_ids) per item
    uniq_go_ids: (G,) LongTensor mapping local uniq index -> global GO id
    dag_parents: dict {child_global_id: [parent_global_id, ...]}
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
        # map global ids for these positives
        glob = uniq_go_ids.index_select(0, loc.to(uniq_go_ids.device))  # (t,)
        id2t = {int(glob[i].item()): i for i in range(t)}
        # for each child present, consider its parents if also present
        for child_gid, child_t in list(id2t.items()):
            parents = dag_parents.get(child_gid, [])
            for pg in parents:
                if pg in id2t:
                    p_t = id2t[pg]
                    # softplus on (child - parent + margin)
                    diff = sp[child_t] - sp[p_t] + margin
                    losses.append(F.softplus(scale * diff))
    if len(losses) == 0:
        return torch.zeros((), device=scores_pos.device)
    return torch.stack(losses).mean()

def topk_maskout_full(H, G, alpha_full, k, model, pad_mask=None):
    """
    Eval-time actual mask-out for full-length case.
    alpha_full: (B, T, L). For each (b,t): pick Top-K residues, zero them (one-by-one), recompute scores.
    Returns delta_full: (B, T, L) with nonzero at tested residues, normalized per (b,t).
    """
    B, T, L = alpha_full.shape
    device = H.device
    delta = torch.zeros_like(alpha_full)
    base_scores, _ = model(H, G, mask=pad_mask, return_alpha=True)  # (B, T)

    for b in range(B):
        for t in range(T):
            topk = min(k, L)
            vals, idx = torch.topk(alpha_full[b, t], k=topk, dim=-1)
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
            # normalize per (b,t)
            m = delta[b, t].amax()
            if m > 0:
                delta[b, t] = delta[b, t] / m
    return delta


# ------------- Trainer -------------
class OppTrainer:
    def __init__(self, cfg: TrainerConfig, attr: AttrConfig, ctx, go_encoder):
        self.cfg, self.attr, self.ctx = cfg, attr, ctx
        self.model = ProteinGoAligner(
            d_h=cfg.d_h,
            d_g=None,          # will be inferred from go encoder
            d_z=cfg.d_z,       # NOTE: expected 768
            go_encoder=go_encoder,  # BioMedBERTEncoder (with LoRA)
            normalize=True,
        )
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self._global_step = 0

        # If VectorResources is present, align its query projector to model's projector
        if hasattr(self.ctx, "vres") and self.ctx.vres is not None:
            self.ctx.vres.query_projector = self.model.proj_p   # nn.Linear 1280 -> 768

        # Safe defaults for miner shortlist K/M
        default_M, default_K = 64, 4
        if getattr(ctx, "scheduler", None) is not None:
            try:
                cur = ctx.scheduler(0)
                default_M = int(getattr(cur.cfg, "shortlist_M", default_M))
                default_K = int(getattr(cur.cfg, "k_hard", default_K))
            except Exception:
                pass

        # Build BatchBuilder if not supplied from context
        if getattr(self.ctx, "batch_builder", None) is None:
            self.builder = BatchBuilder(
                vres=getattr(ctx, "vres", None),
                faiss_index=getattr(ctx, "faiss_index", None) if getattr(ctx, "vres", None) is None else None,
                go_encoder_rerank=go_encoder,
                dag_parents=ctx.dag_parents,
                dag_children=ctx.dag_children,
                scheduler=ctx.scheduler,
                default_M=default_M,
                default_K=default_K,
                all_go_ids=ctx.go_cache.row2id
            )
        else:
            self.builder = ctx.batch_builder

        # W&B setup
        self._wandb_configs()

    def _wandb_configs(self):
        """Initialize W&B run, logger, and per-phase accumulators."""
        self.wandb_run = wandb.init(
            project="protein-go-semantic-align",
            name=self.ctx.run_name,
            config=self.ctx.to_dict()
        )
        self.wlogger = WabLogger(self.wandb_run, project="protein-go-semantic-align", config=self.ctx.to_dict())
        self._phase_acc = defaultdict(lambda: defaultdict(list))
        self._current_phase_id = None

        # Optional: watch model for gradients/parameters (guarded)
        try:
            wandb.watch(self.model, log="all", log_freq=max(100, getattr(self.cfg, "log_every", 50)))
        except Exception:
            pass

    # --------- Wandb helpers (phase bookkeeping) ---------
    def _on_phase_change(self, new_phase_id: int, new_phase_name: str, step: int):
        """Flush summary for previous phase and start a new one."""
        if self._current_phase_id is not None:
            self._flush_phase_table(self._current_phase_id, step)
        self._current_phase_id = new_phase_id
        self._phase_acc[new_phase_id].clear()
        wandb.log({"phase/change_to": new_phase_id, "phase/name": new_phase_name}, step=step)

    def _flush_phase_table(self, phase_id: int, step: int):
        """Aggregate mean/std/min/max for metrics collected within a phase and log a table."""
        import numpy as np
        rows = []
        for k, vals in self._phase_acc[phase_id].items():
            if vals:
                arr = np.array(vals, dtype=float)
                rows.append([phase_id, k, float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()), len(arr)])
        if rows:
            table = wandb.Table(columns=["phase_id", "metric", "mean", "std", "min", "max", "n"], data=rows)
            wandb.log({f"phase_summary/phase_{phase_id}": table}, step=step)

    # --------- Scoring (tensor path) ---------
    def forward_scores(self, H, G, pad_mask, **kwargs):
        """
        Use the model's tensor path to score arbitrary G tensors (e.g., candidates or positives).
        By default returns alpha maps as well unless return_alpha=False is provided.
        """
        return_alpha = kwargs.pop("return_alpha", True)
        return self.model(H=H, G=G, mask=pad_mask, return_alpha=return_alpha, **kwargs)

    # --------- Negative mining ---------
    def _build_negatives(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updated mining flow:
          - coarse shortlist via FAISS (vres.coarse_search)
          - true-pooling rerank using ctx.vres.true_prot_vecs (attention-weighted)
        """
        device = self.cfg.device

        # Resolve positive GO global ids for builder
        if "pos_go_global" in batch:
            pos_go_global = batch["pos_go_global"]
        else:
            # Fallback: map local -> global using uniq_go_ids
            uniq_go_ids = batch["uniq_go_ids"]  # (G,)
            pos_go_global = []
            for loc in batch["pos_go_local"]:
                pos_go_global.append(uniq_go_ids.index_select(0, loc.to(uniq_go_ids.device)))

        # Curriculum params at this step (store for logging)
        self.curriculum_params = None
        new_phase_id, new_phase_name = None, None
        if getattr(self.ctx, "scheduler", None) is not None:
            curr = self.ctx.scheduler(self._global_step)

            # Store as-is for later logging
            self.curriculum_params = curr

            # --- Phase detection (supports both dict-like and dataclass-like schedulers) ---
            # Try common attribute names first
            def _get(attr, *fallbacks):
                for a in (attr, *fallbacks):
                    if isinstance(curr, dict) and a in curr:  # dict scheduler
                        return curr[a]
                    if hasattr(curr, a):  # object
                        return getattr(curr, a)
                    if hasattr(getattr(curr, "cfg", None), a):  # nested cfg
                        return getattr(curr.cfg, a)
                return None

            new_phase_id = _get("phase_id", "phase", "id")
            new_phase_name = _get("phase_name", "name")
            if new_phase_id is not None and new_phase_name is None:
                new_phase_name = f"phase{int(new_phase_id)}"

            # If phase changed, flush previous phase summary and start new one
            if new_phase_id is not None and new_phase_id != self._current_phase_id:
                self._on_phase_change(int(new_phase_id), str(new_phase_name), step=self._global_step)

            # Builder expects a CurriculumConfig? Pass only if it is the right type
            curr_cfg = curr if isinstance(curr, CurriculumConfig) else None
        else:
            curr_cfg = None

        # Preferred: build directly from segmented embeddings (coarse + true rerank inside)
        out = self.builder.build_from_embs(
            prot_emb_pad=batch["prot_emb_pad"].to(device),
            prot_attn_mask=batch["prot_attn_mask"].to(device),
            pos_go_global=pos_go_global,
            zs_mask=batch["zs_mask"].to(device),
            curriculum_config=curr_cfg if isinstance(curr_cfg, CurriculumConfig) else None
        )
        return out

    # --------- Training step (losses + logging) ---------
    def step_losses(self, batch: Dict[str, Any], epoch_idx: int) -> Dict[str, torch.Tensor]:
        self.model.train()
        device = self.cfg.device

        # --- Protein & mask ---
        H = batch["prot_emb_pad"].to(device)               # (B, Lmax, Dh)
        attn_valid = batch["prot_attn_mask"].to(device)    # True = VALID
        pad_mask = ~attn_valid
        pos_local: List[torch.Tensor] = batch["pos_go_local"]
        B = H.size(0)

        # Use attribution loss only during early curriculum epochs
        use_attr = (epoch_idx < self.attr.curriculum_epochs)

        # --- POSITIVES (GoEncoder → uniq_go_embs) ---
        if ("pos_go_tokens" in batch) and (hasattr(self.model, "go_encoder") and self.model.go_encoder is not None):
            toks = batch["pos_go_tokens"]
            uniq_go_embs = self.model.go_encoder(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            )  # [G, Dg], pooled
            try:
                uniq_go_embs = F.normalize(uniq_go_embs, p=2, dim=1)
            except Exception:
                pass
        else:
            # Fallback: ready-made embeddings from collator/cache
            uniq_go_embs = batch["uniq_go_embs"].to(device)  # (G, Dg)

        # --- NEGATIVES (IDs via miner → embeddings via MemoryBank) ---
        mined = self._build_negatives(batch)
        neg_go_ids = mined["neg_go_ids"].to(device)  # (B, k_hard), -1 padded

        # --- Candidate set (positives + mined negatives) using MemoryBank ---
        G_cand, pos_mask, _ = build_candidates_for_batch(
            uniq_go_embs=uniq_go_embs,            # [G, Dg]
            pos_go_local=pos_local,               # List[Tensor]
            neg_go_ids=neg_go_ids,                # [B, K]
            go_cache_embs=self.ctx.memory_bank.embs.to(device),
            id2row=self.ctx.go_cache.id2row,
        )  # -> (B, K, Dg), (B, K), meta

        # Teacher vector from true pooling (optionally with importance weights)
        use_weights = (epoch_idx >= self.attr.curriculum_epochs)
        weights_provider = getattr(self, "weights_provider", None) if use_weights else None
        v_true = self.ctx.vres.true_prot_vecs(H, attn_valid, watti_or_model=weights_provider)  # [B, d?]

        # ====================== Teacher (v_true) → KL DISTILLATION ======================
        # 1) Align v_true to the GO index space if a projector is available
        vt = v_true
        if self.ctx.vres is not None and hasattr(self.ctx.vres, "project_queries_to_index"):
            try:
                vt = self.ctx.vres.project_queries_to_index(vt)  # [B, Dg]
            except Exception:
                pass

        # 2) Normalize for cosine similarity
        vt = F.normalize(vt, dim=-1)       # [B, Dg]
        Gn = F.normalize(G_cand, dim=-1)   # [B, K, Dg]

        # 3) Teacher scores: <v_true, G_k>  → [B, K]
        scores_teacher = torch.einsum("bd,bkd->bk", vt, Gn)

        # ---- Contrastive over candidate set (student) ----
        scores_cand, _ = self.forward_scores(H, G_cand, pad_mask)  # (B, K)
        l_con = multi_positive_infonce_from_candidates(scores_cand, pos_mask, self.attr.temperature)

        # 4) KL distillation: teacher distribution → student logits
        tau = float(self.attr.tau_distill)
        with torch.no_grad():
            p_t = F.softmax(scores_teacher / tau, dim=1)  # teacher target (stopgrad)
        log_p_s = F.log_softmax(scores_cand / tau, dim=1)  # student
        l_con_teacher = F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)

        # Teacher weight (default 0.2 if not specified)
        lambda_v = float(getattr(self.attr, "lambda_vtrue", 0.2))
        # ================================================================================
        # ---- Positives-only tensor for attribution / DAG ----
        T_max = max((int(x.numel()) for x in pos_local), default=1)
        Dg = uniq_go_embs.size(1)
        G_pos = torch.zeros(B, T_max, Dg, device=device, dtype=uniq_go_embs.dtype)
        for b, loc in enumerate(pos_local):
            t = int(loc.numel())
            if t > 0:
                G_pos[b, :t] = uniq_go_embs.index_select(0, loc.to(uniq_go_embs.device))

        scores_pos, alpha_info = self.forward_scores(H, G_pos, pad_mask)

        # ---- Attribution losses (surrogate during training) ----
        if "alpha_full" in alpha_info:
            alpha = alpha_info["alpha_full"]  # (B, T, L)
            if use_attr:
                delta, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, pad_mask)
                l_attr = attribution_loss(alpha, delta, mask=None, reduce="mean")
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(alpha)
            else:
                l_attr = torch.zeros((), device=device); l_ent = torch.zeros((), device=device)
        elif all(k in alpha_info for k in ["alpha_windows", "win_weights", "spans"]):
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
            l_attr = torch.zeros((), device=device); l_ent = torch.zeros((), device=device)

        # ---- DAG loss over positives ----
        uniq_go_ids = batch['uniq_go_ids'].to(device)
        l_dag = dag_consistency_loss_pos(
            scores_pos, pos_local, uniq_go_ids, self.ctx.dag_parents, margin=0.0, scale=1.0
        )

        # ---- Total loss (incl. teacher) ----
        total = (l_con + lambda_v * l_con_teacher) + 0.3 * l_dag + self.attr.lambda_attr * l_attr + l_ent

        # === Step / logging ===
        self._global_step += 1

        # Prepare W&B logging payloads
        losses_log = {
            "total": float(total.detach().item()),
            "contrastive": float(l_con.detach().item()),
            "kl_teacher": float(l_con_teacher.detach().item()),
            "hierarchy": float(l_dag.detach().item()),
            "attr": float(l_attr.detach().item()),
            "entropy": float(l_ent.detach().item()),
        }

        # Curriculum knobs (flatten if CurriculumConfig)
        sched = None
        if getattr(self, "curriculum_params", None) is not None:
            cp = self.curriculum_params
            # try both dataclass.cfg and dict-like access
            source = getattr(cp, "cfg", cp)
            sched = dict(
                hard_frac=float(getattr(source, "hard_frac", getattr(source, "hard_frac_start", 0.0))) if hasattr(source, "hard_frac") or hasattr(source, "hard_frac_start") else None,
                k_hard=int(getattr(source, "k_hard", 0)) if hasattr(source, "k_hard") else None,
                shortlist_M=int(getattr(source, "shortlist_M", 0)) if hasattr(source, "shortlist_M") else None,
                hier_max_hops_up=int(getattr(source, "hier_max_hops_up", 0)) if hasattr(source, "hier_max_hops_up") else None,
                hier_max_hops_down=int(getattr(source, "hier_max_hops_down", 0)) if hasattr(source, "hier_max_hops_down") else None,
                random_k=int(getattr(source, "random_k", 0)) if hasattr(source, "random_k") else None,
                use_inbatch_easy=float(getattr(source, "use_inbatch_easy", 0.0)) if hasattr(source, "use_inbatch_easy") else None,
            )

        # Phase identity (if your context exposes it)
        phase = dict(
            id=getattr(self, "phase_id", getattr(self.ctx, "phase_id", -1)),
            name=getattr(self, "phase_name", getattr(self.ctx, "phase_name", f"phase{getattr(self,'phase_id',-1)}"))
        )

        # Log losses + phase + sched
        self.wlogger.log_losses(losses_log, step=self._global_step, epoch=epoch_idx, phase=phase, sched=sched)

        # Accumulate phase-wise stats for summaries
        pid = int(phase["id"]) if isinstance(phase.get("id", -1), (int,)) else -1
        for k, v in losses_log.items():
            self._phase_acc[pid][f"loss/{k}"].append(v)
        if sched:
            for k, v in sched.items():
                if v is not None:
                    self._phase_acc[pid][f"sched/{k}"].append(float(v))

        # LoRA diagnostics every N steps
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
        device = self.cfg.device
        logs = {"total": 0.0, "contrastive": 0.0, "dag": 0.0, "attr": 0.0, "entropy": 0.0}
        n = 0

        for batch in loader:
            H = batch["prot_emb_pad"].to(device)
            pad_mask = ~batch["prot_attn_mask"].to(device)
            pos_local: List[torch.Tensor] = batch["pos_go_local"]

            # --- POSITIVES (GoEncoder if available, else fallback) ---
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

            # --- NEGATIVES (IDs via miner → embeddings via MemoryBank) ---
            mined = self._build_negatives(batch)
            neg_go_ids = mined["neg_go_ids"].to(device)

            # Candidate set with MemoryBank embeddings
            G_cand, pos_mask, _ = build_candidates_for_batch(
                uniq_go_embs=uniq_go_embs,
                pos_go_local=pos_local,
                neg_go_ids=neg_go_ids,
                go_cache_embs=self.ctx.memory_bank.embs.to(device),
                id2row=self.ctx.go_cache.id2row,
            )
            scores_cand, _ = self.forward_scores(H, G_cand, pad_mask)
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
                # For simplicity, reuse surrogate in eval here; switch to real windowed eval if desired.
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

            logs["total"] += float(total.detach().item())
            logs["contrastive"] += float(l_con.detach().item())
            logs["dag"] += float(l_dag.detach().item())
            logs["attr"] += float(l_attr.detach().item())
            logs["entropy"] += float(l_ent.detach().item())
            n += 1

        for k in logs:
            logs[k] /= max(1, n)
        return logs