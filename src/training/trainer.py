from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F

from src.models.alignment_model import ProteinGoAligner
from src.loss.attribution import attribution_loss, windowed_attr_loss
from src.configs.data_classes import TrainerConfig, AttrConfig, CurriculumConfig

# NEW: use the new builder
from src.training.batch_builder import BatchBuilder


# ------------- Helpers -------------
def l2_normalize(x: torch.Tensor, dim: int=-1, eps: float=1e-12) -> torch.Tensor:
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
    logits = scores / max(1e-8, tau)     # (B,K)
    denom = torch.logsumexp(logits, dim=-1)  # (B,)
    # To avoid -inf when no positives (shouldn't happen), clamp with tiny mask
    pos_logits = logits.masked_fill(~pos_mask, float('-inf'))
    num = torch.logsumexp(pos_logits, dim=-1)  # (B,)
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
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int,int]]]:
    """
    Return:
      G_cand: (B, K, Dg) candidates per item (positives first, then negatives)
      pos_mask: (B, K) booleans for positives
      slices: list of (pos_len, neg_len) per item (for debugging)
    """
    B = len(pos_go_local)
    device = uniq_go_embs.device
    Dg = uniq_go_embs.size(1)
    k_hard = neg_go_ids.size(1)

    # Build per-item positives (from uniq_go_embs)
    pos_embs = []
    pos_lens = []
    for b in range(B):
        if pos_go_local[b].numel() == 0:
            pos_embs.append(torch.zeros(0, Dg, device=device, dtype=uniq_go_embs.dtype))
            pos_lens.append(0)
        else:
            pos_embs.append(uniq_go_embs.index_select(0, pos_go_local[b].to(device)))
            pos_lens.append(int(pos_go_local[b].numel()))

    # Build per-item negatives (from global cache via id2row)
    neg_embs = []
    neg_lens = []
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

    # Pad to common K
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
    H = H.clone().detach().requires_grad_(True)
    scores, alpha_info = model(H, G, mask=pad_mask, return_alpha=True)
    y = scores.mean()
    y.backward(retain_graph=True)
    with torch.no_grad():
        dy_dH = H.grad.norm(dim=-1)  # (B,L)
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
    margin: non-negative margin (default 0)
    scale: optional scaling in softplus
    Returns: scalar loss
    """
    if dag_parents is None or len(pos_local) == 0:
        return torch.zeros((), device=scores_pos.device)

    B, T_max = scores_pos.shape
    losses = []
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
    Eval-time real mask-out for full-length case.
    alpha_full: (B,T,L). For each (b,t): pick Top-K residues, zero them (one-by-one), recompute scores.
    Returns delta_full: (B,T,L) with nonzero at tested residues, normalized per (b,t).
    """
    B, T, L = alpha_full.shape
    device = H.device
    delta = torch.zeros_like(alpha_full)
    base_scores, _ = model(H, G, mask=pad_mask, return_alpha=True)  # (B,T)

    for b in range(B):
        for t in range(T):
            topk = min(k, L)
            vals, idx = torch.topk(alpha_full[b, t], k=topk, dim=-1)
            for i in idx.tolist():
                Hminus = H.clone()
                Hminus[b, i, :] = 0.0
                y_minus, _ = model(Hminus, G[b:b+1], mask=pad_mask[b:b+1] if pad_mask is not None else None, return_alpha=True)
                delta[b, t, i] = (base_scores[b, t] - y_minus.squeeze(0)[t]).clamp_min(0.0)
            # normalize (b,t)
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
            d_g=None,  # from go encoder
            d_z=cfg.d_z, # TODO: bunun 768 olması gerek
            go_encoder=go_encoder,  # BioMedBERTEncoder (LoRA’lı)
            normalize=True,
        )
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self._global_step = 0
    #    self.builder = ctx.batch_builder

        if hasattr(self.ctx, "vres") and self.ctx.vres is not None:
            self.ctx.vres.query_projector = self.model.proj_p   # nn.Linear 1280->768

        if getattr(ctx, "scheduler", None) is not None:
            try:
                cur = ctx.scheduler(0)
                default_M = int(cur.cfg.shortlist_M if hasattr(cur, "shortlist_M") else 64)
                default_K = int(cur.cfg.k_hard if hasattr(cur, "k_hard") else 4)
            except Exception:
                pass
        if self.ctx.batch_builder is None:
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

    def forward_scores(self, H, G, pad_mask=None) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        return self.model(H, G, pad_mask = pad_mask, return_alpha=True)

    def forward_scores(self, H, G, pad_mask, **kwargs):
        """
        Use the model's tensor path to score arbitrary G tensors (e.g., candidates or positives).
        Keeps batch-dict forward (with _broadcast_batch_go) untouched elsewhere.
        """
        # default: return alpha unless explicitly disabled
        return_alpha = kwargs.pop("return_alpha", True)

        return self.model(
            H=H,
            G=G,
            mask=pad_mask,
            return_alpha=return_alpha,
            **kwargs,
        )

    def _build_negatives(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updated mining flow:
          - coarse shortlist via FAISS (vres.coarse_search)
          - true-pooling rerank using ctx.vres.true_prot_vecs (attention-weighted)
        """
        device = self.cfg.device
        # Resolve pos_go_global ids for builder
        if "pos_go_global" in batch:
            pos_go_global = batch["pos_go_global"]
        else:
            # fall back: map local -> global using uniq_go_ids
            uniq_go_ids = batch["uniq_go_ids"]  # (G,)
            pos_go_global = []
            for loc in batch["pos_go_local"]:
                pos_go_global.append(uniq_go_ids.index_select(0, loc.to(uniq_go_ids.device)))

        # Curriculum params
        curr_cfg = None
        if getattr(self.ctx, "scheduler", None) is not None:
            curr_cfg = self.ctx.scheduler(self._global_step)

        # Preferred: build directly from segmented embeddings (coarse + true rerank inside)
        out = self.builder.build_from_embs(
            prot_emb_pad=batch["prot_emb_pad"].to(device),
            prot_attn_mask=batch["prot_attn_mask"].to(device),
            pos_go_global=pos_go_global,
            zs_mask=batch["zs_mask"].to(device),
            curriculum_config=curr_cfg if isinstance(curr_cfg, CurriculumConfig) else None
        )
        return out

    def step_losses(self, batch: Dict[str, Any], epoch_idx: int) -> Dict[str, torch.Tensor]:
        self.model.train()
        device = self.cfg.device

        # --- Protein & mask ---
        H = batch["prot_emb_pad"].to(device)  # (B, Lmax, Dh)
        attn_valid = batch["prot_attn_mask"].to(device)  # True=VALID
        pad_mask = ~attn_valid
        pos_local: List[torch.Tensor] = batch["pos_go_local"]
        B = H.size(0)

        use_attr = (epoch_idx < self.attr.curriculum_epochs)

        # --- POSITIVES (GoEncoder → uniq_go_embs) ---
        if ("pos_go_tokens" in batch) and (hasattr(self.model, "go_encoder") and self.model.go_encoder is not None):
            toks = batch["pos_go_tokens"]
            uniq_go_embs = self.model.go_encoder(
                input_ids=toks["input_ids"].to(device),
                attention_mask=toks["attention_mask"].to(device),
            )  # [G, Dg], pooled
            # normalize (model.normalize'a paralel)
            try:
                uniq_go_embs = F.normalize(uniq_go_embs, p=2, dim=1)
            except Exception:
                pass
        else:
            # Fallback: collator/cache'ten gelen hazır gömüler
            uniq_go_embs = batch["uniq_go_embs"].to(device)  # (G, Dg)

        # ---- NEGATIVES (IDs via miner → embeddings via MemoryBank) ----
        mined = self._build_negatives(batch)
        neg_go_ids = mined["neg_go_ids"].to(device)  # (B, k_hard), -1 padded

        # ---- Candidate set (positives + mined negatives) using MemoryBank ----
        G_cand, pos_mask, _ = build_candidates_for_batch(
            uniq_go_embs=uniq_go_embs,  # [G, Dg]
            pos_go_local=pos_local,  # List[Tensor]
            neg_go_ids=neg_go_ids,  # [B, K]
            go_cache_embs=self.ctx.memory_bank.embs.to(device),
            id2row=self.ctx.go_cache.id2row,
        )  # -> (B, K, Dg), (B, K), meta

        use_weights = (epoch_idx >= self.attr.curriculum_epochs)
        weights_provider = getattr(self, "weights_provider", None) if use_weights else None
        v_true = self.ctx.vres.true_prot_vecs(H, attn_valid, watti_or_model=weights_provider)  # [B, d?]

        # ====================== Teacher (v_true) → KL DISTILLATION ======================


        # 1) Align v_true to the index/GO space (use projection if available)
        vt = v_true
        if self.ctx.vres is not None and hasattr(self.ctx.vres, "project_queries_to_index"):
            try:
                vt = self.ctx.vres.project_queries_to_index(vt)  # [B, Dg]
            except Exception:
                pass

        # 2) cos-sim eşdeğer dot için normalize et
        vt = F.normalize(vt, dim=-1)  # [B, Dg]
        Gn = F.normalize(G_cand, dim=-1)  # [B, K, Dg]

        # 3) öğretmen skorları: <v_true, G_k>  → [B, K]
        scores_teacher = torch.einsum("bd,bkd->bk", vt, Gn)

        # ---- Contrastive over candidate set (student) ----
        scores_cand, _ = self.forward_scores(H, G_cand, pad_mask)  # (B, K)
        l_con = multi_positive_infonce_from_candidates(scores_cand, pos_mask, self.attr.temperature)

        # 4) KL distillation: öğretmen dağılımı → öğrenci skorlarına etki etsin
        tau = float(self.attr.tau_distill)
        with torch.no_grad():
            p_t = F.softmax(scores_teacher / tau, dim=1)  # öğretmen hedefi (stopgrad)
        log_p_s = F.log_softmax(scores_cand / tau, dim=1)  # öğrenci
        l_con_teacher = F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)

        # öğretmen ağırlığı (config yoksa 0.2)
        lambda_v = float(getattr(self.attr, "lambda_vtrue", 0.2))
        # ================================================================================

        # ---- Attribution: positives-only G ----
        T_max = max((int(x.numel()) for x in pos_local), default=1)
        Dg = uniq_go_embs.size(1)
        G_pos = torch.zeros(B, T_max, Dg, device=device, dtype=uniq_go_embs.dtype)
        for b, loc in enumerate(pos_local):
            t = int(loc.numel())
            if t > 0:
                G_pos[b, :t] = uniq_go_embs.index_select(0, loc.to(uniq_go_embs.device))

        scores_pos, alpha_info = self.forward_scores(H, G_pos, pad_mask)

        # ---- Attribution loss (surrogate in train) ----
        if "alpha_full" in alpha_info:
            alpha = alpha_info["alpha_full"]  # (B,T,L)
            if use_attr:
                delta, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, pad_mask)
                l_attr = attribution_loss(alpha, delta, mask=None, reduce="mean")
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(alpha)
            else:
                l_attr = torch.zeros((), device=device)
                l_ent = torch.zeros((), device=device)
        elif all(k in alpha_info for k in ["alpha_windows", "win_weights", "spans"]):
            AW = alpha_info["alpha_windows"]
            Ww = alpha_info["win_weights"]
            spans = alpha_info["spans"]
            if use_attr:
                delta_full, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, pad_mask)
                B_, T_, W, win = AW.shape
                delta_win = torch.zeros_like(AW)
                for wi, (s, e) in enumerate(spans):
                    delta_win[:, :, wi, :e - s] = delta_full[:, :, s:e]
                l_attr = windowed_attr_loss(AW, Ww, spans, delta_win)
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(AW)
            else:
                l_attr = torch.zeros((), device=device)
                l_ent = torch.zeros((), device=device)
        else:
            l_attr = torch.zeros((), device=device)
            l_ent = torch.zeros((), device=device)

        # ---- DAG loss over positives ----
        uniq_go_ids = batch['uniq_go_ids'].to(device)
        l_dag = dag_consistency_loss_pos(scores_pos, pos_local, uniq_go_ids, self.ctx.dag_parents,
                                         margin=0.0, scale=1.0)

        # ---- Total loss (öğretmen dahil) ----
        total = (l_con + lambda_v * l_con_teacher) + 0.3 * l_dag + self.attr.lambda_attr * l_attr + l_ent
        self._global_step += 1

        # (Opsiyonel) W&B gözlem
        try:
            if getattr(self.cfg, "wandb", False) and hasattr(self, "wandb_logger") and self.wandb_logger is not None:
                self.wandb_logger.log({
                    "debug/pos_len_mean": float(
                        torch.tensor([len(x) for x in pos_local], device=device, dtype=torch.float32).mean().item()),
                    "debug/cand_K_mean": float(G_cand.size(1)),
                    "loss/contrastive_teacher": float(l_con_teacher.item()),
                }, step=self._global_step)
        except Exception:
            pass

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
                go_cache_embs=self.ctx.memory_bank.embs.to(device),  # <<< bank
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
                delta = topk_maskout_full(H, G_pos, alpha, k=self.attr.topk_per_window, model=self.model,
                                          pad_mask=pad_mask)
                l_attr = attribution_loss(alpha, delta, mask=None, reduce="mean")
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(alpha)
            elif all(k in alpha_info for k in ["alpha_windows", "win_weights", "spans"]):
                AW = alpha_info["alpha_windows"];
                Ww = alpha_info["win_weights"];
                spans = alpha_info["spans"]
                # Basitlik adına train surrogate tekrar: istersen gerçek windowed util'e geçersin
                delta_full, _ = surrogate_delta_y_from_mask_grad(H, G_pos, self.model, pad_mask)
                delta_win = torch.zeros_like(AW)
                for wi, (s, e) in enumerate(spans):
                    delta_win[:, :, wi, :e - s] = delta_full[:, :, s:e]
                l_attr = windowed_attr_loss(AW, Ww, spans, delta_win)
                l_ent = -self.attr.lambda_entropy_alpha * entropy_regularizer(AW)
            else:
                l_attr = torch.zeros((), device=device)
                l_ent = torch.zeros((), device=device)

            uniq_go_ids = batch['uniq_go_ids'].to(device)
            l_dag = dag_consistency_loss_pos(scores_pos, pos_local, uniq_go_ids, self.ctx.dag_parents,
                                             margin=0.0, scale=1.0)
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


