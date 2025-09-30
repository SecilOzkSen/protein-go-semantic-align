from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

# Utilities
@torch.no_grad()
def _mean_pool(H: torch.Tensor, attn_valid: torch.Tensor) -> torch.Tensor:
    """
    H: [B, Lmax, Dh], attn_valid: [B, Lmax] (True=VALID)
    returns: [B, Dh] L2-normalized
    """
    m = attn_valid.float()  # True=1.0
    denom = torch.clamp(m.sum(dim=1, keepdim=True), min=1.0)  # [B,1]
    pooled = (H * m.unsqueeze(-1)).sum(dim=1) / denom
    return torch.nn.functional.normalize(pooled, p=2, dim=-1)


def _broadcast_zs_mask(zs_mask: torch.Tensor, B: int) -> torch.Tensor:
    """
    zs_mask: [G] or [B,G] (bool) -> [B,G]
    """
    if zs_mask.dim() == 1:
        return zs_mask.view(1, -1).expand(B, -1)
    if zs_mask.dim() == 2 and zs_mask.size(0) in (1, B):
        return zs_mask.expand(B, -1) if zs_mask.size(0) == 1 else zs_mask
    raise ValueError(f"Unexpected zs_mask shape: {zs_mask.shape}")


def _normalize_pos_list(pos_go_global: Union[torch.Tensor, List[torch.Tensor]],
                        B: int, device: torch.device) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if isinstance(pos_go_global, torch.Tensor):
        assert pos_go_global.dim() == 2 and pos_go_global.size(0) == B
        for b in range(B):
            row = pos_go_global[b]
            row = row[row >= 0]
            out.append(row.to(device=device, dtype=torch.long).view(-1))
    else:
        assert len(pos_go_global) == B
        for i in range(B):
            x = pos_go_global[i]
            t = x if isinstance(x, torch.Tensor) else torch.as_tensor(list(x), dtype=torch.long)
            out.append(t.to(device).view(-1))
    return out


# Simplified BatchBuilder (drop-in)
#   - Single entry point used by trainer: build_from_embs(...)
#   - Coarse shortlist via ANN (VectorResources or FAISS)
#   - Filter by zero-shot mask and remove positives
#   - Optional curriculum to read M/K (else defaults)

class BatchBuilder:
    def __init__(self,
                 vres: Optional[object] = None,
                 faiss_index: Optional[object] = None,
                 go_encoder_rerank: Optional[object] = None,
                 dag_parents: Optional[dict] = None,
                 dag_children: Optional[dict] = None,
                 scheduler: Optional[object] = None,
                 default_M: int = 512,
                 default_K: int = 64,
                 all_go_ids: Optional[Union[dict, list]] = None,
                 use_hier_mask: bool = False):
        self.vres = vres
        self.faiss_index = faiss_index
        self.scheduler = scheduler
        self.default_M = int(default_M)
        self.default_K = int(default_K)
        self.all_go_ids = all_go_ids  # id mapping if needed

        # kept only for API compatibility (not used inside simplified flow)
        self.go_encoder_rerank = go_encoder_rerank
        self.dag_parents = dag_parents
        self.dag_children = dag_children
        self.use_hier_mask = use_hier_mask

    # --- internal ANN adaptor ---
    @torch.no_grad()
    def _ann_query(self, q: torch.Tensor, M: int) -> torch.Tensor:
        """
        q: [B, d] (L2-normalized)
        returns LongTensor [B, M] of candidate GO row-ids (global)
        """
        # Option 1: VectorResources API (preferred)
        if self.vres is not None:
            if hasattr(self.vres, "query"):
                return self.vres.query(q, M)  # expected LongTensor [B,M]
            if hasattr(self.vres, "coarse_search"):
                _, I = self.vres.coarse_search(q, M)  # (D, I); take I
                return I.to(q.device, dtype=torch.long)

        # Option 2: raw FAISS index (fallback)
        if self.faiss_index is not None:
            try:
                import numpy as np  # type: ignore
                q_np = q.detach().cpu().numpy().astype("float32")
                D, I = self.faiss_index.search(np.ascontiguousarray(q_np), M)
                return torch.as_tensor(I, dtype=torch.long, device=q.device)
            except Exception:
                pass

        # Option 3: give up -> filled with -1
        B = q.size(0)
        return torch.full((B, M), -1, dtype=torch.long, device=q.device)

    @torch.no_grad()
    def build_from_embs(self,
                        prot_emb_pad: torch.Tensor,  # [B, Lmax, Dh]
                        prot_attn_mask: torch.Tensor,  # [B, Lmax]  True=VALID
                        pos_go_global: Union[torch.Tensor, List[torch.Tensor]],
                        zs_mask: torch.Tensor,  # [G] or [B,G]  (bool; True=ZS→drop)
                        curriculum_config: Optional["CurriculumConfig"] = None,
                        true_vecs: Optional[torch.Tensor] = None,  # optional direct [B,d] query
                        ) -> Dict[str, torch.Tensor]:
        """
        Returns negatives mined via ANN, filtered by zero-shot mask and known positives.

        Output:
          {
            "neg_go_ids": LongTensor [B,K]  (-1 padded),
            "cand_ids":   LongTensor [B,M]  (may include -1),
            "stats":      {"M": int, "K": int}
          }
        Assumptions:
          - zs_mask: True means zero-shot (should be dropped).
          - pos_go_global: [B,P] tensor or list of 1D tensors with positive GO ids (>=0).
        """
        device = prot_emb_pad.device
        B = int(prot_emb_pad.size(0))

        # ---- curriculum M/K (fallback to defaults)
        M = int(getattr(self, "default_M", 128))
        K = int(getattr(self, "default_K", 64))
        if curriculum_config is not None:
            try:
                M = int(curriculum_config.shortlist_M[-1])
                K = int(curriculum_config.k_hard[-1])
            except Exception:
                pass
        M = max(0, M)
        K = max(0, K)

        # ---- 1) Build query vectors
        if true_vecs is not None:
            q = true_vecs.to(device)
        elif self.vres is not None and hasattr(self.vres, "true_prot_vecs"):
            # prefer VectorResources true_prot_vecs if available
            q = self.vres.true_prot_vecs(prot_emb_pad, prot_attn_mask)  # [B,d]
        else:
            q = _mean_pool(prot_emb_pad, prot_attn_mask)  # [B,d]

        # Ensure queries match index dim via VectorResources.project_queries_to_index if available
        if self.vres is not None and hasattr(self.vres, "project_queries_to_index"):
            try:
                q = self.vres.project_queries_to_index(q)
            except Exception:
                # fallback: assume already aligned
                pass

        # ---- 2) shortlist via ANN
        cand_ids = self._ann_query(q, M) if M > 0 else torch.empty(B, 0, dtype=torch.long, device=device)  # [B,M]
        if cand_ids.dtype != torch.long:
            cand_ids = cand_ids.long()
        cand_ids = cand_ids.to(device, non_blocking=True)

        # ---- 3) filter by zs_mask and remove positives (robust)
        zs = _broadcast_zs_mask(zs_mask.to(device).bool(), B)  # [B,G]
        assert zs.dim() == 2 and zs.size(0) == B, f"zs_mask after broadcast must be [B,G], got {tuple(zs.shape)}"
        G = int(zs.size(1))

        pos_list = _normalize_pos_list(pos_go_global, B, device)  # list of [Pi] long

        post_shortlist: List[torch.Tensor] = []
        for b in range(B):
            raw = cand_ids[b]  # [M], may include -1
            # a) drop paddings
            raw = raw[raw >= 0]
            if raw.numel() == 0:
                post_shortlist.append(raw)
                continue

            # b) restrict to valid id range [0, G)
            if G > 0:
                valid = raw < G
                if not torch.all(valid):
                    raw = raw[valid]
                    if raw.numel() == 0:
                        post_shortlist.append(raw)
                        continue
            else:
                # No GO space; everything invalid
                post_shortlist.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            # c) zero-shot filter (True=ZS → drop)
            zs_b = zs[b]  # [G] bool
            keep_mask = ~zs_b[raw]  # [k]
            kept = raw[keep_mask]  # [k_keep]
            if kept.numel() == 0:
                post_shortlist.append(kept)
                continue

            # d) remove known positives
            pos_b = pos_list[b]  # [Pi]
            if pos_b.numel() > 0:
                pos_set = set(int(x) for x in pos_b.tolist())
                if pos_set:
                    keep2 = torch.tensor([int(x) not in pos_set for x in kept.tolist()],
                                         device=device, dtype=torch.bool)
                    kept = kept[keep2]

            post_shortlist.append(kept)

        # ---- 4) top-K from shortlist, with fallback padding (-1)
        final_ids: List[torch.Tensor] = []
        for kept in post_shortlist:
            if kept.numel() >= K:
                final_ids.append(kept[:K])
            else:
                need = K - kept.numel()
                if need > 0:
                    pad = torch.full((need,), -1, dtype=torch.long, device=device)
                    final_ids.append(torch.cat([kept, pad], dim=0))
                else:
                    final_ids.append(kept)

        neg_go_ids = (torch.stack(final_ids, dim=0)
                      if len(final_ids) > 0
                      else torch.empty((B, K), dtype=torch.long, device=device))

        return {
            "neg_go_ids": neg_go_ids,  # [B,K], -1 padded
            "cand_ids": cand_ids,  # [B,M], may contain -1
            "stats": {"M": int(M), "K": int(K)},
        }
