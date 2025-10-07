from __future__ import annotations

from typing import Dict, List, Optional, Union, Set, Deque, Tuple
from collections import deque
import random

import torch

# ---------------- Utilities ----------------
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
    """
    Normalize positives into a list of 1D LongTensors (length Pi per item).
    """
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


def _bfs_within_hops(start_ids: Set[int],
                     graph: Dict[int, List[int]],
                     max_hops: int) -> Set[int]:
    """
    Limited BFS: return all nodes reachable from any start id within 'max_hops' steps.
    If max_hops <= 0 or graph is None/empty, returns empty set.
    """
    if max_hops is None or max_hops <= 0 or not graph:
        return set()
    visited: Set[int] = set()
    q: Deque[Tuple[int, int]] = deque()
    for s in start_ids:
        q.append((s, 0))
        visited.add(s)
    reached: Set[int] = set()
    while q:
        node, d = q.popleft()
        if d == 0:
            # do not include the start nodes themselves as "within hops" by default
            pass
        if d > 0:
            reached.add(node)
        if d >= max_hops:
            continue
        for nb in graph.get(node, []):
            if nb not in visited:
                visited.add(nb)
                q.append((nb, d + 1))
    return reached


# ---------------- BatchBuilder ----------------
# Single entry point used by trainer: build_from_embs(...)
# Knobs (via CurriculumConfig):
#   - shortlist_M: ANN shortlist size
#   - k_hard: number of hard negatives from shortlist
#   - hier_max_hops_up / hier_max_hops_down: exclude candidates that are ancestors/descendants
#   - random_k: additional random easy negatives (after hard negatives)
#   - use_inbatch_easy: optionally fill part of random_k using other items' positives
#
# Output contract:
#   {
#     "neg_go_ids": LongTensor [B, K_total]  (-1 padded where needed),
#     "cand_ids":   LongTensor [B, M],        (may include -1)
#     "stats":      {"M": int, "K": int, "K_hard": int, "random_k": int, "inbatch_used": int}
#   }

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

        # Kept for API parity
        self.go_encoder_rerank = go_encoder_rerank
        self.dag_parents = dag_parents or {}
        self.dag_children = dag_children or {}
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
                        prot_emb_pad: torch.Tensor,            # [B, Lmax, Dh]
                        prot_attn_mask: torch.Tensor,          # [B, Lmax]  True=VALID
                        pos_go_global: Union[torch.Tensor, List[torch.Tensor]],
                        zs_mask: torch.Tensor,                  # [G] or [B,G]  (bool; True=ZS→drop)
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

        # ---- curriculum knobs (fallback to defaults)
        M = int(getattr(self, "default_M", 128))
        K_hard = int(getattr(self, "default_K", 64))
        hops_up = 0
        hops_down = 0
        random_k = 0
        use_inbatch_easy = False

        if curriculum_config is not None:
            # support either arrays (schedules) or scalars
            def _last_or_scalar(x, default=None):
                try:
                    if isinstance(x, (list, tuple)):
                        return x[-1]
                    return x
                except Exception:
                    return default

            M = int(_last_or_scalar(getattr(curriculum_config, "shortlist_M", M), M))
            K_hard = int(_last_or_scalar(getattr(curriculum_config, "k_hard", K_hard), K_hard))
            hops_up = int(_last_or_scalar(getattr(curriculum_config, "hier_max_hops_up", hops_up), hops_up))
            hops_down = int(_last_or_scalar(getattr(curriculum_config, "hier_max_hops_down", hops_down), hops_down))
            random_k = int(_last_or_scalar(getattr(curriculum_config, "random_k", random_k), random_k))
            use_inbatch_easy = bool(getattr(curriculum_config, "use_inbatch_easy", use_inbatch_easy))

        M = max(0, M)
        K_hard = max(0, K_hard)
        random_k = max(0, random_k)

        # ---- 1) Build query vectors
        if true_vecs is not None:
            q = true_vecs.to(device)
        elif self.vres is not None and hasattr(self.vres, "true_prot_vecs"):
            q = self.vres.true_prot_vecs(prot_emb_pad, prot_attn_mask)  # [B,d]
        else:
            q = _mean_pool(prot_emb_pad, prot_attn_mask)  # [B,d]

        # Ensure queries match index dim via VectorResources.project_queries_to_index if available
    #    if self.vres is not None and hasattr(self.vres, "project_queries_to_index"):
    #        try:
    #            q = self.vres.project_queries_to_index(q)
    #        except Exception:
    #            # fallback: assume already aligned
    #            pass

        # ---- 2) shortlist via ANN
        cand_ids = self._ann_query(q, M) if M > 0 else torch.empty(B, 0, dtype=torch.long, device=device)  # [B,M]
        if cand_ids.dtype != torch.long:
            cand_ids = cand_ids.long()
        cand_ids = cand_ids.to(device, non_blocking=True)

        # ---- 3) broadcast ZS mask and normalize positives
        zs = _broadcast_zs_mask(zs_mask.to(device).bool(), B)  # [B,G]
        assert zs.dim() == 2 and zs.size(0) == B, f"zs_mask after broadcast must be [B,G], got {tuple(zs.shape)}"
        G = int(zs.size(1))
        pos_list = _normalize_pos_list(pos_go_global, B, device)  # list of [Pi] long

        # Pre-build per-item positive sets for quick lookup
        pos_sets: List[Set[int]] = []
        for b in range(B):
            pb = pos_list[b]
            pos_sets.append(set(int(x) for x in pb.tolist()))

        # ---- 4) hierarchical masks (exclude close ancestors/descendants)
        # Build "close-to-positives" exclusion sets per item if hops knobs are set
        exclude_hier: List[Set[int]] = [set() for _ in range(B)]
        if (hops_up > 0 or hops_down > 0) and (self.dag_parents or self.dag_children):
            for b in range(B):
                Pset = pos_sets[b]
                if not Pset:
                    continue
                anc = _bfs_within_hops(Pset, self.dag_parents, hops_up) if hops_up > 0 else set()
                des = _bfs_within_hops(Pset, self.dag_children, hops_down) if hops_down > 0 else set()
                # Exclude ancestors/descendants within hops (do not exclude the positives themselves here
                # because they will be removed explicitly below)
                exclude_hier[b] = (anc | des)

        # ---- 5) filter shortlist: drop paddings, out-of-range, zero-shot, positives, hier-close
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
                raw = raw[valid]
                if raw.numel() == 0:
                    post_shortlist.append(raw)
                    continue
            else:
                post_shortlist.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            # c) zero-shot filter (True=ZS → drop)
            zs_b = zs[b]  # [G] bool
            keep_mask = ~zs_b[raw]  # [k]
            kept = raw[keep_mask]
            if kept.numel() == 0:
                post_shortlist.append(kept)
                continue

            # d) remove known positives
            if pos_sets[b]:
                keep2 = torch.tensor([int(x) not in pos_sets[b] for x in kept.tolist()],
                                     device=device, dtype=torch.bool)
                kept = kept[keep2]
                if kept.numel() == 0:
                    post_shortlist.append(kept)
                    continue

            # e) remove hierarchical-close negatives if requested
            if exclude_hier[b]:
                keep3 = torch.tensor([int(x) not in exclude_hier[b] for x in kept.tolist()],
                                     device=device, dtype=torch.bool)
                kept = kept[keep3]

            post_shortlist.append(kept)

        # ---- 6) choose K_hard from shortlist (truncate) ----
        hard_lists: List[torch.Tensor] = []
        for kept in post_shortlist:
            if kept.numel() >= K_hard:
                hard_lists.append(kept[:K_hard])
            else:
                hard_lists.append(kept)

        # ---- 7) add random/easy negatives
        # Build global allowed pool per item: ids in [0,G) that are not ZS, not positives, not already selected,
        # and not hier-close (if enforced).
        K_total_per_item: List[int] = []
        final_ids: List[torch.Tensor] = []
        inbatch_used_counts: List[int] = []

        # Prepare in-batch easy pool: union of other items' positives (per item)
        if use_inbatch_easy:
            all_other_pos: List[Set[int]] = []
            for b in range(B):
                others = set()
                for b2 in range(B):
                    if b2 == b:
                        continue
                    others |= pos_sets[b2]
                all_other_pos.append(others)
        else:
            all_other_pos = [set() for _ in range(B)]

        for b in range(B):
            selected = set(int(x) for x in hard_lists[b].tolist())
            # Base allowed mask
            allowed = []
            if G > 0:
                zs_b = zs[b].tolist()
                # We iterate through kept shortlist union random pool as needed
                # Build a candidate pool of all IDs; we'll filter below
                # NOTE: For speed, we sample later, not list all G always (could be large).
            # In-batch easy fill preference:
            inbatch_fill = []
            if use_inbatch_easy and random_k > 0 and all_other_pos[b]:
                # Prioritize other items' positives as "easy negatives" if they pass filters
                for gid in all_other_pos[b]:
                    if gid < 0 or gid >= G:
                        continue
                    if zs_b[gid]:
                        continue
                    if gid in pos_sets[b]:
                        continue
                    if gid in selected:
                        continue
                    if exclude_hier[b] and gid in exclude_hier[b]:
                        continue
                    inbatch_fill.append(gid)
                    if len(inbatch_fill) >= random_k:
                        break

            # Random pool fill (if still need)
            remaining_rand = max(0, random_k - len(inbatch_fill))
            rand_fill: List[int] = []
            if remaining_rand > 0 and G > 0:
                # Sample without replacement from a filtered pool.
                # To avoid enumerating all G, we attempt randomized trials with an upper cap.
                trials = 0
                cap = max(1000, 20 * remaining_rand)
                while len(rand_fill) < remaining_rand and trials < cap:
                    gid = random.randrange(0, G)
                    trials += 1
                    if zs_b[gid]:
                        continue
                    if gid in pos_sets[b] or gid in selected:
                        continue
                    if exclude_hier[b] and gid in exclude_hier[b]:
                        continue
                    # avoid duplicates against inbatch_fill / rand_fill
                    if gid in rand_fill or gid in inbatch_fill:
                        continue
                    rand_fill.append(gid)

            # Merge: hard -> inbatch -> random
            merged = list(selected)
            # Keep order stable: hard first in original shortlist order
            hard_tensor = hard_lists[b]
            merged = [int(x) for x in hard_tensor.tolist()]
            if len(inbatch_fill) > 0:
                merged += inbatch_fill
            if len(rand_fill) > 0:
                merged += rand_fill

            K_total = len(merged)
            K_total_per_item.append(K_total)
            inbatch_used_counts.append(len(inbatch_fill))

            final_ids.append(torch.as_tensor(merged, dtype=torch.long, device=device))

        # ---- 8) pad to common K_total across batch with -1
        K_max = max(K_total_per_item) if K_total_per_item else 0
        padded_final: List[torch.Tensor] = []
        for ids in final_ids:
            need = K_max - ids.numel()
            if need > 0:
                pad = torch.full((need,), -1, dtype=torch.long, device=device)
                ids = torch.cat([ids, pad], dim=0)
            padded_final.append(ids)

        neg_go_ids = (torch.stack(padded_final, dim=0)
                      if len(padded_final) > 0
                      else torch.empty((B, 0), dtype=torch.long, device=device))

        stats = {
            "M": int(M),
            "K": int(K_max),
            "K_hard": int(K_hard),
            "random_k": int(random_k),
            "inbatch_used": int(sum(inbatch_used_counts)) if inbatch_used_counts else 0
        }

        return {
            "neg_go_ids": neg_go_ids,   # [B,K_max], -1 padded
            "cand_ids": cand_ids,       # [B,M], may contain -1
            "stats": stats,
        }