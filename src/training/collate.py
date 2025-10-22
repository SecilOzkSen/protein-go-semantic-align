from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence
import torch



def fused_collator(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pids = [b["protein_id"] for b in batch]
    Z = torch.stack([b["prot_fused"] for b in batch], dim=0)  # [B,D]
    return {"protein_ids": pids, "prot_fused": Z}

class ContrastiveEmbCollator:
    """
    Collate for embedding-mode protein inputs and CLIP-style bi-directional loss.

    Parameters
    ----------
    go_lookup : Callable[[List[int]], torch.Tensor]
        Maps GO ids -> embedding matrix [G, Dg]. Typically a cache-backed lookup.
    zs_mask_vec : torch.Tensor
        Global zero-shot mask [n_go] (bool). Passed through for miners/filters.
    bidirectional : bool
        If True, also returns GO→Protein match lists for symmetric CLIP losses.
    go_text_store : Optional[object]
        If provided, used to create `pos_go_tokens` via `go_text_store.batch(pos_ids)`.
        Must expose: `.batch(List[int]) -> Dict[str, torch.Tensor]` with keys
        {"input_ids","attention_mask"}.
    faiss_miner : Optional[Callable[[List[int], int, torch.Tensor], List[List[int]]]]
        If provided, used to mine negative GO ids per protein. Signature:
        miner(pos_ids, k, zs_mask) -> List[List[int]]  (length B, each length K).
    neg_k : int
        Number of negatives per protein to request from miner (if miner is set).
    """

    def __init__(self,
                 go_lookup: Callable[[List[int]], torch.Tensor],
                 zs_mask_vec: torch.Tensor,
                 bidirectional: bool = True,
                 go_text_store: Optional[object] = None,
                 faiss_miner: Optional[Callable[[List[int], int, torch.Tensor], List[List[int]]]] = None,
                 neg_k: int = 0,
                 num_labels=None,
                 device: torch.device = torch.device("cpu")):
        self.go_lookup = go_lookup
        self.device = device
        if zs_mask_vec is None:
            if num_labels is None:
                raise ValueError("zs_mask_vec is None -> num_labels mandatory.")
            self.zs_mask_vec = torch.ones(num_labels, dtype=torch.bool, device=self.device)
        else:
            self.zs_mask_vec = zs_mask_vec.to(self.device).bool()
   #     self.zs_mask_vec = zs_mask_vec.bool()
        self.bidirectional = bidirectional
        self.go_text_store = go_text_store
        self.faiss_miner = faiss_miner
        self.neg_k = int(neg_k)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ---------------- Protein padding ----------------
        prot_list = [b["prot_emb"] for b in batch]
        B = len(prot_list)
        if B == 0:
            raise ValueError("Empty batch received.")
        D = int(prot_list[0].shape[1])
        Lmax = max(int(p.shape[0]) for p in prot_list)

        prot_pad = torch.zeros(B, Lmax, D, dtype=prot_list[0].dtype)
        attn_mask = torch.zeros(B, Lmax, dtype=torch.bool)
        for i, P in enumerate(prot_list):
            L = int(P.shape[0])
            prot_pad[i, :L] = P
            attn_mask[i, :L] = True

        # ---------------- Positives → uniq GO set + embeddings ----------------
        pos_lists = [b["pos_go_ids"] for b in batch if b["pos_go_ids"].numel() > 0]
        if len(pos_lists) > 0:
            uniq_go = torch.unique(torch.cat(pos_lists))
            uniq_go_embs = self.go_lookup(uniq_go.tolist())  # [G, Dg]
        else:
            uniq_go = torch.empty(0, dtype=torch.long)
            uniq_go_embs = torch.empty(0)

        # Global GO id -> local index
        idx_map = {int(g): j for j, g in enumerate(uniq_go.tolist())}

        # Per-protein local positive indices + aligned weights
        pos_local: List[torch.Tensor] = []
        pos_local_w: List[torch.Tensor] = []
        # per-batch positive GO id global
        pos_go_ids_flat: List[int] = []
        # weights positives.
        for b in batch:
            ids = b["pos_go_ids"].tolist()
            pos_go_ids_flat.extend(ids)
            wts = b.get("pos_go_weights", None)
            if wts is None:
                wts_list = [1.0] * len(ids)
            else:
                wts_list = b["pos_go_weights"].tolist()

            local_idx, local_w = [], []
            for g, w in zip(ids, wts_list):
                j = idx_map.get(int(g), None)
                if j is not None:
                    local_idx.append(j)
                    local_w.append(float(w))

            pos_local.append(torch.as_tensor(local_idx, dtype=torch.long))
            pos_local_w.append(torch.as_tensor(local_w, dtype=torch.float32))

        # pos_go_tokens (for GoEncoder)
        # Note: We will use the tokens in the forward pass for gradient flow with GoEncoder.
        pos_go_tokens = None
        # Since there can be multiple positives per protein, we don't want to enforce
        # "1 positive per protein" here. Therefore, generating the tokenizer batch
        # over 'uniq_go_ids' is more deterministic and compute-friendly (G items).
        if self.go_text_store is not None and uniq_go.numel() > 0:
            pos_go_tokens = self.go_text_store.batch(uniq_go.tolist())  # dict of [G, L]

        # ---------------- NEW: negative mining (IDs only) ----------------
        neg_go_ids: Optional[List[List[int]]] = None
        if self.faiss_miner is not None and self.neg_k > 0:
            seed_pos_ids: List[int] = []
            for b in batch:
                pids = b["pos_go_ids"].tolist()
                seed_pos_ids.append(int(pids[0]) if len(pids) > 0 else -1)  # -1 → miner filtreleyebilir
            neg_go_ids = self.faiss_miner(seed_pos_ids, self.neg_k, self.zs_mask_vec)  # List[List[int]]

        # ---------------- Output dict ----------------
        out: Dict[str, Any] = dict(
            protein_ids=[b["protein_id"] for b in batch],
            prot_emb_pad=prot_pad,                 # [B, Lmax, D]
            prot_attn_mask=attn_mask,             # [B, Lmax]
            pos_go_local=pos_local,               # List[LongTensor], per protein (local idx in uniq_go)
            pos_go_local_weights=pos_local_w,     # List[FloatTensor], aligned with pos_go_local
            uniq_go_ids=uniq_go,                  # [G]
            uniq_go_embs=uniq_go_embs,            # [G, Dg] (backward-compat; can be ignored when GoEncoder is on)
            zs_mask=self.zs_mask_vec,             # [n_go]
            pos_go_global=[b["pos_go_ids"] for b in batch],  # per protein global ids (backward-compat)
        )

        # NEW fields (optional)
        if pos_go_tokens is not None:
            out["pos_go_tokens"] = pos_go_tokens      # dict: {"input_ids":[G,L], "attention_mask":[G,L]}
        if neg_go_ids is not None:
            out["neg_go_ids"] = neg_go_ids            # List[List[int]]  (B x K)

        # ---------------- Optional: GO→Protein buckets for symmetric loss ----------------
        if self.bidirectional:
            G = int(uniq_go.shape[0])
            buckets: List[List[int]] = [[] for _ in range(G)]
            for pi, local_idxs in enumerate(pos_local):
                for j in local_idxs.tolist():
                    buckets[j].append(pi)
            go2prot_local = [torch.as_tensor(x, dtype=torch.long) for x in buckets]
            out["go2prot_local"] = go2prot_local

        return out
