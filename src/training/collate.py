from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch

class ContrastiveEmbCollator:
    """
    Training collator (LoRA always on):
      - Outputs GO ids + GO tokens only.
      - NEVER outputs cached GO embeddings.

    Required in trainer:
      - go_encoder must exist
      - batch must contain pos_go_tokens
    """

    def __init__(
        self,
        zs_mask_vec: Optional[torch.Tensor],
        go_text_store: object,
        bidirectional: bool = True,
        neg_k: int = 0,
        num_labels=None,
        device: torch.device = torch.device("cpu"),
        go_dropout=None,
    ):
        self.device = device
        self.go_text_store = go_text_store
        self.bidirectional = bidirectional
        self.neg_k = int(neg_k)
        self.go_dropout = go_dropout

        if self.go_text_store is None:
            raise ValueError("go_text_store is required (LoRA training needs GO tokens).")

        if zs_mask_vec is None:
            if num_labels is None:
                raise ValueError("zs_mask_vec is None -> num_labels mandatory.")
            self.zs_mask_vec = torch.ones(num_labels, dtype=torch.bool, device=self.device)
        else:
            self.zs_mask_vec = zs_mask_vec.to(self.device).bool()

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

        # ---------------- Positives → uniq GO ids ----------------
        pos_lists = [b["pos_go_ids"] for b in batch if b["pos_go_ids"].numel() > 0]
        if len(pos_lists) > 0:
            uniq_go = torch.unique(torch.cat(pos_lists))
        else:
            uniq_go = torch.empty(0, dtype=torch.long)

        idx_map = {int(g): j for j, g in enumerate(uniq_go.tolist())}

        # Per-protein local positive indices + aligned weights
        pos_local: List[torch.Tensor] = []
        pos_local_w: List[torch.Tensor] = []
        pos_go_global: List[torch.Tensor] = []

        for b in batch:
            ids_t = b["pos_go_ids"]
            pos_go_global.append(ids_t)

            ids = ids_t.tolist()
            wts = b.get("pos_go_weights", None)
            wts_list = [1.0] * len(ids) if wts is None else b["pos_go_weights"].tolist()

            local_idx, local_w = [], []
            for g, w in zip(ids, wts_list):
                j = idx_map.get(int(g), None)
                if j is not None:
                    local_idx.append(j)
                    local_w.append(float(w))

            pos_local.append(torch.as_tensor(local_idx, dtype=torch.long))
            pos_local_w.append(torch.as_tensor(local_w, dtype=torch.float32))

        # ---------------- Tokens for GoEncoder (mandatory if uniq_go not empty) ----------------
        pos_go_tokens = None
        if uniq_go.numel() > 0:
            pos_go_tokens = self.go_text_store.batch(uniq_go.tolist())
            if self.go_dropout is not None:
                pos_go_tokens["input_ids"], pos_go_tokens["attention_mask"] = self.go_dropout(
                    pos_go_tokens["input_ids"], pos_go_tokens["attention_mask"]
                )
            assert pos_go_tokens["input_ids"].size(0) == uniq_go.numel()
            assert pos_go_tokens["attention_mask"].size(0) == uniq_go.numel()

        out: Dict[str, Any] = dict(
            protein_ids=[b["protein_id"] for b in batch],
            prot_emb_pad=prot_pad,
            prot_attn_mask=attn_mask,
            pos_go_local=pos_local,
            pos_go_local_weights=pos_local_w,
            pos_go_global=pos_go_global,
            uniq_go_ids=uniq_go,
            zs_mask=self.zs_mask_vec,
        )

        if pos_go_tokens is not None:
            out["pos_go_tokens"] = pos_go_tokens

        # symmetric loss buckets
        if self.bidirectional:
            G = int(uniq_go.shape[0])
            buckets: List[List[int]] = [[] for _ in range(G)]
            for pi, local_idxs in enumerate(pos_local):
                for j in local_idxs.tolist():
                    buckets[j].append(pi)
            out["go2prot_local"] = [torch.as_tensor(x, dtype=torch.long) for x in buckets]

        return out