from __future__ import annotations
from typing import Any, Dict, List
import torch

class ContrastiveEmbCollator:
    """
    Retriever v2 training/evaluation collator.

    Responsibilities:
      - Pad variable-length protein residue embeddings.
      - Build the protein attention mask.
      - Preserve each protein's GLOBAL positive GO ids.

    Intentionally NOT responsible for:
      - GO tokenization or GO encoding.
      - Building batch-local unique GO candidate sets.
      - Mapping positives to batch-local GO indices.
      - Negative sampling / queue mining.
      - Positive masks over the active GO universe.

    The trainer owns the active full GO universe and constructs the [B, G]
    positive mask from ``pos_go_global``.
    """

    def __init__(self) -> None:
        # Kept as an explicit constructor so call sites remain clear.
        pass

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch received.")

        # ---------------- Protein padding ----------------
        prot_list = [b["prot_emb"] for b in batch]

        if any(not torch.is_tensor(p) or p.ndim != 2 for p in prot_list):
            raise ValueError(
                "Each batch item must contain prot_emb as a rank-2 tensor [L, D]."
            )

        batch_size = len(prot_list)
        embed_dim = int(prot_list[0].shape[1])

        if any(int(p.shape[1]) != embed_dim for p in prot_list):
            raise ValueError("All protein embeddings in a batch must share D.")

        max_len = max(int(p.shape[0]) for p in prot_list)

        prot_pad = torch.zeros(
            batch_size,
            max_len,
            embed_dim,
            dtype=prot_list[0].dtype,
        )
        prot_attn_mask = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.bool,
        )

        for i, prot_emb in enumerate(prot_list):
            length = int(prot_emb.shape[0])
            prot_pad[i, :length] = prot_emb
            prot_attn_mask[i, :length] = True

        # ---------------- Global positive GO ids ----------------
        pos_go_global: List[torch.Tensor] = []

        for item in batch:
            if "pos_go_ids" not in item:
                raise KeyError("Batch item is missing required key: pos_go_ids")

            pos_ids = item["pos_go_ids"]
            if not torch.is_tensor(pos_ids):
                pos_ids = torch.as_tensor(pos_ids, dtype=torch.long)
            else:
                pos_ids = pos_ids.to(dtype=torch.long)

            pos_ids = pos_ids.reshape(-1)

            if pos_ids.numel() == 0:
                raise ValueError(
                    f"Protein {item.get('protein_id', '<unknown>')} has no positive GO ids."
                )

            # Dataset construction should already provide unique positives, but
            # enforce that invariant here because duplicate positives would
            # distort InfoNCE/PBR normalization.
            if torch.unique(pos_ids).numel() != pos_ids.numel():
                raise ValueError(
                    f"Protein {item.get('protein_id', '<unknown>')} contains duplicate "
                    "positive GO ids."
                )

            pos_go_global.append(pos_ids)

        return {
            "protein_ids": [b["protein_id"] for b in batch],
            "prot_emb_pad": prot_pad,
            "prot_attn_mask": prot_attn_mask,
            "pos_go_global": pos_go_global,
        }
