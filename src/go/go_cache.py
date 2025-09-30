
from __future__ import annotations
from typing import List, Dict, Optional, Sequence, Mapping
import torch

class GoLookupCache:
    """Minimal cache-backed GO lookup for collate/loss."""
    def __init__(self,
                 embs_or_blob,
                 id2row: Optional[dict]=None,
                 row2id: Optional[Sequence[int]]=None):
        if isinstance(embs_or_blob, Mapping):
            b = embs_or_blob
            self.embs   = b["embs"]
            self.id2row = b.get("id2row", id2row)
            _row2id = b.get("row2id", b.get("ids", row2id))
        else:
            self.embs = embs_or_blob
            self.id2row = id2row
            _row2id = row2id

        self.row2id = torch.as_tensor(_row2id, dtype=torch.long)
        self.n_go = int(self.embs.size(0))

    def __call__(self, go_ids: Sequence[int]) -> torch.Tensor:
        """Returns embeddings of given global ids"""
        idxs = self.to_local(go_ids, drop_missing=True)
        if idxs.numel() == 0:
            return torch.empty(0, self.embs.size(1), dtype=self.embs.dtype, device=self.embs.device)
        return self.embs.index_select(0, idxs)

    def to_local(self, go_ids: Sequence[int], *, drop_missing: bool = True) -> torch.LongTensor:
        """Turn global ids into local indexes"""
        if not go_ids:
            return torch.empty(0, dtype=torch.long)
        idxs = [self.id2row.get(int(g), -1) for g in go_ids]
        idxs = torch.tensor(idxs, dtype=torch.long)
        if drop_missing:
            idxs = idxs[idxs >= 0]
        return idxs

    def mask_from_globals(self, terms: Sequence[int]) -> torch.BoolTensor:
        """[n_go] bool mask, given global id's will be True."""
        m = torch.zeros(self.n_go, dtype=torch.bool, device=self.embs.device)
        if not terms:
            return m
        for g in terms:
            j = self.id2row.get(int(g), -1)
            if j >= 0:
                m[j] = True
        return m
