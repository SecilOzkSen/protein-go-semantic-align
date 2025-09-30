"""
ProteinEmbDataset (store-only)
--------------------------------
Dataset for CLIP-style proteins <-> GO training with precomputed residue embeddings

The dataset:
  - Reads residue-level embeddings through a provided `ESMResidueStore`.
  - ZS-sterilizes positives; optionally expands with ancestors (weighted).
  - Provides few-shot flags and per-example sampling weights.

Expected upstream:
  - ESM shards manifest & loader handled by `residue_store.ESMResidueStore`.

Returns per item (dict):
  - protein_id: str
  - prot_emb:   torch.FloatTensor [L, D]
  - pos_go_ids: torch.LongTensor  [P]
  - pos_go_weights: torch.FloatTensor [P]
  - is_fs: bool
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple
from src.configs.parameters import ALLOWED_RELS_FOR_DAG
from src.configs.data_classes import FewZeroConfig
from src.go.go_dag import expand_with_ancestors
from src.go.go_cache import GoLookupCache
from .residue_store import ESMResidueStore


def make_zs_mask(zs_terms: Set[int], n_go: int) -> np.ndarray:
    """Global boolean vector (length n_go) marking ZS terms; used later to filter negatives."""
    m = np.zeros((n_go,), dtype=np.bool_)
    if zs_terms:
        m[list(zs_terms)] = True
    return m


class ProteinEmbDataset(Dataset):
    """
    Dataset for CLIP-style proteins <-> GO training with precomputed embeddings.

    Parameters
    ----------
    protein_ids : Sequence[str]
        Ordered list of proteins ids to index into.
    pid2pos : Dict[str, List[int]]
        Map from protein_id -> positive GO ids (raw labels before ZS filtering).
    n_go : int
        Total number of GO terms (size of the global GO index space).
    fewzero : FewZeroConfig
        Few-shot / Zero-shot control.
    manifest_cache : Optional[str]
        (Unused in store-only mode; kept for API compatibility.)
    shard_cache_items : int
        (Unused in store-only mode; kept for API compatibility.)
    dag_parents : Optional[Mapping[int, Sequence[Tuple[int, str]]]]
        Parent adjacency with relations. If provided and positives are too few,
        ancestors are added as "soft positives" (weighted).
    min_pos_for_expand, max_ancestor_add, max_hops, ancestor_stoplist, ancestor_gamma
        Expansion policy knobs (see file header).
    store : ESMResidueStore
        REQUIRED. All protein embeddings are loaded through this store.
    """

    def __init__(
            self,
            protein_ids: Sequence[str],
            pid2pos: Dict[str, List[int]],
            go_cache: GoLookupCache,
            fewzero: FewZeroConfig,
            *,
            dag_parents: Optional[Mapping[int, Sequence[Tuple[int, str]]]] = None,
            min_pos_for_expand: int = 3,
            max_ancestor_add: int = 4,
            max_hops: Optional[int] = 3,
            ancestor_stoplist: Optional[Set[int]] = None, # nice to set it. Avoids big parent terms to be overlearned.
            ancestor_gamma: float = 0.7,
            store: ESMResidueStore = None,
    ):
        super().__init__()
        if store is None:
            raise ValueError("ProteinEmbDataset requires a non-None 'store' (ESMResidueStore)." )

        self.pids = list(protein_ids)
        self.n_go = go_cache.n_go
        self.fewzero = fewzero
        self.store = store          # single source of truth for embeddings

        # Prepare labels per protein: ZS sterilize positives, then optionally expand with ancestors
        self.pid2pos: Dict[str, List[int]] = pid2pos
        self.pos_weights_map: Dict[str, List[float]] = {}
        self.pos_is_generalized: Dict[str, List[bool]] = {}

        for pid in self.pids:
            orig = self.pid2pos.get(pid, [])
            # ZS sterilization: drop any ZS term from positives (positives only)
            pos = [g for g in orig if g not in fewzero.zero_shot_terms]

            # If too few, optionally expand with nearest ancestors (weighted)
            expanded, weights, is_gen_map = expand_with_ancestors(
                pos_terms=pos,
                parents=dag_parents,
                zs_blocklist=fewzero.zero_shot_terms,
                min_pos=min_pos_for_expand,
                max_add=max_ancestor_add,
                max_hops=max_hops,
                stoplist=ancestor_stoplist,
                gamma=ancestor_gamma,
                allowed_rels=set(ALLOWED_RELS_FOR_DAG)
            )
            self.pos_weights_map[pid] = [float(w) for w in weights]
            self.pos_is_generalized[pid] = [bool(is_gen_map[int(t)]) for t in expanded]

        # Few-shot flag per protein (for sampler)
        self.is_fs: List[bool] = []
        for pid in self.pids:
            labels = set(self.pid2pos.get(pid, []))
            self.is_fs.append(any((g in fewzero.few_shot_terms) for g in labels))

        # Global ZS mask for later negative filtering (e.g., after FAISS shortlist)
        self.zs_mask = go_cache.mask_from_globals(fewzero.zero_shot_terms)

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int):
        """Return one training item for contrastive alignment."""
        pid = self.pids[idx]

        # Single source of truth — load via store (already trimmed if seq_len provided there)
        prot_emb = self.store.get(pid)   # torch.FloatTensor [L, D]

        pos_list = self.pid2pos.get(pid, [])
        pos_wts = self.pos_weights_map.get(pid, [1.0] * len(pos_list))

        return {
            "protein_id": pid,
            "prot_emb": prot_emb,  # [L, D]
            "pos_go_ids": torch.as_tensor(pos_list, dtype=torch.long),  # [P]
            "pos_go_weights": torch.as_tensor(pos_wts, dtype=torch.float32),  # [P]
            "is_fs": self.is_fs[idx],
        }

    def sample_weights(self) -> np.ndarray:
        """
        Per-example weights for WeightedRandomSampler to reach fs_target_ratio FS in batches.
        Strategy:
          Assign w_fs to FS examples and w_cm to others, then renormalize to sum≈N.
        """
        w_fs = float(self.fewzero.fs_target_ratio)
        w_cm = 1.0 - w_fs
        base = np.array([w_fs if fs else w_cm for fs in self.is_fs], dtype=np.float64)
        s = base.sum()
        if s > 0:
            base *= (len(base) / s)
        return base
