# protein_emb_dataset.py
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from src.configs.parameters import ALLOWED_RELS_FOR_DAG
from src.configs.data_classes import FewZeroConfig
from src.go.go_dag import expand_with_ancestors
from src.go.go_cache import GoLookupCache
from .residue_store import ESMResidueStore, ESMFusedStore  # <-- fused import

class ProteinEmbDataset(Dataset):
    """
    CLIP-style protein<->GO training dataset.
    ZORUNLU: residue store (ESMResidueStore), opsiyonel: fused store (sadece yan bilgi).
    Dönen:
      - protein_id: str
      - prot_emb:   FloatTensor [L,D]   (EĞİTİM)
      - pos_go_ids: LongTensor  [P]
      - pos_go_weights: FloatTensor [P]
      - is_fs: bool
      - (opsiyonel) prot_fused: FloatTensor [D]  -- include_fused=True ise
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
        ancestor_stoplist: Optional[Set[int]] = None,
        ancestor_gamma: float = 0.7,
        store: ESMResidueStore,
        fused_store: Optional[ESMFusedStore] = None,
        include_fused: bool = False,
    ):
        super().__init__()
        if store is None:
            raise ValueError("ProteinEmbDataset requires 'store' (ESMResidueStore).")
        if include_fused and fused_store is None:
            raise ValueError("include_fused=True ama fused_store=None.")

        self.pids = list(protein_ids)
        self.n_go = go_cache.n_go
        self.fewzero = fewzero
        self.store = store
        self.fused_store = fused_store
        self.include_fused = bool(include_fused)

        self.pid2pos: Dict[str, List[int]] = pid2pos
        self.pos_weights_map: Dict[str, List[float]] = {}
        self.pos_is_generalized: Dict[str, List[bool]] = {}

        for pid in self.pids:
            orig = self.pid2pos.get(pid, [])
            # zero-shot sterilizasyon
            pos = [g for g in orig if g not in fewzero.zero_shot_terms]
            # az ise atalarla genişlet
            expanded, weights, is_gen_map = expand_with_ancestors(
                pos_terms=pos, parents=dag_parents,
                zs_blocklist=fewzero.zero_shot_terms,
                min_pos=min_pos_for_expand, max_add=max_ancestor_add,
                max_hops=max_hops, stoplist=ancestor_stoplist, gamma=ancestor_gamma,
                allowed_rels=set(ALLOWED_RELS_FOR_DAG)
            )
            self.pid2pos[pid] = expanded
            self.pos_weights_map[pid] = [float(w) for w in weights]
            self.pos_is_generalized[pid] = [bool(is_gen_map[int(t)]) for t in expanded]

        # Few-shot bayrak
        self.is_fs: List[bool] = []
        for pid in self.pids:
            labels = set(self.pid2pos.get(pid, []))
            self.is_fs.append(any((g in fewzero.few_shot_terms) for g in labels))

        # Global ZS maskesi (ileride miner filtreleri için)
        self.zs_mask = go_cache.mask_from_globals(fewzero.zero_shot_terms)

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int):
        pid = self.pids[idx]
        prot_emb = self.store.get(pid)  # [L,D]
        pos_list = self.pid2pos.get(pid, [])
        pos_wts = self.pos_weights_map.get(pid, [1.0] * len(pos_list))

        item = {
            "protein_id": pid,
            "prot_emb": prot_emb,  # [L,D]
            "pos_go_ids": torch.as_tensor(pos_list, dtype=torch.long),
            "pos_go_weights": torch.as_tensor(pos_wts, dtype=torch.float32),
            "is_fs": self.is_fs[idx],
        }
        if self.include_fused:
            z = self.fused_store.get(pid)  # [D]
            item["prot_fused"] = z
        return item

# protein_fused_query_dataset

class ProteinFusedQueryDataset(Dataset):
    """
    Retrieval/Indexing dataset: fused [D] döner.
    Dönen:
      - protein_id: str
      - prot_fused: FloatTensor [D]
    """
    def __init__(self, protein_ids: Sequence[str], fused_store: ESMFusedStore):
        if fused_store is None:
            raise ValueError("ProteinFusedQueryDataset requires fused_store.")
        self.pids = list(protein_ids)
        self.fused_store = fused_store

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int):
        pid = self.pids[idx]
        z = self.fused_store.get(pid)  # [D]
        return {"protein_id": pid, "prot_fused": z}