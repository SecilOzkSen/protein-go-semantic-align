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

        # === CANONICAL GO UNIVERSE ===
        # GoLookupCache row2id -> elimizde embedding/text olan global GO id'ler
        valid_go_ids: Set[int] = set(int(g) for g in go_cache.row2id)

        self.pid2pos: Dict[str, List[int]] = {}
        self.pos_weights_map: Dict[str, List[float]] = {}
        self.pos_is_generalized: Dict[str, List[bool]] = {}

        new_pids: List[str] = []
        dropped_prots = 0
        dropped_terms: Set[int] = set()

        for pid in self.pids:
            orig = pid2pos.get(pid, [])
            # ---- 1) Zero-shot sterilizasyon ----
            pos = [int(g) for g in orig if int(g) not in fewzero.zero_shot_terms]
            # ---- 2) DAG ile genişletme ----
            expanded, weights, is_gen_map = expand_with_ancestors(
                pos_terms=pos,
                parents=dag_parents,
                zs_blocklist=fewzero.zero_shot_terms,
                min_pos=min_pos_for_expand,
                max_add=max_ancestor_add,
                max_hops=max_hops,
                stoplist=ancestor_stoplist,
                gamma=ancestor_gamma,
                allowed_rels=set(ALLOWED_RELS_FOR_DAG),
            )
            # ---- 3) CANONICAL FILTRE: sadece GoTextStore / go_cache'in bildiği id'ler kalsın ----
            expanded_filtered: List[int] = []
            weights_filtered: List[float] = []
            is_gen_filtered: List[bool] = []

            for t, w in zip(expanded, weights):
                t_int = int(t)
                if t_int in valid_go_ids:
                    expanded_filtered.append(t_int)
                    weights_filtered.append(float(w))
                    is_gen_filtered.append(bool(is_gen_map[int(t)]))
                else:
                    dropped_terms.add(t_int)

            # Hiç label kalmadıysa bu proteini dataset'ten at
            if not expanded_filtered:
                dropped_prots += 1
                continue

            new_pids.append(pid)
            self.pid2pos[pid] = expanded_filtered
            self.pos_weights_map[pid] = weights_filtered
            self.pos_is_generalized[pid] = is_gen_filtered

        # Sadece en az bir canonical label'ı kalan proteinleri tut
        self.pids = new_pids
        print(
            f"[ProteinEmbDataset] canonical filter after DAG: "
            f"kept {len(self.pids)} proteins, "
            f"dropped_proteins={dropped_prots}, "
            f"dropped_terms={len(dropped_terms)}"
        )
        if dropped_terms:
            sample = sorted(dropped_terms)[:10]
            print(f"[ProteinEmbDataset] example dropped ancestor terms: {sample}")

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