# src/data/go_cache.py
from __future__ import annotations

from typing import Sequence, Mapping, Optional, Union, Tuple
from collections.abc import Mapping as AbcMapping, Sequence as AbcSequence

import numpy as np
import torch
import torch.nn.functional as F


def _is_seq_not_str(x) -> bool:
    return isinstance(x, AbcSequence) and not isinstance(x, (str, bytes, bytearray))


def _build_row2id_from_id2row(id2row: Mapping[int, int]) -> list[int]:
    if not id2row:
        return []
    max_row = max(int(r) for r in id2row.values())
    row2id = [-1] * (max_row + 1)
    for gid, r in id2row.items():
        row2id[int(r)] = int(gid)
    # sanity: no holes
    if any(v < 0 for v in row2id):
        holes = [i for i, v in enumerate(row2id) if v < 0][:10]
        raise ValueError(f"id2row has holes, cannot build row2id. Example holes: {holes}")
    return row2id


def _validate_id_maps(row2id: Sequence[int], id2row: Mapping[int, int], *, n_checks: int = 25) -> None:
    # quick consistency check: row2id[row] == gid for sampled gid->row
    if not id2row:
        return
    L = len(row2id)
    items = list(id2row.items())
    step = max(1, len(items) // max(1, n_checks))
    for k in range(0, len(items), step):
        gid, r = items[k]
        r = int(r)
        if r < 0 or r >= L:
            raise ValueError(f"id2row out of bounds: gid={gid} row={r} but len(row2id)={L}")
        if int(row2id[r]) != int(gid):
            raise ValueError(
                "id2row and row2id inconsistent. "
                f"At row={r}: row2id[row]={int(row2id[r])} but expected gid={int(gid)}"
            )


class GoMemoryBank:
    """
    Fast GO embedding store on device.
    row2id: Sequence[int] mapping row -> global GO id
    id2row: Mapping[int,int] mapping global GO id -> row
    """

    def __init__(
        self,
        init_embs: Union[torch.Tensor, np.memmap],
        row2id: Sequence[int],
        id2row: Mapping[int, int],
        device: str = "cuda",
        to_device: bool = True,
        device_dtype: torch.dtype = torch.float32,
        pin_memory: bool = False,
        persist_back: bool = True,
    ):
        # ---- HARD GUARDS ----
        if isinstance(row2id, AbcMapping):
            raise TypeError("GoMemoryBank row2id must be a Sequence[int] (row->gid), got dict. Fix cache export.")
        if not _is_seq_not_str(row2id):
            raise TypeError(f"GoMemoryBank row2id must be a Sequence[int], got {type(row2id)}")

        if not isinstance(id2row, AbcMapping):
            raise TypeError(f"GoMemoryBank id2row must be Mapping[int,int], got {type(id2row)}")

        # Copy id2row into a clean int->int dict
        id2row = {int(g): int(r) for g, r in id2row.items()}

        # Validate mapping consistency early
        _validate_id_maps(row2id=row2id, id2row=id2row)

        self.device = torch.device(device)
        self.id2row = id2row
        self.row2id = torch.as_tensor(list(row2id), dtype=torch.long)

        # prepare tensor (CPU)
        if isinstance(init_embs, np.memmap):
            t = torch.from_numpy(np.asarray(init_embs))  # CPU
            self._cpu_mmap = init_embs
        else:
            t = torch.as_tensor(init_embs)
            self._cpu_mmap = None

        if to_device:
            if pin_memory and t.device.type == "cpu":
                t = t.pin_memory()
            t = t.to(self.device, non_blocking=True)
            if device_dtype is not None and t.device.type == "cuda":
                t = t.to(device_dtype)

        self._embs = t.contiguous()

        self.n_go = int(self._embs.size(0))
        if self.n_go != int(self.row2id.numel()):
            raise ValueError(f"emb rows ({self.n_go}) != len(row2id) ({int(self.row2id.numel())}).")

        self._persist_back = bool(persist_back)
        self._device_dtype = device_dtype

    @property
    def embs(self) -> torch.Tensor:
        return self._embs

    def index_select(self, rows: torch.Tensor) -> torch.Tensor:
        rows = rows.to(self._embs.device, non_blocking=True).long()
        return self._embs.index_select(0, rows)

    def to_local(self, go_ids: Sequence[int], *, drop_missing: bool = True) -> torch.LongTensor:
        if not go_ids:
            return torch.empty(0, dtype=torch.long, device=self._embs.device)
        idxs = [self.id2row.get(int(g), -1) for g in go_ids]
        idxs = torch.tensor(idxs, dtype=torch.long, device=self._embs.device)
        return idxs[idxs >= 0] if drop_missing else idxs

    def mask_from_globals(self, terms: Sequence[int]) -> torch.BoolTensor:
        m = torch.zeros(self.n_go, dtype=torch.bool, device=self._embs.device)
        if not terms:
            return m
        for g in terms:
            j = self.id2row.get(int(g), -1)
            if j >= 0:
                m[j] = True
        return m

    def __call__(self, go_ids: Sequence[int]) -> torch.Tensor:
        idxs = self.to_local(go_ids, drop_missing=False)
        if (idxs < 0).any():
            bad = [int(go_ids[i]) for i in (idxs < 0).nonzero(as_tuple=False).view(-1).tolist()[:5]]
            raise KeyError(f"GoMemoryBank missing ids, example: {bad}")
        return self.index_select(idxs)

    @torch.no_grad()
    def update(self, ids: Sequence[int], new_embs: torch.Tensor) -> None:
        if not ids:
            return

        d = int(self._embs.size(1))
        new_embs = torch.as_tensor(new_embs)
        assert new_embs.dim() == 2 and new_embs.size(1) == d, \
            f"new_embs shape {tuple(new_embs.shape)} d={d} ile uyuşmuyor"

        new_embs = torch.nan_to_num(new_embs, nan=0.0, posinf=0.0, neginf=0.0)
        new_embs = new_embs.to(self._embs.device, non_blocking=True)

        if new_embs.dtype != self._embs.dtype:
            new_embs = new_embs.to(self._embs.dtype)

        rows = self.to_local(ids, drop_missing=False)
        ok = rows >= 0
        if ok.any():
            self._embs.index_copy_(0, rows[ok], new_embs[ok])

            if self._persist_back and (self._cpu_mmap is not None):
                cpu_block = new_embs[ok].to(dtype=torch.float32, device="cpu").contiguous()
                np_block = cpu_block.numpy()
                for off, r in enumerate(rows[ok].tolist()):
                    self._cpu_mmap[r] = np_block[off]
                self._cpu_mmap.flush()

    @staticmethod
    def load_memmap(path: str) -> Tuple[np.memmap, Tuple[int, int]]:
        arr = np.load(path, mmap_mode="r+")
        return arr, tuple(arr.shape)


class GoLookupCache:
    def __init__(
        self,
        embs_or_blob: Union[torch.Tensor, Mapping, np.memmap],
        id2row: Optional[dict] = None,
        row2id: Optional[Sequence[int]] = None,
        device: str = "cpu",
    ):
        _id2row = id2row
        _row2id = row2id
        _embs_in = None
        self._mm = None
        self.device = device

        if isinstance(embs_or_blob, AbcMapping):
            b = embs_or_blob
            memmap_path = b.get("memmap_path")

            _id2row = b.get("id2row", _id2row)
            _row2id = b.get("row2id", b.get("ids", _row2id))

            if memmap_path is not None:
                meta = torch.load(str(memmap_path) + ".meta.pt", map_location="cpu", weights_only=False)
                shape = tuple(meta["shape"])
                np_dtype = np.dtype(meta["dtype"])
                self._mm = np.memmap(memmap_path, dtype=np_dtype, mode="r+", shape=shape)
                _embs_in = self._mm
            else:
                _embs_in = b["embs"]
        else:
            _embs_in = embs_or_blob

        # ---- Normalize maps ----
        if _id2row is not None:
            if not isinstance(_id2row, AbcMapping):
                raise TypeError(f"GoLookupCache id2row must be Mapping[int,int], got {type(_id2row)}")
            _id2row = {int(g): int(r) for g, r in _id2row.items()}

        if _row2id is not None and isinstance(_row2id, AbcMapping):
            raise TypeError("GoLookupCache row2id must be a Sequence[int] (row->gid), got dict. Fix cache export.")

        # If row2id missing, build it from id2row
        if _row2id is None:
            if _id2row is None:
                raise ValueError("GoLookupCache: need row2id or id2row.")
            _row2id = _build_row2id_from_id2row(_id2row)

        if not _is_seq_not_str(_row2id):
            raise TypeError(f"GoLookupCache row2id must be a Sequence[int], got {type(_row2id)}")

        # If id2row missing, build from row2id
        if _id2row is None:
            _id2row = {int(gid): int(r) for r, gid in enumerate(_row2id)}

        # Validate consistency before building bank
        _validate_id_maps(row2id=_row2id, id2row=_id2row)

        self._mb = GoMemoryBank(
            _embs_in,
            row2id=_row2id,
            id2row=_id2row,
            device=device,
            to_device=True,
        )

        self.embs = self._mb.embs
        self.id2row = self._mb.id2row
        self.row2id = self._mb.row2id
        self.n_go = self._mb.n_go

    def __call__(self, go_ids: Sequence[int]) -> torch.Tensor:
        return self._mb(go_ids)

    def to_local(self, go_ids: Sequence[int], *, drop_missing: bool = True) -> torch.LongTensor:
        return self._mb.to_local(go_ids, drop_missing=drop_missing)

    def mask_from_globals(self, terms: Sequence[int]) -> torch.BoolTensor:
        return self._mb.mask_from_globals(terms)

    def index_select(self, rows: torch.Tensor) -> torch.Tensor:
        return self._mb.index_select(rows)

    @torch.no_grad()
    def update(self, ids: Sequence[int], new_embs: torch.Tensor) -> None:
        self._mb.update(ids, new_embs)