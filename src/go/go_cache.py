# src/data/go_cache.py
from __future__ import annotations
from typing import Sequence, Mapping, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F

class GoMemoryBank:
    """
    Hızlı GO embedding deposu. (normalize, device'ta)
    """
    def __init__(
        self,
        init_embs: Union[torch.Tensor, np.memmap],
        row2id: Sequence[int],
        device: str = "cuda",
        to_device: bool = True,
        already_normalized: bool = False,
        device_dtype: torch.dtype = torch.float16,   # <= SHARP: GPU'da fp16 varsayılan
        pin_memory: bool = False,                    # <= SHARP: dataloader uyumu
        persist_back: bool = True                   # <= SHARP: update() memmap'e yazsın mı?
    ):
        self.device = torch.device(device)
        self.id2row = {int(i): int(r) for r, i in enumerate(row2id)}
        self.row2id = torch.as_tensor(list(row2id), dtype=torch.long)

        # tensörü hazırla (CPU)
        if isinstance(init_embs, np.memmap):
            t = torch.from_numpy(np.asarray(init_embs))   # CPU
            self._cpu_mmap = init_embs                   # np.memmap handle
        else:
            t = torch.as_tensor(init_embs)               # CPU/GPU olabilir
            self._cpu_mmap = None

        t = t.float()                                    # normalize için güvenli
        if not already_normalized:
            t = F.normalize(t, p=2, dim=1)

        # SHARP: isteğe bağlı pin + half
        if to_device:
            if pin_memory and t.device.type == "cpu":
                t = t.pin_memory()
            t = t.to(self.device, non_blocking=True)
            if device_dtype is not None:
                # sadece GPU'da half'a çevir (CPU memmap'le karışmasın)
                if t.device.type == "cuda":
                    t = t.to(device_dtype)
        self._embs = t.contiguous()

        self.n_go = int(self._embs.size(0))
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
        idxs = self.to_local(go_ids, drop_missing=True)
        if idxs.numel() == 0:
            d = int(self._embs.size(1))
            return torch.empty(0, d, dtype=self._embs.dtype, device=self._embs.device)
        return self.index_select(idxs)

    @torch.no_grad()
    def update(self, ids: Sequence[int], new_embs: torch.Tensor) -> None:
        if not ids:
            return
        # boyut/d eşleşmesi
        d = int(self._embs.size(1))
        new_embs = torch.as_tensor(new_embs)
        assert new_embs.dim() == 2 and new_embs.size(1) == d, \
            f"new_embs shape {tuple(new_embs.shape)} d={d} ile uyuşmuyor"

        # normalize + cihaz
        new_embs = F.normalize(new_embs.float(), p=2, dim=1).to(self._embs.device, non_blocking=True)
        if self._device_dtype is not None and self._embs.device.type == "cuda":
            new_embs = new_embs.to(self._device_dtype)

        rows = self.to_local(ids, drop_missing=False)
        ok = rows >= 0
        if ok.any():
            self._embs.index_copy_(0, rows[ok], new_embs[ok])

            # SHARP: memmap'e kalıcı yaz (isteğe bağlı)
            if self._persist_back and (self._cpu_mmap is not None):
                # sadece güncellenen satırları CPU'ya çekip yaz
                cpu_block = new_embs[ok].to(dtype=torch.float32, device="cpu").contiguous()
                # float16 memmap varsa, cast etmeyi unutma:
                np_block = cpu_block.numpy()
                for off, r in enumerate(rows[ok].tolist()):
                    self._cpu_mmap[r] = np_block[off]
                self._cpu_mmap.flush()

    @staticmethod
    def load_memmap(path: str) -> Tuple[np.memmap, Tuple[int, int]]:
        arr = np.load(path, mmap_mode="r+")  # r+ => update edilebilir
        return arr, tuple(arr.shape)

class GoLookupCache:
    def __init__(self,
                 embs_or_blob: Union[torch.Tensor, Mapping, np.memmap],
                 id2row: Optional[dict] = None,
                 row2id: Optional[Sequence[int]] = None,
                 device: str = "cpu",
                 already_normalized: bool = False):

        # Güvenli başlangıç
        _id2row = id2row
        _row2id = row2id
        _embs_in = None  # GoMemoryBank'e geçeceğimiz tek kaynak
        self._mm = None  # np.memmap handle (varsa)

        if isinstance(embs_or_blob, Mapping):
            b = embs_or_blob
            memmap_path = b.get("memmap_path")

            # id eşlemeleri (blob > param > None)
            _id2row = b.get("id2row", _id2row)
            _row2id = b.get("row2id", b.get("ids", _row2id))

            if memmap_path is not None:
                # Memmap'i meta'dan yükle
                meta = torch.load(str(memmap_path) + ".meta.pt", map_location="cpu", weights_only=False)
                shape = tuple(meta["shape"])
                np_dtype = np.dtype(meta["dtype"])
                # r+ ile açarsak ileride update() kalıcı yazabilir
                self._mm = np.memmap(memmap_path, dtype=np_dtype, mode="r+", shape=shape)
                _embs_in = self._mm  # GoMemoryBank np.memmap alabilir
            else:
                # Doğrudan tensör/dizi bekliyoruz
                _embs_in = b["embs"]
        else:
            # Eski imza: doğrudan tensör/np.memmap
            _embs_in = embs_or_blob
            # _id2row / _row2id parametrelerden gelecek (varsa)

        # Zorunlu: row2id olmalı
        assert _row2id is not None, "GoLookupCache: row2id gerekli."

        # id2row yoksa row2id'den türet
        if _id2row is None:
            _id2row = {int(i): int(r) for r, i in enumerate(_row2id)}

        # Backend – normalize & device yönetimi GoMemoryBank'te
        self._mb = GoMemoryBank(
            _embs_in,
            row2id=_row2id,
            device=device,
            to_device=True,
            already_normalized=already_normalized,
        )

        # Dış API alanları
        self.embs = self._mb.embs
        self.id2row = self._mb.id2row
        self.row2id = self._mb.row2id
        self.n_go = self._mb.n_go

    # Eski metodlar: doğrudan MemoryBank'e delege
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


