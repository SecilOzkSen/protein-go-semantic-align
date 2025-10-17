"""
ESM residue embedding loader (post-hoc stitching / overlap merge) + Drive-aware LRU cache.

- Shards: torch .pt files under embeddings/esm_embed_*.pt
  Each shard is a dict: { protein_id(str): np.ndarray[L_concat, D] or torch.Tensor[L_concat, D] }
- If seq_len_lookup is provided, overlapping windows are stitched back to [L_seq, D] by averaging.

New (Colab/Drive optimizations)
-------------------------------
1) LocalShardCache: when a shard path is on Google Drive (/content/drive/...),
   it is lazily copied once into /content/esm_cache (configurable) and reused.
   An LRU eviction keeps the cache under a size budget (e.g., 12 GB).

2) prefer_fp16=True: cast embeddings to float16 on load to cut I/O and memory ~50%.
   (Safe for downstream cosine/dot sims in most workflows.)

3) Backward-compatible API: get(protein_id) -> torch.FloatTensor [L_seq or L_concat, D]

Usage
-----
    store = ESMResidueStore(
        embed_dir,
        seq_len_lookup=seq_len_dict,
        max_len=1024,
        overlap=256,
        cache_shards=True,            # keep decoded shard dicts in RAM (optional)
        gdrive_cache=True,            # enable Drive->/content lazy caching
        cache_dir="/content/esm_cache",
        cache_gb=12.0,                # local cache budget
        prefer_fp16=True              # cast loaded arrays to fp16
    )
    H = store.get(protein_id)  # torch.FloatTensor [L_seq, D]
"""
from __future__ import annotations
import os
import time
import math
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import json, errno

import torch
from src.configs.paths import GOOGLE_DRIVE_MANIFEST_CACHE
from huggingface_hub import hf_hub_download


class DiskLRU:
    """
    Cap total size of a directory to max_gb by evicting least-recently-used files.
    Uses atime (last access). Works on /content/hf_cache (HF_HOME) or hub_local_dir.
    """
    def __init__(self, root: str, max_gb: float = 12.0, reserve_gb: float = 1.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * (1024**3))
        self.reserve_bytes = int(reserve_gb * (1024**3))  # emniyet payı

    def _scan(self):
        files = []
        total = 0
        for p in self.root.rglob("*"):
            if p.is_file():
                try:
                    st = p.stat()
                    files.append((p, st.st_size, st.st_atime))
                    total += st.st_size
                except FileNotFoundError:
                    pass
        return files, total

    def make_room(self):
        files, total = self._scan()
        limit = max(0, self.max_bytes - self.reserve_bytes)
        if total <= limit:
            return
        # LRU sırala (en eski atan en önce silinsin)
        files.sort(key=lambda x: x[2])  # by atime
        i = 0
        while total > limit and i < len(files):
            p, sz, _ = files[i]
            try:
                p.unlink()
                total -= sz
            except FileNotFoundError:
                pass
            i += 1


# ------------------------- Local LRU Shard Cache ------------------------- #
class LocalShardCache:
    """
    Minimal LRU file cache for Colab:
    On first access, copy a shard from Drive to a fast local dir (/content/esm_cache).
    Keep total size under `max_gb` by evicting least-recently-used files.
    """
    def __init__(self, cache_dir: str = "/content/esm_cache", max_gb: float = 12.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * (1024**3))
        # path(str) -> (size_bytes, last_access_epoch)
        self._meta: Dict[str, Tuple[int, float]] = {}
        self._rebuild_meta()

    def _rebuild_meta(self):
        self._meta.clear()
        for p in self.cache_dir.glob("*"):
            if p.is_file():
                try:
                    st = p.stat()
                    self._meta[str(p)] = (st.st_size, st.st_atime)
                except FileNotFoundError:
                    pass

    def _current_size(self) -> int:
        return sum(size for size, _ in self._meta.values())

    def _touch(self, p: Path):
        now = time.time()
        try:
            os.utime(p, (now, now))
        except Exception:
            pass
        if str(p) in self._meta:
            size, _ = self._meta[str(p)]
            self._meta[str(p)] = (size, now)

    def _evict_if_needed(self, incoming_bytes: int):
        # Evict oldest last-accessed files until there's room
        while self._current_size() + incoming_bytes > self.max_bytes and self._meta:
            victim = sorted(self._meta.items(), key=lambda kv: kv[1][1])[0][0]
            try:
                os.remove(victim)
            except FileNotFoundError:
                pass
            self._meta.pop(victim, None)

    def ensure_local(self, src_path: str) -> str:
        """
        Return a local path for src_path. If not cached, copy it in (rsync if available).
        """
        src = Path(src_path)
        dst = self.cache_dir / src.name
        if dst.exists():
            self._touch(dst)
            return str(dst)

        # make room
        try:
            size_bytes = src.stat().st_size
        except FileNotFoundError:
            size_bytes = 1 * (1024**3)  # fallback estimate
        self._evict_if_needed(size_bytes)

        # copy (prefer rsync for progress/stability)
        if shutil.which("rsync"):
            cmd = ["rsync", "-a", "--info=progress2", str(src), str(dst)]
        else:
            cmd = ["cp", str(src), str(dst)]
        subprocess.run(cmd, check=True)

        try:
            st = dst.stat()
            self._meta[str(dst)] = (st.st_size, time.time())
        except FileNotFoundError:
            pass

        return str(dst)


# --------------------------- ESM Residue Store --------------------------- #
class ESMResidueStore:
    def __init__(
        self,
        embed_dir: str,
        seq_len_lookup: Optional[Dict[str, int]] = None,  # protein_id -> sequence length
        max_len: Optional[int] = None,                    # window size used during extraction
        overlap: Optional[int] = None,                    # overlap used during extraction
        cache_shards: bool = True,
        pro_manifest_file: str = GOOGLE_DRIVE_MANIFEST_CACHE,
        # New (Drive & perf)
        gdrive_cache: bool = False,
        cache_dir: str = "/content/esm_cache",
        cache_gb: float = 12.0,
        prefer_fp16: bool = True,
        # ---- NEW: HF repo info ----
        hub_repo_id: Optional[str] = None,  # "secilozksen/esm-embeddings"
        hub_revision: str = "main",
        hub_repo_type: str = "dataset",  # "model"/"dataset"/"space"
        hub_local_dir: Optional[str] = None,  # cached local path
        hub_symlinks: bool = False,
        hub_cache_max_gb: float = 50.0,
        hub_cache_reserve_gb: float = 10.0,
    ):
        self.embed_dir = embed_dir
        self.seq_len_lookup = seq_len_lookup or {}
        self.max_len = max_len
        self.overlap = overlap
        self.cache_shards = bool(cache_shards)
        self.prefer_fp16 = bool(prefer_fp16)
        self.hub_repo_id = hub_repo_id
        self.hub_revision = hub_revision
        self.hub_repo_type = hub_repo_type
        self.hub_local_dir = hub_local_dir
        self.hub_symlinks = hub_symlinks

        # load manifest mapping protein_id -> shard path (absolute or relative)
        with open(pro_manifest_file, 'rb') as f:
            self.pid2shard: Dict[str, str] = pickle.load(f)

        # normalize to absolute paths if needed
        self.pid2shard = {str(k): self._abspath(v) for k, v in self.pid2shard.items()}

        # optional in-RAM shard dict cache (decoded via torch.load)
        self._shard_ram_cache: Dict[str, dict] = {}

        # local file LRU cache for Drive paths
        self._file_cache = LocalShardCache(cache_dir, cache_gb) if gdrive_cache else None

        cache_root = self.hub_local_dir or os.environ.get("HF_HOME", "/root/.cache/huggingface")
        self._hub_lru = DiskLRU(cache_root, max_gb=hub_cache_max_gb, reserve_gb=hub_cache_reserve_gb)

    # -------------------------- path utilities -------------------------- #
    def _abspath(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return str(Path(self.embed_dir) / p)

    def _maybe_hf_fetch(self, rel_path: str) -> Optional[str]:
        if not self.hub_repo_id:
            return None
        # İndirmeden önce yer aç (proaktif)
        self._hub_lru.make_room()
        try:
            lp = hf_hub_download(
                repo_id=self.hub_repo_id,
                repo_type=self.hub_repo_type,
                revision=self.hub_revision,
                filename=rel_path,
                local_dir=self.hub_local_dir,  # None ise HF_HOME kullanır
                local_dir_use_symlinks=self.hub_symlinks,
            )
            return lp
        except OSError as e:
            # Disk dolu yakalandıysa: agresif temizle ve tekrar dene
            self._hub_lru.make_room()
            lp = hf_hub_download(
                repo_id=self.hub_repo_id,
                repo_type=self.hub_repo_type,
                revision=self.hub_revision,
                filename=rel_path,
                local_dir=self.hub_local_dir,
                local_dir_use_symlinks=self.hub_symlinks,
            )
            return lp

    def _resolve_local_path(self, path_like: str) -> str:
        """
        Unified resolver:
          1) If Google Drive path and Drive cache enabled -> copy into local LRU cache, return that.
          2) Else, compute absolute path w.r.t. embed_dir.
             - If exists -> return abs path.
             - If not exists and we have HF repo info -> lazy download just this file into HF cache and return its local path.
          3) Otherwise -> return abs path (may not exist; caller will raise on load).

        Accepts either a relative path (e.g., 'embeddings/esm_embed_0123.pt')
        or an absolute path (e.g., '/content/drive/.../esm_embed_0123.pt').
        """
        from pathlib import Path
        import os

        raw = str(path_like)

        # ---- 1) Old behavior: Google Drive -> local LRU cache
        if self._file_cache is not None and raw.startswith("/content/drive/"):
            try:
                return self._file_cache.ensure_local(raw)
            except Exception:
                # best-effort; continue with other strategies
                pass

        # ---- 2) Compute absolute path relative to embed_dir (for relative inputs)
        abs_path = raw if os.path.isabs(raw) else str(Path(self.embed_dir) / raw)

        # If already present locally, we're done.
        if os.path.exists(abs_path):
            return abs_path

        # ---- 2b) Not present -> try lazy fetch from HF (relative key)
        # We need a REL path to ask the Hub for this file:
        #  - if the input was relative, use it as-is
        #  - if the input was absolute, try to relativize to embed_dir; fallback to basename
        try:
            rel = raw if not os.path.isabs(raw) else str(Path(abs_path).relative_to(self.embed_dir))
        except Exception:
            rel = Path(raw).name  # last resort

        hf_local = self._maybe_hf_fetch(rel)  # uses hf_hub_download under the hood
        if hf_local and os.path.exists(hf_local):
            return hf_local

        # ---- 3) Give back the absolute path (may still not exist; caller will handle)
        return abs_path

    # --------------------------- shard loading -------------------------- #
    def _load_shard(self, path: str) -> dict:
        """
        Sharp mode (minimal):
          - .npy -> np.memmap (dict sarmal: {'__format__':'npy', 'array': memmap})
          - .pt  -> torch.load (dict sarmal: {'__format__':'pt', ...})
        Drive/HF çözümlemesi, ENOSPC LRU temizliği, RAM cache ve atime korunur.
        Dönen nesne her zaman dict.
        """
        import errno
        import json
        import os
        import numpy as np
        import torch

        # 1) Yol çöz (Drive -> local kopya/HF indirme)
        local_path = self._resolve_local_path(path)

        # 2) RAM cache
        if self.cache_shards and local_path in self._shard_ram_cache:
            return self._shard_ram_cache[local_path]

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Shard not found: {local_path}")

        # 3) Yükleyiciler
        def _torch_load(p):
            try:
                return torch.load(p, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(p, map_location="cpu")

        def _npy_load(p):
            # NPY magic kontrolü (HTML/bozuk dosyayı erken yakala)
            with open(p, "rb") as f:
                if f.read(6) != b"\x93NUMPY":
                    raise RuntimeError(f"Not a valid NPY: {p}")
            # KOPYASIZ yükleme
            return np.load(p, mmap_mode="r")  # allow_pickle=False default

        # 4) Yükleme (disk doluysa LRU sonrası tek tekrar)
        try:
            if local_path.endswith(".npy"):
                arr = _npy_load(local_path)
                shard = {"__format__": "npy", "array": arr}
            else:
                payload = _torch_load(local_path)
                shard = {"__format__": "pt", **payload} if isinstance(payload, dict) else \
                    {"__format__": "pt", "payload": payload}
        except OSError as e:
            if getattr(e, "errno", None) in (errno.ENOSPC,):
                try:
                    if hasattr(self, "_hub_lru") and self._hub_lru is not None:
                        self._hub_lru.make_room()
                except Exception:
                    pass
                # tekrar dene
                if local_path.endswith(".npy"):
                    arr = _npy_load(local_path)
                    shard = {"__format__": "npy", "array": arr}
                else:
                    payload = _torch_load(local_path)
                    shard = {"__format__": "pt", **payload} if isinstance(payload, dict) else \
                        {"__format__": "pt", "payload": payload}
            else:
                raise
        except Exception as e:
            raise RuntimeError(f"Error loading shard {local_path}: {e}")

        # 5) (Opsiyonel) fp16 cast — SADECE .pt içindeki Tensörler için
        if getattr(self, "prefer_fp16", False) and shard.get("__format__") == "pt" and torch.cuda.is_available():
            for k, v in list(shard.items()):
                if k == "__format__":
                    continue
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    shard[k] = v.half()

        # 6) RAM cache’e koy (çözümlenmiş path anahtar)
        if self.cache_shards:
            self._shard_ram_cache[local_path] = shard
            if path != local_path:
                self._shard_ram_cache[path] = shard

        # 7) atime güncelle (DiskLRU)
        try:
            os.utime(local_path, None)
        except Exception:
            pass

        return shard

    # ---------------------- overlap stitching logic --------------------- #
    def _stitch_with_overlap(self, H: torch.Tensor, L_seq: int) -> torch.Tensor:
        """
        Reconstructs residue embeddings from overlapped segments.
        Overlapping positions are averaged to preserve alignment.

        H: [L_concat, D]  -> concatenated segment embeddings
        L_seq: true sequence length
        Returns: [L_seq, D]
        """
        if self.max_len is None or self.overlap is None:
            raise ValueError("max_len and overlap must be provided for stitching.")

        D = H.size(1)
        step = self.max_len - self.overlap
        if step <= 0:
            raise ValueError("Invalid parameters: max_len must be greater than overlap.")

        # Compute segment start indices (based on extraction logic)
        starts = []
        s = 0
        while s < L_seq:
            starts.append(s)
            if s + self.max_len >= L_seq:
                break
            s += step

        # Expected segment lengths (last one may be shorter)
        seg_lens = [min(self.max_len, L_seq - st) for st in starts]

        if sum(seg_lens) != H.size(0):
            raise ValueError(
                f"Concat length mismatch for stitching: "
                f"got {H.size(0)}, expected {sum(seg_lens)}. "
                "Check max_len/overlap/sequence length consistency."
            )

        # Merge overlaps by averaging
        out = H.new_zeros((L_seq, D))
        cnt = H.new_zeros((L_seq, 1))
        cursor = 0
        for st, sl in zip(starts, seg_lens):
            out[st:st+sl] += H[cursor:cursor+sl]
            cnt[st:st+sl] += 1
            cursor += sl
        out = out / cnt.clamp_min(1e-8)
        return out

    # ------------------------------ API ------------------------------- #
    def get(self, protein_id: str) -> torch.Tensor:
        """
        Load residue embeddings for a single protein.
        Returns: torch.FloatTensor or Float16Tensor of shape [L_seq or L_concat, D]
        """
        pid = str(protein_id)
        path = self.pid2shard.get(pid, None)
        if path is None:
            raise KeyError(f"Protein '{pid}' not found in manifest for {self.embed_dir}")

        shard = self._load_shard(path)

        # ---- mmap/satır tabanlı formatları destekle ----
        fmt = shard.get("__format__", None)

        # Yardımcı: pid -> row index haritası gerektiğinde kur
        def _ensure_pid2row():
            # Eğer zaten varsa, kullan
            if hasattr(self, "_pid2row") and isinstance(self._pid2row, dict):
                return
            # Şard'da pid->row veya row->pid ipuçları varsa onları kullan
            if "pid2row" in shard and isinstance(shard["pid2row"], dict):
                self._pid2row = shard["pid2row"]
            elif "row_index" in shard and isinstance(shard["row_index"], dict):
                # row_index: pid -> row
                self._pid2row = shard["row_index"]
            elif "row2pid" in shard and isinstance(shard["row2pid"], (list, tuple)):
                self._pid2row = {str(p): i for i, p in enumerate(shard["row2pid"])}
            else:
                # Son çare: manifest'in global indeksinden üretmeye çalış (varsa)
                # Yoksa eski "dict by pid" yoluna düşeceğiz.
                self._pid2row = None

        H = None
        if fmt == "concat":
            # pid index’ini al
            pid2row = shard.get("pid2row", None)
            if not pid2row or (protein_id not in pid2row and str(protein_id) not in pid2row):
                raise KeyError(f"Protein '{pid}' not indexed in concat shard {path}")
            i = int(pid2row.get(protein_id, pid2row.get(str(protein_id))))

            arrays = shard["arrays"]
            offsets = arrays["offsets"]  # np.ndarray int64
            embs_mm = arrays["embeddings"]  # np.memmap [N_total_res, D]
            start = int(offsets[i]);
            end = int(offsets[i + 1])

            # kopyasız slice -> torch
            view = embs_mm[start:end]  # np.memmap view
            H = torch.from_numpy(view)  # CPU tensor (no copy)

        elif fmt in ("npy", "npz", "pt"):
            _ensure_pid2row()
            if not self._pid2row:
                # pid->row yoksa eski yola dön (dict-by-pid)
                arr = shard.get(pid, shard.get(protein_id, None))
                if arr is None:
                    raise KeyError(f"Protein '{pid}' missing in shard {path}")
                H = torch.as_tensor(arr) if not isinstance(arr, torch.Tensor) else arr
            else:
                # satır tabanlı okuma
                if pid not in self._pid2row:
                    raise KeyError(f"Protein '{pid}' not indexed in shard {path}")
                row = int(self._pid2row[pid])

                if fmt == "npy":
                    # shard["array"] -> np.memmap [N, L(or L_concat), D] veya [N, ...]
                    np_arr = shard["array"]
                    view = np_arr[row]  # memmap slice, kopyasız
                    H = torch.from_numpy(view)  # CPU, zero-copy
                elif fmt == "npz":
                    # shard["arrays"]["embeddings"] -> np.memmap
                    np_arr = shard["arrays"]["embeddings"]
                    view = np_arr[row]
                    H = torch.from_numpy(view)
                else:  # fmt == "pt" satır tabanlı tensor
                    # shard["embs"] -> torch.Tensor [N, L_concat, D] veya [N, D]
                    H = shard["embs"][row]
        else:
            # --- Eski davranış: dict-by-pid ---
            arr = shard.get(pid, None)
            if arr is None:
                # some shards might have non-str keys; attempt fallback
                arr = shard.get(protein_id, None)
            if arr is None:
                raise KeyError(f"Protein '{pid}' missing in shard {path}")
            H = torch.as_tensor(arr) if not isinstance(arr, torch.Tensor) else arr

        # ---- dtype cast (eski davranış korunur) ----
        if self.prefer_fp16 and H.dtype == torch.float32:
            H = H.half()
        else:
            if H.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                H = H.float()

        # ---- overlap stitch (eski davranış korunur) ----
        L_concat = int(H.shape[0])
        L_seq = int(self.seq_len_lookup.get(pid, 0))
        if L_seq > 0 and L_concat != L_seq:
            H = self._stitch_with_overlap(H, L_seq)

        return H
