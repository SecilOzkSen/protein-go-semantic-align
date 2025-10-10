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

import torch
from src.configs.paths import GOOGLE_DRIVE_MANIFEST_CACHE


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
        gdrive_cache: bool = True,
        cache_dir: str = "/content/esm_cache",
        cache_gb: float = 12.0,
        prefer_fp16: bool = False,
    ):
        self.embed_dir = embed_dir
        self.seq_len_lookup = seq_len_lookup or {}
        self.max_len = max_len
        self.overlap = overlap
        self.cache_shards = bool(cache_shards)
        self.prefer_fp16 = bool(prefer_fp16)

        # load manifest mapping protein_id -> shard path (absolute or relative)
        with open(pro_manifest_file, 'rb') as f:
            self.pid2shard: Dict[str, str] = pickle.load(f)

        # normalize to absolute paths if needed
        self.pid2shard = {str(k): self._abspath(v) for k, v in self.pid2shard.items()}

        # optional in-RAM shard dict cache (decoded via torch.load)
        self._shard_ram_cache: Dict[str, dict] = {}

        # local file LRU cache for Drive paths
        self._file_cache = LocalShardCache(cache_dir, cache_gb) if gdrive_cache else None

    # -------------------------- path utilities -------------------------- #
    def _abspath(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return str(Path(self.embed_dir) / p)

    def _resolve_local_path(self, shard_path: str) -> str:
        """
        If shard is on Drive and gdrive_cache is enabled, return local cached path.
        Otherwise return the original path.
        """
        if self._file_cache is not None and shard_path.startswith("/content/drive/"):
            try:
                return self._file_cache.ensure_local(shard_path)
            except Exception:
                # best effort fallback
                return shard_path
        return shard_path

    # --------------------------- shard loading -------------------------- #
    def _load_shard(self, path: str) -> dict:
        # RAM cache hit?
        if self.cache_shards and path in self._shard_ram_cache:
            return self._shard_ram_cache[path]

        # Drive-aware resolution
        local_path = self._resolve_local_path(path)

        try:
            shard = torch.load(local_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Error loading shard {local_path}: {e}")

        # Optional: convert ndarray tensors to desired dtype (fp16) on load
        if self.prefer_fp16:
            for k, v in list(shard.items()):
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float32:
                        shard[k] = v.half()
                # numpy arrays will be cast later at item access

        if self.cache_shards:
            self._shard_ram_cache[path] = shard
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

        arr: Any = shard.get(pid, None)
        if arr is None:
            # some shards might have non-str keys; attempt fallback
            arr = shard.get(protein_id, None)
        if arr is None:
            raise KeyError(f"Protein '{pid}' missing in shard {path}")

        # Convert to tensor (and dtype)
        if isinstance(arr, torch.Tensor):
            H = arr
        else:
            # numpy array path
            H = torch.as_tensor(arr)

        # dtype cast
        if self.prefer_fp16 and H.dtype == torch.float32:
            H = H.half()
        else:
            H = H.float() if H.dtype not in (torch.float32, torch.float16, torch.bfloat16) else H

        # Post-hoc stitching (if needed)
        L_concat = int(H.shape[0])
        L_seq = int(self.seq_len_lookup.get(pid, 0))
        if L_seq > 0 and L_concat != L_seq:
            H = self._stitch_with_overlap(H, L_seq)

        return H
