# ====================== UNIVERSAL RESIDUE / FUSED STORES ======================
# Residue:  [L, D]  (seq-length embeddings)
# Fused:    [D]     (protein-level pooled embedding)
#
# Yeni formatlar:
#   Residue: res_esm1b_*.index.tsv  + res_esm1b_00000.data.npy (+ .meta.json: {"used_rows": N})
#   Fused:   fused_esm1b_*.index.tsv+ fused_esm1b_00000.data.npy
#
# Legacy formatlar (ikisi için de destek):
#   - dict-by-pid .pt  (pid -> Tensor)
#   - row-based .npy/.npz/.pt (pid2row / row2pid / row_index)
#   - concat memmap + offsets (arrays: "embeddings", "offsets", "pid2row")
#
# API:
#   ESMResidueStore.get(pid) -> torch.FloatTensor [L, D]
#   ESMFusedStore.get(pid)   -> torch.FloatTensor [D]
#
# Not: Fused ve Residue manifestleri AYRI tutulur.
# ============================================================================

from __future__ import annotations
import os, time, pickle, shutil, subprocess, json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import torch
import re

# -------- env/path config (senin projenden) --------
try:
    from src.configs.paths import GOOGLE_DRIVE_MANIFEST_CACHE
except Exception:
    GOOGLE_DRIVE_MANIFEST_CACHE = None

# ---helpers

def _parse_shard_id_from_index(p):
    """
    p: Path('.../res_esm1b_00000.index.tsv') veya '.../res_esm1b_00000.index'
    -> 0 tabanlı veya 00000 gibi bir sayı döndürür (int)
    """
    m = re.search(r"res_esm1b_(\d+)\.index(?:\.tsv)?$", p.name)
    if not m:
        raise ValueError(f"Cannot parse shard id from index filename: {p.name}")
    return int(m.group(1))

# ---------------------------- DiskLRU ----------------------------
class DiskLRU:
    def __init__(self, root: str, max_gb: float = 12.0, reserve_gb: float = 1.0):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * (1024**3))
        self.reserve_bytes = int(reserve_gb * (1024**3))

    def _scan(self):
        files, total = [], 0
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
        if total <= limit: return
        files.sort(key=lambda x: x[2])  # by atime
        i = 0
        while total > limit and i < len(files):
            p, sz, _ = files[i]
            try: p.unlink(); total -= sz
            except FileNotFoundError: pass
            i += 1

# ------------------------- LocalShardCache ------------------------
class LocalShardCache:
    def __init__(self, cache_dir: str = "/content/esm_cache", max_gb: float = 12.0):
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * (1024**3))
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
        try: os.utime(p, (now, now))
        except Exception: pass
        if str(p) in self._meta:
            size, _ = self._meta[str(p)]
            self._meta[str(p)] = (size, now)

    def _evict_if_needed(self, incoming_bytes: int):
        while self._current_size() + incoming_bytes > self.max_bytes and self._meta:
            victim = sorted(self._meta.items(), key=lambda kv: kv[1][1])[0][0]
            try: os.remove(victim)
            except FileNotFoundError: pass
            self._meta.pop(victim, None)

    def ensure_local(self, src_path: str) -> str:
        src = Path(src_path); dst = self.cache_dir / src.name
        if dst.exists(): self._touch(dst); return str(dst)
        try: size_bytes = src.stat().st_size
        except FileNotFoundError: size_bytes = 1 * (1024**3)
        self._evict_if_needed(size_bytes)
        cmd = ["rsync", "-a", "--info=progress2", str(src), str(dst)] if shutil.which("rsync") else ["cp", str(src), str(dst)]
        subprocess.run(cmd, check=True)
        try:
            st = dst.stat()
            self._meta[str(dst)] = (st.st_size, time.time())
        except FileNotFoundError:
            pass
        return str(dst)

# ------------------------- Common utilities ------------------------
def _npy_load(path: str):
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise RuntimeError(f"Not a valid NPY: {path}")
    return np.load(path, mmap_mode="r")

def _torch_load(path: str):
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")

# ============================ Base Store ============================
class _BaseStore:
    def __init__(
        self,
        embed_dir: str,
        *,
        pro_manifest_file: Optional[str] = GOOGLE_DRIVE_MANIFEST_CACHE,
        gdrive_cache: bool = False,
        cache_dir: str = "/content/esm_cache",
        cache_gb: float = 12.0,
        prefer_fp16: bool = False,
        hub_repo_id: Optional[str] = None,
        hub_revision: str = "main",
        hub_repo_type: str = "dataset",
        hub_local_dir: Optional[str] = None,
        hub_symlinks: bool = False,
        hub_cache_max_gb: float = 50.0,
        hub_cache_reserve_gb: float = 10.0,
    ):
        self.embed_dir = str(embed_dir)
        self.prefer_fp16 = bool(prefer_fp16)
        self._file_cache = LocalShardCache(cache_dir, cache_gb) if gdrive_cache else None

        cache_root = hub_local_dir or os.environ.get("HF_HOME", "/root/.cache/huggingface")
        self._hub_lru = DiskLRU(cache_root, max_gb=hub_cache_max_gb, reserve_gb=hub_cache_reserve_gb)

        self.hub_repo_id = hub_repo_id
        self.hub_revision = hub_revision
        self.hub_repo_type = hub_repo_type
        self.hub_local_dir = hub_local_dir
        self.hub_symlinks = hub_symlinks

        self._backend = None
        self._pid2shard: Dict[str, str] = {}
        self._shard_ram_cache: Dict[str, dict] = {}
        self._pid2row_cache: Dict[str, Dict[str,int]] = {}
        self._pro_manifest_file = pro_manifest_file

    # --- paths & HF ---
    def _abspath(self, p: str) -> str:
        if os.path.isabs(p): return p
        return str(Path(self.embed_dir) / p)

    def _maybe_hf_fetch(self, rel_path: str) -> Optional[str]:
        if not self.hub_repo_id: return None
        self._hub_lru.make_room()
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(
                repo_id=self.hub_repo_id, repo_type=self.hub_repo_type, revision=self.hub_revision,
                filename=rel_path, local_dir=self.hub_local_dir, local_dir_use_symlinks=self.hub_symlinks)
        except OSError:
            self._hub_lru.make_room()
            return hf_hub_download(
                repo_id=self.hub_repo_id, repo_type=self.hub_repo_type, revision=self.hub_revision,
                filename=rel_path, local_dir=self.hub_local_dir, local_dir_use_symlinks=self.hub_symlinks)

    def _resolve_local_path(self, path_like: str) -> str:
        raw = str(path_like)
        if self._file_cache is not None and raw.startswith("/content/drive/"):
            try: return self._file_cache.ensure_local(raw)
            except Exception: pass
        abs_path = raw if os.path.isabs(raw) else str(Path(self.embed_dir) / raw)
        if os.path.exists(abs_path): return abs_path
        try: rel = raw if not os.path.isabs(raw) else str(Path(abs_path).relative_to(self.embed_dir))
        except Exception: rel = Path(raw).name
        hf_local = self._maybe_hf_fetch(rel)
        if hf_local and os.path.exists(hf_local): return hf_local
        return abs_path

    # --- legacy shard load ---
    def _load_shard_legacy(self, path: str) -> dict:
        local_path = self._resolve_local_path(path)
        if local_path in self._shard_ram_cache: return self._shard_ram_cache[local_path]
        if not os.path.exists(local_path): raise FileNotFoundError(f"Shard not found: {local_path}")

        if local_path.endswith(".npy"):
            arr = _npy_load(local_path)
            shard = {"__format__": "npy", "array": arr}
        elif local_path.endswith(".npz"):
            payload = np.load(local_path, allow_pickle=False)
            shard = {"__format__": "npz", "arrays": dict(payload)}
        else:
            payload = _torch_load(local_path)
            shard = {"__format__": "pt", **payload} if isinstance(payload, dict) else {"__format__": "pt", "payload": payload}

        # optional fp16 cast (sadece .pt tensörleri)
        if self.prefer_fp16 and shard.get("__format__") == "pt":
            for k, v in list(shard.items()):
                if k == "__format__": continue
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    shard[k] = v.half()

        self._shard_ram_cache[local_path] = shard
        if path != local_path: self._shard_ram_cache[path] = shard
        try: os.utime(local_path, None)
        except Exception: pass
        return shard

    def _ensure_pid2row_for(self, shard: dict, path_key: str):
        if path_key in self._pid2row_cache: return self._pid2row_cache[path_key]
        pid2row = None
        if "pid2row" in shard and isinstance(shard["pid2row"], dict): pid2row = shard["pid2row"]
        elif "row_index" in shard and isinstance(shard["row_index"], dict): pid2row = shard["row_index"]
        elif "row2pid" in shard and isinstance(shard["row2pid"], (list, tuple)):
            pid2row = {str(p): i for i, p in enumerate(shard["row2pid"])}
        self._pid2row_cache[path_key] = pid2row
        return pid2row

# ============================ Residue Store ============================
class ESMResidueStore(_BaseStore):
    """
    Backward-compatible residue loader + new residue-concat index desteği.
    get(pid) -> torch.FloatTensor [L, D]
    """
    def __init__(
        self,
        embed_dir: str,
        seq_len_lookup: Optional[Dict[str, int]] = None,
        max_len: Optional[int] = None,
        overlap: Optional[int] = None,
        **kwargs
    ):
        super().__init__(embed_dir, **kwargs)
        self.seq_len_lookup = seq_len_lookup or {}
        self.max_len = max_len; self.overlap = overlap

        self._pid2span: Dict[str, Tuple[int,int,int]] = {}
        self._res_shards: List[np.memmap] = []
        self._res_used: List[int] = []
        self._detect_and_build()

    def _detect_and_build(self):
        root = Path(self.embed_dir)
        print(self.embed_dir)

        # --- NEW FORMAT: res_esm1b_*.index.tsv + .data.npy
        res_indices = sorted(root.glob("res_esm1b_*.index.tsv"))
        if not res_indices:
            res_indices = sorted(root.glob("res_esm1b_*.index"))  # bazı ortamlarda .tsv yok

        res_data = {}
        for p in res_indices:
            shard_id = _parse_shard_id_from_index(p)  # <-- DÜZELTİLEN KISIM
            data_path = p.with_name(f"res_esm1b_{shard_id:05d}.data.npy")
            if not data_path.exists():
                # alt klasörlerde olabilir → kökten ara (opsiyonel güvenlik)
                cand = list(root.rglob(f"res_esm1b_{shard_id:05d}.data.npy"))
                if cand:
                    data_path = cand[0]
            res_data[shard_id] = data_path

        if res_indices and all(p.exists() for p in res_data.values()):
            self._backend = "NEW_RESIDX"
            for idx_path in res_indices:
                sid = _parse_shard_id_from_index(idx_path)  # <-- DÜZELTİLEN KISIM
                data_path = res_data[sid]
                arr = np.load(data_path, mmap_mode="r")
                used = arr.shape[0]
                meta_path = idx_path.with_name(f"res_esm1b_{sid:05d}.meta.json")
                if meta_path.exists():
                    try:
                        m = json.load(open(meta_path, "r"))
                        used = int(m.get("used_rows", used))
                    except Exception:
                        pass
                self._res_shards.append(arr)
                self._res_used.append(used)
                with open(idx_path, "r") as fh:
                    for ln in fh:
                        pid, s, e = ln.strip().split("\t")
                        s, e = int(s), int(e)
                        assert 0 <= s < e <= used, f"Index out of bounds in {idx_path}"
                        self._pid2span[pid] = (len(self._res_shards) - 1, s, e)
            return

        # --- LEGACY MANIFEST (residue tarafı)
        if self._pro_manifest_file and Path(self._pro_manifest_file).exists():
            with open(self._pro_manifest_file, "rb") as f:
                pid2shard: Dict[str, str] = pickle.load(f)
            self._pid2shard = {str(k): self._abspath(v) for k, v in pid2shard.items()}
            self._backend = "LEGACY_MANIFEST"; return

        # --- Fallback: tarama
        fallback = list(root.glob("*.pt")) + list(root.glob("*.npy")) + list(root.glob("*.npz"))
        if fallback:
            self._backend = "LEGACY_MANIFEST"
            for p in fallback: self._pid2shard[str(p)] = str(p)
            return

        raise RuntimeError(f"No supported RESIDUE layout under {self.embed_dir}")

    # --------------------------- Stitching ------------------------------
    def _stitch_with_overlap(self, H: torch.Tensor, L_seq: int) -> torch.Tensor:
        if self.max_len is None or self.overlap is None:
            raise ValueError("max_len and overlap must be provided for stitching.")
        D = H.size(1)
        step = self.max_len - self.overlap
        if step <= 0:
            raise ValueError("Invalid parameters: max_len must be greater than overlap.")
        # segment uzunluklarını L_seq ile eşle
        starts = []
        s = 0
        while s < L_seq:
            starts.append(s)
            if s + self.max_len >= L_seq: break
            s += step
        seg_lens = [min(self.max_len, L_seq - st) for st in starts]
        if sum(seg_lens) != H.size(0):
            raise ValueError(
                f"Concat length mismatch for stitching: got {H.size(0)} vs expected {sum(seg_lens)} "
                f"(L_seq={L_seq}, max_len={self.max_len}, overlap={self.overlap})"
            )
        out = H.new_zeros((L_seq, D))
        cnt = H.new_zeros((L_seq, 1))
        cursor = 0
        for st, sl in zip(starts, seg_lens):
            out[st:st+sl] += H[cursor:cursor+sl]
            cnt[st:st+sl] += 1
            cursor += sl
        out = out / cnt.clamp_min(1e-8)
        return out

    # -------------------------------- API -------------------------------
    @torch.inference_mode()
    def get(self, protein_id: str) -> torch.Tensor:
        pid = str(protein_id)

        # New format: memmap slice -> [L,D]
        if self._backend == "NEW_RESIDX":
            if pid not in self._pid2span:
                raise KeyError(f"Protein '{pid}' not found in residue index under {self.embed_dir}")
            sidx, s, e = self._pid2span[pid]
            used = self._res_used[sidx]
            assert 0 <= s < e <= used
            view = self._res_shards[sidx][s:e]                 # np.memmap
            H = torch.from_numpy(view.copy()).to(torch.float32)       # [L,D] float32 view
            return H

        # Legacy yollar:
        if pid not in self._pid2shard:
            raise KeyError(f"Protein '{pid}' not in residue manifest for {self.embed_dir}")

        path = self._pid2shard[pid]
        shard = self._load_shard_legacy(path)
        fmt = shard.get("__format__", None)

        if fmt == "concat":
            pid2row = shard.get("pid2row", None)
            if not pid2row: raise KeyError(f"concat shard missing pid2row: {path}")
            i = int(pid2row.get(pid, pid2row.get(str(pid), -1)))
            if i < 0: raise KeyError(f"Protein '{pid}' not indexed in concat shard {path}")
            arrays = shard["arrays"]
            offsets = arrays["offsets"]      # [N+1]
            embs_mm = arrays["embeddings"]   # [N_total_res, D]
            start = int(offsets[i]); end = int(offsets[i+1])
            view = embs_mm[start:end]
            H = torch.from_numpy(view)
            if H.dtype != torch.float32: H = H.float()

            L_seq = int(self.seq_len_lookup.get(pid, 0))
            if L_seq > 0 and H.shape[0] != L_seq:
                H = self._stitch_with_overlap(H, L_seq)
            return H

        # row-based veya dict-by-pid
        path_key = path
        pid2row = self._ensure_pid2row_for(shard, path_key)
        if pid2row is not None and pid in pid2row:
            row = int(pid2row[pid])
            if fmt == "npy":
                H = torch.from_numpy(shard["array"][row])
            elif fmt == "npz":
                arr = shard["arrays"].get("embeddings", None)
                if arr is None: raise KeyError(f"npz shard missing 'embeddings': {path}")
                H = torch.from_numpy(arr[row])
            else:  # pt
                embs = shard.get("embs", None)
                if embs is None: raise KeyError(f"pt shard missing 'embs': {path}")
                H = embs[row] if isinstance(embs, torch.Tensor) else torch.as_tensor(embs[row])
        else:
            arr = shard.get(pid, shard.get(str(pid), None))
            if arr is None: raise KeyError(f"Protein '{pid}' missing in shard {path}")
            H = torch.as_tensor(arr) if not isinstance(arr, torch.Tensor) else arr

        if H.ndim != 2:
            raise TypeError(f"Residue store expects [L,D], got shape {tuple(H.shape)} for '{pid}'")
        if H.dtype not in (torch.float32, torch.bfloat16): H = H.float()

        L_seq = int(self.seq_len_lookup.get(pid, 0))
        if L_seq > 0 and H.shape[0] != L_seq:
            H = self._stitch_with_overlap(H, L_seq)
        return H

# ============================== Fused Store ===============================
class ESMFusedStore(_BaseStore):
    """
    Fused (protein-level) embedding loader.
    Yeni format (desteklenen):
      fused_esm1b_*.ids.txt  + fused_esm1b_00000.npy  (+ fused_esm1b_00000.meta.json: {"used_rows": N})
    Ayrıca legacy manifest / npz / pt da çalışır.
    get(pid) -> torch.FloatTensor [D]
    """
    def __init__(self, embed_dir: str, **kwargs):
        super().__init__(embed_dir, **kwargs)
        self._pid2row: Dict[str, Tuple[int,int]] = {}   # pid -> (shard_idx, row)
        self._shards: List[np.memmap] = []              # each [N, D]
        self._used: List[int] = []
        self._detect_and_build()

    def _detect_and_build(self):
        root = Path(self.embed_dir)

        # --- NEW FORMAT: ids.txt + .npy (+ .meta.json)
        ids_txts = sorted(root.glob("fused_esm1b_*.ids.txt"))
        data_npys = {int(p.stem.split("_")[-1].split(".")[0]): root / f"fused_esm1b_{int(p.stem.split('_')[-1].split('.')[0]):05d}.npy"
                     for p in ids_txts}
        if ids_txts and all(p.exists() for p in data_npys.values()):
            self._backend = "NEW_FUSED_IDS_NPY"
            for ids_path in ids_txts:
                sid = int(ids_path.stem.split("_")[-1].split(".")[0])
                data_path = data_npys[sid]
                # npy shape: [N, D] (fp16/fp32)
                arr = np.load(data_path, mmap_mode="r")
                used = arr.shape[0]
                meta_path = root / f"fused_esm1b_{sid:05d}.meta.json"
                if meta_path.exists():
                    try:
                        m = json.load(open(meta_path, "r"))
                        used = int(m.get("used_rows", used))
                    except Exception:
                        pass
                # oku ids.txt -> satır sırası
                with open(ids_path, "r") as fh:
                    ids = [ln.strip() for ln in fh if ln.strip()]
                assert used <= len(ids) <= arr.shape[0], f"ID/used_rows uyumsuz: {ids_path}"
                shard_idx = len(self._shards)
                self._shards.append(arr); self._used.append(used)
                for r, pid in enumerate(ids[:used]):
                    self._pid2row[pid] = (shard_idx, r)
            return

        # --- LEGACY MANIFEST (fused)
        if self._pro_manifest_file and Path(self._pro_manifest_file).exists():
            with open(self._pro_manifest_file, "rb") as f:
                pid2shard: Dict[str, str] = pickle.load(f)
            self._pid2shard = {str(k): self._abspath(v) for k, v in pid2shard.items()}
            self._backend = "LEGACY_MANIFEST"; return

        # --- Fallback tarama
        fallback = list(root.glob("*.pt")) + list(root.glob("*.npy")) + list(root.glob("*.npz"))
        if fallback:
            self._backend = "LEGACY_MANIFEST"
            for p in fallback: self._pid2shard[str(p)] = str(p)
            return

        raise RuntimeError(f"No supported FUSED layout under {self.embed_dir}")

    @torch.inference_mode()
    def get(self, protein_id: str) -> torch.Tensor:
        pid = str(protein_id)

        if self._backend == "NEW_FUSED_IDS_NPY":
            if pid not in self._pid2row:
                raise KeyError(f"Protein '{pid}' not found in fused ids.txt index under {self.embed_dir}")
            sidx, r = self._pid2row[pid]
            used = self._used[sidx]; assert 0 <= r < used
            view = self._shards[sidx][r]                   # np.ndarray [D]
            v = torch.from_numpy(view.copy()).to(torch.float32)
            if v.ndim != 1:
                raise TypeError(f"Fused embedding must be [D], got {tuple(v.shape)} for '{pid}'")
            return v

        # --- LEGACY PATHS (row-based / dict-by-pid / concat) ---
        if pid not in self._pid2shard:
            raise KeyError(f"Protein '{pid}' not in fused manifest for {self.embed_dir}")

        path = self._pid2shard[pid]
        shard = self._load_shard_legacy(path)
        fmt = shard.get("__format__", None)

        if fmt == "concat":
            pid2row = shard.get("pid2row", None)
            if not pid2row: raise KeyError(f"concat(fused) missing pid2row: {path}")
            i = int(pid2row.get(pid, pid2row.get(str(pid), -1)))
            if i < 0: raise KeyError(f"Protein '{pid}' not indexed in concat shard {path}")
            arrays = shard["arrays"]; embs_mm = arrays["embeddings"]  # [N,D]
            v = torch.from_numpy(embs_mm[i])
        else:
            path_key = path
            pid2row = self._ensure_pid2row_for(shard, path_key)
            if pid2row is not None and pid in pid2row:
                row = int(pid2row[pid])
                if fmt == "npy":
                    v = torch.from_numpy(shard["array"][row])
                elif fmt == "npz":
                    arr = shard["arrays"].get("embeddings", None)
                    if arr is None: raise KeyError(f"npz(fused) missing 'embeddings': {path}")
                    v = torch.from_numpy(arr[row])
                else:
                    embs = shard.get("embs", None)
                    if embs is None: raise KeyError(f"pt(fused) missing 'embs': {path}")
                    v = embs[row] if isinstance(embs, torch.Tensor) else torch.as_tensor(embs[row])
            else:
                arr = shard.get(pid, shard.get(str(pid), None))
                if arr is None: raise KeyError(f"Protein '{pid}' missing in shard {path}")
                v = torch.as_tensor(arr) if not isinstance(arr, torch.Tensor) else arr

        if v.ndim != 1:
            raise TypeError(f"Fused store expects [D], got shape {tuple(v.shape)} for '{pid}'")
        if v.dtype not in (torch.float32, torch.bfloat16): v = v.float()
        return v
# ============================== Factory ===============================
def build_esm_stores(
    *,
    residue_dir: str,
    fused_dir: Optional[str] = None,
    residue_manifest: Optional[str] = None,
    fused_manifest: Optional[str] = None,
    seq_len_lookup: Optional[Dict[str,int]] = None,
    max_len: Optional[int] = None,
    overlap: Optional[int] = None,
    gdrive_cache: bool = False,
    cache_dir: str = "/content/esm_cache",
    cache_gb: float = 12.0,
    prefer_fp16: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_revision: str = "main",
    hub_repo_type: str = "dataset",
    hub_local_dir: Optional[str] = None,
    hub_symlinks: bool = False,
) -> Tuple[ESMResidueStore, Optional[ESMFusedStore]]:
    """Tek çağrıda iki store kurar, manifestleri AYRI verir."""
    res_store = ESMResidueStore(
        residue_dir,
        seq_len_lookup=seq_len_lookup,
        max_len=max_len, overlap=overlap,
        pro_manifest_file=residue_manifest,
        gdrive_cache=gdrive_cache, cache_dir=cache_dir, cache_gb=cache_gb,
        prefer_fp16=prefer_fp16,
        hub_repo_id=hub_repo_id, hub_revision=hub_revision, hub_repo_type=hub_repo_type,
        hub_local_dir=hub_local_dir, hub_symlinks=hub_symlinks,
    )
    fused_store = None
    if fused_dir is not None:
        fused_store = ESMFusedStore(
            fused_dir,
            pro_manifest_file=fused_manifest,
            gdrive_cache=gdrive_cache, cache_dir=cache_dir, cache_gb=cache_gb,
            prefer_fp16=prefer_fp16,
            hub_repo_id=hub_repo_id, hub_revision=hub_revision, hub_repo_type=hub_repo_type,
            hub_local_dir=hub_local_dir, hub_symlinks=hub_symlinks,
        )
    return res_store, fused_store