#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toplu protein embedding paketleme (PT -> concat npz + mmap npy + meta json)
- Varsayılan kök: /workspace/embeddings
- Her .pt için:
    <name>.concat.npz
    <name>.embeddings.npy
    <name>.meta.json
- Başarılı doğrulama sonrası .pt silinir (resume-safe).
"""

import os, sys, json, argparse, traceback
from typing import Any, List, Tuple
import numpy as np
import torch

# ---------- low-RAM dönüşüm: dict-by-pid veya {"pid_list","embs"} destekler ----------
def _load_shard_any(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Cannot load shard: {path} :: {e}")

def _as_numpy_2d(x: Any, dtype) -> np.ndarray:
    arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected [L,D] ndarray, got shape={arr.shape}")
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr

def pack_to_concat(shard_path: str, out_npz: str, out_json: str, *, dtype=np.float16) -> Tuple[str, str, str]:
    """
    PT -> (concat.npz, embeddings.npy, meta.json)
    Dönüş: (out_npz, emb_path, out_json)
    """
    sh = _load_shard_any(shard_path)
    if isinstance(sh, dict) and sh.get("__format__") == "concat":
        raise RuntimeError("Input shard already in concat format.")
    # Kaynaktan (pid_list, arrays) çıkar
    if isinstance(sh, dict) and ("pid_list" in sh and "embs" in sh):
        pid_list = [str(x) for x in sh["pid_list"]]
        arr_iter = (sh["embs"][i] for i in range(len(pid_list)))
    elif isinstance(sh, dict):
        # dict-by-pid (str key)
        pid_list = []
        keys = [k for k in sh.keys() if isinstance(k, str)]
        # deterministik için sort (opsiyonel)
        keys.sort()
        pid_list = keys
        arr_iter = (sh[k] for k in keys)
    else:
        raise RuntimeError("Unsupported shard payload; expected dict-like.")

    # ilk array'dan D boyutunu belirlemek için bir kez okuyoruz
    arr_iter = iter(arr_iter)
    first = _as_numpy_2d(next(arr_iter), dtype)
    D = int(first.shape[1])

    arrays: List[np.ndarray] = [first]
    lengths: List[int] = [int(first.shape[0])]
    for a in arr_iter:
        a = _as_numpy_2d(a, dtype)
        if a.shape[1] != D:
            raise ValueError(f"Inconsistent D: got {a.shape[1]}, expected {D}")
        arrays.append(a)
        lengths.append(int(a.shape[0]))

    N_tot = int(sum(lengths))
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)

    emb_path = out_npz.replace(".npz", ".embeddings.npy")
    # atomic yazım için tmp uzantı
    emb_tmp = emb_path + ".tmp"
    npz_tmp = out_npz + ".tmp"
    json_tmp = out_json + ".tmp"

    # büyük tek dosyaya sırayla dök (memmap)
    mm = np.memmap(emb_tmp, mode="w+", dtype=dtype, shape=(N_tot, D))
    pos = 0
    for a in arrays:
        mm[pos:pos + a.shape[0]] = a
        pos += a.shape[0]
    mm.flush(); del mm

    # npz kapsayıcı (sadece meta + tekil path)
    np.savez(npz_tmp,
             __format__=np.array(["concat"], dtype=object),
             embeddings_path=np.array([os.path.basename(emb_path)], dtype=object),
             offsets=offsets)

    pid2row = {pid: i for i, pid in enumerate(pid_list)}
    with open(json_tmp, "w", encoding="utf-8") as f:
        json.dump({"pid2row": pid2row, "row2pid": pid_list}, f)

    # atomic rename
    os.replace(emb_tmp, emb_path)
    os.replace(npz_tmp, out_npz)
    os.replace(json_tmp, out_json)

    return out_npz, emb_path, out_json

def _verify(out_npz: str, emb_path: str, out_json: str) -> bool:
    try:
        meta = np.load(out_npz, allow_pickle=False)
        fmt = str(meta["__format__"][0])
        assert fmt == "concat", f"wrong format: {fmt}"
        offsets = meta["offsets"]
        with open(out_json, "r", encoding="utf-8") as f:
            j = json.load(f)
        pid2row = j["pid2row"]
        assert len(offsets) == (len(pid2row) + 1), "offsets length mismatch"
        mm = np.load(emb_path, mmap_mode="r")
        assert mm.shape[0] == int(offsets[-1]), "embeddings rows != last offset"
        assert mm.ndim == 2, "embeddings must be 2D"
        return True
    except Exception as e:
        print(f"[verify] FAILED for {out_npz}: {e}")
        return False

def find_pt_files(root: str, pattern: str = ".pt") -> List[str]:
    matches = []
    for base, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(pattern):
                matches.append(os.path.join(base, fn))
    matches.sort()
    return matches

def main():
    p = argparse.ArgumentParser(description="Bulk pack PT shards to concat format and delete originals on success.")
    p.add_argument("--root", default="/workspace/embeddings", help="Search root directory")
    p.add_argument("--ext", default=".pt", help="Shard file extension to convert")
    p.add_argument("--dtype", default="fp16", choices=["fp16","fp32"], help="Target dtype for embeddings.npy")
    p.add_argument("--overwrite", action="store_true", help="Overwrite if outputs exist")
    p.add_argument("--dry-run", action="store_true", help="Plan only; don’t write/delete")
    args = p.parse_args()

    dtype = np.float16 if args.dtype == "fp16" else np.float32
    shards = find_pt_files(args.root, args.ext)
    if not shards:
        print(f"No '{args.ext}' files under {args.root}")
        return

    print(f"Found {len(shards)} shard(s). Starting...")
    ok_cnt, skip_cnt, fail_cnt = 0, 0, 0

    for i, spath in enumerate(shards, 1):
        stem = os.path.splitext(spath)[0]
        out_npz  = stem + ".concat.npz"
        out_json = stem + ".meta.json"
        out_npy  = stem + ".embeddings.npy"

        exists_all = os.path.exists(out_npz) and os.path.exists(out_json) and os.path.exists(out_npy)
        if exists_all and not args.overwrite:
            print(f"[{i}/{len(shards)}] SKIP (exists): {spath}")
            skip_cnt += 1
            # resume mode: doğrulayıp .pt’yi silebiliriz
            if _verify(out_npz, out_npy, out_json) and not args.dry_run:
                try:
                    os.remove(spath)
                except Exception:
                    pass
            continue

        print(f"[{i}/{len(shards)}] PACK  -> {spath}")
        try:
            if not args.dry_run:
                out_npz_, out_npy_, out_json_ = pack_to_concat(spath, out_npz, out_json, dtype=dtype)
                if _verify(out_npz_, out_npy_, out_json_):
                    os.remove(spath)
                    ok_cnt += 1
                    print(f"   ✔ OK, removed: {spath}")
                else:
                    fail_cnt += 1
                    print(f"   ✗ Verify failed (kept original): {spath}")
            else:
                print(f"   (dry-run) would write: {out_npz}, {out_npy}, {out_json} and remove {spath}")
        except Exception:
            fail_cnt += 1
            print(f"   ✗ ERROR while packing: {spath}")
            traceback.print_exc()

    print(f"\nDone. ok={ok_cnt}, skip={skip_cnt}, fail={fail_cnt}")

if __name__ == "__main__":
    main()