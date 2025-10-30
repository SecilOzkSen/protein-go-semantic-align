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

import sys, argparse, traceback
from typing import Any, List
import os, json, numpy as np, torch, tempfile
from pathlib import Path

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

def pack_to_concat(src_pt: str, out_npz: str, out_json: str, dtype: str = "float16"):
    """
    src_pt -> {'prot_id': tensor [L,D], ...}
    üret:   out_npz = concat memmap bundle (.npz)
            out_json = index metadata
    Atomic write + tmp dosya aynı klasörde.
    """
    out_dir = os.path.dirname(out_npz) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 1) yükle
    try:
        blob = torch.load(src_pt, map_location="cpu")
    except TypeError:
        blob = torch.load(src_pt, map_location="cpu")

    # 2) tensörleri normalize etme YOK (sadece dtype/np çevirisi)
    entries = []
    total_rows = 0
    D = None
    np_dtype = np.float16 if dtype == "float16" else np.float32

    # sıralı, deterministic: ada göre
    for k in sorted(blob.keys()):
        arr = blob[k]
        if isinstance(arr, torch.Tensor):
            a = arr.detach().cpu().numpy()
        else:
            a = np.asarray(arr)
        if a.dtype not in (np.float16, np.float32, np.float64):
            a = a.astype(np_dtype, copy=False)
        elif dtype == "float16":
            a = a.astype(np.float16, copy=False)
        else:
            a = a.astype(np.float32, copy=False)

        if a.ndim != 2:
            raise ValueError(f"{src_pt}:{k} beklenen [L,D], geldi {a.shape}")

        if D is None:
            D = int(a.shape[1])
        elif D != int(a.shape[1]):
            raise ValueError(f"{src_pt}:{k} D tutarsız: beklenen {D}, gelen {a.shape[1]}")

        L = int(a.shape[0])
        entries.append((k, a, L))
        total_rows += L

    if D is None:
        raise RuntimeError(f"Boş shard: {src_pt}")

    # 3) concat array’i RAM AYIRMADAN diske yaz (memmap + npz)
    #    - .npz içinde tek büyük 'concat' array’i ve id->slice index’i taşıyoruz.
    index = {}
    # npz içine doğrudan memmap yazamayız; bu yüzden geçici .npy (memmap) + npz:
    #   a) tmp .npy’yi yaz
    fd_npy, npy_tmp = tempfile.mkstemp(prefix=Path(out_npz).name, suffix=".concat.npy.tmp", dir=out_dir)
    os.close(fd_npy)
    try:
        mm = np.memmap(npy_tmp, mode="w+", dtype=np_dtype, shape=(total_rows, D))
        cursor = 0
        for pid, a, L in entries:
            mm[cursor:cursor+L] = a
            index[pid] = [int(cursor), int(cursor+L)]  # [start, end)
            cursor += L
        mm.flush()
        del mm  # file handle kapansın

        #   b) tmp .npz’yi yaz (concat’ı np.memmap ile oku, np.load KULLANMA)
        fd_npz, npz_tmp = tempfile.mkstemp(prefix=Path(out_npz).name, suffix=".tmp", dir=out_dir)
        os.close(fd_npz)
        try:
            # HATA VEREN SATIRDI:
            # concat_arr = np.load(npy_tmp, mmap_mode="r")

            # YERİNE ŞUNU KULLAN:
            concat_arr = np.memmap(npy_tmp, mode="r", dtype=np_dtype, shape=(total_rows, D))

            # Not: sıkıştırmasız hızlı; disk alanı kısıtlıysa savez_compressed kullanabilirsin
            np.savez(npz_tmp,
                     concat=concat_arr,
                     dtype=str(np_dtype),
                     dim=int(D),
                     n_rows=int(total_rows))
            #   c) meta json’u tmp yaz
            fd_json, json_tmp = tempfile.mkstemp(prefix=Path(out_json).name, suffix=".tmp", dir=out_dir)
            os.close(fd_json)
            try:
                with open(json_tmp, "w", encoding="utf-8") as f:
                    json.dump({"index": index, "dim": D, "dtype": str(np_dtype), "rows": total_rows}, f)
                #   d) atomik replace
                os.replace(npz_tmp, out_npz)
                os.replace(json_tmp, out_json)
            finally:
                if os.path.exists(json_tmp):
                    try:
                        os.remove(json_tmp)
                    except:
                        pass
        finally:
            if os.path.exists(npz_tmp):
                try:
                    os.remove(npz_tmp)
                except:
                    pass
    finally:
        if os.path.exists(npy_tmp):
            try: os.remove(npy_tmp)
            except: pass

    return out_npz, out_npz.replace(".npz", ".npy"), out_json
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
    p.add_argument("--root", default="/workspace/esm-embeddings", help="Search root directory")
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