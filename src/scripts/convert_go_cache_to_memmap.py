#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faz 1..4 altındaki go_text_embeddings.pt -> memmap .npy (+ .meta.pt) dönüşümü.
Varsayılan kök:
  /workspace/protein-go-semantic-align/src/data/training_ready/go_indexes

Her faz klasöründe beklenen giriş:
  phase{N}/go_text_embeddings.pt

Çıktılar:
  phase{N}/go_text_embeddings.npy
  phase{N}/go_text_embeddings.npy.meta.pt

Başarılı doğrulama sonrası --delete-pt ile .pt silinir (resume-safe).
"""

import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

def convert_one(src_pt: Path, dtype: str = "float16") -> Path:
    blob = torch.load(str(src_pt), map_location="cpu")
    E = blob["embs"].float().numpy()
    if dtype == "float16":
        E = E.astype(np.float16, copy=False)
    elif dtype == "float32":
        E = E.astype(np.float32, copy=False)
    else:
        raise ValueError("--dtype must be float16 or float32")

    # idempotent L2 normalize
    norm = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    E = E / norm

    dst_npy = src_pt.with_suffix(".npy")  # go_text_embeddings.npy
    mm = np.memmap(dst_npy, dtype=E.dtype, mode="w+", shape=E.shape)
    mm[:] = E
    mm.flush()
    del mm

    # id maps
    row2id = blob.get("row2id") or blob.get("ids")
    id2row = blob.get("id2row")
    meta = {"row2id": row2id, "id2row": id2row, "shape": tuple(E.shape), "dtype": str(E.dtype)}
    torch.save(meta, str(dst_npy) + ".meta.pt")
    return dst_npy

def verify(dst_npy: Path) -> bool:
    try:
        arr = np.load(str(dst_npy), mmap_mode="r")
        assert arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0
        # L2 ~ 1 kontrolü (orta değer ~1 etrafında)
        norms = np.linalg.norm(arr[: min(1024, arr.shape[0])], axis=1)
        ok = np.isfinite(norms).all() and (0.8 < norms.mean() < 1.2)
        return bool(ok)
    except Exception as e:
        print(f"[verify] failed for {dst_npy}: {e}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
        default="/workspace/protein-go-semantic-align/src/data/training_ready/go_indexes",
        help="phase dizinlerinin bulunduğu kök.")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=4)
    ap.add_argument("--dtype", default="float16", choices=["float16","float32"])
    ap.add_argument("--delete-pt", action="store_true", help="Doğrulama başarılıysa .pt dosyasını sil.")
    ap.add_argument("--overwrite", action="store_true", help="Var olan .npy/.meta üzerine yaz.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    phases = list(range(args.first, args.last + 1))
    print(f"Root: {root} | phases: {phases} | dtype={args.dtype}")

    ok, skip, fail = 0, 0, 0
    for i in phases:
        d = root / f"phase{i}"
        src = d / "go_text_embeddings.pt"
        dst_npy = d / "go_text_embeddings.npy"
        meta_pt = Path(str(dst_npy) + ".meta.pt")

        if not src.exists():
            print(f"[phase{i}] SKIP (missing): {src}")
            skip += 1
            continue

        if dst_npy.exists() and meta_pt.exists() and not args.overwrite:
            print(f"[phase{i}] SKIP (already converted): {dst_npy}")
            if verify(dst_npy) and args.delete-pt and not args.dry_run:
                try:
                    os.remove(src)
                    print(f"[phase{i}] removed original PT (resume-clean).")
                except Exception:
                    pass
            skip += 1
            continue

        print(f"[phase{i}] CONVERT: {src.name} -> {dst_npy.name}")
        try:
            if args.dry_run:
                print(f"[phase{i}] (dry-run) would write {dst_npy} and {meta_pt}")
                ok += 1
                continue

            out = convert_one(src, dtype=args.dtype)
            if verify(out):
                print(f"[phase{i}] ✔ OK: {out.name}")
                if args.delete-pt:
                    try:
                        os.remove(src)
                        print(f"[phase{i}] deleted: {src.name}")
                    except Exception as e:
                        print(f"[phase{i}] warn: could not delete PT: {e}")
                ok += 1
            else:
                print(f"[phase{i}] ✗ verify failed.")
                fail += 1
        except Exception as e:
            print(f"[phase{i}] ✗ ERROR: {e}")
            fail += 1

    print(f"\nDone. ok={ok}, skip={skip}, fail={fail}")

if __name__ == "__main__":
    main()