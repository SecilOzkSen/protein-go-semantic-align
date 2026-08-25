#!/usr/bin/env python3
"""Read-only audit for GOR2023 GO text/cache/dump ID alignment."""
import argparse, json, re
from pathlib import Path
import numpy as np


def gid(x):
    if isinstance(x, (int, np.integer)): return int(x)
    m = re.search(r"(?:GO:)?0*(\d+)$", str(x).strip())
    if not m: raise ValueError(f"Bad GO ID: {x!r}")
    return int(m.group(1))


def load_json(path):
    with open(path, encoding="utf-8") as f: return json.load(f)


def load_ids(path):
    p = Path(path)
    if p.suffix == ".npy": return [gid(x) for x in np.load(p, mmap_mode="r")]
    obj = load_json(p)
    if isinstance(obj, dict):
        for k in ("ids", "go_ids_int", "go_ids", "row2id"):
            if k in obj: obj = obj[k]; break
    return [gid(x) for x in obj]


def load_texts(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            raw = r.get("go_id", r.get("id", r.get("GO_ID")))
            rows.append((gid(raw), r))
    return rows


def find_sidecar(npy):
    p = Path(npy)
    candidates = [
        p.with_suffix(".ids.json"), Path(str(p) + ".ids.json"),
        p.with_name(p.stem + "_ids.json"), p.with_name(p.stem + ".json"),
        p.with_name("ids.json"), p.with_name("go_ids.json"), p.with_name("row2id.json"),
    ]
    for x in candidates:
        if x.exists(): return x
    raise FileNotFoundError("Cache ID sidecar not found. Tried: " + ", ".join(map(str, candidates)))


def first_mismatches(a, b, n=10):
    return [(i, a[i], b[i]) for i in range(min(len(a), len(b))) if a[i] != b[i]][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--cache_ids", default="")
    ap.add_argument("--go_vocab", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--focus", nargs="*", default=["GO:0032991", "GO:0012505", "GO:0005829", "GO:0005575", "GO:0110165"])
    a = ap.parse_args()

    text_rows = load_texts(a.texts);
    text_ids = [x[0] for x in text_rows]
    cache = np.load(a.cache, mmap_mode="r")
    side = Path(a.cache_ids) if a.cache_ids else find_sidecar(a.cache)
    cache_ids = load_ids(side)
    print(f"texts={len(text_ids)} unique={len(set(text_ids))}")
    print(f"cache_shape={cache.shape} cache_ids={len(cache_ids)} sidecar={side}")
    print("text_order_equals_cache_order:", text_ids == cache_ids)
    print("text/cache first mismatches:", first_mismatches(text_ids, cache_ids))
    print("text_ids_equals_cache_id_set:", set(text_ids) == set(cache_ids))
    print("cache finite:", bool(np.isfinite(cache).all()))

    if a.go_vocab:
        vocab = load_ids(a.go_vocab)
        print(f"go_vocab={len(vocab)} unique={len(set(vocab))}")
        print("vocab_order_equals_cache_order:", vocab == cache_ids)
        print("vocab/cache first mismatches:", first_mismatches(vocab, cache_ids))
        print("vocab_ids_equals_cache_id_set:", set(vocab) == set(cache_ids))

    keyed = dict(text_rows)
    print("\nFOCUS TEXTS")
    for raw in a.focus:
        x = gid(raw);
        r = keyed.get(x)
        print(f"GO:{x:07d}:", json.dumps(r, ensure_ascii=False)[:600] if r else "MISSING")

    if a.dump:
        d = Path(a.dump);
        ev = load_ids(d / "eval_go_ids.npy")
        cols = np.load(d / "top_go_cols.int32.npy", mmap_mode="r")
        print(f"\nDUMP eval_ids={len(ev)} cols_shape={cols.shape} min={cols.min()} max={cols.max()}")
        print("dump_eval_ids_in_cache:", set(ev) <= set(cache_ids))
        saved = d / "top_go_ids.int64.npy"
        if saved.exists():
            top = np.load(saved, mmap_mode="r")
            take = min(1000, cols.shape[0])
            recon = np.asarray(ev, dtype=np.int64)[np.asarray(cols[:take])]
            neq = np.argwhere(recon != np.asarray(top[:take]))
            print("top_go_cols_to_ids_exact_first_rows:", len(neq) == 0)
            print("first dump mismatch:", None if len(neq) == 0 else (neq[0].tolist(), int(recon[tuple(neq[0])]), int(top[tuple(neq[0])])))
        else:
            print("top_go_ids check: SKIPPED, file absent")

    failures = []
    if len(text_ids) != cache.shape[0] or len(cache_ids) != cache.shape[0]: failures.append("cache row count")
    if text_ids != cache_ids: failures.append("text/cache row order")
    if failures: raise SystemExit("\nAUDIT FAILED: " + ", ".join(failures))
    print("\nAUDIT PASSED: GO text rows and cache rows are exactly aligned.")


if __name__ == "__main__": main()
