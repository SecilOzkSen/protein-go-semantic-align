import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        elif any(out_dir.iterdir()):
            raise RuntimeError(f"Output dir exists and is not empty: {out_dir}. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)


def zscore_scores(scores: np.ndarray) -> Tuple[np.ndarray, float, float]:
    arr = np.asarray(scores, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return arr, 0.0, 1.0
    mean = float(arr[finite].mean())
    std = float(arr[finite].std() + 1e-6)
    return (arr - mean) / std, mean, std


def load_true_col_sets(dump: Path) -> Tuple[np.ndarray, List[set[int]]]:
    eval_ids = np.load(dump / "eval_go_ids.npy", mmap_mode="r")
    true_ids = np.load(dump / "true_go_ids.npy", mmap_mode="r")
    id2col = {int(g): i for i, g in enumerate(np.asarray(eval_ids, dtype=np.int64).tolist())}

    out: List[set[int]] = []
    for row in true_ids:
        s = set()
        for gid in row:
            gid = int(gid)
            if gid < 0:
                continue
            j = id2col.get(gid, None)
            if j is not None:
                s.add(int(j))
        out.append(s)
    return np.asarray(eval_ids, dtype=np.int64), out


def main():
    p = argparse.ArgumentParser("Build union candidate dump from two compatible candidate dumps.")
    p.add_argument("--dump_a", type=str, required=True, help="Usually P3a dump.")
    p.add_argument("--dump_b", type=str, required=True, help="Usually ESM-kNN dump.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--topk_a", type=int, default=500)
    p.add_argument("--topk_b", type=int, default=500)
    p.add_argument("--max_out", type=int, default=1000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dump_a = Path(args.dump_a)
    dump_b = Path(args.dump_b)
    out_dir = Path(args.out_dir)
    prepare_output_dir(out_dir, overwrite=bool(args.overwrite))

    # Check row and GO alignment.
    ids_a = read_json(dump_a / "protein_ids.json")
    ids_b = read_json(dump_b / "protein_ids.json")
    if ids_a != ids_b:
        raise RuntimeError("protein_ids mismatch between dumps")

    eval_a = np.load(dump_a / "eval_go_ids.npy", mmap_mode="r")
    eval_b = np.load(dump_b / "eval_go_ids.npy", mmap_mode="r")
    if not np.array_equal(eval_a, eval_b):
        raise RuntimeError("eval_go_ids mismatch between dumps")

    top_cols_a = np.load(dump_a / "top_go_cols.int32.npy", mmap_mode="r")[:, : int(args.topk_a)]
    top_cols_b = np.load(dump_b / "top_go_cols.int32.npy", mmap_mode="r")[:, : int(args.topk_b)]

    scores_a_raw = np.load(dump_a / "top_scores.float32.npy", mmap_mode="r")[:, : int(args.topk_a)]
    scores_b_raw = np.load(dump_b / "top_scores.float32.npy", mmap_mode="r")[:, : int(args.topk_b)]

    scores_a, mean_a, std_a = zscore_scores(scores_a_raw)
    scores_b, mean_b, std_b = zscore_scores(scores_b_raw)

    valid_b = None
    valid_b_path = dump_b / "top_valid.int8.npy"
    if valid_b_path.exists():
        valid_b = np.load(valid_b_path, mmap_mode="r")[:, : int(args.topk_b)]

    N = int(top_cols_a.shape[0])
    K = int(args.max_out)

    top_cols = np.zeros((N, K), dtype=np.int32)
    top_scores = np.full((N, K), -1e6, dtype=np.float32)
    top_labels = np.zeros((N, K), dtype=np.int8)
    in_a = np.zeros((N, K), dtype=np.int8)
    in_b = np.zeros((N, K), dtype=np.int8)

    eval_ids, true_col_sets = load_true_col_sets(dump_a)

    for i in tqdm(range(N), desc="build union candidates"):
        cand: Dict[int, float] = {}
        src_a: Dict[int, int] = {}
        src_b: Dict[int, int] = {}

        for c, sc in zip(top_cols_a[i].tolist(), scores_a[i].tolist()):
            c = int(c)
            cand[c] = max(cand.get(c, -1e9), float(sc))
            src_a[c] = 1

        for j, (c, sc) in enumerate(zip(top_cols_b[i].tolist(), scores_b[i].tolist())):
            if valid_b is not None and int(valid_b[i, j]) == 0:
                continue
            c = int(c)
            cand[c] = max(cand.get(c, -1e9), float(sc))
            src_b[c] = 1

        ranked = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)

        if len(ranked) < K:
            seen = set(c for c, _ in ranked)
            for c in range(len(eval_ids)):
                if c not in seen:
                    ranked.append((int(c), -1e6))
                    seen.add(c)
                if len(ranked) >= K:
                    break

        true_set = true_col_sets[i]
        for j, (c, sc) in enumerate(ranked[:K]):
            c = int(c)
            top_cols[i, j] = c
            top_scores[i, j] = float(sc)
            top_labels[i, j] = 1 if c in true_set else 0
            in_a[i, j] = src_a.get(c, 0)
            in_b[i, j] = src_b.get(c, 0)

    # Copy shared files from dump_a.
    for name in [
        "eval_go_ids.npy",
        "go_z.float16.npy",
        "protein_z.float16.npy",
        "protein_ids.json",
        "true_go_ids.json",
        "true_go_ids.npy",
    ]:
        shutil.copy2(dump_a / name, out_dir / name)

    np.save(out_dir / "top_go_cols.int32.npy", top_cols)
    np.save(out_dir / "top_scores.float32.npy", top_scores)
    np.save(out_dir / "top_labels.int8.npy", top_labels)
    np.save(out_dir / "source_in_a.int8.npy", in_a)
    np.save(out_dir / "source_in_b.int8.npy", in_b)

    metadata = {
        "status": "complete",
        "source": "union",
        "dump_a": str(dump_a),
        "dump_b": str(dump_b),
        "topk_a": int(args.topk_a),
        "topk_b": int(args.topk_b),
        "max_out": int(args.max_out),
        "score_normalization": {
            "a_mean": mean_a,
            "a_std": std_a,
            "b_mean": mean_b,
            "b_std": std_b,
        },
        "files": {
            "top_go_cols": "top_go_cols.int32.npy",
            "top_scores": "top_scores.float32.npy",
            "top_labels": "top_labels.int8.npy",
            "source_in_a": "source_in_a.int8.npy",
            "source_in_b": "source_in_b.int8.npy",
        },
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (out_dir / "DONE").open("w", encoding="utf-8") as f:
        f.write("complete\n")

    print("[union] done:", out_dir)


if __name__ == "__main__":
    main()
