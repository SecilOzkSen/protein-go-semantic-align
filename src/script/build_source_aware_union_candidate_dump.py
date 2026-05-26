'''
python -m src.script.build_source_aware_union_candidate_dump \
  --dump_a /workspace/candidate_dumps/P3a_train_top1000 \
  --dump_b /workspace/candidate_dumps/ESMknn_train_top500_nofill \
  --topk_a 500 \
  --topk_b 500 \
  --max_out 1000 \
  --a_name p3a \
  --b_name esmknn \
  --out_dir /workspace/candidate_dumps/P3a_ESMknn_sourceaware_union_train_top1000 \
  --overwrite
'''

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


def zscore_scores(scores: np.ndarray, valid: np.ndarray | None = None) -> Tuple[np.ndarray, float, float]:
    arr = np.asarray(scores, dtype=np.float32)
    finite = np.isfinite(arr) & (arr > -1e5)
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool)
    if not finite.any():
        return arr.astype(np.float32), 0.0, 1.0
    mean = float(arr[finite].mean())
    std = float(arr[finite].std() + 1e-6)
    out = (arr - mean) / std
    out = np.where(finite, out, -1e6).astype(np.float32)
    return out, mean, std


def load_true_col_sets(dump: Path) -> Tuple[np.ndarray, List[set[int]]]:
    eval_ids = np.load(dump / "eval_go_ids.npy", mmap_mode="r")
    true_ids = np.load(dump / "true_go_ids.npy", mmap_mode="r")
    eval_ids_arr = np.asarray(eval_ids, dtype=np.int64)
    id2col = {int(g): i for i, g in enumerate(eval_ids_arr.tolist())}

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
    return eval_ids_arr, out


def safe_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser("Build source-aware union candidate dump from two compatible candidate dumps.")
    p.add_argument("--dump_a", type=str, required=True, help="Usually P3a dump.")
    p.add_argument("--dump_b", type=str, required=True, help="Usually ESM-kNN dump.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--topk_a", type=int, default=500)
    p.add_argument("--topk_b", type=int, default=500)
    p.add_argument("--max_out", type=int, default=1000)
    p.add_argument("--a_name", type=str, default="p3a")
    p.add_argument("--b_name", type=str, default="esmknn")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dump_a = Path(args.dump_a)
    dump_b = Path(args.dump_b)
    out_dir = Path(args.out_dir)
    prepare_output_dir(out_dir, overwrite=bool(args.overwrite))

    ids_a = read_json(dump_a / "protein_ids.json")
    ids_b = read_json(dump_b / "protein_ids.json")
    if ids_a != ids_b:
        raise RuntimeError("protein_ids mismatch between dumps. Build train/val union from matching splits.")

    eval_a = np.load(dump_a / "eval_go_ids.npy", mmap_mode="r")
    eval_b = np.load(dump_b / "eval_go_ids.npy", mmap_mode="r")
    if not np.array_equal(eval_a, eval_b):
        raise RuntimeError("eval_go_ids mismatch between dumps")

    topk_a = int(args.topk_a)
    topk_b = int(args.topk_b)
    K = int(args.max_out)

    top_cols_a = np.load(dump_a / "top_go_cols.int32.npy", mmap_mode="r")[:, :topk_a]
    top_cols_b = np.load(dump_b / "top_go_cols.int32.npy", mmap_mode="r")[:, :topk_b]

    scores_a_raw = np.load(dump_a / "top_scores.float32.npy", mmap_mode="r")[:, :topk_a]
    scores_b_raw = np.load(dump_b / "top_scores.float32.npy", mmap_mode="r")[:, :topk_b]

    valid_a_path = dump_a / "top_valid.int8.npy"
    valid_b_path = dump_b / "top_valid.int8.npy"
    valid_a = np.load(valid_a_path, mmap_mode="r")[:, :topk_a] if valid_a_path.exists() else None
    valid_b = np.load(valid_b_path, mmap_mode="r")[:, :topk_b] if valid_b_path.exists() else None

    scores_a, mean_a, std_a = zscore_scores(scores_a_raw, valid_a)
    scores_b, mean_b, std_b = zscore_scores(scores_b_raw, valid_b)

    eval_ids, true_col_sets = load_true_col_sets(dump_a)

    N = int(top_cols_a.shape[0])
    if int(top_cols_b.shape[0]) != N:
        raise RuntimeError("row mismatch between dumps")

    top_cols = np.zeros((N, K), dtype=np.int32)
    top_scores = np.full((N, K), -1e6, dtype=np.float32)
    top_labels = np.zeros((N, K), dtype=np.int8)
    top_valid = np.zeros((N, K), dtype=np.int8)

    source_in_a = np.zeros((N, K), dtype=np.int8)
    source_in_b = np.zeros((N, K), dtype=np.int8)
    source_score_a = np.zeros((N, K), dtype=np.float32)
    source_score_b = np.zeros((N, K), dtype=np.float32)
    source_rank_a = np.ones((N, K), dtype=np.float32)
    source_rank_b = np.ones((N, K), dtype=np.float32)

    denom_rank_a = np.log1p(float(max(1, topk_a)))
    denom_rank_b = np.log1p(float(max(1, topk_b)))

    for i in tqdm(range(N), desc="build source-aware union candidates"):
        # candidate col -> properties
        cand: Dict[int, Dict[str, float]] = {}

        for j, (c, sc) in enumerate(zip(top_cols_a[i].tolist(), scores_a[i].tolist())):
            if valid_a is not None and int(valid_a[i, j]) == 0:
                continue
            c = int(c)
            d = cand.setdefault(c, {"score_a": 0.0, "score_b": 0.0, "rank_a": 1.0, "rank_b": 1.0, "in_a": 0.0, "in_b": 0.0})
            d["in_a"] = 1.0
            d["score_a"] = max(float(d["score_a"]), float(sc)) if d["score_a"] != 0.0 else float(sc)
            d["rank_a"] = min(float(d["rank_a"]), float(np.log1p(j + 1.0) / denom_rank_a))

        for j, (c, sc) in enumerate(zip(top_cols_b[i].tolist(), scores_b[i].tolist())):
            if valid_b is not None and int(valid_b[i, j]) == 0:
                continue
            c = int(c)
            d = cand.setdefault(c, {"score_a": 0.0, "score_b": 0.0, "rank_a": 1.0, "rank_b": 1.0, "in_a": 0.0, "in_b": 0.0})
            d["in_b"] = 1.0
            d["score_b"] = max(float(d["score_b"]), float(sc)) if d["score_b"] != 0.0 else float(sc)
            d["rank_b"] = min(float(d["rank_b"]), float(np.log1p(j + 1.0) / denom_rank_b))

        ranked = []
        for c, d in cand.items():
            # Union ranking score: best normalized source score, with a small bonus if both sources retrieved it.
            both = 1.0 if d["in_a"] > 0.5 and d["in_b"] > 0.5 else 0.0
            union_sc = max(float(d["score_a"]) if d["in_a"] > 0.5 else -1e6,
                           float(d["score_b"]) if d["in_b"] > 0.5 else -1e6) + 0.05 * both
            ranked.append((int(c), union_sc, d))
        ranked.sort(key=lambda x: x[1], reverse=True)

        if len(ranked) < K:
            seen = {c for c, _, _ in ranked}
            empty = {"score_a": 0.0, "score_b": 0.0, "rank_a": 1.0, "rank_b": 1.0, "in_a": 0.0, "in_b": 0.0}
            for c in range(len(eval_ids)):
                if c not in seen:
                    ranked.append((int(c), -1e6, empty))
                    seen.add(c)
                if len(ranked) >= K:
                    break

        true_set = true_col_sets[i]
        for j, (c, sc, d) in enumerate(ranked[:K]):
            c = int(c)
            top_cols[i, j] = c
            top_scores[i, j] = float(sc)
            top_labels[i, j] = 1 if c in true_set else 0
            source_in_a[i, j] = int(d["in_a"] > 0.5)
            source_in_b[i, j] = int(d["in_b"] > 0.5)
            source_score_a[i, j] = float(d["score_a"])
            source_score_b[i, j] = float(d["score_b"])
            source_rank_a[i, j] = float(d["rank_a"])
            source_rank_b[i, j] = float(d["rank_b"])
            top_valid[i, j] = 1 if (source_in_a[i, j] or source_in_b[i, j]) else 0

    for name in [
        "eval_go_ids.npy",
        "go_z.float16.npy",
        "protein_z.float16.npy",
        "protein_ids.json",
        "true_go_ids.json",
        "true_go_ids.npy",
    ]:
        safe_copy(dump_a / name, out_dir / name)

    np.save(out_dir / "top_go_cols.int32.npy", top_cols)
    np.save(out_dir / "top_scores.float32.npy", top_scores)
    np.save(out_dir / "top_labels.int8.npy", top_labels)
    np.save(out_dir / "top_valid.int8.npy", top_valid)
    np.save(out_dir / "source_in_a.int8.npy", source_in_a)
    np.save(out_dir / "source_in_b.int8.npy", source_in_b)
    np.save(out_dir / "source_score_a.float32.npy", source_score_a)
    np.save(out_dir / "source_score_b.float32.npy", source_score_b)
    np.save(out_dir / "source_rank_a.float32.npy", source_rank_a)
    np.save(out_dir / "source_rank_b.float32.npy", source_rank_b)

    metadata = {
        "status": "complete",
        "source": "source_aware_union",
        "dump_a": str(dump_a),
        "dump_b": str(dump_b),
        "a_name": str(args.a_name),
        "b_name": str(args.b_name),
        "topk_a": topk_a,
        "topk_b": topk_b,
        "max_out": K,
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
            "top_valid": "top_valid.int8.npy",
            "source_in_a": "source_in_a.int8.npy",
            "source_in_b": "source_in_b.int8.npy",
            "source_score_a": "source_score_a.float32.npy",
            "source_score_b": "source_score_b.float32.npy",
            "source_rank_a": "source_rank_a.float32.npy",
            "source_rank_b": "source_rank_b.float32.npy",
        },
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (out_dir / "DONE").open("w", encoding="utf-8") as f:
        f.write("complete\n")

    print("[source-aware-union] done:", out_dir)


if __name__ == "__main__":
    main()
