from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", type=str, required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    args = p.parse_args()

    dump_dir = Path(args.dump_dir)

    labels_path = dump_dir / "top_labels.int8.npy"
    true_ids_path = dump_dir / "true_go_ids.npy"
    eval_ids_path = dump_dir / "eval_go_ids.npy"
    top_cols_path = dump_dir / "top_go_cols.int32.npy"
    scores_path = dump_dir / "top_scores.float32.npy"

    required = [
        labels_path,
        true_ids_path,
        eval_ids_path,
        top_cols_path,
        scores_path,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required dump file: {path}")

    labels = np.load(labels_path, mmap_mode="r")
    true_ids = np.load(true_ids_path, mmap_mode="r")
    eval_ids = np.load(eval_ids_path, mmap_mode="r")
    top_cols = np.load(top_cols_path, mmap_mode="r")
    scores = np.load(scores_path, mmap_mode="r")

    print("\n[DUMP CHECK]")
    print("dump_dir:", dump_dir)
    print("labels:", labels.shape, labels.dtype)
    print("true_ids:", true_ids.shape, true_ids.dtype)
    print("eval_ids:", eval_ids.shape, eval_ids.dtype)
    print("top_cols:", top_cols.shape, top_cols.dtype)
    print("scores:", scores.shape, scores.dtype)

    if labels.shape != top_cols.shape:
        raise RuntimeError(f"labels shape {labels.shape} != top_cols shape {top_cols.shape}")

    if labels.shape != scores.shape:
        raise RuntimeError(f"labels shape {labels.shape} != scores shape {scores.shape}")

    if top_cols.min() < 0:
        raise RuntimeError("top_cols contains negative indices")

    if top_cols.max() >= len(eval_ids):
        raise RuntimeError(
            f"top_cols max {top_cols.max()} >= len(eval_ids) {len(eval_ids)}"
        )

    # Raw true count, all ids in true_go_ids.
    raw_counts = (true_ids >= 0).sum(axis=1).clip(min=1)

    # Eval-space true count, robust if true_go_ids contains labels outside eval_id_list.
    in_eval = np.isin(true_ids, eval_ids)
    eval_counts = ((true_ids >= 0) & in_eval).sum(axis=1).clip(min=1)

    print("\n[TRUE COUNT]")
    print("raw true count mean:", float(raw_counts.mean()))
    print("eval true count mean:", float(eval_counts.mean()))
    print("rows where raw > eval:", float((raw_counts > eval_counts).mean()))

    print("\n[COVERAGE / ORACLE]")
    for k in args.ks:
        k = min(int(k), labels.shape[1])

        hits = labels[:, :k].sum(axis=1)

        cov_raw = float((hits / raw_counts).mean())
        cov_eval = float((hits / eval_counts).mean())

        any_hit = float((hits > 0).mean())

        oracle_raw = float(
            (2.0 * hits.sum())
            / max(1e-12, 2.0 * hits.sum() + (raw_counts - hits).sum())
        )

        oracle_eval = float(
            (2.0 * hits.sum())
            / max(1e-12, 2.0 * hits.sum() + (eval_counts - hits).sum())
        )

        print(
            f"K={k:<4} "
            f"coverage_raw={cov_raw:.4f} "
            f"coverage_eval={cov_eval:.4f} "
            f"any_hit={any_hit:.4f} "
            f"oracle_raw={oracle_raw:.4f} "
            f"oracle_eval={oracle_eval:.4f}"
        )

    print("\n[EXPECTED FOR P3a VAL, roughly]")
    print("K=200  coverage_eval ≈ 0.4556, oracle_eval ≈ 0.5907")
    print("K=500  coverage_eval ≈ 0.5209, oracle_eval ≈ 0.6591")
    print("K=1000 coverage_eval ≈ 0.5733, oracle_eval ≈ 0.7109")

    print("\nDone.")


if __name__ == "__main__":
    main()