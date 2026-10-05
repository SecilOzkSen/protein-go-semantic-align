from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.metrics.gor2023 import compute_gor2023_wfmax, load_information_accretion
from src.training.retrievercal_trainer import load_gor2023_parents_from_obo

STREAM_FILES = {
    "fused": "top_scores.float32.npy",
    "global": "top_global_scores.float32.npy",
    "local": "top_local_scores.float32.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit raw GOR2023 retriever scores without a reranker")
    parser.add_argument("--dump_root", required=True, help="Directory containing train/valid/test dumps")
    parser.add_argument("--ia_path", required=True)
    parser.add_argument("--go_obo", required=True)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--alpha_step", type=float, default=0.05)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--out_json", default="")
    return parser.parse_args()


def valid_mask(directory: Path, shape: tuple[int, int], topk: int) -> np.ndarray:
    path = directory / "top_valid.int8.npy"
    if path.exists():
        return np.asarray(np.load(path, mmap_mode="r")[:, :topk], dtype=bool)
    return np.ones((shape[0], min(topk, shape[1])), dtype=bool)


def estimate_train_stats(train_dir: Path, filename: str, topk: int) -> tuple[float, float]:
    scores = np.load(train_dir / filename, mmap_mode="r")
    values = np.asarray(scores[:, :topk], dtype=np.float32)
    valid = valid_mask(train_dir, scores.shape, topk)
    selected = values[valid & np.isfinite(values)]
    if selected.size == 0:
        raise ValueError(f"No finite training scores found in {train_dir / filename}")
    mean = float(selected.mean(dtype=np.float64))
    std = float(selected.std(dtype=np.float64))
    return mean, max(std, 1e-6)


def load_truth(directory: Path, eval_go_ids: np.ndarray) -> np.ndarray:
    true_ids = np.asarray(np.load(directory / "true_go_ids.npy", mmap_mode="r"), dtype=np.int64)
    id_to_col = {int(go_id): col for col, go_id in enumerate(eval_go_ids.tolist())}
    truth = np.zeros((true_ids.shape[0], len(eval_go_ids)), dtype=np.int8)
    for row in range(true_ids.shape[0]):
        cols = [id_to_col[int(go_id)] for go_id in true_ids[row] if int(go_id) in id_to_col]
        if cols:
            truth[row, np.asarray(cols, dtype=np.int64)] = 1
    return truth


def load_full_logits(
        directory: Path,
        filename: str,
        topk: int,
        n_terms: int,
        mean: float,
        std: float,
) -> np.ndarray:
    raw = np.load(directory / filename, mmap_mode="r")
    cols = np.load(directory / "top_go_cols.int32.npy", mmap_mode="r")
    k = min(topk, raw.shape[1])
    valid = valid_mask(directory, raw.shape, k)
    # A finite floor is required by the metric implementation. Sigmoid(-30)
    # is effectively zero and remains below the smallest evaluation threshold.
    logits = np.full((raw.shape[0], n_terms), -30.0, dtype=np.float32)
    rows, ranks = np.nonzero(valid[:, :k])
    selected_cols = np.asarray(cols[rows, ranks], dtype=np.int64)
    selected_scores = np.asarray(raw[rows, ranks], dtype=np.float32)
    finite = np.isfinite(selected_scores) & (selected_cols >= 0) & (selected_cols < n_terms)
    logits[rows[finite], selected_cols[finite]] = (
                                                          selected_scores[finite] - float(mean)
                                                  ) / float(std)
    return logits


def compact(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "wfmax": float(metrics["wfmax"]),
        "threshold": float(metrics["threshold"]),
        "weighted_precision": float(metrics["weighted_precision"]),
        "weighted_recall": float(metrics["weighted_recall"]),
        "coverage": float(metrics["coverage"]),
    }


def evaluate(
        truth: np.ndarray,
        logits: np.ndarray,
        eval_go_ids: np.ndarray,
        ia: dict[str, float],
        parents: dict[str, list[str]],
        threshold_step: float,
) -> dict[str, float]:
    return compact(
        compute_gor2023_wfmax(
            y_true=truth,
            y_score=logits,
            go_ids=eval_go_ids.tolist(),
            information_accretion=ia,
            dag_parents=parents,
            threshold_step=threshold_step,
            propagate=True,
        )
    )


def main() -> None:
    args = parse_args()
    root = Path(args.dump_root).expanduser().resolve()
    train_dir = root / "train"
    valid_dir = root / "valid"
    test_dir = root / "test"

    eval_go_ids = np.asarray(np.load(valid_dir / "eval_go_ids.npy"), dtype=np.int64)
    test_eval_ids = np.asarray(np.load(test_dir / "eval_go_ids.npy"), dtype=np.int64)
    if not np.array_equal(eval_go_ids, test_eval_ids):
        raise ValueError("Validation and test eval_go_ids are different")

    stats = {
        name: estimate_train_stats(train_dir, filename, args.topk)
        for name, filename in STREAM_FILES.items()
    }
    print("Training score statistics:")
    for name, (mean, std) in stats.items():
        print(f"  {name:>6}: mean={mean:.6f} std={std:.6f}")

    ia = load_information_accretion(args.ia_path)
    parents = load_gor2023_parents_from_obo(args.go_obo)
    valid_truth = load_truth(valid_dir, eval_go_ids)
    valid_logits = {
        name: load_full_logits(
            valid_dir,
            filename,
            args.topk,
            len(eval_go_ids),
            *stats[name],
        )
        for name, filename in STREAM_FILES.items()
    }

    results: dict[str, object] = {
        "train_score_stats": {
            name: {"mean": mean, "std": std}
            for name, (mean, std) in stats.items()
        },
        "validation": {},
    }

    print("\nVALIDATION RAW STREAMS")
    for name in ("fused", "global", "local"):
        metrics = evaluate(
            valid_truth,
            valid_logits[name],
            eval_go_ids,
            ia,
            parents,
            args.threshold_step,
        )
        results["validation"][name] = metrics
        print(f"  {name:>6}: {json.dumps(metrics, sort_keys=True)}")

    alphas = np.arange(0.0, 1.0 + args.alpha_step / 2.0, args.alpha_step)
    fusion_results: list[dict[str, object]] = []
    best_alpha = 0.0
    best_metrics: dict[str, float] | None = None
    print("\nVALIDATION GLOBAL/LOCAL FUSION")
    for alpha in alphas:
        fusion_logits = (
                float(alpha) * valid_logits["global"]
                + (1.0 - float(alpha)) * valid_logits["local"]
        )
        metrics = evaluate(
            valid_truth,
            fusion_logits,
            eval_go_ids,
            ia,
            parents,
            args.threshold_step,
        )
        fusion_results.append({"alpha_global": float(alpha), **metrics})
        print(f"  alpha_global={alpha:.2f}: wfmax={metrics['wfmax']:.6f}")
        if best_metrics is None or metrics["wfmax"] > best_metrics["wfmax"]:
            best_alpha = float(alpha)
            best_metrics = metrics

    assert best_metrics is not None
    results["validation"]["fusion_grid"] = fusion_results
    results["selected_alpha_global"] = best_alpha
    results["selected_validation_metrics"] = best_metrics

    test_truth = load_truth(test_dir, eval_go_ids)
    test_fused = load_full_logits(
        test_dir,
        STREAM_FILES["fused"],
        args.topk,
        len(eval_go_ids),
        *stats["fused"],
    )
    test_global = load_full_logits(
        test_dir,
        STREAM_FILES["global"],
        args.topk,
        len(eval_go_ids),
        *stats["global"],
    )
    test_local = load_full_logits(
        test_dir,
        STREAM_FILES["local"],
        args.topk,
        len(eval_go_ids),
        *stats["local"],
    )
    test_fusion = best_alpha * test_global + (1.0 - best_alpha) * test_local
    results["test"] = {
        "fused": evaluate(
            test_truth,
            test_fused,
            eval_go_ids,
            ia,
            parents,
            args.threshold_step,
        ),
        "selected_validation_fusion": evaluate(
            test_truth,
            test_fusion,
            eval_go_ids,
            ia,
            parents,
            args.threshold_step,
        ),
    }

    print("\nSELECTION")
    print(f"  best validation alpha_global={best_alpha:.2f}")
    print(f"  validation={json.dumps(best_metrics, sort_keys=True)}")
    print("\nTEST, NO TEST-TIME MODEL OR FUSION SELECTION")
    print(f"  fused={json.dumps(results['test']['fused'], sort_keys=True)}")
    print(
        "  selected_fusion="
        + json.dumps(results["test"]["selected_validation_fusion"], sort_keys=True)
    )

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
