'''
#valid
python -m src.script.eval_candidate_dump_raw \
  --dump_dir /workspace/candidate_dumps/ESMknn_val_top500_nofill \
  --train_dump_for_ic /workspace/candidate_dumps/ESMknn_train_top500_nofill \
  --topk 500 \
  --n_thresholds 501 \
  --out_json /workspace/results/ESMknn_val_top500_raw_metrics.json

#P3a raw
python -m src.script.eval_candidate_dump_raw \
  --dump_dir /workspace/candidate_dumps/P3a_val_top1000 \
  --train_dump_for_ic /workspace/candidate_dumps/ESMknn_train_top500_nofill \
  --topk 500 \
  --n_thresholds 501 \
  --out_json /workspace/results/P3a_val_top500_raw_metrics.json

Union raw:

python -m src.script.eval_candidate_dump_raw \
  --dump_dir /workspace/candidate_dumps/P3a_ESMknn_union_val_top1000 \
  --train_dump_for_ic /workspace/candidate_dumps/ESMknn_train_top500_nofill \
  --topk 1000 \
  --n_thresholds 501 \
  --out_json /workspace/results/P3a_ESMknn_union_val_top1000_raw_metrics.json
'''

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_eval_ids(dump_dir: Path) -> np.ndarray:
    return np.load(dump_dir / "eval_go_ids.npy", mmap_mode="r").astype(np.int64)


def load_true_ids(dump_dir: Path) -> np.ndarray:
    return np.load(dump_dir / "true_go_ids.npy", mmap_mode="r")


def build_id2col(eval_go_ids: np.ndarray) -> Dict[int, int]:
    return {int(g): i for i, g in enumerate(eval_go_ids.tolist())}


def make_ic_from_train_dump(train_dump: Path, eval_go_ids: np.ndarray) -> np.ndarray:
    """
    Empirical IC from train labels:
      IC(g) = -log2((count(g)+1)/(N+2))

    This is not full CAFA conditional IC, but is a useful IC-weighted metric.
    """
    true_ids = load_true_ids(train_dump)
    id2col = build_id2col(eval_go_ids)

    counts = np.zeros(len(eval_go_ids), dtype=np.float64)
    n_rows = int(true_ids.shape[0])

    for row in true_ids:
        seen = set()
        for gid in row:
            gid = int(gid)
            if gid < 0:
                continue
            j = id2col.get(gid)
            if j is not None:
                seen.add(j)
        for j in seen:
            counts[j] += 1.0

    p = (counts + 1.0) / (float(n_rows) + 2.0)
    ic = -np.log2(p)
    ic = np.asarray(ic, dtype=np.float32)

    # Avoid exact zero weights.
    ic = np.maximum(ic, 1e-6)
    return ic


def true_ic_per_row(true_ids: np.ndarray, eval_go_ids: np.ndarray, ic: np.ndarray) -> np.ndarray:
    id2col = build_id2col(eval_go_ids)
    out = np.zeros(true_ids.shape[0], dtype=np.float32)

    for i, row in enumerate(true_ids):
        cols = []
        for gid in row:
            gid = int(gid)
            if gid < 0:
                continue
            j = id2col.get(gid)
            if j is not None:
                cols.append(j)
        if cols:
            out[i] = float(ic[np.asarray(cols, dtype=np.int64)].sum())

    return out


def candidate_oracle(labels: np.ndarray, true_ids: np.ndarray) -> Tuple[float, float]:
    true_counts = (true_ids >= 0).sum(axis=1).clip(min=1)
    hits = labels.sum(axis=1)

    coverage = float(np.mean(hits / true_counts))
    oracle = float((2.0 * hits.sum()) / max(1e-12, 2.0 * hits.sum() + (true_counts - hits).sum()))
    return coverage, oracle


def compute_sparse_fmax_wfmax(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    top_cols: np.ndarray,
    true_ids: np.ndarray,
    eval_go_ids: np.ndarray,
    ic: np.ndarray,
    n_thresholds: int = 501,
) -> Dict[str, float]:
    """
    Candidate-limited full-space Fmax/wFmax.

    Non-candidates are treated as low score and never predicted
    for thresholds over candidate scores.
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int8)
    top_cols = np.asarray(top_cols, dtype=np.int64)

    finite = np.isfinite(scores)
    finite &= scores > -1e5

    if not finite.any():
        return {
            "fmax": 0.0,
            "wfmax": 0.0,
            "best_threshold_f": 0.0,
            "best_threshold_wf": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "wprecision": 0.0,
            "wrecall": 0.0,
        }

    s_min = float(scores[finite].min())
    s_max = float(scores[finite].max())

    if s_min == s_max:
        thresholds = np.asarray([s_min], dtype=np.float32)
    else:
        thresholds = np.linspace(s_min, s_max, int(n_thresholds), dtype=np.float32)

    total_true = int((true_ids >= 0).sum())
    true_ic = true_ic_per_row(true_ids, eval_go_ids, ic)
    valid_true_ic = true_ic > 0

    ic_cand = ic[top_cols]  # [N,K]

    best_f = 0.0
    best_f_t = float(thresholds[0])
    best_prec = 0.0
    best_rec = 0.0

    best_wf = 0.0
    best_wf_t = float(thresholds[0])
    best_wpr = 0.0
    best_wrc = 0.0

    labels_bool = labels.astype(bool)

    for t in thresholds:
        pred = scores >= float(t)

        tp = int((pred & labels_bool).sum())
        pred_count = int(pred.sum())
        fp = pred_count - tp
        fn = total_true - tp

        prec = tp / max(1e-12, tp + fp)
        rec = tp / max(1e-12, tp + fn)
        f = (2.0 * prec * rec) / max(1e-12, prec + rec)

        if f > best_f:
            best_f = float(f)
            best_f_t = float(t)
            best_prec = float(prec)
            best_rec = float(rec)

        # Weighted Fmax, CAFA-like protein-averaged weighted precision/recall.
        pred_ic = (pred * ic_cand).sum(axis=1)
        tp_ic = ((pred & labels_bool) * ic_cand).sum(axis=1)

        has_pred = pred_ic > 0
        if has_pred.any():
            wpr = float(np.mean(tp_ic[has_pred] / np.maximum(pred_ic[has_pred], 1e-12)))
        else:
            wpr = 0.0

        if valid_true_ic.any():
            wrc = float(np.mean(tp_ic[valid_true_ic] / np.maximum(true_ic[valid_true_ic], 1e-12)))
        else:
            wrc = 0.0

        wf = (2.0 * wpr * wrc) / max(1e-12, wpr + wrc)
        if wf > best_wf:
            best_wf = float(wf)
            best_wf_t = float(t)
            best_wpr = float(wpr)
            best_wrc = float(wrc)

    return {
        "fmax": best_f,
        "wfmax": best_wf,
        "best_threshold_f": best_f_t,
        "best_threshold_wf": best_wf_t,
        "precision": best_prec,
        "recall": best_rec,
        "wprecision": best_wpr,
        "wrecall": best_wrc,
    }


def main():
    p = argparse.ArgumentParser("Evaluate raw candidate dump scores with Fmax and empirical wFmax.")
    p.add_argument("--dump_dir", type=str, required=True)
    p.add_argument("--train_dump_for_ic", type=str, required=True)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--n_thresholds", type=int, default=501)
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    dump_dir = Path(args.dump_dir)
    train_dump = Path(args.train_dump_for_ic)

    eval_go_ids = load_eval_ids(dump_dir)
    top_cols = np.load(dump_dir / "top_go_cols.int32.npy", mmap_mode="r")[:, : args.topk]
    scores = np.load(dump_dir / "top_scores.float32.npy", mmap_mode="r")[:, : args.topk]
    labels = np.load(dump_dir / "top_labels.int8.npy", mmap_mode="r")[:, : args.topk]
    true_ids = load_true_ids(dump_dir)

    ic = make_ic_from_train_dump(train_dump, eval_go_ids)

    coverage, oracle = candidate_oracle(labels, true_ids)

    metrics = compute_sparse_fmax_wfmax(
        scores=scores,
        labels=labels,
        top_cols=top_cols,
        true_ids=true_ids,
        eval_go_ids=eval_go_ids,
        ic=ic,
        n_thresholds=args.n_thresholds,
    )

    result = {
        "dump_dir": str(dump_dir),
        "train_dump_for_ic": str(train_dump),
        "topk": int(args.topk),
        "coverage": coverage,
        "oracle_microF": oracle,
        **metrics,
        "ic_note": "Empirical IC from train true_go_ids: IC=-log2((count+1)/(N+2)).",
    }

    print(json.dumps(result, indent=2))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()