#!/usr/bin/env python3
"""Evaluate MZSGO temporal labels from an existing candidate dump.

The dump format matches ``dump_retriever_candidates_global_local.py`` and
``reranker_main.py``. Missing temporal terms outside the dumped top-K receive a
score of zero, which makes retrieval failure explicit in downstream Fmax/AUPR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


def read_go_ids(path: Path) -> List[int]:
    if path.suffix.lower() == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
    else:
        values = path.read_text(encoding="utf-8").splitlines()
    return sorted({int(str(value).strip().replace("GO:", "")) for value in values if str(value).strip()})


def load_true_sets(path: Path, protein_ids: Sequence[str], temporal: Set[int]) -> List[Set[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for protein_id in protein_ids:
        values = raw.get(str(protein_id), [])
        output.append({int(str(value).replace("GO:", "")) for value in values} & temporal)
    return output


def mzsgo_protein_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold_step: float = 0.01
) -> Dict[str, float]:
    keep = y_true.sum(axis=1) > 0
    y_true = y_true[keep].astype(bool)
    y_score = y_score[keep]
    if not y_true.shape[0]:
        raise ValueError("No proteins have temporal positives")
    thresholds = np.arange(0.0, 1.0 + threshold_step / 2.0, threshold_step)
    precisions, recalls, fscores = [], [], []
    for threshold in thresholds:
        pred = y_score >= threshold
        tp = (y_true & pred).sum(axis=1).astype(np.float64)
        pred_count = pred.sum(axis=1)
        has_prediction = pred_count > 0
        precision = float((tp[has_prediction] / pred_count[has_prediction]).mean()) if has_prediction.any() else 0.0
        recall = float((tp / y_true.sum(axis=1)).mean())
        fscore = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
    best = int(np.argmax(fscores))
    order = np.argsort(np.asarray(recalls))
    trapezoid = getattr(np, "trapezoid", np.trapz)
    aupr = float(trapezoid(np.asarray(precisions)[order], np.asarray(recalls)[order]))
    return {
        "fmax": float(fscores[best]),
        "aupr": aupr,
        "precision": float(precisions[best]),
        "recall": float(recalls[best]),
        "threshold": float(thresholds[best]),
        "proteins": int(y_true.shape[0]),
        "positive_annotations": int(y_true.sum()),
    }


def retrieval_metrics(
    ranked_go_ids: np.ndarray,
    true_sets: Sequence[Set[int]],
    temporal_terms: Sequence[int],
    ks: Iterable[int],
) -> Dict[str, float]:
    valid_rows = [idx for idx, truth in enumerate(true_sets) if truth]
    if not valid_rows:
        raise ValueError("No temporal ground-truth annotations found")
    total_true = sum(len(true_sets[idx]) for idx in valid_rows)
    output: Dict[str, float] = {"proteins": len(valid_rows), "true_annotations": total_true}
    temporal_terms = list(temporal_terms)
    for requested_k in ks:
        k = min(int(requested_k), ranked_go_ids.shape[1])
        found_total = 0
        protein_recalls = []
        covered = 0
        term_found = {term: 0 for term in temporal_terms}
        term_total = {term: 0 for term in temporal_terms}
        for row in valid_rows:
            truth = true_sets[row]
            retrieved = set(int(value) for value in ranked_go_ids[row, :k])
            found = truth & retrieved
            found_total += len(found)
            protein_recalls.append(len(found) / len(truth))
            covered += int(bool(found))
            for term in truth:
                term_total[term] += 1
                term_found[term] += int(term in retrieved)
        macro_terms = [term_found[t] / term_total[t] for t in temporal_terms if term_total[t] > 0]
        output[f"unseen_recall@{requested_k}"] = float(np.mean(protein_recalls))
        output[f"candidate_coverage@{requested_k}"] = covered / len(valid_rows)
        output[f"oracle_microF@{requested_k}"] = (2.0 * found_total) / max(1.0, 2.0 * found_total + (total_true - found_total))
        output[f"macro_term_recall@{requested_k}"] = float(np.mean(macro_terms)) if macro_terms else 0.0
        output[f"captured_annotations@{requested_k}"] = found_total
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--temporal-go-ids", type=Path, required=True)
    parser.add_argument("--pid-to-temporal-positives", type=Path, required=True)
    parser.add_argument("--score-kind", choices=("fused", "global", "local"), default="fused")
    parser.add_argument("--max-dump-k", type=int, default=0)
    parser.add_argument("--ks", type=int, nargs="+", default=(10, 50, 100, 200))
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dump = args.dump_dir
    eval_go_ids = np.load(dump / "eval_go_ids.npy", mmap_mode="r").astype(np.int64)
    top_cols = np.load(dump / "top_go_cols.int32.npy", mmap_mode="r")
    score_name = {
        "fused": "top_scores.float32.npy",
        "global": "top_global_scores.float32.npy",
        "local": "top_local_scores.float32.npy",
    }[args.score_kind]
    top_scores = np.load(dump / score_name, mmap_mode="r")
    protein_ids = json.loads((dump / "protein_ids.json").read_text(encoding="utf-8"))
    valid_path = dump / "top_valid.int8.npy"
    top_valid = np.load(valid_path, mmap_mode="r") if valid_path.exists() else None

    k = top_cols.shape[1] if args.max_dump_k <= 0 else min(args.max_dump_k, top_cols.shape[1])
    cols = np.asarray(top_cols[:, :k], dtype=np.int64)
    scores = np.asarray(top_scores[:, :k], dtype=np.float32)
    ranked_ids = eval_go_ids[cols]
    if top_valid is not None:
        valid = np.asarray(top_valid[:, :k], dtype=bool)
        ranked_ids = np.where(valid, ranked_ids, -1)
        scores = np.where(valid, scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

    temporal_terms = read_go_ids(args.temporal_go_ids)
    temporal_set = set(temporal_terms)
    true_sets = load_true_sets(args.pid_to_temporal_positives, protein_ids, temporal_set)
    id_to_temporal_col = {go_id: col for col, go_id in enumerate(temporal_terms)}
    y_true = np.zeros((len(protein_ids), len(temporal_terms)), dtype=np.int8)
    y_score = np.zeros((len(protein_ids), len(temporal_terms)), dtype=np.float32)
    for row, truth in enumerate(true_sets):
        for go_id in truth:
            y_true[row, id_to_temporal_col[go_id]] = 1
        for rank in range(k):
            go_id = int(ranked_ids[row, rank])
            col = id_to_temporal_col.get(go_id)
            if col is not None:
                y_score[row, col] = max(y_score[row, col], float(scores[row, rank]))

    result = {
        "protocol": "MZSGO pure temporal zero-shot",
        "score_kind": args.score_kind,
        "dump_dir": str(dump),
        "temporal_terms": len(temporal_terms),
        "dump_k": k,
        "mzsgo": mzsgo_protein_metrics(y_true, y_score, args.threshold_step),
        "retrieval": retrieval_metrics(ranked_ids, true_sets, temporal_terms, args.ks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
