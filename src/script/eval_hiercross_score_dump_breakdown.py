from __future__ import annotations

"""
Compute branch and frequency breakdowns from a saved HierCross score dump.

Input should be the out_dir produced by eval_hiercross_checkpoint.py.

Recommended use:
  python -m src.script.eval_hiercross_score_dump_breakdown \
    --score_dump_dir /workspace/results/P3a_HierCross_best \
    --train_dump /workspace/candidate_dumps/P3a_train_top1000 \
    --go_basic_json /workspace/data/go_vocab.json \
    --out_json /workspace/results/P3a_HierCross_best/metrics_breakdown.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def go_to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    if not s:
        return None
    if s.startswith("GO:"):
        s = s.split(":", 1)[1]
    if s.startswith("GO_"):
        s = s.split("_", 1)[1]
    try:
        return int(s)
    except Exception:
        return None


def norm_ns(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"mf", "mfo", "molecular_function", "molecular function"}:
        return "MF"
    if s in {"bp", "bpo", "biological_process", "biological process"}:
        return "BP"
    if s in {"cc", "cco", "cellular_component", "cellular component"}:
        return "CC"
    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_go_namespaces(go_basic_json: str | Path) -> Dict[int, str]:
    data = load_json(Path(go_basic_json))
    out: Dict[int, str] = {}

    def add(term: Any, fallback_gid: Any = None) -> None:
        if isinstance(term, dict):
            gid = go_to_int(term.get("id") or term.get("go_id") or term.get("GO") or term.get("go") or fallback_gid)
            ns = norm_ns(term.get("namespace") or term.get("aspect") or term.get("branch") or term.get("ontology"))
        else:
            gid = go_to_int(fallback_gid)
            ns = norm_ns(term)
        if gid is not None and ns is not None:
            out[int(gid)] = ns

    if isinstance(data, list):
        for term in data:
            add(term)
    elif isinstance(data, dict):
        if isinstance(data.get("terms"), list):
            for term in data["terms"]:
                add(term)
        else:
            for k, v in data.items():
                add(v, fallback_gid=k)
    return out


def load_true_rows(dump_dir: str | Path) -> List[List[int]]:
    dump_dir = Path(dump_dir)
    npy = dump_dir / "true_go_ids.npy"
    js = dump_dir / "true_go_ids.json"
    if npy.exists():
        arr = np.load(npy, mmap_mode="r")
        rows: List[List[int]] = []
        for row in arr:
            rows.append([int(x) for x in row if int(x) >= 0])
        return rows
    if js.exists():
        data = load_json(js)
        return [[int(x) for x in row if int(x) >= 0] for row in data]
    raise FileNotFoundError(f"Missing true_go_ids.npy/json in {dump_dir}")


def build_train_counts(train_dump: str | Path, eval_go_ids: np.ndarray) -> Dict[int, int]:
    eval_set = set(int(x) for x in eval_go_ids.tolist())
    counts = {int(g): 0 for g in eval_go_ids.tolist()}
    for row in load_true_rows(train_dump):
        for gid in set(int(x) for x in row if int(x) in eval_set):
            counts[int(gid)] += 1
    return counts


def compute_global_fmax_aupr_from_items(items: List[Tuple[float, int]], n_true_total: int) -> Dict[str, float]:
    if n_true_total <= 0 or len(items) == 0:
        return {"fmax": 0.0, "aupr": 0.0, "precision": 0.0, "recall": 0.0}
    items.sort(key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    best_f = 0.0
    best_p = 0.0
    best_r = 0.0
    aupr = 0.0
    prev_rec = 0.0
    for score, is_true in items:
        if int(is_true) == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, n_true_total)
        f = 0.0 if (prec + rec) == 0 else 2.0 * prec * rec / (prec + rec)
        if f > best_f:
            best_f = f
            best_p = prec
            best_r = rec
        dr = rec - prev_rec
        if dr > 0:
            aupr += prec * dr
            prev_rec = rec
    return {"fmax": float(best_f), "aupr": float(aupr), "precision": float(best_p), "recall": float(best_r)}


def true_count_for_group(true_go_ids: np.ndarray, group_ids: Set[int]) -> np.ndarray:
    out = np.zeros(true_go_ids.shape[0], dtype=np.int32)
    for i, row in enumerate(true_go_ids):
        out[i] = sum(1 for x in row if int(x) >= 0 and int(x) in group_ids)
    return out


def group_metrics(
    *,
    name: str,
    group_ids: Set[int],
    scores: np.ndarray,
    cand_ids: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    true_go_ids: np.ndarray,
) -> Dict[str, Any]:
    if not group_ids:
        return {
            "group": name,
            "term_count": 0,
            "proteins_with_positive": 0,
            "positive_annotations": 0,
            "fmax_topk": None,
            "aupr_topk": None,
            "fmax_full": None,
            "aupr_full": None,
            "retrieval_recall@K": None,
            "oracle_microF@K": None,
        }

    true_counts = true_count_for_group(true_go_ids, group_ids)
    n_true_full = int(true_counts.sum())
    proteins_with_pos = int((true_counts > 0).sum())

    mask_group = np.isin(cand_ids, np.fromiter(group_ids, dtype=np.int64))
    mask = (valid > 0) & mask_group
    n_true_candidate = int(((labels > 0) & mask).sum())

    items: List[Tuple[float, int]] = []
    rows, cols = np.where(mask)
    for i, j in zip(rows.tolist(), cols.tolist()):
        items.append((float(scores[i, j]), int(labels[i, j])))

    m_topk = compute_global_fmax_aupr_from_items(items.copy(), n_true_candidate)
    m_full = compute_global_fmax_aupr_from_items(items.copy(), n_true_full)
    recall = float(n_true_candidate) / max(1.0, float(n_true_full))
    oracle = float((2.0 * n_true_candidate) / max(1e-12, 2.0 * n_true_candidate + (n_true_full - n_true_candidate)))

    return {
        "group": name,
        "term_count": int(len(group_ids)),
        "proteins_with_positive": proteins_with_pos,
        "positive_annotations": n_true_full,
        "n_true_candidate": n_true_candidate,
        "n_pairs_scored": int(len(items)),
        "retrieval_recall@K": recall,
        "oracle_microF@K": oracle,
        "fmax_topk": m_topk["fmax"],
        "aupr_topk": m_topk["aupr"],
        "fmax_full": m_full["fmax"],
        "aupr_full": m_full["aupr"],
        "precision_full_at_fmax": m_full["precision"],
        "recall_full_at_fmax": m_full["recall"],
    }


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return ""
        return f"{x:.4f}"
    return str(x)


def print_table(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    print(f"\n[{title}]")
    cols = [
        "group", "term_count", "proteins_with_positive", "positive_annotations",
        "retrieval_recall@K", "oracle_microF@K", "fmax_topk", "aupr_topk", "fmax_full", "aupr_full",
    ]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(fmt(r.get(c)) for c in cols))


def main() -> None:
    parser = argparse.ArgumentParser("Break down HierCross score dump by branch and frequency.")
    parser.add_argument("--score_dump_dir", type=str, required=True)
    parser.add_argument("--train_dump", type=str, required=True)
    parser.add_argument("--go_basic_json", type=str, default="/workspace/data/go_vocab.json")
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--out_tsv", type=str, default=None)
    args = parser.parse_args()

    d = Path(args.score_dump_dir)
    scores = np.load(d / "scores.float32.npy", mmap_mode="r").astype(np.float32)
    cand_ids = np.load(d / "cand_ids.int64.npy", mmap_mode="r").astype(np.int64)
    labels = np.load(d / "labels.int8.npy", mmap_mode="r").astype(np.int8)
    valid = np.load(d / "valid.int8.npy", mmap_mode="r").astype(np.int8)
    true_go_ids = np.load(d / "true_go_ids.npy", mmap_mode="r").astype(np.int64)
    eval_go_ids = np.load(d / "eval_go_ids.npy", mmap_mode="r").astype(np.int64)

    eval_set = set(int(x) for x in eval_go_ids.tolist())
    ns_map = load_go_namespaces(args.go_basic_json)
    counts = build_train_counts(args.train_dump, eval_go_ids)

    groups: Dict[str, Set[int]] = {}
    groups["overall"] = set(eval_set)
    groups["MF"] = {g for g in eval_set if ns_map.get(g) == "MF"}
    groups["BP"] = {g for g in eval_set if ns_map.get(g) == "BP"}
    groups["CC"] = {g for g in eval_set if ns_map.get(g) == "CC"}

    groups["zero_count_0"] = {g for g in eval_set if counts.get(g, 0) == 0}
    groups["few_1_19"] = {g for g in eval_set if 1 <= counts.get(g, 0) < 20}
    groups["rare_lt20_including_zero"] = {g for g in eval_set if counts.get(g, 0) < 20}
    groups["mid_20_99"] = {g for g in eval_set if 20 <= counts.get(g, 0) < 100}
    groups["common_100_plus"] = {g for g in eval_set if counts.get(g, 0) >= 100}

    namespace_rows = [
        group_metrics(name=x, group_ids=groups[x], scores=scores, cand_ids=cand_ids, labels=labels, valid=valid, true_go_ids=true_go_ids)
        for x in ["overall", "MF", "BP", "CC"]
    ]
    frequency_rows = [
        group_metrics(name=x, group_ids=groups[x], scores=scores, cand_ids=cand_ids, labels=labels, valid=valid, true_go_ids=true_go_ids)
        for x in ["zero_count_0", "few_1_19", "rare_lt20_including_zero", "mid_20_99", "common_100_plus"]
    ]

    cross_rows: List[Dict[str, Any]] = []
    for ns in ["MF", "BP", "CC"]:
        for bucket in ["zero_count_0", "few_1_19", "mid_20_99", "common_100_plus"]:
            nm = f"{ns}__{bucket}"
            cross_rows.append(
                group_metrics(name=nm, group_ids=groups[ns] & groups[bucket], scores=scores, cand_ids=cand_ids, labels=labels, valid=valid, true_go_ids=true_go_ids)
            )

    # Branch macro uses full-space Fmax over MF/BP/CC.
    branch_vals = [r["fmax_full"] for r in namespace_rows if r["group"] in {"MF", "BP", "CC"} and r["fmax_full"] is not None]
    summary = {
        "score_dump_dir": str(d),
        "train_dump": str(args.train_dump),
        "n_proteins": int(scores.shape[0]),
        "topk": int(scores.shape[1]),
        "branch_macro_fmax_full": float(np.mean(branch_vals)) if branch_vals else None,
        "branch_min_fmax_full": float(np.min(branch_vals)) if branch_vals else None,
        "namespace": namespace_rows,
        "frequency": frequency_rows,
        "namespace_x_frequency": cross_rows,
    }

    print_table("NAMESPACE", namespace_rows)
    print_table("FREQUENCY", frequency_rows)
    print_table("NAMESPACE_X_FREQUENCY", cross_rows)
    print("\n[SUMMARY]")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"namespace", "frequency", "namespace_x_frequency"}}, indent=2))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if args.out_tsv:
        out = Path(args.out_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "section", "group", "term_count", "proteins_with_positive", "positive_annotations",
            "retrieval_recall@K", "oracle_microF@K", "fmax_topk", "aupr_topk", "fmax_full", "aupr_full",
        ]
        with out.open("w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for section, rows in [("namespace", namespace_rows), ("frequency", frequency_rows), ("namespace_x_frequency", cross_rows)]:
                for r in rows:
                    f.write("\t".join(fmt(section if c == "section" else r.get(c)) for c in cols) + "\n")


if __name__ == "__main__":
    main()
