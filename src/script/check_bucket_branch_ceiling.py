import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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

    # Handles strings like "GO_0008150" if they appear.
    if s.startswith("GO_"):
        s = s.split("_", 1)[1]

    # Handles accidental float-like strings.
    if "." in s:
        try:
            return int(float(s))
        except Exception:
            pass

    try:
        return int(s)
    except Exception:
        return None


def normalize_namespace(ns: Any) -> Optional[str]:
    if ns is None:
        return None

    s = str(ns).strip().lower()

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


def load_go_namespaces(go_basic_json: Optional[str | Path]) -> Dict[int, str]:
    """
    Robust namespace loader.

    Supports common formats:
      1. {"GO:0008150": {"namespace": "biological_process", ...}, ...}
      2. {"8150": {"namespace": "biological_process", ...}, ...}
      3. {"terms": [{"id": "GO:0008150", "namespace": "..."}]}
      4. [{"id": "GO:0008150", "namespace": "..."}]
    """
    if go_basic_json is None:
        return {}

    path = Path(go_basic_json)
    if not path.exists():
        raise FileNotFoundError(f"go_basic_json not found: {path}")

    data = load_json(path)
    out: Dict[int, str] = {}

    def add_term(term: Any, fallback_gid: Any = None):
        if not isinstance(term, dict):
            return

        raw_gid = (
            term.get("id")
            or term.get("go_id")
            or term.get("GO")
            or term.get("go")
            or fallback_gid
        )
        gid = go_to_int(raw_gid)

        ns = (
            term.get("namespace")
            or term.get("aspect")
            or term.get("branch")
            or term.get("ontology")
        )
        ns_norm = normalize_namespace(ns)

        if gid is not None and ns_norm is not None:
            out[int(gid)] = ns_norm

    if isinstance(data, list):
        for term in data:
            add_term(term)

    elif isinstance(data, dict):
        if isinstance(data.get("terms"), list):
            for term in data["terms"]:
                add_term(term)
        else:
            for k, v in data.items():
                if isinstance(v, dict):
                    add_term(v, fallback_gid=k)
                else:
                    gid = go_to_int(k)
                    ns_norm = normalize_namespace(v)
                    if gid is not None and ns_norm is not None:
                        out[int(gid)] = ns_norm

    return out


def load_true_rows(dump_dir: str | Path) -> List[List[int]]:
    dump_dir = Path(dump_dir)

    npy_path = dump_dir / "true_go_ids.npy"
    json_path = dump_dir / "true_go_ids.json"

    if npy_path.exists():
        arr = np.load(npy_path, mmap_mode="r")
        rows: List[List[int]] = []
        for row in arr:
            xs = []
            for x in row:
                gid = int(x)
                if gid >= 0:
                    xs.append(gid)
            rows.append(xs)
        return rows

    if json_path.exists():
        data = load_json(json_path)
        rows = []
        for row in data:
            xs = []
            for x in row:
                gid = go_to_int(x)
                if gid is not None and gid >= 0:
                    xs.append(int(gid))
            rows.append(xs)
        return rows

    raise FileNotFoundError(f"No true_go_ids.npy or true_go_ids.json found in {dump_dir}")


def load_candidate_arrays(dump_dir: str | Path, max_k: int):
    dump_dir = Path(dump_dir)

    eval_go_ids = np.load(dump_dir / "eval_go_ids.npy", mmap_mode="r").astype(np.int64)
    top_cols = np.load(dump_dir / "top_go_cols.int32.npy", mmap_mode="r")[:, :max_k].astype(np.int64)
    labels = np.load(dump_dir / "top_labels.int8.npy", mmap_mode="r")[:, :max_k].astype(np.int8)

    valid_path = dump_dir / "top_valid.int8.npy"
    if valid_path.exists():
        valid = np.load(valid_path, mmap_mode="r")[:, :max_k].astype(np.int8)
    else:
        valid = np.ones_like(labels, dtype=np.int8)

    # Convert candidate columns to global GO ids.
    top_ids = eval_go_ids[top_cols]

    # Ensure filler / invalid candidates do not count as hits.
    labels = (labels * valid).astype(np.int8)

    return eval_go_ids, top_cols, top_ids, labels, valid


def build_train_counts(train_dump: str | Path, eval_go_ids: np.ndarray) -> Dict[int, int]:
    """
    Count train proteins per GO term over eval_go_ids universe.
    Counts are protein-level, one count per protein per GO.
    """
    eval_set = set(int(x) for x in eval_go_ids.tolist())
    rows = load_true_rows(train_dump)

    counts: Dict[int, int] = {int(g): 0 for g in eval_go_ids.tolist()}

    for row in rows:
        uniq = set(int(g) for g in row if int(g) in eval_set)
        for gid in uniq:
            counts[gid] += 1

    return counts


def group_true_counts(true_rows: List[List[int]], group_ids: Set[int]) -> np.ndarray:
    out = np.zeros(len(true_rows), dtype=np.int32)
    for i, row in enumerate(true_rows):
        if row:
            out[i] = sum(1 for gid in row if int(gid) in group_ids)
    return out


def compute_group_metrics(
    *,
    group_name: str,
    group_ids: Set[int],
    eval_go_ids: np.ndarray,
    top_ids: np.ndarray,
    labels: np.ndarray,
    true_rows: List[List[int]],
    ks: Sequence[int],
) -> Dict[str, Any]:
    max_k = labels.shape[1]
    ks = [min(int(k), max_k) for k in ks]

    true_counts = group_true_counts(true_rows, group_ids)
    eligible = true_counts > 0

    term_count = int(sum(1 for gid in eval_go_ids.tolist() if int(gid) in group_ids))
    protein_count = int(eligible.sum())
    pos_total = int(true_counts.sum())

    out: Dict[str, Any] = {
        "group": group_name,
        "term_count_in_eval_space": term_count,
        "proteins_with_positive": protein_count,
        "positive_annotations": pos_total,
        "by_k": {},
    }

    if protein_count == 0 or pos_total == 0 or term_count == 0:
        for k in ks:
            out["by_k"][str(k)] = {
                "coverage_mean_nonempty": None,
                "any_hit_nonempty": None,
                "micro_recall": None,
                "oracle_microF": None,
                "hits": 0,
            }
        return out

    # Candidate GO ids inside this group.
    group_mask_all = np.isin(top_ids[:, : max(ks)], np.fromiter(group_ids, dtype=np.int64))

    for k in ks:
        group_mask = group_mask_all[:, :k]
        y = labels[:, :k].astype(np.int32)

        hits_per_row = (y * group_mask.astype(np.int32)).sum(axis=1)
        hits_eligible = hits_per_row[eligible]
        true_eligible = true_counts[eligible]

        hits_total = int(hits_eligible.sum())
        true_total = int(true_eligible.sum())
        fn_total = true_total - hits_total

        coverage_mean = float(np.mean(hits_eligible / np.maximum(true_eligible, 1)))
        any_hit = float(np.mean(hits_eligible > 0))
        micro_recall = float(hits_total / max(1, true_total))
        oracle_micro_f = float((2.0 * hits_total) / max(1e-12, 2.0 * hits_total + fn_total))

        out["by_k"][str(k)] = {
            "coverage_mean_nonempty": coverage_mean,
            "any_hit_nonempty": any_hit,
            "micro_recall": micro_recall,
            "oracle_microF": oracle_micro_f,
            "hits": hits_total,
        }

    return out


def format_float(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NA"
    return f"{float(x):.4f}"


def print_group_table(title: str, results: List[Dict[str, Any]], ks: Sequence[int]) -> None:
    print(f"\n[{title}]")
    header = [
        "group",
        "terms",
        "proteins",
        "positives",
    ]
    for k in ks:
        header += [f"cov@{k}", f"oracle@{k}", f"any@{k}"]

    print("\t".join(header))

    for r in results:
        row = [
            str(r["group"]),
            str(r["term_count_in_eval_space"]),
            str(r["proteins_with_positive"]),
            str(r["positive_annotations"]),
        ]
        for k in ks:
            m = r["by_k"][str(k)]
            row += [
                format_float(m["coverage_mean_nonempty"]),
                format_float(m["oracle_microF"]),
                format_float(m["any_hit_nonempty"]),
            ]
        print("\t".join(row))


def main():
    p = argparse.ArgumentParser("Check P3a candidate ceiling by namespace and frequency bucket.")
    p.add_argument("--dump_dir", type=str, required=True, help="Candidate dump to evaluate, e.g. P3a_val_top1000.")
    p.add_argument("--train_dump", type=str, required=True, help="Train dump used to compute GO train counts.")
    p.add_argument("--go_basic_json", type=str, default="/workspace/data/go_vocab.json")
    p.add_argument("--ks", type=int, nargs="+", default=[200, 500, 1000])
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--no_cross", action="store_true", help="Skip namespace x frequency bucket cross table.")
    args = p.parse_args()

    dump_dir = Path(args.dump_dir)
    train_dump = Path(args.train_dump)
    ks = sorted(set(int(k) for k in args.ks))
    max_k = max(ks)

    eval_go_ids, top_cols, top_ids, labels, valid = load_candidate_arrays(dump_dir, max_k=max_k)
    true_rows = load_true_rows(dump_dir)
    train_counts = build_train_counts(train_dump, eval_go_ids)
    namespace_map = load_go_namespaces(args.go_basic_json)

    if labels.shape[0] != len(true_rows):
        raise RuntimeError(f"labels rows {labels.shape[0]} != true rows {len(true_rows)}")

    eval_set: Set[int] = set(int(x) for x in eval_go_ids.tolist())

    # Overall.
    groups: Dict[str, Set[int]] = {
        "overall": set(eval_set),
    }

    # Namespaces.
    ns_groups: Dict[str, Set[int]] = {
        "MF": set(),
        "BP": set(),
        "CC": set(),
        "UNKNOWN_NS": set(),
    }

    for gid in eval_set:
        ns = namespace_map.get(int(gid))
        if ns in {"MF", "BP", "CC"}:
            ns_groups[ns].add(int(gid))
        else:
            ns_groups["UNKNOWN_NS"].add(int(gid))

    # Frequency buckets.
    zero = {gid for gid in eval_set if train_counts.get(gid, 0) == 0}
    few = {gid for gid in eval_set if 1 <= train_counts.get(gid, 0) < 20}
    mid = {gid for gid in eval_set if 20 <= train_counts.get(gid, 0) < 100}
    common = {gid for gid in eval_set if train_counts.get(gid, 0) >= 100}
    rare_lt20 = zero | few

    bucket_groups: Dict[str, Set[int]] = {
        "zero_count_0": zero,
        "few_1_19": few,
        "rare_lt20_including_zero": rare_lt20,
        "mid_20_99": mid,
        "common_100_plus": common,
    }

    print("\n[P3A BUCKET / BRANCH CEILING CHECK]")
    print("dump_dir:", str(dump_dir))
    print("train_dump:", str(train_dump))
    print("go_basic_json:", str(args.go_basic_json))
    print("n_proteins:", labels.shape[0])
    print("n_eval_go:", len(eval_go_ids))
    print("topk_available:", labels.shape[1])
    print("ks:", ks)
    print("namespace_map_size:", len(namespace_map))

    overall_results = [
        compute_group_metrics(
            group_name="overall",
            group_ids=groups["overall"],
            eval_go_ids=eval_go_ids,
            top_ids=top_ids,
            labels=labels,
            true_rows=true_rows,
            ks=ks,
        )
    ]

    ns_results = []
    for name in ["MF", "BP", "CC", "UNKNOWN_NS"]:
        ns_results.append(
            compute_group_metrics(
                group_name=name,
                group_ids=ns_groups[name],
                eval_go_ids=eval_go_ids,
                top_ids=top_ids,
                labels=labels,
                true_rows=true_rows,
                ks=ks,
            )
        )

    bucket_results = []
    for name in ["zero_count_0", "few_1_19", "rare_lt20_including_zero", "mid_20_99", "common_100_plus"]:
        bucket_results.append(
            compute_group_metrics(
                group_name=name,
                group_ids=bucket_groups[name],
                eval_go_ids=eval_go_ids,
                top_ids=top_ids,
                labels=labels,
                true_rows=true_rows,
                ks=ks,
            )
        )

    print_group_table("OVERALL", overall_results, ks)
    print_group_table("NAMESPACE", ns_results, ks)
    print_group_table("FREQUENCY_BUCKET", bucket_results, ks)

    cross_results = []
    if not args.no_cross:
        for ns_name in ["MF", "BP", "CC"]:
            for bucket_name in ["zero_count_0", "few_1_19", "mid_20_99", "common_100_plus"]:
                group_ids = ns_groups[ns_name] & bucket_groups[bucket_name]
                cross_results.append(
                    compute_group_metrics(
                        group_name=f"{ns_name}__{bucket_name}",
                        group_ids=group_ids,
                        eval_go_ids=eval_go_ids,
                        top_ids=top_ids,
                        labels=labels,
                        true_rows=true_rows,
                        ks=ks,
                    )
                )
        print_group_table("NAMESPACE_X_FREQUENCY_BUCKET", cross_results, ks)

    result = {
        "dump_dir": str(dump_dir),
        "train_dump": str(train_dump),
        "go_basic_json": str(args.go_basic_json),
        "n_proteins": int(labels.shape[0]),
        "n_eval_go": int(len(eval_go_ids)),
        "topk_available": int(labels.shape[1]),
        "ks": ks,
        "namespace_map_size": int(len(namespace_map)),
        "overall": overall_results,
        "namespace": ns_results,
        "frequency_bucket": bucket_results,
        "namespace_x_frequency_bucket": cross_results,
        "bucket_definition": {
            "zero_count_0": "GO terms with train_count == 0.",
            "few_1_19": "GO terms with 1 <= train_count < 20.",
            "rare_lt20_including_zero": "GO terms with train_count < 20, including zero-shot.",
            "mid_20_99": "GO terms with 20 <= train_count < 100.",
            "common_100_plus": "GO terms with train_count >= 100.",
        },
        "metric_definition": {
            "coverage_mean_nonempty": "Mean over proteins with at least one true label in the group: hits@K / true_count_group.",
            "any_hit_nonempty": "Fraction of proteins with at least one hit@K among proteins with at least one true label in the group.",
            "micro_recall": "Total hits@K divided by total true annotations in the group.",
            "oracle_microF": "2TP / (2TP + FN), assuming the oracle reranker selects only true candidates and no false positives.",
        },
    }

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("\nSaved:", str(out_path))


if __name__ == "__main__":
    main()