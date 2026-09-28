#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_TERMS = {
    "GO:0008150",  # biological_process
    "GO:0003674",  # molecular_function
    "GO:0005575",  # cellular_component
}


def normalize_go_id(x):
    """
    Supports:
      GO:0001234
      1234
      "1234"
    """
    if isinstance(x, int):
        return f"GO:{x:07d}"

    x = str(x).strip()

    if x.startswith("GO:"):
        return x

    try:
        return f"GO:{int(x):07d}"
    except ValueError:
        return x


def load_split(path):
    with open(path) as f:
        return [
            line.strip().split()[0]
            for line in f
            if line.strip()
        ]


def load_pid_to_positives(path):
    with open(path) as f:
        raw = json.load(f)

    out = {}

    for pid, gos in raw.items():
        out[str(pid)] = {
            normalize_go_id(g)
            for g in gos
        }

    return out


def load_go_obo(path):
    """
    Minimal OBO parser.

    Keeps:
      id
      namespace
      is_a parents

    For this first diagnostic we deliberately use is_a only.
    """

    parents = defaultdict(set)
    namespace = {}

    current_id = None
    obsolete = False

    def reset():
        return None, False

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()

            if line == "[Term]":
                current_id, obsolete = reset()
                continue

            if not line:
                continue

            if line.startswith("id: GO:"):
                current_id = line.split("id:", 1)[1].strip()
                continue

            if current_id is None:
                continue

            if line == "is_obsolete: true":
                obsolete = True
                continue

            if obsolete:
                continue

            if line.startswith("namespace:"):
                namespace[current_id] = (
                    line.split("namespace:", 1)[1].strip()
                )

            elif line.startswith("is_a: GO:"):
                parent = line.split("is_a:", 1)[1].split()[0]
                parents[current_id].add(parent)

    return dict(parents), namespace


def get_ancestors(go_id, parents, cache):
    if go_id in cache:
        return cache[go_id]

    result = set()
    stack = list(parents.get(go_id, []))

    while stack:
        p = stack.pop()

        if p in result:
            continue

        result.add(p)
        stack.extend(parents.get(p, []))

    cache[go_id] = result
    return result


def compute_depths(parents):
    """
    Minimum is_a distance from a GO root.

    Root depth = 0.
    """

    memo = {}

    def depth(go_id, visiting=None):
        if go_id in memo:
            return memo[go_id]

        if go_id in ROOT_TERMS:
            memo[go_id] = 0
            return 0

        if visiting is None:
            visiting = set()

        if go_id in visiting:
            return np.nan

        ps = parents.get(go_id, set())

        if not ps:
            return np.nan

        visiting = visiting | {go_id}

        parent_depths = [
            depth(p, visiting)
            for p in ps
        ]

        parent_depths = [
            d for d in parent_depths
            if not np.isnan(d)
        ]

        if not parent_depths:
            return np.nan

        value = 1 + min(parent_depths)
        memo[go_id] = value
        return value

    all_terms = set(parents.keys())

    for ps in parents.values():
        all_terms.update(ps)

    all_terms.update(ROOT_TERMS)

    return {
        go: depth(go)
        for go in all_terms
    }


def compute_train_ic(
        train_pids,
        pid2pos,
        parents,
):
    """
    Returns:
      direct_ic
      propagated_ic
      direct_counts
      propagated_counts

    Frequencies are based ONLY on train proteins.
    """

    direct_counts = defaultdict(int)
    propagated_counts = defaultdict(int)

    ancestor_cache = {}

    valid_train = [
        pid for pid in train_pids
        if pid in pid2pos
    ]

    n_train = len(valid_train)

    if n_train == 0:
        raise RuntimeError("No train proteins found in pid2pos.")

    for pid in valid_train:
        direct = set(pid2pos[pid])

        for go in direct:
            direct_counts[go] += 1

        propagated = set(direct)

        for go in direct:
            propagated.update(
                get_ancestors(
                    go,
                    parents,
                    ancestor_cache,
                )
            )

        for go in propagated:
            propagated_counts[go] += 1

    def counts_to_ic(counts):
        ic = {}

        for go, count in counts.items():
            p = count / n_train

            if p > 0:
                ic[go] = -math.log(p)

        return ic

    return (
        counts_to_ic(direct_counts),
        counts_to_ic(propagated_counts),
        dict(direct_counts),
        dict(propagated_counts),
    )


def safe_stats(values):
    values = [
        float(v)
        for v in values
        if v is not None and np.isfinite(v)
    ]

    if not values:
        return {
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    arr = np.asarray(values)

    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_protein_rows(
        split_name,
        pids,
        pid2pos,
        depths,
        direct_ic,
        propagated_ic,
        namespace,
):
    rows = []

    missing_pid = 0

    for pid in pids:
        gos = pid2pos.get(pid)

        if not gos:
            missing_pid += 1
            continue

        gos = sorted(set(gos))

        depth_values = [
            depths.get(go, np.nan)
            for go in gos
        ]

        direct_values = [
            direct_ic.get(go, np.nan)
            for go in gos
        ]

        propagated_values = [
            propagated_ic.get(go, np.nan)
            for go in gos
        ]

        depth_stats = safe_stats(depth_values)
        direct_stats = safe_stats(direct_values)
        prop_stats = safe_stats(propagated_values)

        namespace_counts = defaultdict(int)

        for go in gos:
            namespace_counts[
                namespace.get(go, "unknown")
            ] += 1

        rows.append(
            {
                "protein_id": pid,
                "split": split_name,
                "n_go": len(gos),

                "mean_depth": depth_stats["mean"],
                "median_depth": depth_stats["median"],
                "min_depth": depth_stats["min"],
                "max_depth": depth_stats["max"],

                "mean_ic_direct": direct_stats["mean"],
                "median_ic_direct": direct_stats["median"],
                "total_ic_direct": float(
                    np.nansum(direct_values)
                ),

                "mean_ic_propagated": prop_stats["mean"],
                "median_ic_propagated": prop_stats["median"],
                "total_ic_propagated": float(
                    np.nansum(propagated_values)
                ),

                "n_bp": namespace_counts[
                    "biological_process"
                ],
                "n_mf": namespace_counts[
                    "molecular_function"
                ],
                "n_cc": namespace_counts[
                    "cellular_component"
                ],
            }
        )

    print(
        f"[{split_name}] proteins={len(rows)} "
        f"missing/no-label={missing_pid}"
    )

    return rows


def summarize(df):
    metrics = [
        "n_go",
        "mean_depth",
        "median_depth",
        "mean_ic_direct",
        "median_ic_direct",
        "total_ic_direct",
        "mean_ic_propagated",
        "median_ic_propagated",
        "total_ic_propagated",
    ]

    rows = []

    for split, sub in df.groupby("split"):
        row = {
            "split": split,
            "n_proteins": len(sub),
        }

        for metric in metrics:
            values = sub[metric].dropna()

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_median"] = values.median()

        # Is cardinality merely tracking semantic load?
        valid = sub[
            ["n_go", "total_ic_propagated"]
        ].dropna()

        if len(valid) > 2:
            row["corr_n_go_total_ic"] = (
                valid["n_go"].corr(
                    valid["total_ic_propagated"],
                    method="spearman",
                )
            )
        else:
            row["corr_n_go_total_ic"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pid2pos",
        default="/workspace/data_pfresgo/processed/pid_to_positives_bp.json",
    )

    ap.add_argument(
        "--train",
        default="/workspace/stargo/datasets/pfresgo/train.txt",
    )

    ap.add_argument(
        "--valid",
        default="/workspace/stargo/datasets/pfresgo/valid.txt",
    )

    ap.add_argument(
        "--test",
        default="/workspace/stargo/datasets/pfresgo/test.txt",
    )

    ap.add_argument(
        "--obo",
        required=True,
        help="Path to the GO .obo file used for this benchmark.",
    )

    ap.add_argument(
        "--outdir",
        default="/workspace/data_pfresgo/diagnostics/go_specificity_shift_bp",
    )

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    pid2pos = load_pid_to_positives(args.pid2pos)

    splits = {
        "train": load_split(args.train),
        "valid": load_split(args.valid),
        "test": load_split(args.test),
    }

    print("Loading GO ontology...")

    parents, namespace = load_go_obo(args.obo)

    print("Computing GO depths...")

    depths = compute_depths(parents)

    print("Computing train-derived IC...")

    (
        direct_ic,
        propagated_ic,
        direct_counts,
        propagated_counts,
    ) = compute_train_ic(
        splits["train"],
        pid2pos,
        parents,
    )

    # GO-level table
    all_go = (
            set(direct_counts)
            | set(propagated_counts)
            | set(depths)
    )

    go_rows = []

    for go in sorted(all_go):
        go_rows.append(
            {
                "go_id": go,
                "namespace": namespace.get(
                    go,
                    "unknown",
                ),
                "depth": depths.get(go, np.nan),
                "train_count_direct":
                    direct_counts.get(go, 0),
                "train_count_propagated":
                    propagated_counts.get(go, 0),
                "ic_direct":
                    direct_ic.get(go, np.nan),
                "ic_propagated":
                    propagated_ic.get(go, np.nan),
            }
        )

    go_df = pd.DataFrame(go_rows)

    go_path = outdir / "go_information_content.csv"
    go_df.to_csv(go_path, index=False)

    # Protein-level table
    rows = []

    for split_name, pids in splits.items():
        rows.extend(
            build_protein_rows(
                split_name,
                pids,
                pid2pos,
                depths,
                direct_ic,
                propagated_ic,
                namespace,
            )
        )

    protein_df = pd.DataFrame(rows)

    protein_path = (
            outdir
            / "protein_annotation_specificity.csv"
    )

    protein_df.to_csv(
        protein_path,
        index=False,
    )

    summary_df = summarize(protein_df)

    summary_path = outdir / "split_summary.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n==============================")
    print("SPLIT SUMMARY")
    print("==============================")

    cols = [
        "split",
        "n_proteins",
        "n_go_mean",
        "n_go_median",
        "mean_depth_mean",
        "mean_ic_propagated_mean",
        "total_ic_propagated_mean",
        "total_ic_propagated_median",
        "corr_n_go_total_ic",
    ]

    print(
        summary_df[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved:")
    print(protein_path)
    print(go_path)
    print(summary_path)


if __name__ == "__main__":
    main()