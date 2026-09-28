#!/usr/bin/env python3

"""
D3: Specificity-controlled retrieval generalization.

Question:
    Does the VALID -> TEST retrieval gap become larger for
    more specific / higher-information-content GO terms?

Inputs:
    1. VALID full-ranking retriever dump
    2. TEST full-ranking retriever dump
    3. GO information-content table produced by D2

Outputs:
    - gold_pair_ranks.csv
        One row per protein-GO gold pair.

    - specificity_summary.csv
        Retrieval performance by IC specificity bin and split.

    - specificity_gap.csv
        VALID vs TEST gap within each IC bin.

Important:
    IC bins are defined from TRAIN-derived propagated IC values.
    The script does NOT recompute IC from validation or test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KS = [50, 100, 200, 500, 1000]


def normalize_go_id(x) -> str:
    """
    Convert integer / string GO IDs to canonical GO:XXXXXXX strings.
    """
    if isinstance(x, (int, np.integer)):
        return f"GO:{int(x):07d}"

    x = str(x).strip()

    if x.startswith("GO:"):
        return x

    try:
        return f"GO:{int(x):07d}"
    except ValueError:
        return x


def load_dump(dump_dir: Path, split_name: str) -> pd.DataFrame:
    """
    Convert a full-ranking dump into one row per GOLD protein-GO pair.

    Required files:
        protein_ids.json
        true_go_ids.json
        top_go_ids.int64.npy

    Since top_go_ids is stored in descending retriever-score order:
        position 0 -> rank 1
        position 1 -> rank 2
        ...
    """

    dump_dir = Path(dump_dir)

    with open(dump_dir / "protein_ids.json", "r") as f:
        protein_ids = json.load(f)

    with open(dump_dir / "true_go_ids.json", "r") as f:
        true_go_ids = json.load(f)

    ranking = np.load(
        dump_dir / "top_go_ids.int64.npy",
        mmap_mode="r",
    )

    if len(protein_ids) != len(true_go_ids):
        raise RuntimeError(
            f"{split_name}: protein_ids and true_go_ids "
            f"have different lengths."
        )

    if len(protein_ids) != ranking.shape[0]:
        raise RuntimeError(
            f"{split_name}: protein count={len(protein_ids)} "
            f"but ranking rows={ranking.shape[0]}"
        )

    print(
        f"[{split_name}] proteins={len(protein_ids)} "
        f"ranking_width={ranking.shape[1]}"
    )

    rows = []

    missing_gold = 0
    total_gold = 0

    for i, pid in enumerate(protein_ids):

        golds = [
            int(g)
            for g in true_go_ids[i]
            if int(g) >= 0
        ]

        n_gold = len(golds)

        if n_gold == 0:
            continue

        # Full ranking for this protein.
        ranked_ids = np.asarray(
            ranking[i],
            dtype=np.int64,
        )

        # GO ID -> 1-based rank
        #
        # Full BP space is only ~1943 terms, so this is cheap
        # and keeps the logic transparent.
        rank_map = {
            int(go_id): rank + 1
            for rank, go_id in enumerate(ranked_ids)
        }

        for go_id in golds:
            total_gold += 1

            rank = rank_map.get(go_id)

            if rank is None:
                missing_gold += 1
                rank_value = np.nan
            else:
                rank_value = int(rank)

            row = {
                "protein_id": str(pid),
                "split": split_name,
                "go_id": normalize_go_id(go_id),
                "go_id_int": int(go_id),
                "n_gold": n_gold,
                "rank": rank_value,
            }

            for k in KS:
                row[f"hit@{k}"] = (
                    int(rank <= k)
                    if rank is not None
                    else 0
                )

            rows.append(row)

    print(
        f"[{split_name}] gold_pairs={total_gold} "
        f"missing_from_full_ranking={missing_gold}"
    )

    if missing_gold > 0:
        print(
            f"[WARN] {split_name}: {missing_gold} gold labels "
            "were not found in the full ranking."
        )

    return pd.DataFrame(rows)


def load_ic_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "go_id",
        "depth",
        "ic_propagated",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"IC table missing columns: {sorted(missing)}"
        )

    df = df.copy()

    df["go_id"] = df["go_id"].map(normalize_go_id)

    return df[
        [
            "go_id",
            "depth",
            "train_count_direct",
            "train_count_propagated",
            "ic_direct",
            "ic_propagated",
        ]
    ].drop_duplicates("go_id")


def assign_ic_quantiles(
        pair_df: pd.DataFrame,
        ic_df: pd.DataFrame,
        n_bins: int = 4,
):
    """
    Define specificity bins from the TRAIN-derived GO IC table.

    Important:
        We do NOT define quantile cutoffs separately on VALID and TEST.

        Otherwise the meaning of "high IC" would differ between splits.

    We use unique finite GO-level propagated IC values to define
    common global boundaries.
    """

    finite_ic = (
        ic_df["ic_propagated"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(finite_ic) == 0:
        raise RuntimeError(
            "No finite propagated IC values found."
        )

    quantiles = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    edges = finite_ic.quantile(
        quantiles
    ).to_numpy(dtype=float)

    # Ensure strictly increasing boundaries.
    edges = np.unique(edges)

    if len(edges) < 3:
        raise RuntimeError(
            f"Too few unique IC boundaries: {edges}"
        )

    # Make sure extreme values are included.
    edges[0] = -np.inf
    edges[-1] = np.inf

    n_actual_bins = len(edges) - 1

    if n_actual_bins == 4:
        labels = [
            "Q1_general",
            "Q2",
            "Q3",
            "Q4_specific",
        ]
    else:
        labels = [
            f"Q{i + 1}"
            for i in range(n_actual_bins)
        ]

    pair_df = pair_df.copy()

    pair_df["ic_bin"] = pd.cut(
        pair_df["ic_propagated"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    print("\nIC bin boundaries:")
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]

        print(
            f"  {labels[i]:12s}: "
            f"({lo:.4f}, {hi:.4f}]"
        )

    return pair_df, labels


def summarize_pairs(
        df: pd.DataFrame,
        bin_labels,
) -> pd.DataFrame:
    rows = []

    for split in ["valid", "test"]:

        split_df = df[
            df["split"] == split
            ]

        for ic_bin in bin_labels:

            sub = split_df[
                split_df["ic_bin"] == ic_bin
                ]

            if len(sub) == 0:
                continue

            valid_rank = sub["rank"].dropna()

            row = {
                "split": split,
                "ic_bin": str(ic_bin),

                "n_gold_pairs": len(sub),

                "n_proteins": (
                    sub["protein_id"]
                    .nunique()
                ),

                "mean_ic": (
                    sub["ic_propagated"]
                    .mean()
                ),

                "median_ic": (
                    sub["ic_propagated"]
                    .median()
                ),

                "mean_depth": (
                    sub["depth"]
                    .mean()
                ),

                "mean_n_gold": (
                    sub["n_gold"]
                    .mean()
                ),

                "median_n_gold": (
                    sub["n_gold"]
                    .median()
                ),

                "median_gold_rank": (
                    valid_rank.median()
                    if len(valid_rank)
                    else np.nan
                ),

                "mean_gold_rank": (
                    valid_rank.mean()
                    if len(valid_rank)
                    else np.nan
                ),

                "gold_mrr": (
                    (1.0 / valid_rank).mean()
                    if len(valid_rank)
                    else np.nan
                ),
            }

            for k in KS:
                row[f"recall@{k}"] = (
                    sub[f"hit@{k}"].mean()
                )

            rows.append(row)

    return pd.DataFrame(rows)


def make_gap_table(
        summary: pd.DataFrame,
) -> pd.DataFrame:
    valid = (
        summary[
            summary["split"] == "valid"
            ]
        .set_index("ic_bin")
    )

    test = (
        summary[
            summary["split"] == "test"
            ]
        .set_index("ic_bin")
    )

    common = [
        x
        for x in valid.index
        if x in test.index
    ]

    rows = []

    for ic_bin in common:

        v = valid.loc[ic_bin]
        t = test.loc[ic_bin]

        row = {
            "ic_bin": ic_bin,

            "valid_n_pairs":
                int(v["n_gold_pairs"]),

            "test_n_pairs":
                int(t["n_gold_pairs"]),

            "valid_mean_ic":
                float(v["mean_ic"]),

            "test_mean_ic":
                float(t["mean_ic"]),

            "valid_mean_n_gold":
                float(v["mean_n_gold"]),

            "test_mean_n_gold":
                float(t["mean_n_gold"]),

            "valid_median_rank":
                float(v["median_gold_rank"]),

            "test_median_rank":
                float(t["median_gold_rank"]),

            # Positive means TEST gold terms
            # are ranked lower / worse.
            "rank_gap_test_minus_valid":
                float(
                    t["median_gold_rank"]
                    - v["median_gold_rank"]
                ),
        }

        for k in KS:
            vr = float(v[f"recall@{k}"])
            tr = float(t[f"recall@{k}"])

            row[f"valid_R@{k}"] = vr
            row[f"test_R@{k}"] = tr

            # Positive = validation advantage.
            row[f"gap_R@{k}"] = vr - tr

        rows.append(row)

    return pd.DataFrame(rows)


def print_main_table(gap_df: pd.DataFrame):
    print("\n")
    print("=" * 100)
    print(
        "D3: SPECIFICITY-CONTROLLED "
        "VALID -> TEST GENERALIZATION"
    )
    print("=" * 100)

    cols = [
        "ic_bin",
        "valid_n_pairs",
        "test_n_pairs",
        "valid_mean_ic",
        "test_mean_ic",
        "valid_mean_n_gold",
        "test_mean_n_gold",
        "valid_R@50",
        "test_R@50",
        "gap_R@50",
        "valid_R@200",
        "test_R@200",
        "gap_R@200",
        "valid_R@500",
        "test_R@500",
        "gap_R@500",
        "valid_median_rank",
        "test_median_rank",
    ]

    print(
        gap_df[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--valid_dump",
        default=(
            "/workspace/data_pfresgo/diagnostics/"
            "coverage_top200_full_ranking_valid"
        ),
    )

    ap.add_argument(
        "--test_dump",
        default=(
            "/workspace/data_pfresgo/diagnostics/"
            "coverage_top200_full_ranking_test"
        ),
    )

    ap.add_argument(
        "--ic_csv",
        default=(
            "/workspace/data_pfresgo/diagnostics/"
            "go_specificity_shift_bp/"
            "go_information_content.csv"
        ),
    )

    ap.add_argument(
        "--outdir",
        default=(
            "/workspace/data_pfresgo/diagnostics/"
            "specificity_generalization_bp"
        ),
    )

    ap.add_argument(
        "--n_bins",
        type=int,
        default=4,
    )

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading full-ranking dumps...")

    valid_df = load_dump(
        Path(args.valid_dump),
        "valid",
    )

    test_df = load_dump(
        Path(args.test_dump),
        "test",
    )

    pair_df = pd.concat(
        [valid_df, test_df],
        ignore_index=True,
    )

    print("\nLoading train-derived GO information content...")

    ic_df = load_ic_table(
        Path(args.ic_csv)
    )

    pair_df = pair_df.merge(
        ic_df,
        on="go_id",
        how="left",
        validate="many_to_one",
    )

    n_missing_ic = int(
        pair_df["ic_propagated"]
        .isna()
        .sum()
    )

    print(
        f"Gold pairs without propagated IC: "
        f"{n_missing_ic}/{len(pair_df)}"
    )

    if n_missing_ic > 0:
        print(
            "[WARN] Gold pairs without train-derived IC "
            "cannot be assigned to an IC bin."
        )

    pair_df, bin_labels = assign_ic_quantiles(
        pair_df,
        ic_df,
        n_bins=args.n_bins,
    )

    summary_df = summarize_pairs(
        pair_df,
        bin_labels,
    )

    gap_df = make_gap_table(
        summary_df
    )

    pair_path = (
            outdir
            / "gold_pair_ranks_with_ic.csv"
    )

    summary_path = (
            outdir
            / "specificity_summary.csv"
    )

    gap_path = (
            outdir
            / "specificity_gap.csv"
    )

    pair_df.to_csv(
        pair_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    gap_df.to_csv(
        gap_path,
        index=False,
    )

    print_main_table(gap_df)

    print("\nSaved:")
    print(pair_path)
    print(summary_path)
    print(gap_path)


if __name__ == "__main__":
    main()