from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CARD_BINS = [
    (1, 5, "1-5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 40, "21-40"),
    (41, 80, "41-80"),
    (81, 160, "81-160"),
    (161, 10 ** 9, "161+"),
]


def parse_args():
    p = argparse.ArgumentParser(
        "Analyze train/val/test candidate distributions "
        "and create a test-matched validation subset."
    )

    p.add_argument("--train_dump", required=True)
    p.add_argument("--val_dump", required=True)
    p.add_argument("--test_dump", required=True)

    p.add_argument(
        "--out_dir",
        required=True,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Optional fixed matched-val size.
    # If omitted, script uses the largest feasible subset
    # that preserves the test cardinality proportions.
    p.add_argument(
        "--matched_val_size",
        type=int,
        default=None,
    )

    return p.parse_args()


def cardinality_bin(n: int) -> str:
    for lo, hi, name in CARD_BINS:
        if lo <= n <= hi:
            return name
    return "0"


def load_dump(
        dump_dir: str | Path,
        split: str,
) -> pd.DataFrame:
    dump_dir = Path(dump_dir)

    with (dump_dir / "protein_ids.json").open(
            "r",
            encoding="utf-8",
    ) as f:
        protein_ids = [
            str(x)
            for x in json.load(f)
        ]

    true_go_ids = np.load(
        dump_dir / "true_go_ids.npy",
        mmap_mode="r",
    )

    top_labels = np.load(
        dump_dir / "top_labels.int8.npy",
        mmap_mode="r",
    )

    if len(protein_ids) != true_go_ids.shape[0]:
        raise RuntimeError(
            f"{split}: protein_ids={len(protein_ids)} "
            f"but true_go_ids rows={true_go_ids.shape[0]}"
        )

    if top_labels.shape[0] != len(protein_ids):
        raise RuntimeError(
            f"{split}: top_labels rows={top_labels.shape[0]} "
            f"but proteins={len(protein_ids)}"
        )

    rows = []

    for i, pid in enumerate(protein_ids):

        true_row = np.asarray(
            true_go_ids[i],
            dtype=np.int64,
        )

        full_cardinality = int(
            np.sum(true_row >= 0)
        )

        candidate_positives = int(
            np.asarray(
                top_labels[i],
                dtype=np.int64,
            ).sum()
        )

        if full_cardinality > 0:
            candidate_recall = (
                    candidate_positives
                    / full_cardinality
            )
        else:
            candidate_recall = np.nan

        rows.append(
            {
                "split": split,
                "row_index": i,
                "protein_id": pid,
                "full_cardinality": full_cardinality,
                "candidate_positives": candidate_positives,
                "candidate_recall": candidate_recall,
                "cardinality_bin": cardinality_bin(
                    full_cardinality
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_split(
        df: pd.DataFrame,
) -> dict:
    valid = df[
        df["full_cardinality"] > 0
        ].copy()

    summary = {
        "n_proteins": int(len(df)),
        "n_valid_proteins": int(len(valid)),
        "full_cardinality": {
            "mean": float(
                valid["full_cardinality"].mean()
            ),
            "median": float(
                valid["full_cardinality"].median()
            ),
            "p25": float(
                valid["full_cardinality"].quantile(0.25)
            ),
            "p75": float(
                valid["full_cardinality"].quantile(0.75)
            ),
            "p90": float(
                valid["full_cardinality"].quantile(0.90)
            ),
        },
        "candidate_positives": {
            "mean": float(
                valid["candidate_positives"].mean()
            ),
            "median": float(
                valid["candidate_positives"].median()
            ),
            "p25": float(
                valid["candidate_positives"].quantile(0.25)
            ),
            "p75": float(
                valid["candidate_positives"].quantile(0.75)
            ),
            "p90": float(
                valid["candidate_positives"].quantile(0.90)
            ),
        },
        "candidate_recall": {
            "macro_mean": float(
                valid["candidate_recall"].mean()
            ),
            "median": float(
                valid["candidate_recall"].median()
            ),
        },
    }

    total_true = int(
        valid["full_cardinality"].sum()
    )

    total_cand_true = int(
        valid["candidate_positives"].sum()
    )

    summary[
        "candidate_recall"
    ][
        "micro"
    ] = (
        total_cand_true / total_true
        if total_true > 0
        else 0.0
    )

    return summary


def summarize_bins(
        df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    total = len(df)

    for _, _, bin_name in CARD_BINS:

        x = df[
            df["cardinality_bin"] == bin_name
            ]

        if len(x) == 0:
            rows.append(
                {
                    "cardinality_bin": bin_name,
                    "N": 0,
                    "fraction": 0.0,
                    "mean_full_cardinality": np.nan,
                    "mean_candidate_positives": np.nan,
                    "macro_candidate_recall": np.nan,
                    "micro_candidate_recall": np.nan,
                }
            )
            continue

        full_sum = x[
            "full_cardinality"
        ].sum()

        cand_sum = x[
            "candidate_positives"
        ].sum()

        rows.append(
            {
                "cardinality_bin": bin_name,
                "N": int(len(x)),
                "fraction": float(
                    len(x) / max(1, total)
                ),
                "mean_full_cardinality": float(
                    x["full_cardinality"].mean()
                ),
                "mean_candidate_positives": float(
                    x["candidate_positives"].mean()
                ),
                "macro_candidate_recall": float(
                    x["candidate_recall"].mean()
                ),
                "micro_candidate_recall": float(
                    cand_sum / full_sum
                )
                if full_sum > 0
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def determine_matched_size(
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
) -> int:
    """
    Largest validation subset that can approximately
    preserve the test cardinality-bin proportions.
    """

    test_counts = (
        test_df[
            "cardinality_bin"
        ]
        .value_counts()
        .to_dict()
    )

    val_counts = (
        val_df[
            "cardinality_bin"
        ]
        .value_counts()
        .to_dict()
    )

    test_total = len(test_df)

    possible_sizes = []

    for _, _, bin_name in CARD_BINS:

        t = int(
            test_counts.get(
                bin_name,
                0,
            )
        )

        v = int(
            val_counts.get(
                bin_name,
                0,
            )
        )

        if t <= 0:
            continue

        p = t / test_total

        if p <= 0:
            continue

        # n * p <= v
        possible_sizes.append(
            int(np.floor(v / p))
        )

    if not possible_sizes:
        raise RuntimeError(
            "Could not determine a matched "
            "validation subset size."
        )

    return min(
        min(possible_sizes),
        len(val_df),
    )


def make_matched_validation(
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_size: int,
        seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    test_counts = (
        test_df[
            "cardinality_bin"
        ]
        .value_counts()
        .to_dict()
    )

    test_total = len(test_df)

    selected = []

    print("\nMatched validation sampling:")
    print(
        f"{'Bin':<12}"
        f"{'Test frac':>12}"
        f"{'Target N':>12}"
        f"{'Available':>12}"
        f"{'Selected':>12}"
    )

    for _, _, bin_name in CARD_BINS:

        test_n = int(
            test_counts.get(
                bin_name,
                0,
            )
        )

        test_frac = (
            test_n / test_total
            if test_total > 0
            else 0.0
        )

        target_n = int(
            round(
                target_size
                * test_frac
            )
        )

        candidates = val_df[
            val_df[
                "cardinality_bin"
            ] == bin_name
            ]

        available = len(candidates)

        n_select = min(
            target_n,
            available,
        )

        if n_select > 0:
            chosen_positions = rng.choice(
                available,
                size=n_select,
                replace=False,
            )

            chosen = candidates.iloc[
                chosen_positions
            ].copy()

            selected.append(chosen)

        print(
            f"{bin_name:<12}"
            f"{test_frac:>12.4f}"
            f"{target_n:>12}"
            f"{available:>12}"
            f"{n_select:>12}"
        )

    if not selected:
        raise RuntimeError(
            "Matched validation subset is empty."
        )

    matched = pd.concat(
        selected,
        ignore_index=True,
    )

    # deterministic shuffle
    matched = matched.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return matched


def print_summary(
        name: str,
        summary: dict,
):
    print(
        "\n========================================"
    )
    print(name.upper())
    print(
        "========================================"
    )

    print(
        "proteins:",
        summary["n_proteins"],
    )

    print(
        "full cardinality "
        f"mean={summary['full_cardinality']['mean']:.2f} "
        f"median={summary['full_cardinality']['median']:.2f} "
        f"p75={summary['full_cardinality']['p75']:.2f} "
        f"p90={summary['full_cardinality']['p90']:.2f}"
    )

    print(
        "candidate positives "
        f"mean={summary['candidate_positives']['mean']:.2f} "
        f"median={summary['candidate_positives']['median']:.2f}"
    )

    print(
        "candidate recall "
        f"macro={summary['candidate_recall']['macro_mean']:.4f} "
        f"micro={summary['candidate_recall']['micro']:.4f}"
    )


def main():
    args = parse_args()

    out_dir = Path(
        args.out_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading dumps...")

    train_df = load_dump(
        args.train_dump,
        "train",
    )

    val_df = load_dump(
        args.val_dump,
        "val",
    )

    test_df = load_dump(
        args.test_dump,
        "test",
    )

    # ------------------------------------------------
    # Per-protein files
    # ------------------------------------------------

    train_df.to_csv(
        out_dir / "train_per_protein.csv",
        index=False,
    )

    val_df.to_csv(
        out_dir / "val_per_protein.csv",
        index=False,
    )

    test_df.to_csv(
        out_dir / "test_per_protein.csv",
        index=False,
    )

    # ------------------------------------------------
    # Overall summaries
    # ------------------------------------------------

    summaries = {
        "train": summarize_split(
            train_df
        ),
        "val": summarize_split(
            val_df
        ),
        "test": summarize_split(
            test_df
        ),
    }

    with (
            out_dir
            / "distribution_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summaries,
            f,
            indent=2,
        )

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        print_summary(
            split_name,
            summaries[split_name],
        )

    # ------------------------------------------------
    # Cardinality-bin summaries
    # ------------------------------------------------

    train_bins = summarize_bins(
        train_df
    )
    val_bins = summarize_bins(
        val_df
    )
    test_bins = summarize_bins(
        test_df
    )

    train_bins.to_csv(
        out_dir / "train_cardinality_bins.csv",
        index=False,
    )

    val_bins.to_csv(
        out_dir / "val_cardinality_bins.csv",
        index=False,
    )

    test_bins.to_csv(
        out_dir / "test_cardinality_bins.csv",
        index=False,
    )

    print(
        "\n\n========================================"
    )
    print("VAL VS TEST CARDINALITY DISTRIBUTION")
    print(
        "========================================\n"
    )

    comparison = val_bins[
        [
            "cardinality_bin",
            "N",
            "fraction",
            "macro_candidate_recall",
            "micro_candidate_recall",
        ]
    ].rename(
        columns={
            "N": "val_N",
            "fraction": "val_fraction",
            "macro_candidate_recall": "val_macro_recall",
            "micro_candidate_recall": "val_micro_recall",
        }
    ).merge(
        test_bins[
            [
                "cardinality_bin",
                "N",
                "fraction",
                "macro_candidate_recall",
                "micro_candidate_recall",
            ]
        ].rename(
            columns={
                "N": "test_N",
                "fraction": "test_fraction",
                "macro_candidate_recall": "test_macro_recall",
                "micro_candidate_recall": "test_micro_recall",
            }
        ),
        on="cardinality_bin",
        how="outer",
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison.to_csv(
        out_dir
        / "val_vs_test_cardinality.csv",
        index=False,
    )

    # ------------------------------------------------
    # Matched validation subset
    # ------------------------------------------------

    if args.matched_val_size is None:
        matched_size = (
            determine_matched_size(
                val_df,
                test_df,
            )
        )
    else:
        matched_size = int(
            args.matched_val_size
        )

    print(
        "\nLargest/requested matched "
        f"validation size: {matched_size}"
    )

    matched_val = (
        make_matched_validation(
            val_df=val_df,
            test_df=test_df,
            target_size=matched_size,
            seed=args.seed,
        )
    )

    matched_val.to_csv(
        out_dir
        / "matched_val_per_protein.csv",
        index=False,
    )

    indices = (
        matched_val[
            "row_index"
        ]
        .astype(int)
        .tolist()
    )

    protein_ids = (
        matched_val[
            "protein_id"
        ]
        .astype(str)
        .tolist()
    )

    with (
            out_dir
            / "matched_val_indices.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            indices,
            f,
            indent=2,
        )

    with (
            out_dir
            / "matched_val_protein_ids.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            protein_ids,
            f,
            indent=2,
        )

    matched_summary = (
        summarize_split(
            matched_val
        )
    )

    with (
            out_dir
            / "matched_val_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            matched_summary,
            f,
            indent=2,
        )

    matched_bins = (
        summarize_bins(
            matched_val
        )
    )

    matched_bins.to_csv(
        out_dir
        / "matched_val_cardinality_bins.csv",
        index=False,
    )

    print_summary(
        "matched validation",
        matched_summary,
    )

    print(
        "\nMatched validation cardinality bins:\n"
    )

    print(
        matched_bins.to_string(
            index=False
        )
    )

    print(
        "\nSaved under:",
        out_dir,
    )


if __name__ == "__main__":
    main()