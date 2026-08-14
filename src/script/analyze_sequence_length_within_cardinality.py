# src/script/analyze_sequence_length_within_cardinality.py

from __future__ import annotations

import argparse
import json
import pickle
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

QUARTILE_LABELS = [
    "Q1_shortest",
    "Q2",
    "Q3",
    "Q4_longest",
]


def parse_args():
    p = argparse.ArgumentParser(
        "Analyze sequence-length effects within cardinality bins."
    )

    p.add_argument("--val_dump", required=True)
    p.add_argument("--test_dump", required=True)
    p.add_argument("--seq_len_lookup", required=True)
    p.add_argument("--out_dir", required=True)

    return p.parse_args()


def cardinality_bin(n: int) -> str:
    for lo, hi, name in CARD_BINS:
        if lo <= n <= hi:
            return name
    return "0"


def load_seq_lengths(path: str | Path) -> dict[str, int]:
    path = Path(path)

    with path.open("rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected dict in seq_len_lookup, got {type(obj)}"
        )

    return {
        str(k): int(v)
        for k, v in obj.items()
    }


def safe_mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    return float(x.mean())


def safe_median(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    return float(np.median(x))


def load_dump(
        dump_dir: str | Path,
        split: str,
        seq_lengths: dict[str, int],
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

    rows = []

    missing_seq = 0

    for i, pid in enumerate(protein_ids):

        true_row = np.asarray(
            true_go_ids[i],
            dtype=np.int64,
        )

        full_cardinality = int(
            np.sum(true_row >= 0)
        )

        labels = np.asarray(
            top_labels[i],
            dtype=np.int8,
        )

        candidate_positives = int(
            (labels > 0).sum()
        )

        candidate_recall = (
            candidate_positives / full_cardinality
            if full_cardinality > 0
            else np.nan
        )

        seq_len = seq_lengths.get(
            pid,
            None,
        )

        if seq_len is None:
            missing_seq += 1
            seq_len = np.nan

        rows.append(
            {
                "split": split,
                "row_index": i,
                "protein_id": pid,
                "full_cardinality": full_cardinality,
                "candidate_positives": candidate_positives,
                "candidate_recall": candidate_recall,
                "sequence_length": seq_len,
                "cardinality_bin": cardinality_bin(
                    full_cardinality
                ),
            }
        )

    print(
        f"[{split}] proteins={len(rows)} "
        f"missing_seq_lengths={missing_seq}"
    )

    return pd.DataFrame(rows)


def assign_length_quartiles(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign quartiles separately within each split + cardinality bin.

    This answers:
      within proteins of comparable annotation cardinality,
      are longer sequences harder?
    """

    out = df.copy()
    out["length_quartile"] = None

    for _, _, bin_name in CARD_BINS:

        mask = (
                (out["cardinality_bin"] == bin_name)
                & out["sequence_length"].notna()
        )

        x = out.loc[
            mask,
            "sequence_length",
        ]

        if len(x) < 4:
            continue

        # rank(method="first") avoids qcut failure when many proteins
        # share exactly the same sequence length.
        ranked = x.rank(
            method="first"
        )

        quartiles = pd.qcut(
            ranked,
            q=4,
            labels=QUARTILE_LABELS,
        )

        out.loc[
            mask,
            "length_quartile",
        ] = quartiles.astype(str).values

    return out


def summarize(
        df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, _, bin_name in CARD_BINS:

        for quartile in QUARTILE_LABELS:

            x = df[
                (df["cardinality_bin"] == bin_name)
                & (df["length_quartile"] == quartile)
                ]

            if len(x) == 0:
                rows.append(
                    {
                        "split": (
                            str(df["split"].iloc[0])
                            if len(df) > 0
                            else "unknown"
                        ),
                        "cardinality_bin": bin_name,
                        "length_quartile": quartile,
                        "N": 0,
                        "mean_seq_len": np.nan,
                        "median_seq_len": np.nan,
                        "mean_cardinality": np.nan,
                        "mean_candidate_positives": np.nan,
                        "macro_candidate_recall": np.nan,
                        "micro_candidate_recall": np.nan,
                    }
                )
                continue

            total_true = float(
                x["full_cardinality"].sum()
            )

            total_hits = float(
                x["candidate_positives"].sum()
            )

            rows.append(
                {
                    "split": str(
                        x["split"].iloc[0]
                    ),
                    "cardinality_bin": bin_name,
                    "length_quartile": quartile,
                    "N": int(len(x)),
                    "mean_seq_len": safe_mean(
                        x["sequence_length"]
                    ),
                    "median_seq_len": safe_median(
                        x["sequence_length"]
                    ),
                    "mean_cardinality": safe_mean(
                        x["full_cardinality"]
                    ),
                    "mean_candidate_positives": safe_mean(
                        x["candidate_positives"]
                    ),
                    "macro_candidate_recall": safe_mean(
                        x["candidate_recall"]
                    ),
                    "micro_candidate_recall": (
                        total_hits / total_true
                        if total_true > 0
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def print_split_table(
        summary: pd.DataFrame,
        split: str,
):
    print(
        "\n========================================"
    )
    print(
        f"{split.upper()} LENGTH EFFECT WITHIN CARDINALITY"
    )
    print(
        "========================================\n"
    )

    x = summary[
        summary["split"] == split
        ]

    cols = [
        "cardinality_bin",
        "length_quartile",
        "N",
        "mean_seq_len",
        "mean_cardinality",
        "macro_candidate_recall",
        "micro_candidate_recall",
    ]

    print(
        x[cols].to_string(
            index=False
        )
    )


def build_q1_q4_effect_table(
        summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main diagnostic:
      recall(longest quartile) - recall(shortest quartile)

    Negative values indicate worse retrieval for longer proteins.
    """

    rows = []

    for split in ["val", "test"]:

        sx = summary[
            summary["split"] == split
            ]

        for _, _, bin_name in CARD_BINS:

            q1 = sx[
                (sx["cardinality_bin"] == bin_name)
                & (
                        sx["length_quartile"]
                        == "Q1_shortest"
                )
                ]

            q4 = sx[
                (sx["cardinality_bin"] == bin_name)
                & (
                        sx["length_quartile"]
                        == "Q4_longest"
                )
                ]

            if (
                    len(q1) == 0
                    or len(q4) == 0
                    or int(q1.iloc[0]["N"]) == 0
                    or int(q4.iloc[0]["N"]) == 0
            ):
                continue

            q1_row = q1.iloc[0]
            q4_row = q4.iloc[0]

            rows.append(
                {
                    "split": split,
                    "cardinality_bin": bin_name,

                    "q1_N": int(
                        q1_row["N"]
                    ),
                    "q4_N": int(
                        q4_row["N"]
                    ),

                    "q1_mean_seq_len": float(
                        q1_row["mean_seq_len"]
                    ),
                    "q4_mean_seq_len": float(
                        q4_row["mean_seq_len"]
                    ),

                    "q1_macro_recall": float(
                        q1_row[
                            "macro_candidate_recall"
                        ]
                    ),
                    "q4_macro_recall": float(
                        q4_row[
                            "macro_candidate_recall"
                        ]
                    ),

                    "delta_q4_minus_q1": float(
                        q4_row[
                            "macro_candidate_recall"
                        ]
                        - q1_row[
                            "macro_candidate_recall"
                        ]
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_val_test_effect_comparison(
        effect_df: pd.DataFrame,
) -> pd.DataFrame:
    val = effect_df[
        effect_df["split"] == "val"
        ].drop(
        columns=["split"]
    )

    test = effect_df[
        effect_df["split"] == "test"
        ].drop(
        columns=["split"]
    )

    val = val.rename(
        columns={
            c: f"val_{c}"
            for c in val.columns
            if c != "cardinality_bin"
        }
    )

    test = test.rename(
        columns={
            c: f"test_{c}"
            for c in test.columns
            if c != "cardinality_bin"
        }
    )

    merged = val.merge(
        test,
        on="cardinality_bin",
        how="outer",
    )

    if (
            "val_delta_q4_minus_q1"
            in merged.columns
            and "test_delta_q4_minus_q1"
            in merged.columns
    ):
        merged[
            "test_minus_val_length_effect"
        ] = (
                merged[
                    "test_delta_q4_minus_q1"
                ]
                - merged[
                    "val_delta_q4_minus_q1"
                ]
        )

    return merged


def main():
    args = parse_args()

    out_dir = Path(
        args.out_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading sequence-length lookup...")

    seq_lengths = load_seq_lengths(
        args.seq_len_lookup
    )

    print(
        "Sequence lengths loaded:",
        len(seq_lengths),
    )

    val_df = load_dump(
        args.val_dump,
        "val",
        seq_lengths,
    )

    test_df = load_dump(
        args.test_dump,
        "test",
        seq_lengths,
    )

    val_df = assign_length_quartiles(
        val_df
    )

    test_df = assign_length_quartiles(
        test_df
    )

    val_df.to_csv(
        out_dir / "val_per_protein_length.csv",
        index=False,
    )

    test_df.to_csv(
        out_dir / "test_per_protein_length.csv",
        index=False,
    )

    val_summary = summarize(
        val_df
    )

    test_summary = summarize(
        test_df
    )

    summary = pd.concat(
        [
            val_summary,
            test_summary,
        ],
        ignore_index=True,
    )

    summary.to_csv(
        out_dir
        / "length_quartile_summary.csv",
        index=False,
    )

    print_split_table(
        summary,
        "val",
    )

    print_split_table(
        summary,
        "test",
    )

    effect_df = (
        build_q1_q4_effect_table(
            summary
        )
    )

    effect_df.to_csv(
        out_dir
        / "q1_vs_q4_length_effect.csv",
        index=False,
    )

    comparison = (
        build_val_test_effect_comparison(
            effect_df
        )
    )

    comparison.to_csv(
        out_dir
        / "val_vs_test_length_effect.csv",
        index=False,
    )

    print(
        "\n========================================"
    )
    print("SHORTEST vs LONGEST QUARTILE")
    print(
        "========================================\n"
    )

    if len(effect_df) > 0:
        print(
            effect_df.to_string(
                index=False
            )
        )

    print(
        "\n========================================"
    )
    print("VAL vs TEST LENGTH-EFFECT COMPARISON")
    print(
        "========================================\n"
    )

    if len(comparison) > 0:
        print(
            comparison.to_string(
                index=False
            )
        )

    print(
        "\nInterpretation:"
    )
    print(
        "delta_q4_minus_q1 < 0 means the longest "
        "proteins have lower candidate recall than "
        "the shortest proteins within the SAME "
        "cardinality bin."
    )

    print(
        "\nSaved under:",
        out_dir,
    )


if __name__ == "__main__":
    main()