# src/script/analyze_within_cardinality_shift.py

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from collections import Counter

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
        "Analyze residual validation-test shift within cardinality bins."
    )

    p.add_argument("--val_dump", required=True)
    p.add_argument("--test_dump", required=True)

    p.add_argument(
        "--train_pid2pos",
        default="",
        help="JSON mapping protein -> training positive GO ids.",
    )

    p.add_argument(
        "--seq_len_lookup",
        default="",
        help="Optional pickle mapping protein id -> sequence length.",
    )

    p.add_argument("--out_dir", required=True)

    return p.parse_args()


def card_bin(n: int) -> str:
    for lo, hi, name in CARD_BINS:
        if lo <= n <= hi:
            return name
    return "0"


def go_to_int(x):
    if isinstance(x, (int, np.integer)):
        return int(x)

    s = str(x).strip()

    if s.startswith("GO:"):
        s = s.split(":", 1)[1]

    try:
        return int(s)
    except Exception:
        return None


def load_train_go_frequency(path: str) -> Counter:
    freq = Counter()

    if not path:
        return freq

    with open(path, "r", encoding="utf-8") as f:
        pid2pos = json.load(f)

    for _, gos in pid2pos.items():
        unique = set()

        for g in gos:
            gi = go_to_int(g)

            if gi is not None:
                unique.add(gi)

        for gi in unique:
            freq[gi] += 1

    return freq


def load_seq_lengths(path: str):
    if not path:
        return {}

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(
            f"seq_len_lookup expected dict, got {type(obj)}"
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
        go_freq: Counter,
        seq_lengths: dict,
) -> pd.DataFrame:
    d = Path(dump_dir)

    with (d / "protein_ids.json").open(
            "r",
            encoding="utf-8",
    ) as f:
        pids = [str(x) for x in json.load(f)]

    true_ids = np.load(
        d / "true_go_ids.npy",
        mmap_mode="r",
    )

    top_labels = np.load(
        d / "top_labels.int8.npy",
        mmap_mode="r",
    )

    top_scores = np.load(
        d / "top_scores.float32.npy",
        mmap_mode="r",
    )

    rows = []

    for i, pid in enumerate(pids):

        true_row = np.asarray(
            true_ids[i],
            dtype=np.int64,
        )

        true_gos = [
            int(x)
            for x in true_row
            if int(x) >= 0
        ]

        full_card = len(true_gos)

        labels = np.asarray(
            top_labels[i],
            dtype=np.int8,
        )

        scores = np.asarray(
            top_scores[i],
            dtype=np.float32,
        )

        n_cand_pos = int(
            (labels > 0).sum()
        )

        cand_recall = (
            n_cand_pos / full_card
            if full_card > 0
            else np.nan
        )

        # -----------------------------
        # Retriever score statistics
        # -----------------------------

        finite = np.isfinite(scores)
        finite_scores = scores[finite]

        top1_score = (
            float(finite_scores[0])
            if finite_scores.size > 0
            else np.nan
        )

        top10_mean = safe_mean(
            finite_scores[:10]
        )

        top50_mean = safe_mean(
            finite_scores[:50]
        )

        top500_mean = safe_mean(
            finite_scores
        )

        score_std = (
            float(np.std(finite_scores))
            if finite_scores.size > 0
            else np.nan
        )

        # -----------------------------
        # Positive/negative score gap
        # -----------------------------

        pos_scores = scores[
            (labels > 0) & finite
            ]

        neg_scores = scores[
            (labels <= 0) & finite
            ]

        mean_pos_score = safe_mean(
            pos_scores
        )

        mean_neg_score = safe_mean(
            neg_scores
        )

        score_margin = (
            mean_pos_score - mean_neg_score
            if np.isfinite(mean_pos_score)
               and np.isfinite(mean_neg_score)
            else np.nan
        )

        # -----------------------------
        # GO training frequency
        # -----------------------------

        go_freqs = np.asarray(
            [
                int(go_freq.get(g, 0))
                for g in true_gos
            ],
            dtype=np.float32,
        )

        if go_freqs.size > 0:
            mean_go_freq = float(
                go_freqs.mean()
            )
            median_go_freq = float(
                np.median(go_freqs)
            )

            rare_lt20_frac = float(
                np.mean(go_freqs < 20)
            )

            rare_lt50_frac = float(
                np.mean(go_freqs < 50)
            )

            freq_ge100_frac = float(
                np.mean(go_freqs >= 100)
            )
        else:
            mean_go_freq = np.nan
            median_go_freq = np.nan
            rare_lt20_frac = np.nan
            rare_lt50_frac = np.nan
            freq_ge100_frac = np.nan

        rows.append(
            {
                "split": split,
                "row_index": i,
                "protein_id": pid,

                "full_cardinality": full_card,
                "cardinality_bin": card_bin(full_card),

                "candidate_positives": n_cand_pos,
                "candidate_recall": cand_recall,

                "sequence_length": seq_lengths.get(
                    pid,
                    np.nan,
                ),

                "top1_score": top1_score,
                "top10_mean_score": top10_mean,
                "top50_mean_score": top50_mean,
                "top500_mean_score": top500_mean,
                "score_std": score_std,

                "mean_positive_score": mean_pos_score,
                "mean_negative_score": mean_neg_score,
                "positive_negative_margin": score_margin,

                "mean_true_go_train_freq": mean_go_freq,
                "median_true_go_train_freq": median_go_freq,

                "rare_lt20_fraction": rare_lt20_frac,
                "rare_lt50_fraction": rare_lt50_frac,
                "freq_ge100_fraction": freq_ge100_frac,
            }
        )

    return pd.DataFrame(rows)


def summarize_bin(
        df: pd.DataFrame,
        split: str,
        bin_name: str,
) -> dict:
    x = df[
        df["cardinality_bin"] == bin_name
        ].copy()

    if len(x) == 0:
        return {
            "split": split,
            "cardinality_bin": bin_name,
            "N": 0,
        }

    total_true = float(
        x["full_cardinality"].sum()
    )

    total_hit = float(
        x["candidate_positives"].sum()
    )

    return {
        "split": split,
        "cardinality_bin": bin_name,
        "N": int(len(x)),

        "mean_cardinality":
            safe_mean(x["full_cardinality"]),

        "median_cardinality":
            safe_median(x["full_cardinality"]),

        "macro_candidate_recall":
            safe_mean(x["candidate_recall"]),

        "micro_candidate_recall":
            (
                total_hit / total_true
                if total_true > 0
                else np.nan
            ),

        "mean_candidate_positives":
            safe_mean(x["candidate_positives"]),

        "mean_sequence_length":
            safe_mean(x["sequence_length"]),

        "median_sequence_length":
            safe_median(x["sequence_length"]),

        "mean_top1_score":
            safe_mean(x["top1_score"]),

        "mean_top10_score":
            safe_mean(x["top10_mean_score"]),

        "mean_top50_score":
            safe_mean(x["top50_mean_score"]),

        "mean_top500_score":
            safe_mean(x["top500_mean_score"]),

        "mean_score_std":
            safe_mean(x["score_std"]),

        "mean_positive_score":
            safe_mean(x["mean_positive_score"]),

        "mean_negative_score":
            safe_mean(x["mean_negative_score"]),

        "mean_positive_negative_margin":
            safe_mean(
                x["positive_negative_margin"]
            ),

        "mean_true_go_train_freq":
            safe_mean(
                x["mean_true_go_train_freq"]
            ),

        "median_true_go_train_freq":
            safe_median(
                x["median_true_go_train_freq"]
            ),

        "mean_rare_lt20_fraction":
            safe_mean(
                x["rare_lt20_fraction"]
            ),

        "mean_rare_lt50_fraction":
            safe_mean(
                x["rare_lt50_fraction"]
            ),

        "mean_freq_ge100_fraction":
            safe_mean(
                x["freq_ge100_fraction"]
            ),
    }


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading training GO frequencies...")
    go_freq = load_train_go_frequency(
        args.train_pid2pos
    )

    print(
        "GO terms with training frequency:",
        len(go_freq),
    )

    print("Loading sequence lengths...")
    seq_lengths = load_seq_lengths(
        args.seq_len_lookup
    )

    print(
        "Sequence lengths loaded:",
        len(seq_lengths),
    )

    print("Loading validation dump...")
    val_df = load_dump(
        args.val_dump,
        "val",
        go_freq,
        seq_lengths,
    )

    print("Loading test dump...")
    test_df = load_dump(
        args.test_dump,
        "test",
        go_freq,
        seq_lengths,
    )

    val_df.to_csv(
        out_dir / "val_per_protein.csv",
        index=False,
    )

    test_df.to_csv(
        out_dir / "test_per_protein.csv",
        index=False,
    )

    rows = []

    for _, _, bin_name in CARD_BINS:
        rows.append(
            summarize_bin(
                val_df,
                "val",
                bin_name,
            )
        )

        rows.append(
            summarize_bin(
                test_df,
                "test",
                bin_name,
            )
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        out_dir / "within_cardinality_summary.csv",
        index=False,
    )

    # ---------------------------------------------
    # Wide comparison table
    # ---------------------------------------------

    val_sum = summary[
        summary["split"] == "val"
        ].drop(
        columns=["split"]
    )

    test_sum = summary[
        summary["split"] == "test"
        ].drop(
        columns=["split"]
    )

    val_sum = val_sum.rename(
        columns={
            c: f"val_{c}"
            for c in val_sum.columns
            if c != "cardinality_bin"
        }
    )

    test_sum = test_sum.rename(
        columns={
            c: f"test_{c}"
            for c in test_sum.columns
            if c != "cardinality_bin"
        }
    )

    wide = val_sum.merge(
        test_sum,
        on="cardinality_bin",
        how="outer",
    )

    # Useful explicit gaps
    if (
            "val_macro_candidate_recall"
            in wide.columns
            and "test_macro_candidate_recall"
            in wide.columns
    ):
        wide["delta_macro_recall_test_minus_val"] = (
                wide["test_macro_candidate_recall"]
                - wide["val_macro_candidate_recall"]
        )

    if (
            "val_mean_positive_negative_margin"
            in wide.columns
            and "test_mean_positive_negative_margin"
            in wide.columns
    ):
        wide["delta_margin_test_minus_val"] = (
                wide["test_mean_positive_negative_margin"]
                - wide["val_mean_positive_negative_margin"]
        )

    if (
            "val_mean_sequence_length"
            in wide.columns
            and "test_mean_sequence_length"
            in wide.columns
    ):
        wide["delta_seq_len_test_minus_val"] = (
                wide["test_mean_sequence_length"]
                - wide["val_mean_sequence_length"]
        )

    if (
            "val_mean_true_go_train_freq"
            in wide.columns
            and "test_mean_true_go_train_freq"
            in wide.columns
    ):
        wide["delta_go_freq_test_minus_val"] = (
                wide["test_mean_true_go_train_freq"]
                - wide["val_mean_true_go_train_freq"]
        )

    wide.to_csv(
        out_dir / "val_vs_test_within_cardinality.csv",
        index=False,
    )

    print(
        "\n========================================"
    )
    print("WITHIN-CARDINALITY VAL vs TEST")
    print(
        "========================================\n"
    )

    cols = [
        "cardinality_bin",

        "val_N",
        "test_N",

        "val_macro_candidate_recall",
        "test_macro_candidate_recall",
        "delta_macro_recall_test_minus_val",

        "val_mean_sequence_length",
        "test_mean_sequence_length",

        "val_mean_true_go_train_freq",
        "test_mean_true_go_train_freq",

        "val_mean_rare_lt20_fraction",
        "test_mean_rare_lt20_fraction",

        "val_mean_positive_negative_margin",
        "test_mean_positive_negative_margin",
        "delta_margin_test_minus_val",
    ]

    cols = [
        c for c in cols
        if c in wide.columns
    ]

    print(
        wide[cols].to_string(
            index=False
        )
    )

    print(
        "\nSaved analysis to:",
        out_dir,
    )


if __name__ == "__main__":
    main()