#!/usr/bin/env python3

"""
D10: Does distance from the learned TRAIN protein manifold explain
gold-GO retrieval failure?

Inputs
------
1. D9B per-protein learned-space proximity:
   learned_proximity_per_protein.csv

2. Full-ranking VALID and TEST dumps produced from the SAME checkpoint.

For each protein we compute:
    - gold recall / coverage @ 50, 200, 500
    - median gold rank
    - mean gold rank

Then relate retrieval quality to:
    - global NN1 train proximity
    - global Top10 train proximity
    - local-slot mean train proximity
    - local-slot weakest-slot proximity

Analyses
--------
A. Overall Spearman correlations
B. Correlations within protein-cardinality bins
C. Train-proximity quartiles -> retrieval quality
D. Proximity quartiles within cardinality bins

Interpretation
--------------
If proximity strongly predicts retrieval even within cardinality bins,
learned protein-space generalization is likely an important bottleneck.

If proximity has weak association after cardinality control, then the
major failure likely emerges specifically in protein <-> GO correspondence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KS = [50, 200, 500]

CARD_ORDER = [
    "1_5",
    "6_10",
    "11_20",
    "21_40",
    "41_80",
    "81_160",
    "161plus",
]

PROXIMITY_METRICS = [
    "global_nn1",
    "global_top10",
    "local_slot_nn_mean",
    "local_slot_nn_min",
]


# ---------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------

def find_file(directory: Path, candidates):
    for name in candidates:
        p = directory / name
        if p.exists():
            return p

    raise FileNotFoundError(
        f"None of {candidates} found under {directory}"
    )


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_ranking_dump(dump_dir: Path):
    """
    Expected full-ranking dump.

    We deliberately support several names because earlier diagnostics
    used slightly different dump naming conventions.
    """

    pid_path = find_file(
        dump_dir,
        [
            "protein_ids.json",
            "pids.json",
        ],
    )

    rank_path = find_file(
        dump_dir,
        [
            "top_go_cols.int32.npy",
            "top_indices.int32.npy",
            "top_indices.npy",
            "ranking_indices.int32.npy",
            "ranking_indices.npy",
        ],
    )

    go_path = find_file(
        dump_dir,
        [
            "eval_go_ids.npy",
            "go_ids.npy",
        ],
    )

    pids = load_json(pid_path)

    go_ids = np.load(
        go_path,
        allow_pickle=True,
    ).tolist()

    ranking = np.load(
        rank_path,
        mmap_mode="r",
    )

    if ranking.shape[0] != len(pids):
        raise RuntimeError(
            f"Ranking/protein mismatch: "
            f"{ranking.shape[0]} vs {len(pids)}"
        )

    print(
        f"{dump_dir.name}: "
        f"N={len(pids)} "
        f"ranking_width={ranking.shape[1]} "
        f"GO={len(go_ids)}"
    )

    return pids, go_ids, ranking


# ---------------------------------------------------------------------
# Gold ranks
# ---------------------------------------------------------------------

def build_gold_rank_df(
        split,
        dump_dir,
        pid2pos,
):
    pids, go_ids, ranking = load_ranking_dump(
        dump_dir
    )

    go_to_idx = {
        str(g): i
        for i, g in enumerate(go_ids)
    }

    rows = []

    missing_gold_total = 0

    for i, pid in enumerate(pids):

        gold = [
            str(g)
            for g in pid2pos.get(pid, [])
            if str(g) in go_to_idx
        ]

        if len(gold) == 0:
            continue

        ranked_indices = np.asarray(
            ranking[i],
            dtype=np.int64,
        )

        # inverse rank:
        # GO index -> 1-based rank
        inverse = np.empty(
            len(go_ids),
            dtype=np.int32,
        )

        inverse[ranked_indices] = (
                np.arange(
                    len(ranked_indices),
                    dtype=np.int32,
                )
                + 1
        )

        gold_indices = np.asarray(
            [go_to_idx[g] for g in gold],
            dtype=np.int64,
        )

        gold_ranks = inverse[
            gold_indices
        ]

        missing_gold_total += int(
            np.sum(gold_ranks <= 0)
        )

        row = {
            "protein_id": pid,
            "split": split,
            "n_gold_eval": len(gold),
            "median_gold_rank":
                float(np.median(gold_ranks)),
            "mean_gold_rank":
                float(np.mean(gold_ranks)),
        }

        for k in KS:
            row[f"coverage@{k}"] = float(
                np.mean(
                    gold_ranks <= k
                )
            )

            row[f"any_hit@{k}"] = float(
                np.any(
                    gold_ranks <= k
                )
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    print(
        f"[{split}] proteins={len(df)} "
        f"gold_pairs={df['n_gold_eval'].sum()} "
        f"missing={missing_gold_total}"
    )

    return df


# ---------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------

def spearman_table(df):
    """
    Overall split-level correlations.
    """

    outcomes = [
        "coverage@50",
        "coverage@200",
        "coverage@500",
        "median_gold_rank",
    ]

    rows = []

    for split in ["valid", "test"]:

        d = df[
            df["split"] == split
            ]

        for prox in PROXIMITY_METRICS:
            for outcome in outcomes:
                sub = d[
                    [prox, outcome]
                ].dropna()

                rho = sub[
                    prox
                ].corr(
                    sub[outcome],
                    method="spearman",
                )

                rows.append({
                    "split": split,
                    "proximity_metric": prox,
                    "retrieval_metric": outcome,
                    "n": len(sub),
                    "spearman_rho": rho,
                })

    return pd.DataFrame(rows)


def cardinality_spearman(df):
    """
    Correlations within cardinality bins.
    """

    outcomes = [
        "coverage@50",
        "coverage@200",
        "coverage@500",
        "median_gold_rank",
    ]

    rows = []

    for split in ["valid", "test"]:

        d = df[
            df["split"] == split
            ]

        for card in CARD_ORDER:

            sub_card = d[
                d["card_bin"] == card
                ]

            if len(sub_card) < 10:
                continue

            for prox in PROXIMITY_METRICS:
                for outcome in outcomes:

                    sub = sub_card[
                        [prox, outcome]
                    ].dropna()

                    if len(sub) < 10:
                        continue

                    rho = sub[
                        prox
                    ].corr(
                        sub[outcome],
                        method="spearman",
                    )

                    rows.append({
                        "split": split,
                        "card_bin": card,
                        "proximity_metric": prox,
                        "retrieval_metric": outcome,
                        "n": len(sub),
                        "spearman_rho": rho,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Quartile analysis
# ---------------------------------------------------------------------

def add_split_quartiles(df):
    """
    Q1 = least train-like
    Q4 = most train-like.

    Quartiles are defined independently within VALID and TEST so that
    we test the association inside each distribution.
    """

    out = df.copy()

    for prox in PROXIMITY_METRICS:

        out[f"{prox}_quartile"] = None

        for split in [
            "valid",
            "test",
        ]:

            mask = (
                    out["split"] == split
            )

            values = out.loc[
                mask,
                prox,
            ]

            try:
                q = pd.qcut(
                    values,
                    q=4,
                    labels=[
                        "Q1_least_train_like",
                        "Q2",
                        "Q3",
                        "Q4_most_train_like",
                    ],
                    duplicates="drop",
                )

                out.loc[
                    mask,
                    f"{prox}_quartile",
                ] = q.astype(str)

            except ValueError:
                pass

    return out


def quartile_summary(df):
    rows = []

    for split in [
        "valid",
        "test",
    ]:

        d = df[
            df["split"] == split
            ]

        for prox in PROXIMITY_METRICS:

            qcol = (
                f"{prox}_quartile"
            )

            for quartile in [
                "Q1_least_train_like",
                "Q2",
                "Q3",
                "Q4_most_train_like",
            ]:

                sub = d[
                    d[qcol] == quartile
                    ]

                if len(sub) == 0:
                    continue

                rows.append({
                    "split": split,
                    "proximity_metric": prox,
                    "quartile": quartile,
                    "n": len(sub),

                    "mean_proximity":
                        sub[prox].mean(),

                    "mean_n_gold":
                        sub["n_gold"].mean(),

                    "coverage@50":
                        sub["coverage@50"].mean(),

                    "coverage@200":
                        sub["coverage@200"].mean(),

                    "coverage@500":
                        sub["coverage@500"].mean(),

                    "median_gold_rank":
                        sub[
                            "median_gold_rank"
                        ].median(),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Cardinality-controlled quartiles
# ---------------------------------------------------------------------

def cardinality_controlled_quartiles(df):
    """
    Define train-proximity quartiles separately inside every
    split x cardinality bin.

    This removes the gross cardinality shift from the comparison.
    """

    rows = []

    for split in [
        "valid",
        "test",
    ]:

        split_df = df[
            df["split"] == split
            ]

        for card in CARD_ORDER:

            card_df = split_df[
                split_df["card_bin"] == card
                ].copy()

            # Need enough proteins for quartiles.
            if len(card_df) < 20:
                continue

            for prox in PROXIMITY_METRICS:

                try:
                    card_df[
                        "_quartile"
                    ] = pd.qcut(
                        card_df[prox],
                        q=4,
                        labels=[
                            "Q1_least_train_like",
                            "Q2",
                            "Q3",
                            "Q4_most_train_like",
                        ],
                        duplicates="drop",
                    )

                except ValueError:
                    continue

                for quartile in [
                    "Q1_least_train_like",
                    "Q2",
                    "Q3",
                    "Q4_most_train_like",
                ]:

                    sub = card_df[
                        card_df[
                            "_quartile"
                        ].astype(str)
                        == quartile
                        ]

                    if len(sub) == 0:
                        continue

                    rows.append({
                        "split": split,
                        "card_bin": card,
                        "proximity_metric": prox,
                        "quartile": quartile,
                        "n": len(sub),

                        "mean_proximity":
                            sub[prox].mean(),

                        "mean_n_gold":
                            sub["n_gold"].mean(),

                        "coverage@50":
                            sub[
                                "coverage@50"
                            ].mean(),

                        "coverage@200":
                            sub[
                                "coverage@200"
                            ].mean(),

                        "coverage@500":
                            sub[
                                "coverage@500"
                            ].mean(),

                        "median_gold_rank":
                            sub[
                                "median_gold_rank"
                            ].median(),
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    base = Path(
        "/workspace/data_pfresgo/"
        "diagnostics"
    )

    ap.add_argument(
        "--proximity_csv",
        default=str(
            base
            / "learned_protein_proximity_bp"
            / "learned_proximity_per_protein.csv"
        ),
    )

    ap.add_argument(
        "--valid_dump",
        default=str(
            base
            / "slotdiv005_step70572_fullranking_valid"
        ),
    )

    ap.add_argument(
        "--test_dump",
        default=str(
            base
            / "slotdiv005_step70572_fullranking_test"
        ),
    )

    ap.add_argument(
        "--pid2pos",
        default=(
            "/workspace/data_pfresgo/"
            "processed/"
            "pid_to_positives_bp.json"
        ),
    )

    ap.add_argument(
        "--outdir",
        default=str(
            base
            / "proximity_vs_retrieval_bp"
        ),
    )

    args = ap.parse_args()

    print(
        "Loading learned-space proximity..."
    )

    proximity = pd.read_csv(
        args.proximity_csv
    )

    print(
        "Proximity rows:",
        len(proximity),
    )

    with open(
            args.pid2pos
    ) as f:
        pid2pos = json.load(f)

    print(
        "\nComputing VALID "
        "per-protein retrieval..."
    )

    valid_retrieval = (
        build_gold_rank_df(
            "valid",
            Path(args.valid_dump),
            pid2pos,
        )
    )

    print(
        "\nComputing TEST "
        "per-protein retrieval..."
    )

    test_retrieval = (
        build_gold_rank_df(
            "test",
            Path(args.test_dump),
            pid2pos,
        )
    )

    retrieval = pd.concat(
        [
            valid_retrieval,
            test_retrieval,
        ],
        ignore_index=True,
    )

    print(
        "\nJoining proximity "
        "and retrieval..."
    )

    merged = proximity.merge(
        retrieval,
        on=[
            "protein_id",
            "split",
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        "Merged proteins:",
        len(merged),
    )

    print(
        "VALID:",
        int(
            (
                    merged.split
                    == "valid"
            ).sum()
        ),
    )

    print(
        "TEST:",
        int(
            (
                    merged.split
                    == "test"
            ).sum()
        ),
    )

    # -------------------------------------------------------------
    # Overall correlations
    # -------------------------------------------------------------

    corr = spearman_table(
        merged
    )

    print("\n")
    print("=" * 110)
    print(
        "D10A: TRAIN PROXIMITY "
        "vs RETRIEVAL"
    )
    print("=" * 110)

    print(
        corr.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -------------------------------------------------------------
    # Cardinality-controlled correlations
    # -------------------------------------------------------------

    card_corr = (
        cardinality_spearman(
            merged
        )
    )

    print("\n")
    print("=" * 110)
    print(
        "D10B: WITHIN-CARDINALITY "
        "CORRELATIONS"
    )
    print("=" * 110)

    important = card_corr[
        (
                card_corr[
                    "retrieval_metric"
                ]
                == "coverage@200"
        )
        &
        (
            card_corr[
                "proximity_metric"
            ].isin([
                "global_nn1",
                "global_top10",
                "local_slot_nn_mean",
            ])
        )
        ]

    print(
        important.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -------------------------------------------------------------
    # Quartiles
    # -------------------------------------------------------------

    merged_q = (
        add_split_quartiles(
            merged
        )
    )

    quartiles = (
        quartile_summary(
            merged_q
        )
    )

    print("\n")
    print("=" * 110)
    print(
        "D10C: PROXIMITY QUARTILES"
    )
    print("=" * 110)

    display_q = quartiles[
        quartiles[
            "proximity_metric"
        ].isin([
            "global_top10",
            "local_slot_nn_mean",
        ])
    ]

    print(
        display_q.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -------------------------------------------------------------
    # Strongest control:
    # quartiles inside cardinality bins
    # -------------------------------------------------------------

    controlled = (
        cardinality_controlled_quartiles(
            merged
        )
    )

    print("\n")
    print("=" * 110)
    print(
        "D10D: CARDINALITY-CONTROLLED "
        "PROXIMITY QUARTILES"
    )
    print("=" * 110)

    display_controlled = controlled[
        (
                controlled[
                    "split"
                ] == "test"
        )
        &
        (
                controlled[
                    "proximity_metric"
                ]
                == "global_top10"
        )
        ]

    print(
        display_controlled.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        outdir
        / "proximity_retrieval_per_protein.csv",
        index=False,
    )

    corr.to_csv(
        outdir
        / "proximity_retrieval_correlations.csv",
        index=False,
    )

    card_corr.to_csv(
        outdir
        / "proximity_retrieval_correlations_by_cardinality.csv",
        index=False,
    )

    quartiles.to_csv(
        outdir
        / "proximity_retrieval_quartiles.csv",
        index=False,
    )

    controlled.to_csv(
        outdir
        / "proximity_retrieval_cardinality_controlled_quartiles.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        outdir
        / "proximity_retrieval_per_protein.csv"
    )
    print(
        outdir
        / "proximity_retrieval_correlations.csv"
    )
    print(
        outdir
        / "proximity_retrieval_correlations_by_cardinality.csv"
    )
    print(
        outdir
        / "proximity_retrieval_quartiles.csv"
    )
    print(
        outdir
        / "proximity_retrieval_cardinality_controlled_quartiles.csv"
    )

if __name__ == "__main__":
    main()