#!/usr/bin/env python3

"""
D6: Gold GO -> local protein slot utilization diagnostic.

Question:
    When proteins have many gold GO annotations, does the local expert:

    A) use all/most of its slots, suggesting a possible capacity ceiling,

    B) collapse many gold GO terms onto only a few slots, suggesting
       poor slot specialization,

    C) use slots differently on VALID vs TEST, suggesting an OOD
       generalization failure of the local representation,

    D) show similar slot utilization everywhere, suggesting that slot
       utilization itself does not explain the retrieval failure.

Inputs:
    Existing full-ranking dumps containing:
        protein_ids.json
        true_go_ids.json
        eval_go_ids.npy
        go_z.float16.npy
        protein_local_z.float16.npy

No model inference is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CARD_BINS = [
    (1, 5, "1_5"),
    (6, 10, "6_10"),
    (11, 20, "11_20"),
    (21, 40, "21_40"),
    (41, 80, "41_80"),
    (81, 160, "81_160"),
    (161, None, "161plus"),
]


def cardinality_bin(n_gold: int) -> str:
    for lo, hi, label in CARD_BINS:
        if hi is None:
            if n_gold >= lo:
                return label
        elif lo <= n_gold <= hi:
            return label

    return "unknown"


def load_dump(dump_dir: Path):
    dump_dir = Path(dump_dir)

    with open(dump_dir / "protein_ids.json") as f:
        protein_ids = json.load(f)

    with open(dump_dir / "true_go_ids.json") as f:
        true_go_ids = json.load(f)

    eval_go_ids = np.load(
        dump_dir / "eval_go_ids.npy"
    ).astype(np.int64)

    go_z = np.load(
        dump_dir / "go_z.float16.npy"
    ).astype(np.float32)

    protein_local_z = np.load(
        dump_dir / "protein_local_z.float16.npy",
        mmap_mode="r",
    )

    if len(protein_ids) != len(true_go_ids):
        raise RuntimeError(
            "protein_ids / true_go_ids length mismatch"
        )

    if protein_local_z.shape[0] != len(protein_ids):
        raise RuntimeError(
            "protein_local_z / protein_ids mismatch"
        )

    if go_z.shape[0] != len(eval_go_ids):
        raise RuntimeError(
            "go_z / eval_go_ids mismatch"
        )

    if protein_local_z.ndim != 3:
        raise RuntimeError(
            "Expected protein_local_z shape [N, S, D], "
            f"got {protein_local_z.shape}"
        )

    # Numerical safety.
    go_norm = np.linalg.norm(
        go_z,
        axis=1,
        keepdims=True,
    )

    go_z = go_z / np.clip(
        go_norm,
        1e-8,
        None,
    )

    return {
        "protein_ids": protein_ids,
        "true_go_ids": true_go_ids,
        "eval_go_ids": eval_go_ids,
        "go_z": go_z,
        "protein_local_z": protein_local_z,
    }


def normalized_entropy(counts: np.ndarray) -> float:
    """
    Entropy of gold->slot assignments normalized to [0,1].

    0:
        all gold GO terms assigned to one slot

    1:
        assignments uniformly distributed across all available slots
    """

    counts = counts.astype(np.float64)

    total = counts.sum()

    if total <= 0:
        return np.nan

    p = counts / total
    p = p[p > 0]

    entropy = -np.sum(
        p * np.log(p)
    )

    n_slots = len(counts)

    if n_slots <= 1:
        return 0.0

    return float(
        entropy / np.log(n_slots)
    )


def effective_slot_number(counts: np.ndarray) -> float:
    """
    Entropy-based effective number of slots:

        exp(H)

    Examples:

        [80,0,0,...] -> 1

        [10,10,...,10] across 8 slots -> 8

    Unlike raw active-slot count, this penalizes highly imbalanced use.
    """

    counts = counts.astype(np.float64)

    total = counts.sum()

    if total <= 0:
        return np.nan

    p = counts / total
    p = p[p > 0]

    entropy = -np.sum(
        p * np.log(p)
    )

    return float(
        np.exp(entropy)
    )


def analyze_split(
        dump,
        split_name: str,
):
    protein_ids = dump["protein_ids"]
    true_go_ids = dump["true_go_ids"]
    eval_go_ids = dump["eval_go_ids"]
    go_z = dump["go_z"]
    protein_local_z = dump["protein_local_z"]

    n_slots = protein_local_z.shape[1]

    id_to_col = {
        int(go_id): idx
        for idx, go_id in enumerate(eval_go_ids)
    }

    rows = []

    print(
        f"\n[{split_name}] "
        f"proteins={len(protein_ids)} "
        f"slots={n_slots}"
    )

    for i, pid in enumerate(protein_ids):

        gold_ids = [
            int(g)
            for g in true_go_ids[i]
            if int(g) >= 0
        ]

        gold_cols = [
            id_to_col[g]
            for g in gold_ids
            if g in id_to_col
        ]

        n_gold = len(gold_cols)

        if n_gold == 0:
            continue

        # [S, D]
        local = np.asarray(
            protein_local_z[i],
            dtype=np.float32,
        )

        local_norm = np.linalg.norm(
            local,
            axis=1,
            keepdims=True,
        )

        local = local / np.clip(
            local_norm,
            1e-8,
            None,
        )

        # Only gold GO embeddings.
        #
        # [G_gold, D]
        gold_go = go_z[
            np.asarray(
                gold_cols,
                dtype=np.int64,
            )
        ]

        # [S, G_gold]
        slot_gold_scores = (
                local @ gold_go.T
        )

        # For each gold GO:
        # which slot has highest cosine?
        #
        # [G_gold]
        winner_slots = np.argmax(
            slot_gold_scores,
            axis=0,
        )

        counts = np.bincount(
            winner_slots,
            minlength=n_slots,
        )

        active_slots = int(
            np.sum(counts > 0)
        )

        dominant_count = int(
            counts.max()
        )

        dominant_fraction = float(
            dominant_count / n_gold
        )

        entropy_norm = normalized_entropy(
            counts
        )

        effective_slots = effective_slot_number(
            counts
        )

        # How strong is the winning slot for gold GO terms?
        best_gold_scores = np.max(
            slot_gold_scores,
            axis=0,
        )

        # How decisive is the winner?
        # Difference between best and second-best slot.
        if n_slots >= 2:
            sorted_scores = np.sort(
                slot_gold_scores,
                axis=0,
            )

            winner_gap = (
                    sorted_scores[-1]
                    - sorted_scores[-2]
            )
        else:
            winner_gap = np.zeros(
                n_gold,
                dtype=np.float32,
            )

        row = {
            "protein_id": str(pid),
            "split": split_name,
            "n_gold": n_gold,
            "card_bin":
                cardinality_bin(n_gold),

            "n_slots": n_slots,

            # Raw number of slots winning >=1 gold GO.
            "active_slots":
                active_slots,

            "active_slot_fraction":
                active_slots / n_slots,

            # Entropy-based effective slot count.
            "effective_slots":
                effective_slots,

            "effective_slot_fraction":
                effective_slots / n_slots,

            # Largest fraction of gold GO terms
            # captured by one slot.
            "dominant_slot_fraction":
                dominant_fraction,

            # 0 = collapsed, 1 = uniform across slots.
            "assignment_entropy":
                entropy_norm,

            # Gold-slot alignment strength.
            "mean_best_gold_slot_score":
                float(
                    np.mean(
                        best_gold_scores
                    )
                ),

            "median_best_gold_slot_score":
                float(
                    np.median(
                        best_gold_scores
                    )
                ),

            # How much better winner is than
            # second-best slot.
            "mean_winner_gap":
                float(
                    np.mean(
                        winner_gap
                    )
                ),

            "median_winner_gap":
                float(
                    np.median(
                        winner_gap
                    )
                ),
        }

        # Save per-slot assignment fractions.
        for s in range(n_slots):
            row[f"slot_{s}_count"] = int(
                counts[s]
            )

            row[f"slot_{s}_fraction"] = float(
                counts[s] / n_gold
            )

        rows.append(row)

        if (
                (i + 1) % 500 == 0
                or i + 1 == len(protein_ids)
        ):
            print(
                f"[{split_name}] "
                f"{i + 1}/{len(protein_ids)}"
            )

    return pd.DataFrame(rows)


def summarize(
        df: pd.DataFrame,
):
    rows = []

    labels = [
        x[2]
        for x in CARD_BINS
    ]

    for split in ["valid", "test"]:

        split_df = df[
            df["split"] == split
            ]

        for card_bin in labels:

            sub = split_df[
                split_df["card_bin"]
                == card_bin
                ]

            if len(sub) == 0:
                continue

            rows.append(
                {
                    "split":
                        split,

                    "card_bin":
                        card_bin,

                    "n_proteins":
                        len(sub),

                    "mean_n_gold":
                        sub["n_gold"].mean(),

                    "mean_active_slots":
                        sub[
                            "active_slots"
                        ].mean(),

                    "mean_effective_slots":
                        sub[
                            "effective_slots"
                        ].mean(),

                    "mean_effective_slot_fraction":
                        sub[
                            "effective_slot_fraction"
                        ].mean(),

                    "mean_dominant_slot_fraction":
                        sub[
                            "dominant_slot_fraction"
                        ].mean(),

                    "mean_assignment_entropy":
                        sub[
                            "assignment_entropy"
                        ].mean(),

                    "mean_best_gold_slot_score":
                        sub[
                            "mean_best_gold_slot_score"
                        ].mean(),

                    "mean_winner_gap":
                        sub[
                            "mean_winner_gap"
                        ].mean(),
                }
            )

    return pd.DataFrame(rows)


def make_gap_table(
        summary: pd.DataFrame,
):
    valid = (
        summary[
            summary["split"] == "valid"
            ]
        .set_index("card_bin")
    )

    test = (
        summary[
            summary["split"] == "test"
            ]
        .set_index("card_bin")
    )

    rows = []

    for _, _, card_bin in CARD_BINS:

        if (
                card_bin not in valid.index
                or card_bin not in test.index
        ):
            continue

        v = valid.loc[card_bin]
        t = test.loc[card_bin]

        row = {
            "card_bin":
                card_bin,

            "valid_n":
                int(
                    v["n_proteins"]
                ),

            "test_n":
                int(
                    t["n_proteins"]
                ),

            "valid_mean_n_gold":
                float(
                    v["mean_n_gold"]
                ),

            "test_mean_n_gold":
                float(
                    t["mean_n_gold"]
                ),
        }

        metrics = [
            "mean_active_slots",
            "mean_effective_slots",
            "mean_effective_slot_fraction",
            "mean_dominant_slot_fraction",
            "mean_assignment_entropy",
            "mean_best_gold_slot_score",
            "mean_winner_gap",
        ]

        for metric in metrics:
            vv = float(v[metric])
            tt = float(t[metric])

            row[
                f"valid_{metric}"
            ] = vv

            row[
                f"test_{metric}"
            ] = tt

            row[
                f"test_minus_valid_{metric}"
            ] = tt - vv

        rows.append(row)

    return pd.DataFrame(rows)


def print_results(
        gap_df: pd.DataFrame,
):
    print("\n")
    print("=" * 115)
    print(
        "D6A: SLOT UTILIZATION BY CARDINALITY"
    )
    print("=" * 115)

    cols = [
        "card_bin",
        "valid_n",
        "test_n",

        "valid_mean_active_slots",
        "test_mean_active_slots",

        "valid_mean_effective_slots",
        "test_mean_effective_slots",

        "valid_mean_dominant_slot_fraction",
        "test_mean_dominant_slot_fraction",

        "valid_mean_assignment_entropy",
        "test_mean_assignment_entropy",
    ]

    print(
        gap_df[
            cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n")
    print("=" * 115)
    print(
        "D6B: GOLD-SLOT ALIGNMENT QUALITY"
    )
    print("=" * 115)

    cols = [
        "card_bin",

        "valid_mean_best_gold_slot_score",
        "test_mean_best_gold_slot_score",

        "test_minus_valid_mean_best_gold_slot_score",

        "valid_mean_winner_gap",
        "test_mean_winner_gap",
    ]

    print(
        gap_df[
            cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--valid_dump",
        default=(
            "/workspace/data_pfresgo/"
            "diagnostics/"
            "coverage_top200_full_ranking_valid"
        ),
    )

    ap.add_argument(
        "--test_dump",
        default=(
            "/workspace/data_pfresgo/"
            "diagnostics/"
            "coverage_top200_full_ranking_test"
        ),
    )

    ap.add_argument(
        "--outdir",
        default=(
            "/workspace/data_pfresgo/"
            "diagnostics/"
            "slot_utilization_bp"
        ),
    )

    args = ap.parse_args()

    print("Loading VALID dump...")

    valid_dump = load_dump(
        Path(args.valid_dump)
    )

    print("Loading TEST dump...")

    test_dump = load_dump(
        Path(args.test_dump)
    )

    valid_df = analyze_split(
        valid_dump,
        "valid",
    )

    test_df = analyze_split(
        test_dump,
        "test",
    )

    per_protein = pd.concat(
        [
            valid_df,
            test_df,
        ],
        ignore_index=True,
    )

    summary = summarize(
        per_protein
    )

    gap = make_gap_table(
        summary
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_protein_path = (
            outdir
            / "slot_utilization_per_protein.csv"
    )

    summary_path = (
            outdir
            / "slot_utilization_by_cardinality.csv"
    )

    gap_path = (
            outdir
            / "slot_utilization_valid_test_gap.csv"
    )

    per_protein.to_csv(
        per_protein_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    gap.to_csv(
        gap_path,
        index=False,
    )

    print_results(
        gap
    )

    print("\nSaved:")
    print(per_protein_path)
    print(summary_path)
    print(gap_path)


if __name__ == "__main__":
    main()