#!/usr/bin/env python3

"""
D7: Local-slot geometry / redundancy diagnostic.

Question:
    Are the 8 local protein slots actually distinct representations,
    or are they nearly identical?

Why:
    D6 showed that:
      - high-cardinality proteins appear to use multiple slots,
      - but best-vs-second-best GO-slot score gap is ~1e-4.

    Therefore the apparent slot utilization may be misleading if
    the slots themselves are nearly identical.

For each protein:
    - L2-normalize its S local slots
    - compute all pairwise slot-slot cosine similarities
    - summarize mean / median / min / max similarity
    - estimate effective slot geometry
    - compare VALID vs TEST across cardinality bins

No model inference or GPU required.
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

    with open(
            dump_dir / "protein_ids.json"
    ) as f:
        protein_ids = json.load(f)

    with open(
            dump_dir / "true_go_ids.json"
    ) as f:
        true_go_ids = json.load(f)

    local_z = np.load(
        dump_dir / "protein_local_z.float16.npy",
        mmap_mode="r",
    )

    if local_z.ndim != 3:
        raise RuntimeError(
            "Expected protein_local_z shape [N,S,D], "
            f"got {local_z.shape}"
        )

    if local_z.shape[0] != len(protein_ids):
        raise RuntimeError(
            "protein_local_z / protein_ids mismatch"
        )

    if len(true_go_ids) != len(protein_ids):
        raise RuntimeError(
            "true_go_ids / protein_ids mismatch"
        )

    return {
        "protein_ids": protein_ids,
        "true_go_ids": true_go_ids,
        "local_z": local_z,
    }


def effective_rank_from_gram(
        gram: np.ndarray,
) -> float:
    """
    Effective rank based on eigenvalue entropy.

    If all slots are identical:
        effective rank ~ 1

    If slots span many independent directions:
        effective rank approaches number of slots.
    """

    eigvals = np.linalg.eigvalsh(
        gram.astype(np.float64)
    )

    eigvals = np.clip(
        eigvals,
        0.0,
        None,
    )

    total = eigvals.sum()

    if total <= 1e-12:
        return np.nan

    p = eigvals / total
    p = p[p > 1e-12]

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
    local_z = dump["local_z"]

    n_slots = local_z.shape[1]

    tri_i, tri_j = np.triu_indices(
        n_slots,
        k=1,
    )

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

        n_gold = len(gold_ids)

        if n_gold == 0:
            continue

        # [S,D]
        slots = np.asarray(
            local_z[i],
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            slots,
            axis=1,
            keepdims=True,
        )

        slots = slots / np.clip(
            norms,
            1e-8,
            None,
        )

        # [S,S]
        gram = slots @ slots.T

        gram = np.clip(
            gram,
            -1.0,
            1.0,
        )

        pairwise = gram[
            tri_i,
            tri_j,
        ]

        # Distance equivalent for normalized vectors.
        pairwise_distance = (
                1.0 - pairwise
        )

        # For each slot, find its most similar OTHER slot.
        no_self = gram.copy()

        np.fill_diagonal(
            no_self,
            -np.inf,
        )

        nearest_slot_cos = np.max(
            no_self,
            axis=1,
        )

        eff_rank = effective_rank_from_gram(
            gram
        )

        rows.append(
            {
                "protein_id": str(pid),
                "split": split_name,

                "n_gold": n_gold,
                "card_bin":
                    cardinality_bin(n_gold),

                "n_slots": n_slots,

                "mean_slot_cosine":
                    float(
                        pairwise.mean()
                    ),

                "median_slot_cosine":
                    float(
                        np.median(pairwise)
                    ),

                "min_slot_cosine":
                    float(
                        pairwise.min()
                    ),

                "max_slot_cosine":
                    float(
                        pairwise.max()
                    ),

                "std_slot_cosine":
                    float(
                        pairwise.std()
                    ),

                "mean_slot_distance":
                    float(
                        pairwise_distance.mean()
                    ),

                "mean_nearest_slot_cosine":
                    float(
                        nearest_slot_cos.mean()
                    ),

                "max_nearest_slot_cosine":
                    float(
                        nearest_slot_cos.max()
                    ),

                "effective_rank":
                    eff_rank,

                "effective_rank_fraction":
                    float(
                        eff_rank / n_slots
                    ),
            }
        )

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

                    "mean_slot_cosine":
                        sub[
                            "mean_slot_cosine"
                        ].mean(),

                    "median_slot_cosine":
                        sub[
                            "median_slot_cosine"
                        ].mean(),

                    "mean_max_slot_cosine":
                        sub[
                            "max_slot_cosine"
                        ].mean(),

                    "mean_min_slot_cosine":
                        sub[
                            "min_slot_cosine"
                        ].mean(),

                    "mean_nearest_slot_cosine":
                        sub[
                            "mean_nearest_slot_cosine"
                        ].mean(),

                    "mean_slot_distance":
                        sub[
                            "mean_slot_distance"
                        ].mean(),

                    "mean_effective_rank":
                        sub[
                            "effective_rank"
                        ].mean(),

                    "mean_effective_rank_fraction":
                        sub[
                            "effective_rank_fraction"
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

    metrics = [
        "mean_slot_cosine",
        "median_slot_cosine",
        "mean_max_slot_cosine",
        "mean_min_slot_cosine",
        "mean_nearest_slot_cosine",
        "mean_slot_distance",
        "mean_effective_rank",
        "mean_effective_rank_fraction",
    ]

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
        }

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
        gap: pd.DataFrame,
):
    print("\n")
    print("=" * 120)
    print(
        "D7A: LOCAL SLOT REDUNDANCY BY CARDINALITY"
    )
    print("=" * 120)

    cols = [
        "card_bin",
        "valid_n",
        "test_n",

        "valid_mean_slot_cosine",
        "test_mean_slot_cosine",

        "valid_mean_nearest_slot_cosine",
        "test_mean_nearest_slot_cosine",

        "valid_mean_max_slot_cosine",
        "test_mean_max_slot_cosine",
    ]

    print(
        gap[
            cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n")
    print("=" * 120)
    print(
        "D7B: EFFECTIVE SLOT GEOMETRY"
    )
    print("=" * 120)

    cols = [
        "card_bin",

        "valid_mean_slot_distance",
        "test_mean_slot_distance",

        "valid_mean_effective_rank",
        "test_mean_effective_rank",

        "valid_mean_effective_rank_fraction",
        "test_mean_effective_rank_fraction",
    ]

    print(
        gap[
            cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
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
            "slot_geometry_bp"
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
            / "slot_geometry_per_protein.csv"
    )

    summary_path = (
            outdir
            / "slot_geometry_by_cardinality.csv"
    )

    gap_path = (
            outdir
            / "slot_geometry_valid_test_gap.csv"
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