#!/usr/bin/env python3

"""
D4C: Cross-modal protein <-> GO alignment diagnostic.

Question:
    Does protein-GO alignment deteriorate from VALID to TEST,
    particularly as annotation cardinality increases?

Uses frozen dump artifacts only. No model inference.

For each protein:
    - score all GO terms with global expert
    - score all GO terms with local-slot expert
    - reconstruct fused score
    - measure mean gold score
    - measure weakest gold score
    - measure hard-negative boundary
    - measure gold-vs-negative margins

Then summarize by protein annotation cardinality.
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

    eval_go_ids = np.load(
        dump_dir / "eval_go_ids.npy"
    ).astype(np.int64)

    go_z = np.load(
        dump_dir / "go_z.float16.npy"
    ).astype(np.float32)

    protein_global_z = np.load(
        dump_dir / "protein_global_z.float16.npy",
        mmap_mode="r",
    )

    protein_local_z = np.load(
        dump_dir / "protein_local_z.float16.npy",
        mmap_mode="r",
    )

    with open(
            dump_dir / "metadata.json"
    ) as f:
        metadata = json.load(f)

    n = len(protein_ids)

    if len(true_go_ids) != n:
        raise RuntimeError(
            "protein_ids / true_go_ids length mismatch"
        )

    if protein_global_z.shape[0] != n:
        raise RuntimeError(
            "protein_global_z row mismatch"
        )

    if protein_local_z.shape[0] != n:
        raise RuntimeError(
            "protein_local_z row mismatch"
        )

    if go_z.shape[0] != len(eval_go_ids):
        raise RuntimeError(
            "go_z / eval_go_ids mismatch"
        )

    # Normalize again for numerical safety.
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

    global_weight = metadata.get(
        "expert_global_weight",
        None,
    )

    if global_weight is None:
        raise RuntimeError(
            "metadata.json has no expert_global_weight"
        )

    global_weight = float(global_weight)

    return {
        "protein_ids": protein_ids,
        "true_go_ids": true_go_ids,
        "eval_go_ids": eval_go_ids,
        "go_z": go_z,
        "protein_global_z": protein_global_z,
        "protein_local_z": protein_local_z,
        "global_weight": global_weight,
        "metadata": metadata,
    }


def cardinality_bin(n_gold: int) -> str:
    for lo, hi, label in CARD_BINS:

        if hi is None:
            if n_gold >= lo:
                return label

        elif lo <= n_gold <= hi:
            return label

    return "unknown"


def lse_pool(scores, tau: float):
    """
    Local slot aggregation used by the retriever:

        tau * logsumexp(scores / tau)

    scores shape:
        [S, G]
    """

    x = scores / tau

    m = np.max(
        x,
        axis=0,
        keepdims=True,
    )

    lse = (
            m
            + np.log(
        np.exp(x - m).sum(
            axis=0,
            keepdims=True,
        )
    )
    )

    return (
            tau * lse.squeeze(0)
    )


def summarize_scores(
        scores: np.ndarray,
        gold_mask: np.ndarray,
        target_k: int = 200,
):
    """
    Per-protein alignment diagnostics.

    Boundary:
        target_k-th highest NON-GOLD score.

    Margins:
        mean gold - boundary
        weakest gold - boundary
    """

    gold_scores = scores[
        gold_mask
    ]

    neg_scores = scores[
        ~gold_mask
    ]

    if len(gold_scores) == 0:
        return None

    if len(neg_scores) == 0:
        return None

    k = min(
        int(target_k),
        len(neg_scores),
    )

    # kth highest non-gold.
    boundary = np.partition(
        neg_scores,
        len(neg_scores) - k,
    )[len(neg_scores) - k]

    mean_gold = float(
        np.mean(gold_scores)
    )

    median_gold = float(
        np.median(gold_scores)
    )

    weakest_gold = float(
        np.min(gold_scores)
    )

    strongest_gold = float(
        np.max(gold_scores)
    )

    mean_neg = float(
        np.mean(neg_scores)
    )

    max_neg = float(
        np.max(neg_scores)
    )

    return {
        "mean_gold": mean_gold,
        "median_gold": median_gold,
        "weakest_gold": weakest_gold,
        "strongest_gold": strongest_gold,

        "mean_neg": mean_neg,
        "max_neg": max_neg,

        "boundary": float(boundary),

        "mean_margin":
            mean_gold - boundary,

        "weakest_margin":
            weakest_gold - boundary,

        "gold_above_boundary_frac":
            float(
                np.mean(
                    gold_scores > boundary
                )
            ),
    }


def analyze_split(
        dump,
        split_name: str,
        local_tau: float,
        target_k: int,
):
    protein_ids = dump[
        "protein_ids"
    ]

    true_go_ids = dump[
        "true_go_ids"
    ]

    eval_go_ids = dump[
        "eval_go_ids"
    ]

    go_z = dump[
        "go_z"
    ]

    protein_global_z = dump[
        "protein_global_z"
    ]

    protein_local_z = dump[
        "protein_local_z"
    ]

    global_weight = dump[
        "global_weight"
    ]

    id_to_col = {
        int(g): i
        for i, g in enumerate(
            eval_go_ids
        )
    }

    rows = []

    print(
        f"\n[{split_name}] "
        f"N={len(protein_ids)} "
        f"global_weight={global_weight:.4f}"
    )

    for i, pid in enumerate(
            protein_ids
    ):

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

        gold_mask = np.zeros(
            len(eval_go_ids),
            dtype=bool,
        )

        gold_mask[
            gold_cols
        ] = True

        # -------------------------
        # GLOBAL
        # -------------------------

        pg = np.asarray(
            protein_global_z[i],
            dtype=np.float32,
        )

        pg = pg / max(
            np.linalg.norm(pg),
            1e-8,
        )

        global_scores = (
                go_z @ pg
        )

        # -------------------------
        # LOCAL
        # -------------------------

        pl = np.asarray(
            protein_local_z[i],
            dtype=np.float32,
        )

        pl_norm = np.linalg.norm(
            pl,
            axis=1,
            keepdims=True,
        )

        pl = pl / np.clip(
            pl_norm,
            1e-8,
            None,
        )

        # [S,D] @ [D,G] -> [S,G]
        slot_scores = (
                pl @ go_z.T
        )

        local_scores = lse_pool(
            slot_scores,
            tau=local_tau,
        )

        # -------------------------
        # FUSED
        # -------------------------

        fused_scores = (
                global_weight
                * global_scores
                +
                (1.0 - global_weight)
                * local_scores
        )

        expert_scores = {
            "global": global_scores,
            "local": local_scores,
            "fused": fused_scores,
        }

        base = {
            "protein_id": str(pid),
            "split": split_name,
            "n_gold": n_gold,
            "card_bin":
                cardinality_bin(n_gold),
        }

        for expert, scores in (
                expert_scores.items()
        ):

            stats = summarize_scores(
                scores,
                gold_mask,
                target_k=target_k,
            )

            if stats is None:
                continue

            for key, value in (
                    stats.items()
            ):
                base[
                    f"{expert}_{key}"
                ] = value

        rows.append(base)

        if (
                (i + 1) % 500 == 0
                or i + 1 == len(protein_ids)
        ):
            print(
                f"[{split_name}] "
                f"{i + 1}/{len(protein_ids)}"
            )

    return pd.DataFrame(rows)


def summarize_by_cardinality(
        df: pd.DataFrame,
):
    rows = []

    labels = [
        x[2]
        for x in CARD_BINS
    ]

    for split in [
        "valid",
        "test",
    ]:

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

            row = {
                "split": split,
                "card_bin": card_bin,
                "n_proteins": len(sub),
                "mean_n_gold":
                    sub["n_gold"].mean(),
                "median_n_gold":
                    sub["n_gold"].median(),
            }

            for expert in [
                "global",
                "local",
                "fused",
            ]:

                for metric in [
                    "mean_gold",
                    "boundary",
                    "mean_margin",
                    "weakest_margin",
                    "gold_above_boundary_frac",
                ]:
                    col = (
                        f"{expert}_{metric}"
                    )

                    row[
                        f"{expert}_{metric}"
                    ] = sub[col].mean()

            rows.append(row)

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

    labels = [
        x[2]
        for x in CARD_BINS
    ]

    rows = []

    for card_bin in labels:

        if (
                card_bin not in valid.index
                or card_bin not in test.index
        ):
            continue

        v = valid.loc[
            card_bin
        ]

        t = test.loc[
            card_bin
        ]

        row = {
            "card_bin": card_bin,

            "valid_n":
                int(v["n_proteins"]),

            "test_n":
                int(t["n_proteins"]),

            "valid_mean_n_gold":
                float(v["mean_n_gold"]),

            "test_mean_n_gold":
                float(t["mean_n_gold"]),
        }

        for expert in [
            "global",
            "local",
            "fused",
        ]:

            for metric in [
                "mean_gold",
                "boundary",
                "mean_margin",
                "weakest_margin",
                "gold_above_boundary_frac",
            ]:
                col = (
                    f"{expert}_{metric}"
                )

                vv = float(
                    v[col]
                )

                tt = float(
                    t[col]
                )

                row[
                    f"valid_{expert}_{metric}"
                ] = vv

                row[
                    f"test_{expert}_{metric}"
                ] = tt

                row[
                    f"gap_{expert}_{metric}"
                ] = vv - tt

        rows.append(row)

    return pd.DataFrame(rows)


def print_main_table(
        gap_df: pd.DataFrame,
):
    print("\n")
    print("=" * 115)
    print(
        "D4C: CROSS-MODAL ALIGNMENT "
        "BY PROTEIN CARDINALITY"
    )
    print("=" * 115)

    cols = [
        "card_bin",
        "valid_n",
        "test_n",

        "valid_fused_mean_gold",
        "test_fused_mean_gold",

        "valid_fused_boundary",
        "test_fused_boundary",

        "valid_fused_mean_margin",
        "test_fused_mean_margin",

        "valid_fused_gold_above_boundary_frac",
        "test_fused_gold_above_boundary_frac",
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
        "GLOBAL vs LOCAL MEAN MARGIN"
    )
    print("=" * 115)

    cols2 = [
        "card_bin",

        "valid_global_mean_margin",
        "test_global_mean_margin",

        "valid_local_mean_margin",
        "test_local_mean_margin",

        "valid_fused_mean_margin",
        "test_fused_mean_margin",
    ]

    print(
        gap_df[
            cols2
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
            "cross_modal_alignment_bp"
        ),
    )

    ap.add_argument(
        "--local_tau",
        type=float,
        default=0.10,
    )

    ap.add_argument(
        "--target_k",
        type=int,
        default=200,
    )

    args = ap.parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading VALID dump..."
    )

    valid_dump = load_dump(
        Path(args.valid_dump)
    )

    print(
        "Loading TEST dump..."
    )

    test_dump = load_dump(
        Path(args.test_dump)
    )

    print(
        "\nCheckpoint expert weights:"
    )

    print(
        "VALID global weight:",
        valid_dump["global_weight"],
    )

    print(
        "TEST global weight:",
        test_dump["global_weight"],
    )

    valid_df = analyze_split(
        valid_dump,
        split_name="valid",
        local_tau=args.local_tau,
        target_k=args.target_k,
    )

    test_df = analyze_split(
        test_dump,
        split_name="test",
        local_tau=args.local_tau,
        target_k=args.target_k,
    )

    protein_df = pd.concat(
        [
            valid_df,
            test_df,
        ],
        ignore_index=True,
    )

    summary_df = (
        summarize_by_cardinality(
            protein_df
        )
    )

    gap_df = make_gap_table(
        summary_df
    )

    protein_path = (
            outdir
            / "cross_modal_per_protein.csv"
    )

    summary_path = (
            outdir
            / "cross_modal_by_cardinality.csv"
    )

    gap_path = (
            outdir
            / "cross_modal_gap.csv"
    )

    protein_df.to_csv(
        protein_path,
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

    print_main_table(
        gap_df
    )

    print("\nSaved:")
    print(protein_path)
    print(summary_path)
    print(gap_path)


if __name__ == "__main__":
    main()