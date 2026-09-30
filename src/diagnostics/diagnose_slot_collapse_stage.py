#!/usr/bin/env python3

"""
D8: Locate where local-slot collapse occurs.

We already know that final projected local slots are nearly identical.
This diagnostic asks where that collapse begins:

    residue attention
        ->
    raw slot vectors
        ->
    protein_ln
        ->
    proj_p
        ->
    normalized projected slots

This script works directly from an existing full-ranking dump.

IMPORTANT:
The current dump contains final projected slots, but not raw slots or
attention maps. Therefore, if the dump does not contain the intermediate
tensors, this script will report exactly what is available instead of
pretending to reconstruct them.

This first version also performs several geometry checks on the tensors
we already have, including:
    - final projected slot cosine
    - slot difference magnitude
    - per-dimension variance across slots
    - effective rank
    - float16 quantization sanity check

If raw/attention dump tensors exist, they are analyzed automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize(x: np.ndarray, axis=-1, eps=1e-8):
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def pairwise_offdiag_cos(x: np.ndarray):
    """
    x: [S, D]

    Returns off-diagonal cosine similarities.
    """
    z = normalize(x)
    sim = z @ z.T

    s = sim.shape[0]
    mask = ~np.eye(s, dtype=bool)

    return sim[mask]


def effective_rank(x: np.ndarray):
    """
    Entropy-based effective rank of slot matrix.

    x: [S, D]

    Returns ~1 when all slots span one direction.
    """
    x = x.astype(np.float64)

    # Normalize rows because our retrieval geometry is cosine-based.
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    x = x / np.clip(n, 1e-12, None)

    gram = x @ x.T

    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.clip(eigvals, 0.0, None)

    total = eigvals.sum()

    if total <= 1e-12:
        return np.nan

    p = eigvals / total
    p = p[p > 1e-12]

    h = -np.sum(p * np.log(p))

    return float(np.exp(h))


def slot_geometry(x: np.ndarray):
    """
    x: [S,D]

    Compute multiple measures so we do not rely only on cosine.
    """
    x32 = x.astype(np.float32)

    cos = pairwise_offdiag_cos(x32)

    # Difference between every pair of slots.
    diffs = []

    S = x32.shape[0]

    for i in range(S):
        for j in range(i + 1, S):
            diffs.append(
                np.linalg.norm(x32[i] - x32[j])
            )

    diffs = np.asarray(diffs, dtype=np.float32)

    # How much does each embedding dimension vary across slots?
    dim_var = np.var(
        x32,
        axis=0,
    )

    return {
        "mean_cos":
            float(np.mean(cos)),

        "median_cos":
            float(np.median(cos)),

        "min_cos":
            float(np.min(cos)),

        "max_cos":
            float(np.max(cos)),

        "mean_pair_l2":
            float(np.mean(diffs)),

        "max_pair_l2":
            float(np.max(diffs)),

        "mean_dim_variance":
            float(np.mean(dim_var)),

        "max_dim_variance":
            float(np.max(dim_var)),

        "effective_rank":
            effective_rank(x32),
    }


def summarize_stage(
        tensor: np.ndarray,
        stage_name: str,
        max_proteins: int | None,
):
    """
    tensor expected:
        [N,S,D]

    For attention maps this can also be [N,S,T].
    """

    if tensor.ndim != 3:
        raise RuntimeError(
            f"{stage_name}: expected [N,S,D/T], "
            f"got {tensor.shape}"
        )

    N = tensor.shape[0]

    if max_proteins is not None:
        N = min(N, max_proteins)

    rows = []

    for i in range(N):
        x = np.asarray(
            tensor[i],
            dtype=np.float32,
        )

        geom = slot_geometry(x)

        geom["protein_index"] = i
        geom["stage"] = stage_name

        rows.append(geom)

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame):
    rows = []

    for stage, sub in df.groupby("stage"):
        rows.append(
            {
                "stage": stage,
                "n_proteins": len(sub),

                "mean_slot_cosine":
                    sub["mean_cos"].mean(),

                "median_slot_cosine":
                    sub["median_cos"].mean(),

                "mean_min_slot_cosine":
                    sub["min_cos"].mean(),

                "mean_max_slot_cosine":
                    sub["max_cos"].mean(),

                "mean_pair_l2":
                    sub["mean_pair_l2"].mean(),

                "mean_max_pair_l2":
                    sub["max_pair_l2"].mean(),

                "mean_dim_variance":
                    sub["mean_dim_variance"].mean(),

                "mean_effective_rank":
                    sub["effective_rank"].mean(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dump_dir",
        default=(
            "/workspace/data_pfresgo/"
            "diagnostics/"
            "coverage_top200_full_ranking_valid"
        ),
    )

    ap.add_argument(
        "--max_proteins",
        type=int,
        default=500,
    )

    ap.add_argument(
        "--outdir",
        default=(
            "/workspace/data_pfresgo/"
            "diagnostics/"
            "slot_collapse_stage_bp"
        ),
    )

    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("D8: SLOT COLLAPSE STAGE DIAGNOSTIC")
    print("=" * 80)

    print("\nDump:")
    print(dump_dir)

    # -------------------------------------------------------------
    # Find available tensors
    # -------------------------------------------------------------

    candidate_files = {
        "attention": [
            "protein_slot_attn.float16.npy",
            "slot_attn.float16.npy",
            "protein_local_attn.float16.npy",
        ],

        "raw_slots": [
            "protein_raw_slots.float16.npy",
            "protein_slots_raw.float16.npy",
            "raw_slots.float16.npy",
        ],

        "projected_slots": [
            "protein_local_z.float16.npy",
        ],
    }

    available = {}

    print("\nSearching for intermediate tensors...")

    for stage, names in candidate_files.items():

        found = None

        for name in names:
            p = dump_dir / name

            if p.exists():
                found = p
                break

        available[stage] = found

        print(
            f"{stage:18s}: "
            f"{str(found) if found else 'NOT SAVED'}"
        )

    # -------------------------------------------------------------
    # We must at least have projected slots.
    # -------------------------------------------------------------

    if available["projected_slots"] is None:
        raise RuntimeError(
            "protein_local_z.float16.npy not found."
        )

    all_rows = []

    # -------------------------------------------------------------
    # Analyze anything that exists.
    # -------------------------------------------------------------

    for stage in [
        "attention",
        "raw_slots",
        "projected_slots",
    ]:

        path = available[stage]

        if path is None:
            continue

        print(f"\nLoading {stage}:")
        print(path)

        tensor = np.load(
            path,
            mmap_mode="r",
        )

        print(
            f"shape={tensor.shape} "
            f"dtype={tensor.dtype}"
        )

        df = summarize_stage(
            tensor=tensor,
            stage_name=stage,
            max_proteins=args.max_proteins,
        )

        all_rows.append(df)

    all_df = pd.concat(
        all_rows,
        ignore_index=True,
    )

    summary = aggregate(all_df)

    # -------------------------------------------------------------
    # Float16 sanity test
    # -------------------------------------------------------------
    #
    # D7 used a float16 dump. We should make sure the near-identical
    # geometry is not merely an artifact of loading fp16 values.
    #
    # This cannot recover information already lost before dumping,
    # but it quantifies the actual numerical differences remaining.
    # -------------------------------------------------------------

    proj_path = available["projected_slots"]

    projected = np.load(
        proj_path,
        mmap_mode="r",
    )

    n_check = min(
        args.max_proteins,
        projected.shape[0],
    )

    raw_abs_diffs = []

    relative_diffs = []

    for i in range(n_check):

        x = np.asarray(
            projected[i],
            dtype=np.float32,
        )

        S = x.shape[0]

        for a in range(S):
            for b in range(a + 1, S):
                d = np.linalg.norm(
                    x[a] - x[b]
                )

                base = (
                               np.linalg.norm(x[a])
                               + np.linalg.norm(x[b])
                       ) / 2.0

                raw_abs_diffs.append(d)

                relative_diffs.append(
                    d / max(base, 1e-8)
                )

    raw_abs_diffs = np.asarray(
        raw_abs_diffs
    )

    relative_diffs = np.asarray(
        relative_diffs
    )

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("D8A: GEOMETRY BY REPRESENTATION STAGE")
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    print("\n")
    print("=" * 100)
    print("D8B: PROJECTED SLOT NUMERICAL DIFFERENCES")
    print("=" * 100)

    print(
        f"Proteins checked: {n_check}"
    )

    print(
        "Mean pair absolute L2 difference: "
        f"{raw_abs_diffs.mean():.8e}"
    )

    print(
        "Median pair absolute L2 difference: "
        f"{np.median(raw_abs_diffs):.8e}"
    )

    print(
        "Max pair absolute L2 difference: "
        f"{raw_abs_diffs.max():.8e}"
    )

    print(
        "Mean relative pair difference: "
        f"{relative_diffs.mean():.8e}"
    )

    print(
        "Median relative pair difference: "
        f"{np.median(relative_diffs):.8e}"
    )

    # -------------------------------------------------------------
    # Interpretation helper
    # -------------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("D8C: WHAT CAN CURRENT DUMP TELL US?")
    print("=" * 100)

    has_attn = (
            available["attention"] is not None
    )

    has_raw = (
            available["raw_slots"] is not None
    )

    if has_attn and has_raw:

        print(
            "Attention, raw slots and projected slots are all available."
        )

        print(
            "We can directly localize where collapse begins."
        )

    elif has_raw:

        print(
            "Raw and projected slots are available, "
            "but attention maps are not."
        )

        print(
            "We can distinguish raw-slot collapse from "
            "projection collapse, but not attention collapse."
        )

    else:

        print(
            "Only final projected slots are available."
        )

        print(
            "The existing dump confirms final rank-one collapse, "
            "but cannot determine whether collapse starts in "
            "attention, raw slot extraction, or projection."
        )

        print(
            "\nNEXT REQUIRED ACTION:"
        )

        print(
            "Dump slot attention and raw slot vectors for a small "
            "sample from the checkpoint. No retraining is required."
        )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    per_path = (
            outdir
            / "slot_collapse_stage_per_protein.csv"
    )

    summary_path = (
            outdir
            / "slot_collapse_stage_summary.csv"
    )

    all_df.to_csv(
        per_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nSaved:")
    print(per_path)
    print(summary_path)


if __name__ == "__main__":
    main()