#!/usr/bin/env python3

"""
D4A: GO-side shared embedding geometry diagnostic.

Question:
    Are projected GO representations sufficiently separated in the
    retriever's shared embedding space?

    In particular:
        - Is the GO space globally collapsed / anisotropic?
        - Are specific high-IC GO terms more crowded than general terms?
        - Does GO specificity correlate with local embedding density?

Inputs:
    Retriever full-ranking dump:
        go_z.float16.npy
        eval_go_ids.npy

    D2 GO information-content table:
        go_information_content.csv

No model inference is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_go_id(x) -> str:
    if isinstance(x, (int, np.integer)):
        return f"GO:{int(x):07d}"

    x = str(x).strip()

    if x.startswith("GO:"):
        return x

    try:
        return f"GO:{int(x):07d}"
    except ValueError:
        return x


def load_data(
        dump_dir: Path,
        ic_csv: Path,
):
    dump_dir = Path(dump_dir)

    go_z = np.load(
        dump_dir / "go_z.float16.npy"
    ).astype(np.float32)

    go_ids = np.load(
        dump_dir / "eval_go_ids.npy"
    ).astype(np.int64)

    if go_z.shape[0] != len(go_ids):
        raise RuntimeError(
            f"go_z rows={go_z.shape[0]} "
            f"but eval_go_ids={len(go_ids)}"
        )

    # Safety: normalize again.
    norms = np.linalg.norm(
        go_z,
        axis=1,
        keepdims=True,
    )

    zero_mask = norms.squeeze(1) < 1e-8

    if zero_mask.any():
        raise RuntimeError(
            f"Found {zero_mask.sum()} zero GO vectors."
        )

    go_z = go_z / norms

    df = pd.DataFrame(
        {
            "go_id_int": go_ids,
            "go_id": [
                normalize_go_id(x)
                for x in go_ids
            ],
        }
    )

    ic = pd.read_csv(ic_csv)
    ic["go_id"] = ic["go_id"].map(
        normalize_go_id
    )

    keep_cols = [
        "go_id",
        "depth",
        "train_count_direct",
        "train_count_propagated",
        "ic_direct",
        "ic_propagated",
    ]

    ic = ic[
        [c for c in keep_cols if c in ic.columns]
    ].drop_duplicates("go_id")

    df = df.merge(
        ic,
        on="go_id",
        how="left",
        validate="one_to_one",
    )

    return go_z, df


def cosine_geometry(go_z):
    """
    Since vectors are normalized:
        cosine = Z @ Z.T
    """

    sim = go_z @ go_z.T

    # Numerical cleanup.
    sim = np.clip(
        sim,
        -1.0,
        1.0,
    )

    n = sim.shape[0]

    # Remove self similarity for neighbour calculations.
    sim_no_self = sim.copy()

    np.fill_diagonal(
        sim_no_self,
        -np.inf,
    )

    # All unique off-diagonal pairs.
    tri_i, tri_j = np.triu_indices(
        n,
        k=1,
    )

    offdiag = sim[
        tri_i,
        tri_j,
    ]

    return sim_no_self, offdiag


def compute_local_density(
        sim_no_self,
        ks=(1, 5, 10, 50),
):
    """
    For every GO:
      nearest-neighbour cosine
      mean cosine of top-5 / top-10 / top-50 neighbours
    """

    n = sim_no_self.shape[0]

    max_k = min(
        max(ks),
        n - 1,
    )

    # Largest max_k similarities per GO.
    top = np.partition(
        sim_no_self,
        kth=n - max_k,
        axis=1,
    )[:, -max_k:]

    # Sort descending only within the selected neighbours.
    top = np.sort(
        top,
        axis=1,
    )[:, ::-1]

    out = {}

    for k in ks:
        kk = min(k, top.shape[1])

        out[f"nn_mean_top{k}"] = (
            top[:, :kk].mean(axis=1)
        )

    out["nearest_neighbor_cosine"] = (
        top[:, 0]
    )

    return out


def assign_ic_bins(
        df,
        n_bins=4,
):
    finite = (
        df["ic_propagated"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if len(finite) == 0:
        raise RuntimeError(
            "No finite ic_propagated values."
        )

    quantiles = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    edges = finite.quantile(
        quantiles
    ).to_numpy(dtype=float)

    edges = np.unique(edges)

    if len(edges) < 3:
        raise RuntimeError(
            f"Too few unique IC boundaries: {edges}"
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    n_actual = len(edges) - 1

    if n_actual == 4:
        labels = [
            "Q1_general",
            "Q2",
            "Q3",
            "Q4_specific",
        ]
    else:
        labels = [
            f"Q{i + 1}"
            for i in range(n_actual)
        ]

    df = df.copy()

    df["ic_bin"] = pd.cut(
        df["ic_propagated"],
        bins=edges,
        labels=labels,
        include_lowest=True,
    )

    return df, labels, edges


def summarize_by_ic(
        df,
        labels,
):
    rows = []

    for label in labels:

        sub = df[
            df["ic_bin"] == label
            ]

        if len(sub) == 0:
            continue

        rows.append(
            {
                "ic_bin": str(label),

                "n_go": len(sub),

                "mean_ic":
                    sub["ic_propagated"].mean(),

                "mean_depth":
                    sub["depth"].mean(),

                "mean_nn1":
                    sub[
                        "nearest_neighbor_cosine"
                    ].mean(),

                "median_nn1":
                    sub[
                        "nearest_neighbor_cosine"
                    ].median(),

                "mean_top5_cos":
                    sub[
                        "nn_mean_top5"
                    ].mean(),

                "mean_top10_cos":
                    sub[
                        "nn_mean_top10"
                    ].mean(),

                "mean_top50_cos":
                    sub[
                        "nn_mean_top50"
                    ].mean(),
            }
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dump_dir",
        default=(
            "/workspace/data_pfresgo/diagnostics/"
            "coverage_top200_full_ranking_valid"
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
            "go_space_geometry_bp"
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

    print("Loading projected GO space...")

    go_z, df = load_data(
        Path(args.dump_dir),
        Path(args.ic_csv),
    )

    print(
        f"GO embeddings: {go_z.shape}"
    )

    print("\nComputing full GO-GO cosine matrix...")

    sim_no_self, offdiag = (
        cosine_geometry(go_z)
    )

    print("\n==============================")
    print("GLOBAL GO SPACE")
    print("==============================")

    print(
        f"Off-diagonal pairs: "
        f"{len(offdiag):,}"
    )

    print(
        f"Mean cosine:   "
        f"{offdiag.mean():.4f}"
    )

    print(
        f"Median cosine: "
        f"{np.median(offdiag):.4f}"
    )

    print(
        f"Std cosine:    "
        f"{offdiag.std():.4f}"
    )

    for q in [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    ]:
        print(
            f"P{int(q * 100):02d}:           "
            f"{np.quantile(offdiag, q):.4f}"
        )

    print(
        f"Min cosine:    "
        f"{offdiag.min():.4f}"
    )

    print(
        f"Max cosine:    "
        f"{offdiag.max():.4f}"
    )

    density = compute_local_density(
        sim_no_self,
        ks=(1, 5, 10, 50),
    )

    for key, values in density.items():
        df[key] = values

    df, labels, edges = assign_ic_bins(
        df,
        n_bins=args.n_bins,
    )

    print("\n==============================")
    print("IC BIN BOUNDARIES")
    print("==============================")

    for i, label in enumerate(labels):
        print(
            f"{label:12s}: "
            f"({edges[i]:.4f}, "
            f"{edges[i + 1]:.4f}]"
        )

    summary = summarize_by_ic(
        df,
        labels,
    )

    print("\n")
    print("=" * 95)
    print(
        "D4A: GO SPACE DENSITY BY SPECIFICITY"
    )
    print("=" * 95)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Correlations.
    corr_df = df[
        [
            "ic_propagated",
            "depth",
            "nearest_neighbor_cosine",
            "nn_mean_top10",
            "nn_mean_top50",
        ]
    ].dropna()

    print("\n==============================")
    print("SPEARMAN CORRELATIONS")
    print("==============================")

    if len(corr_df) > 2:
        corr = corr_df.corr(
            method="spearman"
        )

        print(
            "IC vs nearest-neighbour cosine: "
            f"{corr.loc['ic_propagated', 'nearest_neighbor_cosine']:.4f}"
        )

        print(
            "IC vs top-10 density:           "
            f"{corr.loc['ic_propagated', 'nn_mean_top10']:.4f}"
        )

        print(
            "IC vs top-50 density:           "
            f"{corr.loc['ic_propagated', 'nn_mean_top50']:.4f}"
        )

        print(
            "Depth vs nearest-neighbour:     "
            f"{corr.loc['depth', 'nearest_neighbor_cosine']:.4f}"
        )

    # Save.
    per_go_path = (
            outdir
            / "go_geometry_per_term.csv"
    )

    summary_path = (
            outdir
            / "go_geometry_by_specificity.csv"
    )

    df.to_csv(
        per_go_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nSaved:")
    print(per_go_path)
    print(summary_path)


if __name__ == "__main__":
    main()