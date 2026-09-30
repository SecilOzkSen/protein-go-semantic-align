#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

STAGES = [
    ("query", "slot_stage_query.float32.npy"),
    ("attention", "slot_stage_attention.float32.npy"),
    ("weighted_raw", "slot_stage_weighted_raw.float32.npy"),
    ("extractor_out", "slot_stage_extractor_out.float32.npy"),
    ("protein_ln", "slot_stage_protein_ln.float32.npy"),
    ("projected", "slot_stage_projected.float32.npy"),
    ("normalized", "slot_stage_normalized.float32.npy"),
]


def normalize(x, eps=1e-12):
    x = x.astype(np.float64, copy=False)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def effective_rank(x):
    """
    x: [S,D]
    Effective rank of normalized slot geometry.
    """
    z = normalize(x)
    gram = z @ z.T

    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.clip(eigvals, 0.0, None)

    total = eigvals.sum()
    if total <= 1e-12:
        return np.nan

    p = eigvals / total
    p = p[p > 1e-12]

    return float(
        np.exp(
            -np.sum(p * np.log(p))
        )
    )


def geometry_for_protein(x):
    """
    x: [S,D]
    """
    z = normalize(x)

    sim = z @ z.T

    S = sim.shape[0]
    tri = np.triu_indices(S, k=1)

    pair_cos = sim[tri]

    pair_l2 = []

    for i in range(S):
        for j in range(i + 1, S):
            pair_l2.append(
                np.linalg.norm(x[i] - x[j])
            )

    pair_l2 = np.asarray(pair_l2)

    return {
        "mean_cos": float(pair_cos.mean()),
        "median_cos": float(np.median(pair_cos)),
        "min_cos": float(pair_cos.min()),
        "max_cos": float(pair_cos.max()),
        "mean_pair_l2": float(pair_l2.mean()),
        "max_pair_l2": float(pair_l2.max()),
        "effective_rank": effective_rank(x),
    }


def analyze_stage(name, path):
    arr = np.load(path)

    if arr.ndim != 3:
        raise RuntimeError(
            f"{name}: expected [N,S,D], got {arr.shape}"
        )

    rows = []

    for i in range(arr.shape[0]):
        g = geometry_for_protein(
            arr[i].astype(np.float64)
        )
        g["protein_index"] = i
        g["stage"] = name
        rows.append(g)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dump_dir",
        default=(
            "/workspace/data_pfresgo/"
            "diagnostics/"
            "d8_slot_stages_valid"
        ),
    )

    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)

    all_rows = []

    for name, filename in STAGES:

        path = dump_dir / filename

        if not path.exists():
            raise RuntimeError(
                f"Missing stage: {path}"
            )

        print(f"Analyzing {name}...")

        df = analyze_stage(
            name,
            path,
        )

        all_rows.append(df)

    per_protein = pd.concat(
        all_rows,
        ignore_index=True,
    )

    summary_rows = []

    for name, _ in STAGES:
        sub = per_protein[
            per_protein["stage"] == name
            ]

        summary_rows.append(
            {
                "stage": name,
                "n": len(sub),

                "mean_slot_cosine":
                    sub["mean_cos"].mean(),

                "median_slot_cosine":
                    sub["median_cos"].mean(),

                "mean_min_cosine":
                    sub["min_cos"].mean(),

                "mean_max_cosine":
                    sub["max_cos"].mean(),

                "mean_pair_l2":
                    sub["mean_pair_l2"].mean(),

                "mean_max_pair_l2":
                    sub["max_pair_l2"].mean(),

                "mean_effective_rank":
                    sub["effective_rank"].mean(),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print("\n")
    print("=" * 115)
    print("D8: SLOT COLLAPSE LOCATION")
    print("=" * 115)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    print("\n")
    print("=" * 115)
    print("STAGE-TO-STAGE CHANGE")
    print("=" * 115)

    for i in range(1, len(summary)):
        prev = summary.iloc[i - 1]
        cur = summary.iloc[i]

        print(
            f"{prev['stage']:15s} -> "
            f"{cur['stage']:15s} | "
            f"cos {prev['mean_slot_cosine']:.6f}"
            f" -> {cur['mean_slot_cosine']:.6f} | "
            f"rank {prev['mean_effective_rank']:.4f}"
            f" -> {cur['mean_effective_rank']:.4f}"
        )

    outdir = (
            dump_dir
            / "stage_analysis"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_protein.to_csv(
        outdir
        / "slot_stage_geometry_per_protein.csv",
        index=False,
    )

    summary.to_csv(
        outdir
        / "slot_stage_geometry_summary.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        outdir
        / "slot_stage_geometry_per_protein.csv"
    )
    print(
        outdir
        / "slot_stage_geometry_summary.csv"
    )


if __name__ == "__main__":
    main()