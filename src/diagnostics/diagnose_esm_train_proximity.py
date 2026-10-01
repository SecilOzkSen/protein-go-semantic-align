#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CARD_BINS = [
    (1, 5, "1_5"),
    (6, 10, "6_10"),
    (11, 20, "11_20"),
    (21, 40, "21_40"),
    (41, 80, "41_80"),
    (81, 160, "81_160"),
    (161, None, "161plus"),
]


def card_bin(n):
    for lo, hi, label in CARD_BINS:
        if hi is None:
            if n >= lo:
                return label
        elif lo <= n <= hi:
            return label
    return "unknown"


def read_ids(path):
    with open(path) as f:
        return [
            line.strip().split()[0]
            for line in f
            if line.strip()
        ]


def load_pid2pos(path):
    with open(path) as f:
        return json.load(f)


def load_embedding_file(path):
    """
    Supports the common formats likely used in the residue store.
    """
    path = Path(path)

    if path.suffix == ".npy":
        x = np.load(path)

    elif path.suffix == ".npz":
        z = np.load(path)

        if "embedding" in z:
            x = z["embedding"]
        elif "embeddings" in z:
            x = z["embeddings"]
        else:
            # first array
            x = z[z.files[0]]

    elif path.suffix in {".pt", ".pth"}:
        obj = torch.load(
            path,
            map_location="cpu",
        )

        if torch.is_tensor(obj):
            x = obj.cpu().numpy()

        elif isinstance(obj, dict):
            for key in [
                "embedding",
                "embeddings",
                "residue_embedding",
                "residue_embeddings",
                "representations",
            ]:
                if key in obj:
                    val = obj[key]

                    if isinstance(val, dict):
                        # ESM-style representation dict.
                        val = val[
                            max(val.keys())
                        ]

                    if torch.is_tensor(val):
                        x = val.cpu().numpy()
                    else:
                        x = np.asarray(val)

                    break
            else:
                raise RuntimeError(
                    f"Cannot identify embedding in {path}"
                )
        else:
            raise RuntimeError(
                f"Unsupported torch object in {path}"
            )

    else:
        raise RuntimeError(
            f"Unsupported embedding format: {path}"
        )

    x = np.asarray(x)

    # Remove batch dimension if present.
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    if x.ndim != 2:
        raise RuntimeError(
            f"Expected [L,D], got {x.shape} from {path}"
        )

    return x.astype(
        np.float32,
        copy=False,
    )


def find_embedding(embed_dir, pid):
    candidates = [
        embed_dir / f"{pid}.npy",
        embed_dir / f"{pid}.npz",
        embed_dir / f"{pid}.pt",
        embed_dir / f"{pid}.pth",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def build_vectors(
        pids,
        embed_dir,
        pid2pos,
        split,
):
    vectors = []
    kept_pids = []
    n_gold = []

    missing = 0

    for i, pid in enumerate(pids):

        path = find_embedding(
            embed_dir,
            pid,
        )

        if path is None:
            missing += 1
            continue

        x = load_embedding_file(path)

        # Mean residue representation.
        v = x.mean(axis=0)

        norm = np.linalg.norm(v)

        if norm < 1e-8:
            continue

        v = v / norm

        positives = pid2pos.get(
            pid,
            [],
        )

        vectors.append(v)
        kept_pids.append(pid)
        n_gold.append(len(positives))

        if (
                (i + 1) % 2000 == 0
                or i + 1 == len(pids)
        ):
            print(
                f"[{split}] "
                f"{i + 1}/{len(pids)}"
            )

    if not vectors:
        raise RuntimeError(
            f"No embeddings loaded for {split}"
        )

    print(
        f"[{split}] kept={len(vectors)} "
        f"missing={missing}"
    )

    return (
        np.stack(vectors).astype(np.float32),
        kept_pids,
        np.asarray(n_gold),
    )


def nearest_train(
        query_vectors,
        train_vectors,
        device,
        query_bs=256,
        train_bs=4096,
):
    ks = [1, 5, 10, 50]
    max_k = max(ks)

    train = torch.from_numpy(
        train_vectors
    ).to(device)

    all_top = []

    for start in range(
            0,
            len(query_vectors),
            query_bs,
    ):

        q_np = query_vectors[
            start:start + query_bs
        ]

        q = torch.from_numpy(
            q_np
        ).to(device)

        # Keep global top-50 while scanning train chunks.
        running = None

        for ts in range(
                0,
                len(train_vectors),
                train_bs,
        ):

            t = train[
                ts:ts + train_bs
            ]

            sim = q @ t.T

            k_here = min(
                max_k,
                sim.shape[1],
            )

            vals = torch.topk(
                sim,
                k=k_here,
                dim=1,
            ).values

            if running is None:
                running = vals
            else:
                merged = torch.cat(
                    [running, vals],
                    dim=1,
                )

                running = torch.topk(
                    merged,
                    k=min(
                        max_k,
                        merged.shape[1],
                    ),
                    dim=1,
                ).values

        all_top.append(
            running.cpu().numpy()
        )

        print(
            f"[NN] "
            f"{min(start + query_bs, len(query_vectors))}"
            f"/{len(query_vectors)}"
        )

    top = np.concatenate(
        all_top,
        axis=0,
    )

    result = {}

    for k in ks:
        result[f"nn{k}_mean"] = (
            top[:, :k].mean(axis=1)
        )

    result["nn1"] = top[:, 0]

    return result


def make_df(
        split,
        pids,
        n_gold,
        nn,
):
    df = pd.DataFrame({
        "protein_id": pids,
        "split": split,
        "n_gold": n_gold,
        "nn1_cosine": nn["nn1"],
        "top5_train_cosine": nn["nn5_mean"],
        "top10_train_cosine": nn["nn10_mean"],
        "top50_train_cosine": nn["nn50_mean"],
    })

    df["card_bin"] = [
        card_bin(int(x))
        for x in df["n_gold"]
    ]

    return df


def summarize(df):
    rows = []

    for split in [
        "valid",
        "test",
    ]:

        d = df[
            df["split"] == split
            ]

        # Overall.
        rows.append({
            "split": split,
            "card_bin": "ALL",
            "n": len(d),
            "mean_n_gold":
                d.n_gold.mean(),
            "mean_nn1":
                d.nn1_cosine.mean(),
            "median_nn1":
                d.nn1_cosine.median(),
            "mean_top5":
                d.top5_train_cosine.mean(),
            "mean_top10":
                d.top10_train_cosine.mean(),
            "mean_top50":
                d.top50_train_cosine.mean(),
        })

        for _, _, label in CARD_BINS:

            sub = d[
                d["card_bin"] == label
                ]

            if len(sub) == 0:
                continue

            rows.append({
                "split": split,
                "card_bin": label,
                "n": len(sub),
                "mean_n_gold":
                    sub.n_gold.mean(),
                "mean_nn1":
                    sub.nn1_cosine.mean(),
                "median_nn1":
                    sub.nn1_cosine.median(),
                "mean_top5":
                    sub.top5_train_cosine.mean(),
                "mean_top10":
                    sub.top10_train_cosine.mean(),
                "mean_top50":
                    sub.top50_train_cosine.mean(),
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train_ids",
        default=(
            "/workspace/stargo/"
            "datasets/pfresgo/train.txt"
        ),
    )

    ap.add_argument(
        "--valid_ids",
        default=(
            "/workspace/stargo/"
            "datasets/pfresgo/valid.txt"
        ),
    )

    ap.add_argument(
        "--test_ids",
        default=(
            "/workspace/stargo/"
            "datasets/pfresgo/test.txt"
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
        "--embed_dir",
        default=(
            "/workspace/data_pfresgo/"
            "protein_embeddings/"
            "esm1b_residue"
        ),
    )

    ap.add_argument(
        "--outdir",
        default=(
            "/workspace/data_pfresgo/"
            "diagnostics/"
            "esm_train_proximity_bp"
        ),
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    args = ap.parse_args()

    embed_dir = Path(
        args.embed_dir
    )

    print("Loading IDs...")

    train_ids = read_ids(
        args.train_ids
    )

    valid_ids = read_ids(
        args.valid_ids
    )

    test_ids = read_ids(
        args.test_ids
    )

    print("Loading annotations...")

    pid2pos = load_pid2pos(
        args.pid2pos
    )

    print("\nBuilding TRAIN vectors...")

    train_vec, train_pid, train_ng = (
        build_vectors(
            train_ids,
            embed_dir,
            pid2pos,
            "train",
        )
    )

    print("\nBuilding VALID vectors...")

    valid_vec, valid_pid, valid_ng = (
        build_vectors(
            valid_ids,
            embed_dir,
            pid2pos,
            "valid",
        )
    )

    print("\nBuilding TEST vectors...")

    test_vec, test_pid, test_ng = (
        build_vectors(
            test_ids,
            embed_dir,
            pid2pos,
            "test",
        )
    )

    print(
        "\nComputing VALID -> TRAIN proximity..."
    )

    valid_nn = nearest_train(
        valid_vec,
        train_vec,
        args.device,
    )

    print(
        "\nComputing TEST -> TRAIN proximity..."
    )

    test_nn = nearest_train(
        test_vec,
        train_vec,
        args.device,
    )

    valid_df = make_df(
        "valid",
        valid_pid,
        valid_ng,
        valid_nn,
    )

    test_df = make_df(
        "test",
        test_pid,
        test_ng,
        test_nn,
    )

    all_df = pd.concat(
        [valid_df, test_df],
        ignore_index=True,
    )

    summary = summarize(
        all_df
    )

    print("\n")
    print("=" * 100)
    print(
        "D9A: ESM-1B PROXIMITY TO TRAINING PROTEINS"
    )
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Correlation with cardinality.
    print("\n")
    print("=" * 100)
    print(
        "CARDINALITY vs TRAIN PROXIMITY"
    )
    print("=" * 100)

    for split in [
        "valid",
        "test",
    ]:
        d = all_df[
            all_df["split"] == split
            ]

        corr = d[
            [
                "n_gold",
                "nn1_cosine",
                "top10_train_cosine",
            ]
        ].corr(
            method="spearman"
        )

        print(
            f"\n[{split}]"
        )

        print(
            "Spearman n_gold vs NN1: "
            f"{corr.loc['n_gold', 'nn1_cosine']:.4f}"
        )

        print(
            "Spearman n_gold vs Top10: "
            f"{corr.loc['n_gold', 'top10_train_cosine']:.4f}"
        )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_df.to_csv(
        outdir
        / "esm_train_proximity_per_protein.csv",
        index=False,
    )

    summary.to_csv(
        outdir
        / "esm_train_proximity_summary.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        outdir
        / "esm_train_proximity_per_protein.csv"
    )
    print(
        outdir
        / "esm_train_proximity_summary.csv"
    )


if __name__ == "__main__":
    main()