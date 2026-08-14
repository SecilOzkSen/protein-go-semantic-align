# src/script/analyze_train_similarity_shift.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
        "Analyze train-nearest protein similarity vs retrieval recall."
    )

    p.add_argument("--train_dump", required=True)
    p.add_argument("--val_dump", required=True)
    p.add_argument("--test_dump", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--query_batch_size", type=int, default=512)
    p.add_argument("--train_chunk_size", type=int, default=4096)

    return p.parse_args()


def card_bin(n: int) -> str:
    for lo, hi, name in CARD_BINS:
        if lo <= n <= hi:
            return name
    return "0"


def load_split(dump_dir: str | Path, split: str):
    d = Path(dump_dir)

    with (d / "protein_ids.json").open("r", encoding="utf-8") as f:
        pids = [str(x) for x in json.load(f)]

    protein_z = np.load(
        d / "protein_z.float16.npy",
        mmap_mode="r",
    )

    true_go_ids = np.load(
        d / "true_go_ids.npy",
        mmap_mode="r",
    )

    top_labels = np.load(
        d / "top_labels.int8.npy",
        mmap_mode="r",
    )

    rows = []

    for i, pid in enumerate(pids):
        true_row = np.asarray(true_go_ids[i], dtype=np.int64)
        full_card = int((true_row >= 0).sum())

        labels = np.asarray(top_labels[i], dtype=np.int8)
        cand_pos = int((labels > 0).sum())

        recall = (
            cand_pos / full_card
            if full_card > 0
            else np.nan
        )

        rows.append({
            "split": split,
            "row_index": i,
            "protein_id": pid,
            "full_cardinality": full_card,
            "candidate_positives": cand_pos,
            "candidate_recall": recall,
            "cardinality_bin": card_bin(full_card),
        })

    return pd.DataFrame(rows), protein_z


@torch.no_grad()
def nearest_train_similarity(
        query_z: np.ndarray,
        train_z: np.ndarray,
        device: torch.device,
        query_batch_size: int,
        train_chunk_size: int,
):
    train_np = np.asarray(train_z, dtype=np.float32)
    query_np = np.asarray(query_z, dtype=np.float32)

    n_query = query_np.shape[0]
    out = np.full(n_query, -1.0, dtype=np.float32)

    for qs in range(0, n_query, query_batch_size):
        qe = min(qs + query_batch_size, n_query)

        q = torch.from_numpy(
            query_np[qs:qe]
        ).to(device)

        q = F.normalize(q, dim=-1)

        best = torch.full(
            (q.size(0),),
            -1.0,
            dtype=torch.float32,
            device=device,
        )

        for ts in range(0, train_np.shape[0], train_chunk_size):
            te = min(ts + train_chunk_size, train_np.shape[0])

            t = torch.from_numpy(
                train_np[ts:te]
            ).to(device)

            t = F.normalize(t, dim=-1)

            sims = q @ t.T
            chunk_best = sims.max(dim=1).values
            best = torch.maximum(best, chunk_best)

        out[qs:qe] = best.detach().cpu().numpy()

        print(
            f"processed queries {qe}/{n_query}"
        )

    return out


def summarize_bins(df: pd.DataFrame):
    rows = []

    for split in ["val", "test"]:
        sx = df[df["split"] == split]

        for _, _, bin_name in CARD_BINS:
            x = sx[
                sx["cardinality_bin"] == bin_name
                ]

            if len(x) == 0:
                continue

            rows.append({
                "split": split,
                "cardinality_bin": bin_name,
                "N": int(len(x)),
                "mean_nearest_train_sim": float(
                    x["nearest_train_similarity"].mean()
                ),
                "median_nearest_train_sim": float(
                    x["nearest_train_similarity"].median()
                ),
                "macro_candidate_recall": float(
                    x["candidate_recall"].mean()
                ),
            })

    return pd.DataFrame(rows)


def add_similarity_quartiles(df: pd.DataFrame):
    out = df.copy()
    out["similarity_quartile"] = None

    for split in ["val", "test"]:
        for _, _, bin_name in CARD_BINS:
            mask = (
                    (out["split"] == split)
                    & (out["cardinality_bin"] == bin_name)
            )

            x = out.loc[
                mask,
                "nearest_train_similarity",
            ]

            if len(x) < 4:
                continue

            ranked = x.rank(method="first")

            q = pd.qcut(
                ranked,
                q=4,
                labels=[
                    "Q1_farthest",
                    "Q2",
                    "Q3",
                    "Q4_nearest",
                ],
            )

            out.loc[
                mask,
                "similarity_quartile",
            ] = q.astype(str).values

    return out


def summarize_similarity_quartiles(df: pd.DataFrame):
    rows = []

    for split in ["val", "test"]:
        sx = df[df["split"] == split]

        for _, _, bin_name in CARD_BINS:
            for q in [
                "Q1_farthest",
                "Q2",
                "Q3",
                "Q4_nearest",
            ]:
                x = sx[
                    (sx["cardinality_bin"] == bin_name)
                    & (sx["similarity_quartile"] == q)
                    ]

                if len(x) == 0:
                    continue

                rows.append({
                    "split": split,
                    "cardinality_bin": bin_name,
                    "similarity_quartile": q,
                    "N": int(len(x)),
                    "mean_similarity": float(
                        x["nearest_train_similarity"].mean()
                    ),
                    "macro_candidate_recall": float(
                        x["candidate_recall"].mean()
                    ),
                })

    return pd.DataFrame(rows)


def spearman_per_bin(df: pd.DataFrame):
    rows = []

    for split in ["val", "test"]:
        sx = df[df["split"] == split]

        for _, _, bin_name in CARD_BINS:
            x = sx[
                sx["cardinality_bin"] == bin_name
                ]

            if len(x) < 5:
                continue

            corr = x[
                [
                    "nearest_train_similarity",
                    "candidate_recall",
                ]
            ].corr(method="spearman").iloc[0, 1]

            rows.append({
                "split": split,
                "cardinality_bin": bin_name,
                "N": int(len(x)),
                "spearman_similarity_vs_recall": float(corr),
            })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Loading train...")
    train_df, train_z = load_split(
        args.train_dump,
        "train",
    )

    print("Loading val...")
    val_df, val_z = load_split(
        args.val_dump,
        "val",
    )

    print("Loading test...")
    test_df, test_z = load_split(
        args.test_dump,
        "test",
    )

    print("Computing nearest-train similarity for val...")
    val_sim = nearest_train_similarity(
        val_z,
        train_z,
        device=device,
        query_batch_size=args.query_batch_size,
        train_chunk_size=args.train_chunk_size,
    )

    print("Computing nearest-train similarity for test...")
    test_sim = nearest_train_similarity(
        test_z,
        train_z,
        device=device,
        query_batch_size=args.query_batch_size,
        train_chunk_size=args.train_chunk_size,
    )

    val_df["nearest_train_similarity"] = val_sim
    test_df["nearest_train_similarity"] = test_sim

    combined = pd.concat(
        [val_df, test_df],
        ignore_index=True,
    )

    combined.to_csv(
        out_dir / "per_protein_similarity.csv",
        index=False,
    )

    bin_summary = summarize_bins(
        combined
    )

    bin_summary.to_csv(
        out_dir / "similarity_by_cardinality.csv",
        index=False,
    )

    combined_q = add_similarity_quartiles(
        combined
    )

    quart_summary = summarize_similarity_quartiles(
        combined_q
    )

    quart_summary.to_csv(
        out_dir / "similarity_quartiles.csv",
        index=False,
    )

    corr_df = spearman_per_bin(
        combined
    )

    corr_df.to_csv(
        out_dir / "similarity_recall_spearman.csv",
        index=False,
    )

    print("\n========================================")
    print("NEAREST-TRAIN SIMILARITY BY CARDINALITY")
    print("========================================\n")
    print(bin_summary.to_string(index=False))

    print("\n========================================")
    print("SIMILARITY QUARTILES")
    print("========================================\n")
    print(quart_summary.to_string(index=False))

    print("\n========================================")
    print("SPEARMAN: TRAIN SIMILARITY vs RECALL")
    print("========================================\n")
    print(corr_df.to_string(index=False))

    print("\nSaved under:", out_dir)


if __name__ == "__main__":
    main()