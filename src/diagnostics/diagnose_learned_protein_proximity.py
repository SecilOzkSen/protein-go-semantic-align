#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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


def load_dump(path: Path):
    with open(path / "protein_ids.json") as f:
        pids = json.load(f)

    global_z = np.load(
        path / "protein_global_z.float16.npy",
        mmap_mode="r",
    )

    local_z = np.load(
        path / "protein_local_z.float16.npy",
        mmap_mode="r",
    )

    if len(pids) != global_z.shape[0]:
        raise RuntimeError(
            f"{path}: PID/global mismatch"
        )

    if len(pids) != local_z.shape[0]:
        raise RuntimeError(
            f"{path}: PID/local mismatch"
        )

    print(
        f"{path.name}: "
        f"N={len(pids)} "
        f"global={global_z.shape} "
        f"local={local_z.shape}"
    )

    return pids, global_z, local_z


def normalize(x):
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    n = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )

    return x / np.clip(
        n,
        1e-8,
        None,
    )


def global_proximity(
        query,
        train,
        device,
        query_bs=256,
        train_bs=4096,
):
    query = normalize(query)
    train = normalize(train)

    train_t = torch.from_numpy(
        train
    ).to(device)

    ks = [1, 5, 10, 50]
    max_k = max(ks)

    output = []

    for qs in range(
            0,
            len(query),
            query_bs,
    ):

        q = torch.from_numpy(
            query[
                qs:qs + query_bs
            ]
        ).to(device)

        running = None

        for ts in range(
                0,
                len(train),
                train_bs,
        ):

            t = train_t[
                ts:ts + train_bs
            ]

            sim = q @ t.T

            vals = torch.topk(
                sim,
                k=min(
                    max_k,
                    sim.shape[1],
                ),
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

        output.append(
            running.cpu().numpy()
        )

        print(
            f"[global NN] "
            f"{min(qs + query_bs, len(query))}"
            f"/{len(query)}"
        )

    top = np.concatenate(
        output,
        axis=0,
    )

    return {
        "global_nn1": top[:, 0],
        "global_top5": top[:, :5].mean(1),
        "global_top10": top[:, :10].mean(1),
        "global_top50": top[:, :50].mean(1),
    }


def local_proximity(
        query_local,
        train_local,
        device,
        query_bs=32,
        train_protein_bs=512,
):
    """
    For each query protein:

    For every query slot:
        find its best matching slot among ALL slots
        of ALL training proteins.

    Then summarize across the query's slots.

    This preserves the multi-vector nature of the
    learned local representation.
    """

    q = normalize(query_local)
    t = normalize(train_local)

    Nq, S, D = q.shape
    Nt = t.shape[0]

    # Flatten all TRAIN slots:
    # [Nt, S, D] -> [Nt*S, D]
    t_flat = t.reshape(
        Nt * S,
        D,
    )

    t_tensor = torch.from_numpy(
        t_flat
    ).to(device)

    rows = []

    # Number of train slots per chunk.
    slot_bs = train_protein_bs * S

    for qs in range(
            0,
            Nq,
            query_bs,
    ):

        q_np = q[
            qs:qs + query_bs
        ]

        B = q_np.shape[0]

        # [B,S,D] -> [B*S,D]
        q_flat = torch.from_numpy(
            q_np.reshape(
                B * S,
                D,
            )
        ).to(device)

        best = torch.full(
            (B * S,),
            -float("inf"),
            device=device,
        )

        for ts in range(
                0,
                len(t_flat),
                slot_bs,
        ):
            train_chunk = t_tensor[
                ts:ts + slot_bs
            ]

            sim = (
                    q_flat
                    @ train_chunk.T
            )

            best = torch.maximum(
                best,
                sim.max(
                    dim=1
                ).values,
            )

        # [B*S] -> [B,S]
        best = best.reshape(
            B,
            S,
        )

        rows.append(
            {
                "mean":
                    best.mean(
                        dim=1
                    ).cpu().numpy(),

                "min":
                    best.min(
                        dim=1
                    ).values.cpu().numpy(),

                "max":
                    best.max(
                        dim=1
                    ).values.cpu().numpy(),
            }
        )

        print(
            f"[local NN] "
            f"{min(qs + query_bs, Nq)}"
            f"/{Nq}"
        )

    return {
        "local_slot_nn_mean":
            np.concatenate(
                [x["mean"] for x in rows]
            ),

        "local_slot_nn_min":
            np.concatenate(
                [x["min"] for x in rows]
            ),

        "local_slot_nn_max":
            np.concatenate(
                [x["max"] for x in rows]
            ),
    }


def make_df(
        split,
        pids,
        pid2pos,
        global_metrics,
        local_metrics,
):
    n_gold = np.asarray([
        len(
            pid2pos.get(
                pid,
                [],
            )
        )
        for pid in pids
    ])

    df = pd.DataFrame({
        "protein_id": pids,
        "split": split,
        "n_gold": n_gold,
    })

    for key, values in (
            global_metrics.items()
    ):
        df[key] = values

    for key, values in (
            local_metrics.items()
    ):
        df[key] = values

    df["card_bin"] = [
        card_bin(int(x))
        for x in n_gold
    ]

    return df


def summarize(df):
    metrics = [
        "global_nn1",
        "global_top5",
        "global_top10",
        "global_top50",
        "local_slot_nn_mean",
        "local_slot_nn_min",
        "local_slot_nn_max",
    ]

    rows = []

    for split in [
        "valid",
        "test",
    ]:

        d = df[
            df.split == split
            ]

        groups = [
            ("ALL", d)
        ]

        for _, _, label in CARD_BINS:
            groups.append(
                (
                    label,
                    d[
                        d.card_bin == label
                        ],
                )
            )

        for label, sub in groups:

            if len(sub) == 0:
                continue

            row = {
                "split": split,
                "card_bin": label,
                "n": len(sub),
                "mean_n_gold":
                    sub.n_gold.mean(),
            }

            for metric in metrics:
                row[
                    f"mean_{metric}"
                ] = sub[metric].mean()

            rows.append(row)

    return pd.DataFrame(rows)


def make_gap(summary):
    v = (
        summary[
            summary.split == "valid"
            ]
        .set_index("card_bin")
    )

    t = (
        summary[
            summary.split == "test"
            ]
        .set_index("card_bin")
    )

    metrics = [
        "mean_global_nn1",
        "mean_global_top10",
        "mean_global_top50",
        "mean_local_slot_nn_mean",
        "mean_local_slot_nn_min",
    ]

    rows = []

    labels = [
        "ALL",
        "1_5",
        "6_10",
        "11_20",
        "21_40",
        "41_80",
        "81_160",
        "161plus",
    ]

    for label in labels:

        if (
                label not in v.index
                or label not in t.index
        ):
            continue

        row = {
            "card_bin": label,
            "valid_n":
                int(v.loc[label, "n"]),
            "test_n":
                int(t.loc[label, "n"]),
        }

        for metric in metrics:
            vv = float(
                v.loc[label, metric]
            )

            tt = float(
                t.loc[label, metric]
            )

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


def main():
    ap = argparse.ArgumentParser()

    base = (
        "/workspace/data_pfresgo/"
        "diagnostics"
    )

    ap.add_argument(
        "--train_dump",
        default=(
            f"{base}/"
            "slotdiv005_step70572_train"
        ),
    )

    ap.add_argument(
        "--valid_dump",
        default=(
            f"{base}/"
            "slotdiv005_step70572_valid"
        ),
    )

    ap.add_argument(
        "--test_dump",
        default=(
            f"{base}/"
            "slotdiv005_step70572_test"
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
        default=(
            f"{base}/"
            "learned_protein_proximity_bp"
        ),
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    args = ap.parse_args()

    print("Loading dumps...")

    (
        train_pids,
        train_global,
        train_local,
    ) = load_dump(
        Path(args.train_dump)
    )

    (
        valid_pids,
        valid_global,
        valid_local,
    ) = load_dump(
        Path(args.valid_dump)
    )

    (
        test_pids,
        test_global,
        test_local,
    ) = load_dump(
        Path(args.test_dump)
    )

    print("\nProtein counts:")
    print("TRAIN:", len(train_pids))
    print("VALID:", len(valid_pids))
    print("TEST :", len(test_pids))

    with open(args.pid2pos) as f:
        pid2pos = json.load(f)

    print(
        "\nVALID -> TRAIN global..."
    )

    vg = global_proximity(
        valid_global,
        train_global,
        args.device,
    )

    print(
        "\nTEST -> TRAIN global..."
    )

    tg = global_proximity(
        test_global,
        train_global,
        args.device,
    )

    print(
        "\nVALID -> TRAIN local slots..."
    )

    vl = local_proximity(
        valid_local,
        train_local,
        args.device,
    )

    print(
        "\nTEST -> TRAIN local slots..."
    )

    tl = local_proximity(
        test_local,
        train_local,
        args.device,
    )

    valid_df = make_df(
        "valid",
        valid_pids,
        pid2pos,
        vg,
        vl,
    )

    test_df = make_df(
        "test",
        test_pids,
        pid2pos,
        tg,
        tl,
    )

    all_df = pd.concat(
        [valid_df, test_df],
        ignore_index=True,
    )

    summary = summarize(
        all_df
    )

    gap = make_gap(
        summary
    )

    print("\n")
    print("=" * 120)
    print(
        "D9B: LEARNED PROTEIN SPACE "
        "PROXIMITY TO TRAIN"
    )
    print("=" * 120)

    display_cols = [
        "split",
        "card_bin",
        "n",
        "mean_global_nn1",
        "mean_global_top10",
        "mean_local_slot_nn_mean",
        "mean_local_slot_nn_min",
    ]

    print(
        summary[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n")
    print("=" * 120)
    print(
        "D9B: TEST - VALID GAP"
    )
    print("=" * 120)

    gap_cols = [
        "card_bin",
        "valid_n",
        "test_n",
        "test_minus_valid_mean_global_nn1",
        "test_minus_valid_mean_global_top10",
        "test_minus_valid_mean_local_slot_nn_mean",
        "test_minus_valid_mean_local_slot_nn_min",
    ]

    print(
        gap[
            gap_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Cardinality correlation.
    print("\n")
    print("=" * 120)
    print(
        "CARDINALITY vs LEARNED-SPACE "
        "TRAIN PROXIMITY"
    )
    print("=" * 120)

    for split in [
        "valid",
        "test",
    ]:

        d = all_df[
            all_df.split == split
            ]

        for metric in [
            "global_nn1",
            "global_top10",
            "local_slot_nn_mean",
            "local_slot_nn_min",
        ]:
            rho = d[
                ["n_gold", metric]
            ].corr(
                method="spearman"
            ).iloc[0, 1]

            print(
                f"{split:5s} "
                f"n_gold vs {metric:24s} "
                f"rho={rho:+.4f}"
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
        / "learned_proximity_per_protein.csv",
        index=False,
    )

    summary.to_csv(
        outdir
        / "learned_proximity_summary.csv",
        index=False,
    )

    gap.to_csv(
        outdir
        / "learned_proximity_gap.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        outdir
        / "learned_proximity_per_protein.csv"
    )
    print(
        outdir
        / "learned_proximity_summary.csv"
    )
    print(
        outdir
        / "learned_proximity_gap.csv"
    )


if __name__ == "__main__":
    main()