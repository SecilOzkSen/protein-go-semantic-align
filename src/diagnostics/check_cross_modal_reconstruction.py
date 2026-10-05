import argparse
import json
from pathlib import Path

import numpy as np


def lse_pool(scores, tau=0.10):
    x = scores / tau
    m = np.max(x, axis=0, keepdims=True)

    return (
            tau * (
            m + np.log(
        np.exp(x - m).sum(
            axis=0,
            keepdims=True
        )
    )
    )
    ).squeeze(0)


def check_dump(dump_dir, tau=0.10, k=200):
    d = Path(dump_dir)

    with open(d / "metadata.json") as f:
        meta = json.load(f)

    with open(d / "true_go_ids.json") as f:
        true_ids = json.load(f)

    eval_ids = np.load(
        d / "eval_go_ids.npy"
    ).astype(np.int64)

    go_z = np.load(
        d / "go_z.float16.npy"
    ).astype(np.float32)

    global_z = np.load(
        d / "protein_global_z.float16.npy",
        mmap_mode="r",
    )

    local_z = np.load(
        d / "protein_local_z.float16.npy",
        mmap_mode="r",
    )

    actual_top_ids = np.load(
        d / "top_go_ids.int64.npy",
        mmap_mode="r",
    )

    # Numerical safety
    go_z /= np.clip(
        np.linalg.norm(
            go_z,
            axis=1,
            keepdims=True,
        ),
        1e-8,
        None,
    )

    w = float(
        meta["expert_global_weight"]
    )

    id_to_col = {
        int(g): i
        for i, g in enumerate(eval_ids)
    }

    overlaps = []
    reconstructed_coverages = []

    for i in range(len(true_ids)):

        pg = np.asarray(
            global_z[i],
            dtype=np.float32,
        )

        pg /= max(
            np.linalg.norm(pg),
            1e-8,
        )

        pl = np.asarray(
            local_z[i],
            dtype=np.float32,
        )

        pl /= np.clip(
            np.linalg.norm(
                pl,
                axis=1,
                keepdims=True,
            ),
            1e-8,
            None,
        )

        global_scores = go_z @ pg

        slot_scores = pl @ go_z.T

        local_scores = lse_pool(
            slot_scores,
            tau=tau,
        )

        fused_scores = (
                w * global_scores
                + (1.0 - w) * local_scores
        )

        # reconstructed Top-K
        top_cols = np.argpartition(
            fused_scores,
            -k,
        )[-k:]

        recon_ids = set(
            eval_ids[top_cols].tolist()
        )

        actual_ids = set(
            np.asarray(
                actual_top_ids[i, :k]
            ).astype(int).tolist()
        )

        overlap = (
                len(recon_ids & actual_ids) / k
        )

        overlaps.append(overlap)

        gold = {
            int(g)
            for g in true_ids[i]
            if int(g) >= 0
        }

        if gold:
            reconstructed_coverages.append(
                len(gold & recon_ids)
                / len(gold)
            )

    print(
        f"N proteins: {len(true_ids)}"
    )

    print(
        f"Mean Top-{k} overlap with actual dump: "
        f"{np.mean(overlaps):.6f}"
    )

    print(
        f"Median Top-{k} overlap: "
        f"{np.median(overlaps):.6f}"
    )

    print(
        f"Min Top-{k} overlap: "
        f"{np.min(overlaps):.6f}"
    )

    print(
        f"Reconstructed Coverage@{k}: "
        f"{np.mean(reconstructed_coverages):.6f}"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dump_dir",
        required=True,
    )

    p.add_argument(
        "--tau",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--k",
        type=int,
        default=200,
    )

    args = p.parse_args()

    check_dump(
        args.dump_dir,
        tau=args.tau,
        k=args.k,
    )


if __name__ == "__main__":
    main()