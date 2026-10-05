import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", required=True)
    args = p.parse_args()

    d = Path(args.dump_dir)

    top_labels = np.load(
        d / "top_labels.int8.npy",
        mmap_mode="r",
    )

    with open(d / "true_go_ids.json") as f:
        true_ids = json.load(f)

    n_true = np.asarray(
        [len(x) for x in true_ids],
        dtype=np.float64,
    )

    assert len(n_true) == top_labels.shape[0]

    print(f"N proteins: {len(n_true)}")
    print(f"Ranking width: {top_labels.shape[1]}")
    print(f"Mean gold/protein: {n_true.mean():.4f}")
    print(f"Median gold/protein: {np.median(n_true):.4f}")
    print()

    for k in [50, 100, 200, 500, 1000]:
        kk = min(k, top_labels.shape[1])

        hits = top_labels[:, :kk].sum(axis=1)

        valid = n_true > 0

        coverage_per_protein = (
                hits[valid] / n_true[valid]
        )

        coverage = coverage_per_protein.mean()

        any_hit = (
                hits[valid] > 0
        ).mean()

        print(
            f"K={k:4d}  "
            f"coverage={coverage:.4f}  "
            f"any_hit={any_hit:.4f}"
        )


if __name__ == "__main__":
    main()