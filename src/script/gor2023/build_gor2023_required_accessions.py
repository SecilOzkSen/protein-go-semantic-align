#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

BRANCHES = ("mf", "bp", "cc")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_ids(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return {
            line.strip().split()[0]
            for line in f
            if line.strip()
        }


def save_ids(ids, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pid in sorted(ids):
            f.write(pid + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    all_train = set()
    all_test = set()

    branch_stats = {}

    for branch in BRANCHES:
        train_path = (
                args.processed_dir
                / branch
                / "pid_to_go_train_released.json"
        )

        test_path = (
                args.processed_dir
                / branch
                / "test_ids.txt"
        )

        train = set(load_json(train_path))
        test = read_ids(test_path)

        all_train.update(train)
        all_test.update(test)

        branch_stats[branch] = {
            "train": len(train),
            "test": len(test),
        }

    required = all_train | all_test

    save_ids(required, args.out)

    print("=" * 70)
    print("GOR2023 REQUIRED ACCESSIONS")
    print("=" * 70)

    for branch in BRANCHES:
        print(
            f"{branch.upper()}: "
            f"train={branch_stats[branch]['train']:,} | "
            f"test={branch_stats[branch]['test']:,}"
        )

    print()
    print(f"Unique train proteins: {len(all_train):,}")
    print(f"Unique test proteins:  {len(all_test):,}")
    print(f"Train/test overlap:    {len(all_train & all_test):,}")
    print(f"Total required:        {len(required):,}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()