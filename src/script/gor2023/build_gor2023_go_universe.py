#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

BRANCHES = ("mf", "bp", "cc")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_ids(ids, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for go_id in sorted(ids):
            f.write(go_id + "\n")


def collect_terms(pid_to_go):
    terms = set()
    counts = Counter()

    for go_ids in pid_to_go.values():
        for go_id in go_ids:
            terms.add(go_id)
            counts[go_id] += 1

    return terms, counts


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build branch-specific GO term universes for GOR2023 "
            "from released training annotations and exact test gold labels."
        )
    )

    parser.add_argument(
        "--processed_dir",
        type=Path,
        required=True,
        help="Processed GOR2023 directory.",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for GO universe files.",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GOR2023 GO UNIVERSE CONSTRUCTION")
    print("=" * 72)

    all_train_terms = set()
    all_test_terms = set()
    all_required_terms = set()

    summary = {}

    for branch in BRANCHES:

        train_path = (
                args.processed_dir
                / branch
                / "pid_to_go_train_released.json"
        )

        test_path = (
                args.processed_dir
                / branch
                / "pid_to_go_test.json"
        )

        if not train_path.exists():
            raise FileNotFoundError(train_path)

        if not test_path.exists():
            raise FileNotFoundError(test_path)

        train = load_json(train_path)
        test = load_json(test_path)

        train_terms, train_counts = collect_terms(train)
        test_terms, test_counts = collect_terms(test)

        required_terms = train_terms | test_terms

        test_only = test_terms - train_terms
        train_only = train_terms - test_terms
        shared = train_terms & test_terms

        all_train_terms |= train_terms
        all_test_terms |= test_terms
        all_required_terms |= required_terms

        branch_dir = args.out_dir / branch
        branch_dir.mkdir(parents=True, exist_ok=True)

        save_ids(
            train_terms,
            branch_dir / "train_go_ids.txt",
        )

        save_ids(
            test_terms,
            branch_dir / "test_go_ids.txt",
        )

        save_ids(
            required_terms,
            branch_dir / "required_go_ids.txt",
        )

        save_ids(
            test_only,
            branch_dir / "test_only_go_ids.txt",
        )

        save_json(
            dict(sorted(train_counts.items())),
            branch_dir / "train_go_frequency.json",
        )

        save_json(
            dict(sorted(test_counts.items())),
            branch_dir / "test_go_frequency.json",
        )

        summary[branch] = {
            "train_proteins": len(train),
            "test_proteins": len(test),
            "train_go_terms": len(train_terms),
            "test_go_terms": len(test_terms),
            "required_go_terms": len(required_terms),
            "shared_train_test_go_terms": len(shared),
            "train_only_go_terms": len(train_only),
            "test_only_go_terms": len(test_only),
            "test_only_go_examples": sorted(test_only)[:50],
        }

        print(f"\n{branch.upper()}")
        print(f"  train proteins:       {len(train):,}")
        print(f"  test proteins:        {len(test):,}")
        print(f"  train GO terms:       {len(train_terms):,}")
        print(f"  test GO terms:        {len(test_terms):,}")
        print(f"  required GO terms:    {len(required_terms):,}")
        print(f"  shared GO terms:      {len(shared):,}")
        print(f"  train-only GO terms:  {len(train_only):,}")
        print(f"  test-only GO terms:   {len(test_only):,}")

    #
    # Cross-branch sanity checks.
    #
    mf_terms = set(
        (args.out_dir / "mf" / "required_go_ids.txt")
        .read_text()
        .splitlines()
    )

    bp_terms = set(
        (args.out_dir / "bp" / "required_go_ids.txt")
        .read_text()
        .splitlines()
    )

    cc_terms = set(
        (args.out_dir / "cc" / "required_go_ids.txt")
        .read_text()
        .splitlines()
    )

    cross_branch_overlap = {
        "mf_bp": sorted(mf_terms & bp_terms),
        "mf_cc": sorted(mf_terms & cc_terms),
        "bp_cc": sorted(bp_terms & cc_terms),
    }

    summary["global"] = {
        "unique_train_go_terms": len(all_train_terms),
        "unique_test_go_terms": len(all_test_terms),
        "unique_required_go_terms": len(all_required_terms),
        "mf_bp_overlap": len(cross_branch_overlap["mf_bp"]),
        "mf_cc_overlap": len(cross_branch_overlap["mf_cc"]),
        "bp_cc_overlap": len(cross_branch_overlap["bp_cc"]),
    }

    save_ids(
        all_required_terms,
        args.out_dir / "all_required_go_ids.txt",
    )

    save_json(
        summary,
        args.out_dir / "go_universe_summary.json",
    )

    save_json(
        cross_branch_overlap,
        args.out_dir / "cross_branch_overlap.json",
    )

    print("\n" + "=" * 72)
    print("GLOBAL SUMMARY")
    print("=" * 72)

    print(
        f"Unique train GO terms:    "
        f"{len(all_train_terms):,}"
    )
    print(
        f"Unique test GO terms:     "
        f"{len(all_test_terms):,}"
    )
    print(
        f"Unique required GO terms: "
        f"{len(all_required_terms):,}"
    )

    print()
    print(
        f"MF/BP GO overlap: {len(cross_branch_overlap['mf_bp']):,}"
    )
    print(
        f"MF/CC GO overlap: {len(cross_branch_overlap['mf_cc']):,}"
    )
    print(
        f"BP/CC GO overlap: {len(cross_branch_overlap['bp_cc']):,}"
    )

    # GO IDs should belong to exactly one namespace.
    if (
            cross_branch_overlap["mf_bp"]
            or cross_branch_overlap["mf_cc"]
            or cross_branch_overlap["bp_cc"]
    ):
        print(
            "\n[WARNING] GO IDs occur across multiple branch universes. "
            "This must be checked against the ontology snapshot."
        )
    else:
        print("\nCross-branch GO ID sanity check: PASS")

    print(f"\nOutputs -> {args.out_dir}")


if __name__ == "__main__":
    main()