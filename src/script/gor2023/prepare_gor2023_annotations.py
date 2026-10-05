#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

BRANCHES = ("mf", "bp", "cc")


def parse_train_file(path: Path):
    """
    Expected input format:
        Protein_ID    GO_ID    PMID

    Multiple rows may exist for the same PID-GO pair because the same
    annotation can be associated with multiple PubMed records.

    Returns:
        pid_to_go: dict[str, list[str]]
        stats: dict
    """

    pid_to_go_sets = defaultdict(set)

    total_rows = 0
    valid_rows = 0
    malformed_rows = 0

    unique_pid_go_pairs = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            total_rows += 1

            parts = line.split()

            if len(parts) < 2:
                malformed_rows += 1
                print(
                    f"[WARN] malformed row in {path} "
                    f"at line {line_no}: {line}"
                )
                continue

            pid = parts[0]
            go_id = parts[1]

            if not go_id.startswith("GO:"):
                malformed_rows += 1
                print(
                    f"[WARN] invalid GO ID in {path} "
                    f"at line {line_no}: {go_id}"
                )
                continue

            valid_rows += 1

            pid_to_go_sets[pid].add(go_id)
            unique_pid_go_pairs.add((pid, go_id))

    pid_to_go = {
        pid: sorted(go_ids)
        for pid, go_ids in sorted(pid_to_go_sets.items())
    }

    label_counts = [len(go_ids) for go_ids in pid_to_go.values()]

    unique_go_terms = {
        go_id
        for go_ids in pid_to_go.values()
        for go_id in go_ids
    }

    duplicate_pid_go_rows = valid_rows - len(unique_pid_go_pairs)

    stats = {
        "source_file": str(path),
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "malformed_rows": malformed_rows,
        "unique_proteins": len(pid_to_go),
        "unique_go_terms": len(unique_go_terms),
        "unique_pid_go_pairs": len(unique_pid_go_pairs),
        "duplicate_pid_go_rows": duplicate_pid_go_rows,
        "labels_per_protein": {
            "mean": mean(label_counts) if label_counts else 0.0,
            "median": median(label_counts) if label_counts else 0.0,
            "min": min(label_counts) if label_counts else 0,
            "max": max(label_counts) if label_counts else 0,
        },
    }

    return pid_to_go, stats


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert released GORetriever branch-specific training files "
            "into PID -> GO annotation dictionaries."
        )
    )

    parser.add_argument(
        "--gor_data_dir",
        type=Path,
        required=True,
        help=(
            "Directory containing mf_train.txt, bp_train.txt, cc_train.txt"
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for processed annotation dictionaries.",
    )

    args = parser.parse_args()

    all_stats = {}

    print("=" * 70)
    print("GOR2023 TRAIN ANNOTATION PREPROCESSING")
    print("=" * 70)

    for branch in BRANCHES:
        input_path = args.gor_data_dir / f"{branch}_train.txt"

        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing training file for {branch.upper()}: {input_path}"
            )

        print(f"\n[{branch.upper()}] reading {input_path}")

        pid_to_go, stats = parse_train_file(input_path)

        output_path = (
                args.out_dir
                / branch
                / "pid_to_go_train_released.json"
        )

        save_json(pid_to_go, output_path)

        all_stats[branch] = stats

        print(f"  proteins:             {stats['unique_proteins']:,}")
        print(f"  unique GO terms:      {stats['unique_go_terms']:,}")
        print(f"  PID-GO pairs:         {stats['unique_pid_go_pairs']:,}")
        print(f"  duplicate PID-GO:     {stats['duplicate_pid_go_rows']:,}")
        print(
            f"  labels/protein mean:  "
            f"{stats['labels_per_protein']['mean']:.2f}"
        )
        print(
            f"  labels/protein median:"
            f" {stats['labels_per_protein']['median']:.1f}"
        )
        print(
            f"  labels/protein range: "
            f"{stats['labels_per_protein']['min']} - "
            f"{stats['labels_per_protein']['max']}"
        )
        print(f"  saved -> {output_path}")

    audit_path = args.out_dir / "audit" / "train_annotation_stats.json"
    save_json(all_stats, audit_path)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for branch in BRANCHES:
        s = all_stats[branch]
        print(
            f"{branch.upper():>2}: "
            f"{s['unique_proteins']:,} proteins | "
            f"{s['unique_go_terms']:,} GO terms | "
            f"{s['unique_pid_go_pairs']:,} PID-GO pairs"
        )

    print(f"\nAudit saved -> {audit_path}")


if __name__ == "__main__":
    main()