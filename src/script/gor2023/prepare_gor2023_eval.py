#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

BRANCHES = ("mf", "bp", "cc")


def read_pid_file(path: Path):
    pids = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if not parts:
                continue

            pids.append(parts[0])

    return pids


def read_gold_file(path: Path):
    """
    Expected format:
        Protein_ID    GO_ID    ontology    numeric_id

    Returns:
        branch_to_pid_go:
            {
                "mf": {pid: set(go_ids)},
                "bp": {pid: set(go_ids)},
                "cc": {pid: set(go_ids)}
            }
    """

    branch_to_pid_go = {
        branch: defaultdict(set)
        for branch in BRANCHES
    }

    total_rows = 0
    malformed_rows = 0
    invalid_branch_rows = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            total_rows += 1
            parts = line.split()

            if len(parts) < 3:
                malformed_rows += 1
                print(
                    f"[WARN] malformed gold row at line {line_no}: {line}"
                )
                continue

            pid = parts[0]
            go_id = parts[1]
            branch = parts[2].lower()

            if branch not in BRANCHES:
                invalid_branch_rows += 1
                print(
                    f"[WARN] invalid ontology at line {line_no}: {branch}"
                )
                continue

            if not go_id.startswith("GO:"):
                malformed_rows += 1
                print(
                    f"[WARN] invalid GO ID at line {line_no}: {go_id}"
                )
                continue

            branch_to_pid_go[branch][pid].add(go_id)

    stats = {
        "total_rows": total_rows,
        "malformed_rows": malformed_rows,
        "invalid_branch_rows": invalid_branch_rows,
    }

    return branch_to_pid_go, stats


def load_train_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )


def save_id_list(ids, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for pid in sorted(ids):
            f.write(pid + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare exact GOR2023 standard evaluation sets, difficult subsets, "
            "gold annotation dictionaries, and protein-level leakage audits."
        )
    )

    parser.add_argument(
        "--gor_data_dir",
        type=Path,
        required=True,
        help=(
            "Root directory containing golden/pid_go.txt, "
            "golden/text/*.pid, golden/diffcult/*.pid"
        ),
    )

    parser.add_argument(
        "--processed_dir",
        type=Path,
        required=True,
        help=(
            "Processed GOR2023 directory containing branch-specific "
            "pid_to_go_train_released.json files."
        ),
    )

    args = parser.parse_args()

    golden_dir = args.gor_data_dir / "golden"
    gold_path = golden_dir / "pid_go.txt"

    if not gold_path.exists():
        raise FileNotFoundError(f"Missing gold file: {gold_path}")

    print("=" * 72)
    print("GOR2023 EVALUATION DATASET PREPARATION")
    print("=" * 72)

    branch_gold, raw_gold_stats = read_gold_file(gold_path)

    eval_stats = {}
    leakage_stats = {}

    for branch in BRANCHES:
        print(f"\n[{branch.upper()}]")

        test_pid_path = golden_dir / "text" / f"{branch}.pid"
        difficult_pid_path = golden_dir / "diffcult" / f"{branch}.pid"

        train_json_path = (
                args.processed_dir
                / branch
                / "pid_to_go_train_released.json"
        )

        for required_path in [
            test_pid_path,
            difficult_pid_path,
            train_json_path,
        ]:
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Missing required file: {required_path}"
                )

        test_ids_raw = read_pid_file(test_pid_path)
        difficult_ids_raw = read_pid_file(difficult_pid_path)

        test_ids = set(test_ids_raw)
        difficult_ids = set(difficult_ids_raw)

        train_dict = load_train_json(train_json_path)
        train_ids = set(train_dict)

        full_gold_dict = branch_gold[branch]
        full_gold_ids = set(full_gold_dict)

        test_gold_dict = {
            pid: sorted(full_gold_dict[pid])
            for pid in sorted(test_ids)
            if pid in full_gold_dict
        }

        test_gold_ids = set(test_gold_dict)

        difficult_gold_dict = {
            pid: sorted(full_gold_dict[pid])
            for pid in sorted(difficult_ids)
            if pid in full_gold_dict
        }

        train_test_overlap = train_ids & test_ids
        train_difficult_overlap = train_ids & difficult_ids

        test_without_gold = test_ids - full_gold_ids
        gold_without_test = full_gold_ids - test_ids

        difficult_without_test = difficult_ids - test_ids
        difficult_without_gold = difficult_ids - full_gold_ids

        duplicate_test_ids = len(test_ids_raw) - len(test_ids)
        duplicate_difficult_ids = (
                len(difficult_ids_raw) - len(difficult_ids)
        )

        n_test_gold_pairs = sum(
            len(go_ids)
            for go_ids in test_gold_dict.values()
        )

        n_difficult_gold_pairs = sum(
            len(go_ids)
            for go_ids in difficult_gold_dict.values()
        )

        branch_eval_stats = {
            "train_proteins": len(train_ids),
            "standard_test_proteins": len(test_ids),
            "difficult_proteins": len(difficult_ids),
            "gold_pool_proteins": len(full_gold_ids),
            "standard_test_gold_proteins": len(test_gold_ids),
            "standard_test_gold_pairs": n_test_gold_pairs,
            "difficult_gold_pairs": n_difficult_gold_pairs,
            "gold_pool_without_standard_test": len(gold_without_test),
            "duplicate_standard_test_ids": duplicate_test_ids,
            "duplicate_difficult_ids": duplicate_difficult_ids,
        }

        branch_leakage_stats = {
            "train_test_overlap_count": len(train_test_overlap),
            "train_difficult_overlap_count": len(
                train_difficult_overlap
            ),
            "test_without_gold_count": len(test_without_gold),
            "gold_without_test_count": len(gold_without_test),
            "difficult_without_test_count": len(
                difficult_without_test
            ),
            "difficult_without_gold_count": len(
                difficult_without_gold
            ),
            "difficult_is_subset_of_test": (
                    difficult_ids <= test_ids
            ),
            "train_test_overlap_examples": sorted(
                train_test_overlap
            )[:20],
            "train_difficult_overlap_examples": sorted(
                train_difficult_overlap
            )[:20],
            "test_without_gold_examples": sorted(
                test_without_gold
            )[:20],
            "difficult_without_test_examples": sorted(
                difficult_without_test
            )[:20],
            "difficult_without_gold_examples": sorted(
                difficult_without_gold
            )[:20],
        }

        eval_stats[branch] = branch_eval_stats
        leakage_stats[branch] = branch_leakage_stats

        branch_dir = args.processed_dir / branch

        save_json(
            test_gold_dict,
            branch_dir / "pid_to_go_test.json",
        )

        save_json(
            difficult_gold_dict,
            branch_dir / "pid_to_go_difficult.json",
        )

        save_id_list(
            test_ids,
            branch_dir / "test_ids.txt",
        )

        save_id_list(
            difficult_ids,
            branch_dir / "difficult_ids.txt",
        )

        print(f"  train proteins:            {len(train_ids):,}")
        print(f"  standard test proteins:    {len(test_ids):,}")
        print(f"  difficult proteins:        {len(difficult_ids):,}")
        print(f"  gold-pool proteins:        {len(full_gold_ids):,}")
        print(f"  test gold PID-GO pairs:    {n_test_gold_pairs:,}")
        print(f"  train ∩ test:              {len(train_test_overlap):,}")
        print(
            f"  train ∩ difficult:         "
            f"{len(train_difficult_overlap):,}"
        )
        print(f"  test without gold:         {len(test_without_gold):,}")
        print(
            f"  difficult subset of test:  "
            f"{difficult_ids <= test_ids}"
        )

        #
        # Hard safety assertions
        #
        assert len(train_test_overlap) == 0, (
            f"{branch.upper()}: train/test protein leakage detected."
        )

        assert len(train_difficult_overlap) == 0, (
            f"{branch.upper()}: train/difficult leakage detected."
        )

        assert len(test_without_gold) == 0, (
            f"{branch.upper()}: standard test proteins without gold labels."
        )

        assert len(difficult_without_test) == 0, (
            f"{branch.upper()}: difficult set is not a subset of standard test."
        )

        assert len(difficult_without_gold) == 0, (
            f"{branch.upper()}: difficult proteins missing gold labels."
        )

    audit_dir = args.processed_dir / "audit"

    save_json(
        {
            "raw_gold_file": raw_gold_stats,
            "branches": eval_stats,
        },
        audit_dir / "eval_dataset_stats.json",
    )

    save_json(
        leakage_stats,
        audit_dir / "leakage_report.json",
    )

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    for branch in BRANCHES:
        e = eval_stats[branch]
        l = leakage_stats[branch]

        print(
            f"{branch.upper():>2}: "
            f"train={e['train_proteins']:,} | "
            f"test={e['standard_test_proteins']:,} | "
            f"difficult={e['difficult_proteins']:,} | "
            f"train∩test={l['train_test_overlap_count']}"
        )

    print("\nEvaluation dataset preparation completed successfully.")
    print(f"Processed data -> {args.processed_dir}")
    print(f"Audit reports  -> {audit_dir}")


if __name__ == "__main__":
    main()