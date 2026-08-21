#!/usr/bin/env python3
"""Canonicalize GOR2023 protein-to-GO annotations against GO 2023-01-01."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BRANCHES = ("mf", "bp", "cc")
EXPECTED_PROTEINS = {
    "mf": {"train": 78670, "test": 882},
    "bp": {"train": 88038, "test": 861},
    "cc": {"train": 80757, "test": 811},
}
EXPECTED_CANONICAL_TERMS = {
    "mf": 7228,
    "bp": 21308,
    "cc": 2945,
}


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, obj):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def read_ids(path):
    with path.open(encoding="utf-8") as handle:
        return {
            line.strip()
            for line in handle
            if line.strip() and not line.startswith("#")
        }


def load_branch_mappings(audit_tsv):
    mappings = defaultdict(dict)
    dropped = defaultdict(dict)

    with audit_tsv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["subset"] != "required":
                continue

            branch = row["branch"].strip().lower()
            input_id = row["input_id"].strip()
            resolved_id = row["resolved_id"].strip()

            if resolved_id:
                mappings[branch][input_id] = resolved_id
            else:
                dropped[branch][input_id] = {
                    "status": row["status"],
                    "consider": row["consider"],
                }

    return mappings, dropped


def canonicalize_split(
        annotations,
        mapping,
        dropped,
        canonical_universe,
        branch,
        split,
):
    output = {}
    stats = Counter()
    changed_pairs = Counter()
    dropped_pairs = Counter()
    zero_label_proteins = []

    for protein_id, raw_go_ids in annotations.items():
        stats["input_proteins"] += 1
        stats["input_annotations"] += len(raw_go_ids)

        canonical_go_ids = []

        for go_id in raw_go_ids:
            if go_id in dropped:
                stats["dropped_annotations"] += 1
                dropped_pairs[go_id] += 1
                continue

            if go_id not in mapping:
                raise RuntimeError(
                    f"{branch.upper()} {split}: GO ID absent from audit "
                    f"mapping: protein={protein_id}, go_id={go_id}"
                )

            canonical_id = mapping[go_id]

            if canonical_id != go_id:
                stats["canonicalized_annotations"] += 1
                changed_pairs[(go_id, canonical_id)] += 1

            if canonical_id not in canonical_universe:
                raise RuntimeError(
                    f"{branch.upper()} {split}: canonical GO ID is outside "
                    f"branch universe: {go_id} -> {canonical_id}"
                )

            canonical_go_ids.append(canonical_id)

        unique_go_ids = sorted(set(canonical_go_ids))
        stats["deduplicated_annotations"] += (
                len(canonical_go_ids) - len(unique_go_ids)
        )

        if not unique_go_ids:
            zero_label_proteins.append(protein_id)
            continue

        output[protein_id] = unique_go_ids
        stats["output_annotations"] += len(unique_go_ids)

    stats["output_proteins"] = len(output)
    stats["zero_label_proteins"] = len(zero_label_proteins)

    return {
        "annotations": output,
        "stats": dict(stats),
        "changed_pairs": {
            f"{source}->{target}": count
            for (source, target), count in sorted(changed_pairs.items())
        },
        "dropped_pairs": dict(sorted(dropped_pairs.items())),
        "zero_label_protein_ids": sorted(zero_label_proteins),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/GOR2023"),
    )
    parser.add_argument(
        "--audit-tsv",
        type=Path,
        default=Path(
            "/workspace/GOR2023/go/coverage_audit/"
            "go_coverage_details.tsv"
        ),
    )
    args = parser.parse_args()

    mappings, dropped = load_branch_mappings(args.audit_tsv)
    global_summary = {
        "ontology_release": "2023-01-01",
        "branches": {},
    }

    for branch in BRANCHES:
        branch_dir = args.root / branch
        canonical_ids_path = (
                args.root
                / "go"
                / "canonical"
                / f"{branch}_canonical_go_ids.txt"
        )
        canonical_universe = read_ids(canonical_ids_path)

        if len(canonical_universe) != EXPECTED_CANONICAL_TERMS[branch]:
            raise RuntimeError(
                f"{branch.upper()} canonical universe size mismatch: "
                f"{len(canonical_universe)} != "
                f"{EXPECTED_CANONICAL_TERMS[branch]}"
            )

        branch_summary = {
            "canonical_universe_terms": len(canonical_universe),
            "declared_dropped_terms": dropped[branch],
            "splits": {},
        }

        split_specs = {
            "train": (
                branch_dir / "pid_to_go_train_released.json",
                branch_dir / "pid_to_go_train_canonical.json",
            ),
            "test": (
                branch_dir / "pid_to_go_test.json",
                branch_dir / "pid_to_go_test_canonical.json",
            ),
        }

        for split, (input_path, output_path) in split_specs.items():
            raw = read_json(input_path)

            expected_proteins = EXPECTED_PROTEINS[branch][split]
            if len(raw) != expected_proteins:
                raise RuntimeError(
                    f"{branch.upper()} {split} protein count mismatch: "
                    f"{len(raw)} != {expected_proteins}"
                )

            result = canonicalize_split(
                annotations=raw,
                mapping=mappings[branch],
                dropped=dropped[branch],
                canonical_universe=canonical_universe,
                branch=branch,
                split=split,
            )

            if result["zero_label_protein_ids"]:
                raise RuntimeError(
                    f"{branch.upper()} {split}: proteins lost all labels: "
                    f"{result['zero_label_protein_ids'][:20]}"
                )

            if result["stats"]["output_proteins"] != expected_proteins:
                raise RuntimeError(
                    f"{branch.upper()} {split}: output protein count changed"
                )

            if split == "test":
                raw_normalized = {
                    pid: sorted(set(go_ids))
                    for pid, go_ids in raw.items()
                }
                if result["annotations"] != raw_normalized:
                    raise RuntimeError(
                        f"{branch.upper()} test gold changed during "
                        "canonicalization"
                    )

                for key in (
                        "canonicalized_annotations",
                        "dropped_annotations",
                        "deduplicated_annotations",
                ):
                    if result["stats"].get(key, 0) != 0:
                        raise RuntimeError(
                            f"{branch.upper()} test gold unexpectedly has "
                            f"{key}={result['stats'].get(key, 0)}"
                        )

            write_json(output_path, result["annotations"])

            branch_summary["splits"][split] = {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "stats": result["stats"],
                "changed_pairs": result["changed_pairs"],
                "dropped_pairs": result["dropped_pairs"],
            }

        summary_path = branch_dir / "canonicalization_summary.json"
        write_json(summary_path, branch_summary)
        global_summary["branches"][branch.upper()] = branch_summary

    global_summary_path = (
            args.root / "go" / "canonical" / "annotation_summary.json"
    )
    write_json(global_summary_path, global_summary)

    print("=" * 72)
    print("GOR2023 ANNOTATION CANONICALIZATION")
    print("=" * 72)

    for branch in BRANCHES:
        print(f"\n{branch.upper()}")
        for split in ("train", "test"):
            stats = global_summary["branches"][branch]["splits"][split][
                "stats"
            ]
            print(
                f"  {split:5s}: proteins "
                f"{stats['input_proteins']:,} -> "
                f"{stats['output_proteins']:,} | annotations "
                f"{stats['input_annotations']:,} -> "
                f"{stats['output_annotations']:,} | mapped="
                f"{stats.get('canonicalized_annotations', 0):,} | "
                f"dropped={stats.get('dropped_annotations', 0):,} | "
                f"deduplicated="
                f"{stats.get('deduplicated_annotations', 0):,}"
            )

    print(f"\nSummary: {global_summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
