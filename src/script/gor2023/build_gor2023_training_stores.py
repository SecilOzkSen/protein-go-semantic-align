#!/usr/bin/env python3
"""Build GOR2023 retriever store artifacts and deterministic splits."""

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

BRANCHES = ("mf", "bp", "cc")
EXPECTED = {
    "mf": {"train_source": 78670, "test": 882},
    "bp": {"train_source": 88038, "test": 861},
    "cc": {"train_source": 80757, "test": 811},
}


def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def extract_quoted(value):
    value = clean_text(value)
    match = re.search(r'"(.*?)"', value)
    return clean_text(match.group(1)) if match else value


def parse_obo(path):
    terms = {}
    current = None

    def commit():
        if current and current.get("id"):
            for key in (
                    "is_a",
                    "part_of",
                    "alt_id",
                    "replaced_by",
                    "consider",
                    "synonym",
            ):
                current[key] = sorted(set(current[key]))
            terms[current["id"]] = current.copy()

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if line == "[Term]":
                commit()
                current = {
                    "id": None,
                    "name": "",
                    "namespace": "",
                    "definition": "",
                    "is_a": [],
                    "part_of": [],
                    "alt_id": [],
                    "replaced_by": [],
                    "consider": [],
                    "synonym": [],
                    "is_obsolete": False,
                }
                continue

            if line.startswith("[") and line.endswith("]"):
                commit()
                current = None
                continue

            if current is None or not line:
                continue

            if line.startswith("id: "):
                current["id"] = line[4:].strip()
            elif line.startswith("name: "):
                current["name"] = clean_text(line[6:])
            elif line.startswith("namespace: "):
                current["namespace"] = clean_text(line[11:])
            elif line.startswith("def: "):
                current["definition"] = extract_quoted(line[5:])
            elif line.startswith("is_a: "):
                current["is_a"].append(
                    line[6:].split(" ! ", 1)[0].strip()
                )
            elif line.startswith("relationship: part_of "):
                current["part_of"].append(
                    line[len("relationship: part_of "):]
                    .split(" ! ", 1)[0]
                    .strip()
                )
            elif line.startswith("alt_id: "):
                current["alt_id"].append(line[8:].strip())
            elif line.startswith("replaced_by: "):
                current["replaced_by"].append(
                    line[13:].split(" ! ", 1)[0].strip()
                )
            elif line.startswith("consider: "):
                current["consider"].append(
                    line[10:].split(" ! ", 1)[0].strip()
                )
            elif line.startswith("synonym: "):
                synonym = extract_quoted(line[9:])
                if synonym:
                    current["synonym"].append(synonym)
            elif line == "is_obsolete: true":
                current["is_obsolete"] = True

    commit()
    return terms


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, obj, pretty=False):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            obj,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
        )
        handle.write("\n")


def write_ids(path, ids):
    with path.open("w", encoding="utf-8") as handle:
        for protein_id in sorted(ids):
            handle.write(protein_id + "\n")


def constrained_validation_split(annotations, fraction, seed):
    """Select validation proteins while retaining every label in train."""
    protein_ids = sorted(annotations)
    target_size = round(len(protein_ids) * fraction)

    remaining_label_counts = Counter()
    for go_ids in annotations.values():
        remaining_label_counts.update(set(go_ids))

    # Prefer proteins whose labels are frequent, then use seeded random order
    # within equal rarity groups. This protects rare labels from being removed.
    rng = random.Random(seed)
    random_tiebreak = {protein_id: rng.random() for protein_id in protein_ids}

    def rarity_key(protein_id):
        counts = [
            remaining_label_counts[go_id]
            for go_id in set(annotations[protein_id])
        ]
        return (
            min(counts) if counts else 0,
            sum(1.0 / count for count in counts if count > 0),
            random_tiebreak[protein_id],
        )

    candidates = sorted(protein_ids, key=rarity_key, reverse=True)
    validation = set()

    for protein_id in candidates:
        if len(validation) >= target_size:
            break

        labels = set(annotations[protein_id])
        if all(remaining_label_counts[label] > 1 for label in labels):
            validation.add(protein_id)
            for label in labels:
                remaining_label_counts[label] -= 1

    if len(validation) != target_size:
        raise RuntimeError(
            f"Could not construct requested validation size: "
            f"selected={len(validation)}, target={target_size}"
        )

    train = set(protein_ids) - validation
    train_labels = {
        go_id
        for protein_id in train
        for go_id in annotations[protein_id]
    }
    validation_labels = {
        go_id
        for protein_id in validation
        for go_id in annotations[protein_id]
    }
    missing_from_train = sorted(validation_labels - train_labels)
    if missing_from_train:
        raise RuntimeError(
            "Validation contains labels absent from train: "
            f"{missing_from_train[:20]}"
        )

    return train, validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/GOR2023"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed = args.root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    obo_path = args.root / "go" / "ontology" / "go-2023-01-01.obo"
    go_vocab = parse_obo(obo_path)
    go_vocab_path = processed / "go_vocab.json"
    write_json(go_vocab_path, go_vocab)

    markerless_source = (
            args.root
            / "go"
            / "canonical"
            / "go_texts_canonical_markerless.jsonl"
    )
    markerless_target = processed / "go_texts_canonical.jsonl"
    shutil.copyfile(markerless_source, markerless_target)

    summary = {
        "seed": args.seed,
        "validation_fraction": args.val_fraction,
        "go_vocab_terms": len(go_vocab),
        "go_vocab_path": str(go_vocab_path),
        "go_text_path": str(markerless_target),
        "branches": {},
    }

    for offset, branch in enumerate(BRANCHES):
        branch_source = args.root / branch
        branch_out = processed / branch
        branch_out.mkdir(parents=True, exist_ok=True)

        train_annotations = read_json(
            branch_source / "pid_to_go_train_canonical.json"
        )
        test_annotations = read_json(
            branch_source / "pid_to_go_test_canonical.json"
        )

        if len(train_annotations) != EXPECTED[branch]["train_source"]:
            raise RuntimeError(f"{branch.upper()} train source count mismatch")
        if len(test_annotations) != EXPECTED[branch]["test"]:
            raise RuntimeError(f"{branch.upper()} test count mismatch")

        overlap = set(train_annotations).intersection(test_annotations)
        if overlap:
            raise RuntimeError(
                f"{branch.upper()} train/test overlap: {sorted(overlap)[:20]}"
            )

        train_ids, valid_ids = constrained_validation_split(
            annotations=train_annotations,
            fraction=args.val_fraction,
            seed=args.seed + offset,
        )
        test_ids = set(test_annotations)

        write_ids(branch_out / "train.txt", train_ids)
        write_ids(branch_out / "valid.txt", valid_ids)
        write_ids(branch_out / "test.txt", test_ids)

        combined_positives = dict(train_annotations)
        combined_positives.update(test_annotations)
        positives_path = branch_out / f"pid_to_positives_{branch}.json"
        write_json(positives_path, combined_positives)

        train_labels = {
            go_id
            for protein_id in train_ids
            for go_id in train_annotations[protein_id]
        }
        valid_labels = {
            go_id
            for protein_id in valid_ids
            for go_id in train_annotations[protein_id]
        }
        test_labels = {
            go_id
            for go_ids in test_annotations.values()
            for go_id in go_ids
        }

        branch_summary = {
            "train": len(train_ids),
            "valid": len(valid_ids),
            "test": len(test_ids),
            "train_labels": len(train_labels),
            "valid_labels": len(valid_labels),
            "test_labels": len(test_labels),
            "validation_labels_missing_from_train": len(
                valid_labels - train_labels
            ),
            "paths": {
                "train_ids": str(branch_out / "train.txt"),
                "valid_ids": str(branch_out / "valid.txt"),
                "test_ids": str(branch_out / "test.txt"),
                "pid_to_positives": str(positives_path),
            },
        }
        summary["branches"][branch.upper()] = branch_summary
        write_json(
            branch_out / "store_summary.json",
            branch_summary,
            pretty=True,
        )

    write_json(processed / "training_store_summary.json", summary, pretty=True)

    print("=" * 72)
    print("GOR2023 TRAINING STORE BUILD")
    print("=" * 72)
    print(f"GO vocabulary terms: {len(go_vocab):,}")
    print("GO segment format: markerless_v1")
    for branch in BRANCHES:
        stats = summary["branches"][branch.upper()]
        print(
            f"{branch.upper()}: train={stats['train']:,} | "
            f"valid={stats['valid']:,} | test={stats['test']:,} | "
            "val labels missing from train="
            f"{stats['validation_labels_missing_from_train']}"
        )
    print(f"Outputs: {processed}")
    print("=" * 72)


if __name__ == "__main__":
    main()
