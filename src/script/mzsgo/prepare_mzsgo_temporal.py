#!/usr/bin/env python3
"""Prepare MZSGO's pure temporal benchmark for the protein-GO pipeline.

The input repository is the public MZSGO release. Its sequence files use two
lines per record: a FASTA header containing semicolon-separated numeric GO IDs,
followed by the protein sequence. This script deliberately consumes the
released ``zero_shot_below30.txt`` split instead of claiming to reproduce the
unreleased DIAMOND construction command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


BRANCH_TO_NAMESPACE = {
    "mf": "molecular_function",
    "bp": "biological_process",
    "cc": "cellular_component",
}
EXPECTED = {
    "mf": {"train_proteins": 50874, "temporal_proteins": 555, "temporal_labels": 35},
    "bp": {"train_proteins": 51618, "temporal_proteins": 3687, "temporal_labels": 34},
    "cc": {"train_proteins": 51743, "temporal_proteins": 17, "temporal_labels": 6},
}
ACTIVE_SEGMENTS = ("name", "definition", "is_a")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def extract_quoted(value: str) -> str:
    match = re.search(r'"(.*?)"', value)
    return clean_text(match.group(1) if match else value)


def ensure_sentence(value: str) -> str:
    value = clean_text(value)
    if not value:
        return "none."
    return value if value[-1] in ".!?" else value + "."


def parse_obo(path: Path) -> Dict[str, dict]:
    terms: Dict[str, dict] = {}
    current = None

    def commit() -> None:
        nonlocal current
        if current and current.get("id"):
            for key in ("is_a", "part_of", "alt_id", "replaced_by", "consider"):
                current[key] = sorted(set(current[key]))
            terms[current["id"]] = dict(current)

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
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
                current["is_a"].append(line[6:].split(" ! ", 1)[0].strip())
            elif line.startswith("relationship: part_of "):
                current["part_of"].append(
                    line[len("relationship: part_of "):].split(" ! ", 1)[0].strip()
                )
            elif line.startswith("alt_id: "):
                current["alt_id"].append(line[8:].strip())
            elif line.startswith("replaced_by: "):
                current["replaced_by"].append(line[13:].split(" ! ", 1)[0].strip())
            elif line.startswith("consider: "):
                current["consider"].append(line[10:].split(" ! ", 1)[0].strip())
            elif line == "is_obsolete: true":
                current["is_obsolete"] = True
    commit()
    if not terms:
        raise ValueError(f"No GO terms parsed from {path}")
    return terms


def active_primary_ids(terms: Mapping[str, dict]) -> Set[str]:
    return {go_id for go_id, term in terms.items() if not term["is_obsolete"]}


def read_sequence_records(path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    header = None
    sequence_parts: List[str] = []

    def commit() -> None:
        nonlocal header, sequence_parts
        if header is None:
            return
        fields = header.split()
        protein_id = fields[0]
        if protein_id in records:
            raise ValueError(f"Duplicate protein ID in {path}: {protein_id}")
        raw_go_ids = fields[1].split(";") if len(fields) > 1 and fields[1] else []
        records[protein_id] = {
            "go_ids": [f"GO:{item}" if not item.startswith("GO:") else item for item in raw_go_ids],
            "sequence": "".join(sequence_parts),
        }

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                header = line[1:]
                sequence_parts = []
            else:
                if header is None:
                    raise ValueError(f"Sequence before first header in {path}")
                sequence_parts.append(line)
    commit()
    return records


def branch_annotations(
    records: Mapping[str, dict],
    terms: Mapping[str, dict],
    namespace: str,
    allowed_ids: Set[str],
) -> Dict[str, List[str]]:
    output = {}
    for protein_id, record in records.items():
        labels = sorted({
            go_id
            for go_id in record["go_ids"]
            if go_id in allowed_ids and terms.get(go_id, {}).get("namespace") == namespace
        })
        if labels:
            output[protein_id] = labels
    return output


def constrained_validation_split(
    annotations: Mapping[str, Sequence[str]], fraction: float, seed: int
) -> Tuple[Set[str], Set[str]]:
    protein_ids = sorted(annotations)
    target = round(len(protein_ids) * fraction)
    counts = Counter(label for labels in annotations.values() for label in set(labels))
    candidates = protein_ids.copy()
    random.Random(seed).shuffle(candidates)
    valid: Set[str] = set()
    for protein_id in candidates:
        if len(valid) >= target:
            break
        labels = set(annotations[protein_id])
        if all(counts[label] > 1 for label in labels):
            valid.add(protein_id)
            for label in labels:
                counts[label] -= 1
    if len(valid) != target:
        raise RuntimeError(f"Could not create validation split: {len(valid)} != {target}")
    return set(protein_ids) - valid, valid


def go_text_entry(go_id: str, term: dict, terms: Mapping[str, dict]) -> dict:
    parent_names = sorted({
        clean_text(terms[parent]["name"])
        for parent in term["is_a"]
        if parent in terms and not terms[parent]["is_obsolete"] and clean_text(terms[parent]["name"])
    })[:3]
    segments = {
        "name": ensure_sentence(term["name"]),
        "namespace": ensure_sentence(term["namespace"].replace("_", " ")),
        "definition": ensure_sentence(term["definition"]),
        "is_a": ensure_sentence("; ".join(parent_names) if parent_names else "none"),
        "part_of": ensure_sentence("none"),
    }
    return {
        "go_id": go_id,
        "domain": {
            "molecular_function": "Molecular Function",
            "biological_process": "Biological Process",
            "cellular_component": "Cellular Component",
        }[term["namespace"]],
        "namespace": term["namespace"].replace("_", " "),
        "name": term["name"],
        "definition": term["definition"],
        "is_a_parent_ids": term["is_a"],
        "part_of_parent_ids": term["part_of"],
        "is_a_parents": parent_names,
        "part_of_parents": [],
        "segments": segments,
        "segment_order": list(ACTIVE_SEGMENTS),
        "segment_format": "markerless_v1",
        "text": "\n".join(segments[key] for key in ACTIVE_SEGMENTS),
    }


def write_json(path: Path, value, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2 if pretty else None)
        handle.write("\n")


def write_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_ids(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in sorted(set(values)):
            handle.write(str(value) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_fasta(path: Path, records: Mapping[str, dict], width: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein_id in sorted(records):
            sequence = records[protein_id]["sequence"]
            if not sequence:
                raise ValueError(f"Empty sequence for {protein_id}")
            handle.write(f">{protein_id}\n")
            for start in range(0, len(sequence), width):
                handle.write(sequence[start:start + width] + "\n")


def sequence_hashes(records: Mapping[str, dict]) -> Set[str]:
    return {
        hashlib.sha256(record["sequence"].encode("ascii")).hexdigest()
        for record in records.values()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mzsgo-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-expected-count-checks", action="store_true")
    args = parser.parse_args()

    data = args.mzsgo_root / "data"
    old_obo = data / "go_2023_01_01.obo"
    new_obo = data / "go_2025_10_10.obo"
    train_path = data / "sequence" / "cafa5_train_in_swissprot.txt"
    temporal_path = data / "sequence" / "zero_shot_below30.txt"
    for required in (old_obo, new_obo, train_path, temporal_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    old_terms = parse_obo(old_obo)
    new_terms = parse_obo(new_obo)
    old_active = active_primary_ids(old_terms)
    new_active = active_primary_ids(new_terms)
    temporal_ids = new_active - old_active
    train_records = read_sequence_records(train_path)
    temporal_records = read_sequence_records(temporal_path)

    id_overlap = sorted(set(train_records) & set(temporal_records))
    exact_sequence_overlap = len(sequence_hashes(train_records) & sequence_hashes(temporal_records))
    if id_overlap:
        raise RuntimeError(f"Train/temporal protein ID leakage: {id_overlap[:20]}")
    if exact_sequence_overlap:
        raise RuntimeError(f"Train/temporal exact sequence leakage: {exact_sequence_overlap}")

    processed = args.out_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    write_json(processed / "go_vocab.json", new_terms)
    combined_sequence_records = dict(train_records)
    combined_sequence_records.update(temporal_records)
    write_fasta(args.out_root / "sequences" / "all_train_temporal.fasta", combined_sequence_records)

    summary = {
        "protocol": "MZSGO pure temporal zero-shot",
        "old_ontology": str(old_obo),
        "new_ontology": str(new_obo),
        "released_temporal_split": str(temporal_path),
        "identity_filter_claim": "released filename/paper claim only; DIAMOND command not public",
        "train_records": len(train_records),
        "temporal_records": len(temporal_records),
        "train_temporal_id_overlap": len(id_overlap),
        "train_temporal_exact_sequence_overlap": exact_sequence_overlap,
        "sequence_fasta": str(args.out_root / "sequences" / "all_train_temporal.fasta"),
        "branches": {},
    }

    all_candidate_ids: Set[str] = set()
    for offset, (branch, namespace) in enumerate(BRANCH_TO_NAMESPACE.items()):
        train_ann = branch_annotations(train_records, old_terms, namespace, old_active)
        temporal_ann = branch_annotations(temporal_records, new_terms, namespace, temporal_ids)
        temporal_label_set = {label for labels in temporal_ann.values() for label in labels}
        train_label_set = {label for labels in train_ann.values() for label in labels}
        if train_label_set & temporal_label_set:
            raise RuntimeError(f"{branch.upper()} temporal labels leaked into training")

        if not args.skip_expected_count_checks:
            expected = EXPECTED[branch]
            observed = {
                "train_proteins": len(train_ann),
                "temporal_proteins": len(temporal_ann),
                "temporal_labels": len(temporal_label_set),
            }
            if observed != expected:
                raise RuntimeError(f"{branch.upper()} count mismatch: {observed} != {expected}")

        train_ids, valid_ids = constrained_validation_split(
            train_ann, args.val_fraction, args.seed + offset
        )
        branch_dir = processed / branch
        write_ids(branch_dir / "train.txt", train_ids)
        write_ids(branch_dir / "valid.txt", valid_ids)
        write_ids(branch_dir / "test.txt", temporal_ann)
        write_ids(branch_dir / f"temporal_go_ids_{branch}.txt", temporal_label_set)

        combined = dict(train_ann)
        combined.update(temporal_ann)
        write_json(branch_dir / f"pid_to_positives_{branch}.json", combined)
        write_json(branch_dir / f"pid_to_go_train_{branch}.json", train_ann)
        write_json(branch_dir / f"pid_to_go_temporal_{branch}.json", temporal_ann)

        candidate_ids = train_label_set | temporal_label_set
        all_candidate_ids |= candidate_ids
        write_json(branch_dir / f"candidate_go_ids_{branch}.json", sorted(candidate_ids))
        write_json(branch_dir / f"temporal_go_ids_{branch}.json", sorted(temporal_label_set))
        write_pickle(branch_dir / f"candidate_go_ids_{branch}.pkl", sorted(candidate_ids))
        write_pickle(branch_dir / f"seen_go_ids_{branch}.pkl", sorted(train_label_set))
        write_pickle(branch_dir / f"temporal_go_ids_{branch}.pkl", sorted(temporal_label_set))
        write_pickle(branch_dir / "empty_go_ids.pkl", [])

        term_frequency = Counter(label for protein_id in train_ids for label in train_ann[protein_id])
        write_json(branch_dir / f"train_go_frequency_{branch}.json", dict(term_frequency), pretty=True)

        summary["branches"][branch] = {
            "namespace": namespace,
            "train_source_proteins": len(train_ann),
            "train_proteins": len(train_ids),
            "valid_proteins": len(valid_ids),
            "temporal_proteins": len(temporal_ann),
            "train_labels_before_validation_split": len(train_label_set),
            "candidate_labels": len(candidate_ids),
            "temporal_labels": len(temporal_label_set),
            "temporal_annotations": sum(map(len, temporal_ann.values())),
            "temporal_labels_in_training": len(train_label_set & temporal_label_set),
        }

    text_rows = [
        go_text_entry(go_id, new_terms[go_id], new_terms)
        for go_id in sorted(all_candidate_ids)
    ]
    write_jsonl(processed / "go_texts_canonical.jsonl", text_rows)
    write_json(processed / "mzsgo_temporal_summary.json", summary, pretty=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
