#!/usr/bin/env python3
"""Build markerless canonical GO texts for the GOR2023 benchmark."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BRANCHES = ("mf", "bp", "cc")
EXPECTED_NAMESPACES = {
    "mf": "molecular_function",
    "bp": "biological_process",
    "cc": "cellular_component",
}
EXPECTED_COUNTS = {
    "mf": 7228,
    "bp": 21308,
    "cc": 2945,
    "global": 31481,
}
ACTIVE_SEGMENTS = ["name", "definition", "is_a"]
MAX_IS_A_PARENTS = 3
MAX_PART_OF_PARENTS = 3


def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def strip_leading_label(text, label):
    """Remove one redundant source-level field label, case-insensitively."""
    text = clean_text(text)
    return re.sub(
        rf"^{re.escape(label)}\s*:\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def ensure_sentence(text):
    text = clean_text(text)
    if not text:
        return "none."
    if text[-1] in ".!?":
        return text
    return text + "."


def extract_quoted_text(value):
    value = clean_text(value)
    match = re.search(r'"(.*?)"', value)
    return clean_text(match.group(1)) if match else value


def parse_obo(path):
    terms = {}
    current = None

    def commit():
        if current and current.get("id"):
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
                current["definition"] = extract_quoted_text(line[5:])
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
            elif line == "is_obsolete: true":
                current["is_obsolete"] = True

    commit()
    return terms


def load_canonical_mapping(audit_tsv):
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


def get_parent_names(parent_ids, terms, max_parents):
    names = []

    for parent_id in sorted(set(parent_ids)):
        parent = terms.get(parent_id)
        if not parent or parent["is_obsolete"]:
            continue

        name = clean_text(parent["name"])
        if name:
            names.append(name)

    return sorted(set(names))[:max_parents]


def build_entry(go_id, term, terms):
    name = strip_leading_label(term["name"], "Name")
    definition = strip_leading_label(term["definition"], "Definition")
    namespace = strip_leading_label(
        term["namespace"].replace("_", " "),
        "Namespace",
    )

    is_a_ids = sorted(set(term["is_a"]))
    part_of_ids = sorted(set(term["part_of"]))

    is_a_names = get_parent_names(
        is_a_ids,
        terms,
        MAX_IS_A_PARENTS,
    )
    part_of_names = get_parent_names(
        part_of_ids,
        terms,
        MAX_PART_OF_PARENTS,
    )

    segments = {
        "name": ensure_sentence(name),
        "namespace": ensure_sentence(namespace),
        "definition": ensure_sentence(definition),
        "is_a": ensure_sentence(
            "; ".join(is_a_names) if is_a_names else "none"
        ),
        "part_of": ensure_sentence(
            "; ".join(part_of_names) if part_of_names else "none"
        ),
    }

    domain = {
        "molecular_function": "Molecular Function",
        "biological_process": "Biological Process",
        "cellular_component": "Cellular Component",
    }[term["namespace"]]

    return {
        "go_id": go_id,
        "domain": domain,
        "namespace": namespace,
        "name": name,
        "definition": definition,
        "is_a_parent_ids": is_a_ids,
        "part_of_parent_ids": part_of_ids,
        "is_a_parents": is_a_names,
        "part_of_parents": part_of_names,
        "segments": segments,
        "segment_order": ACTIVE_SEGMENTS,
        "segment_format": "markerless_v1",
        "text": "\n".join(segments[key] for key in ACTIVE_SEGMENTS),
    }


def write_jsonl(path, entries):
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_ids(path, ids):
    with path.open("w", encoding="utf-8") as handle:
        for go_id in sorted(ids):
            handle.write(go_id + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--go-root",
        type=Path,
        default=Path("/workspace/GOR2023/go"),
    )
    parser.add_argument(
        "--obo",
        type=Path,
        default=Path(
            "/workspace/GOR2023/go/ontology/go-2023-01-01.obo"
        ),
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

    terms = parse_obo(args.obo)
    mappings, dropped = load_canonical_mapping(args.audit_tsv)

    output_dir = args.go_root / "canonical"
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_ids = {}
    branch_entries = {}
    all_original_to_canonical = {}

    for branch in BRANCHES:
        canonical_ids = set(mappings[branch].values())
        branch_ids[branch] = canonical_ids

        if len(canonical_ids) != EXPECTED_COUNTS[branch]:
            raise RuntimeError(
                f"{branch.upper()} canonical count mismatch: "
                f"{len(canonical_ids)} != {EXPECTED_COUNTS[branch]}"
            )

        entries = []
        for go_id in sorted(canonical_ids):
            term = terms.get(go_id)
            if term is None:
                raise RuntimeError(f"Missing canonical GO term: {go_id}")
            if term["is_obsolete"]:
                raise RuntimeError(f"Canonical GO term is obsolete: {go_id}")
            if term["namespace"] != EXPECTED_NAMESPACES[branch]:
                raise RuntimeError(
                    f"Namespace mismatch for {go_id}: "
                    f"{term['namespace']} != "
                    f"{EXPECTED_NAMESPACES[branch]}"
                )
            entries.append(build_entry(go_id, term, terms))

        branch_entries[branch] = entries
        write_ids(
            output_dir / f"{branch}_canonical_go_ids.txt",
            canonical_ids,
        )
        write_jsonl(
            output_dir
            / f"{branch}_go_texts_canonical_markerless.jsonl",
            entries,
        )

        for original_id, canonical_id in mappings[branch].items():
            all_original_to_canonical[original_id] = canonical_id

    global_ids = set().union(*branch_ids.values())
    if len(global_ids) != EXPECTED_COUNTS["global"]:
        raise RuntimeError(
            f"Global canonical count mismatch: {len(global_ids)} != "
            f"{EXPECTED_COUNTS['global']}"
        )

    global_entries = sorted(
        (
            entry
            for branch in BRANCHES
            for entry in branch_entries[branch]
        ),
        key=lambda entry: entry["go_id"],
    )

    quality = Counter()
    for entry in global_entries:
        if not entry["name"]:
            quality["missing_name"] += 1
        if not entry["definition"]:
            quality["missing_definition"] += 1
        if not entry["is_a_parents"]:
            quality["missing_is_a_parents"] += 1

        for segment_name in ACTIVE_SEGMENTS:
            segment = entry["segments"][segment_name]
            forbidden_prefixes = (
                "Name:",
                "Namespace:",
                "Definition:",
                "Is-a parents:",
                "Part-of parents:",
            )
            if segment.startswith(forbidden_prefixes):
                raise RuntimeError(
                    f"Legacy marker detected in {entry['go_id']} "
                    f"segment={segment_name}: {segment!r}"
                )

    write_ids(output_dir / "all_canonical_go_ids.txt", global_ids)
    write_jsonl(
        output_dir / "go_texts_canonical_markerless.jsonl",
        global_entries,
    )

    mapping_output = {
        "segment_format": "markerless_v1",
        "active_segments": ACTIVE_SEGMENTS,
        "original_to_canonical": dict(
            sorted(all_original_to_canonical.items())
        ),
        "dropped": {
            branch.upper(): dropped[branch]
            for branch in BRANCHES
        },
    }
    with (
            output_dir / "go_id_canonicalization.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(mapping_output, handle, indent=2, sort_keys=True)

    summary = {
        "ontology": str(args.obo),
        "segment_format": "markerless_v1",
        "active_segments": ACTIVE_SEGMENTS,
        "max_is_a_parents": MAX_IS_A_PARENTS,
        "max_part_of_parents": MAX_PART_OF_PARENTS,
        "canonical_counts": {
            "MF": len(branch_entries["mf"]),
            "BP": len(branch_entries["bp"]),
            "CC": len(branch_entries["cc"]),
            "GLOBAL": len(global_entries),
        },
        "text_quality": dict(sorted(quality.items())),
    }
    with (
            output_dir / "go_text_build_summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("=" * 72)
    print("GOR2023 MARKERLESS CANONICAL GO TEXT BUILD")
    print("=" * 72)
    for branch in BRANCHES:
        print(f"{branch.upper()}: {len(branch_entries[branch]):,}")
    print(f"GLOBAL: {len(global_entries):,}")
    print(f"Active segments: {ACTIVE_SEGMENTS}")
    print("Segment format: markerless_v1")
    print(f"Text quality: {dict(sorted(quality.items()))}")
    print(f"Outputs: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
