#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

EXPECTED_NAMESPACES = {
    "mf": "molecular_function",
    "bp": "biological_process",
    "cc": "cellular_component",
}


def read_ids(path):
    with path.open() as handle:
        return {
            line.strip()
            for line in handle
            if line.strip() and not line.startswith("#")
        }


def parse_obo(path):
    terms = {}
    current = None

    def commit():
        if current is not None and current.get("id"):
            terms[current["id"]] = current.copy()

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if line == "[Term]":
                commit()
                current = {
                    "id": None,
                    "name": None,
                    "namespace": None,
                    "definition": None,
                    "alt_ids": [],
                    "is_obsolete": False,
                    "replaced_by": [],
                    "consider": [],
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
                current["name"] = line[6:].strip()
            elif line.startswith("namespace: "):
                current["namespace"] = line[11:].strip()
            elif line.startswith("def: "):
                current["definition"] = line[5:].strip()
            elif line.startswith("alt_id: "):
                current["alt_ids"].append(line[8:].strip())
            elif line == "is_obsolete: true":
                current["is_obsolete"] = True
            elif line.startswith("replaced_by: "):
                current["replaced_by"].append(
                    line[13:].split(" ! ", 1)[0].strip()
                )
            elif line.startswith("consider: "):
                current["consider"].append(
                    line[10:].split(" ! ", 1)[0].strip()
                )

    commit()

    alt_to_primary = {}
    for primary_id, term in terms.items():
        for alt_id in term["alt_ids"]:
            if alt_id in alt_to_primary:
                raise ValueError(
                    f"Duplicate alt_id mapping: {alt_id} maps to both "
                    f"{alt_to_primary[alt_id]} and {primary_id}"
                )
            alt_to_primary[alt_id] = primary_id

    return terms, alt_to_primary


def resolve_go_id(go_id, terms, alt_to_primary):
    original_id = go_id
    resolution_path = []
    seen = set()
    used_alt_id = False
    used_replacement = False

    while True:
        if go_id in seen:
            return {
                "input_id": original_id,
                "resolved_id": None,
                "status": "resolution_cycle",
                "path": resolution_path + [go_id],
                "term": None,
                "consider": [],
            }

        seen.add(go_id)
        resolution_path.append(go_id)

        if go_id in alt_to_primary:
            used_alt_id = True
            go_id = alt_to_primary[go_id]
            continue

        term = terms.get(go_id)

        if term is None:
            return {
                "input_id": original_id,
                "resolved_id": None,
                "status": "unresolved",
                "path": resolution_path,
                "term": None,
                "consider": [],
            }

        if not term["is_obsolete"]:
            if used_replacement and used_alt_id:
                status = "resolved_replaced_and_alt"
            elif used_replacement:
                status = "resolved_replaced_by"
            elif used_alt_id:
                status = "resolved_alt_id"
            else:
                status = "active"

            return {
                "input_id": original_id,
                "resolved_id": go_id,
                "status": status,
                "path": resolution_path,
                "term": term,
                "consider": [],
            }

        if term["replaced_by"]:
            used_replacement = True
            go_id = term["replaced_by"][0]
            continue

        if term["consider"]:
            return {
                "input_id": original_id,
                "resolved_id": None,
                "status": "obsolete_consider_only",
                "path": resolution_path,
                "term": term,
                "consider": term["consider"],
            }

        return {
            "input_id": original_id,
            "resolved_id": None,
            "status": "obsolete_unresolved",
            "path": resolution_path,
            "term": term,
            "consider": [],
        }


def audit_id_set(ids, expected_namespace, terms, alt_to_primary):
    rows = []
    counts = Counter()

    for go_id in sorted(ids):
        result = resolve_go_id(go_id, terms, alt_to_primary)
        term = result["term"]
        resolved_namespace = term["namespace"] if term else None

        namespace_match = (
                result["resolved_id"] is not None
                and resolved_namespace == expected_namespace
        )

        counts[result["status"]] += 1

        if result["resolved_id"] is not None:
            counts["resolved_to_active"] += 1
            if namespace_match:
                counts["namespace_match"] += 1
            else:
                counts["namespace_mismatch"] += 1
        else:
            counts["not_resolved_to_active"] += 1

        rows.append({
            "input_id": go_id,
            "status": result["status"],
            "resolved_id": result["resolved_id"] or "",
            "name": term["name"] if term else "",
            "namespace": resolved_namespace or "",
            "expected_namespace": expected_namespace,
            "namespace_match": namespace_match,
            "resolution_path": " -> ".join(result["path"]),
            "consider": "|".join(result["consider"]),
        })

    summary = {
        "total": len(ids),
        "resolved_to_active": counts["resolved_to_active"],
        "not_resolved_to_active": counts["not_resolved_to_active"],
        "coverage_percent": round(
            100.0 * counts["resolved_to_active"] / len(ids), 6
        ) if ids else 100.0,
        "namespace_match": counts["namespace_match"],
        "namespace_mismatch": counts["namespace_mismatch"],
        "status_counts": dict(sorted(
            (key, value)
            for key, value in counts.items()
            if key not in {
                "resolved_to_active",
                "not_resolved_to_active",
                "namespace_match",
                "namespace_mismatch",
            }
        )),
    }

    return summary, rows


def write_tsv(path, rows):
    columns = [
        "branch",
        "subset",
        "input_id",
        "status",
        "resolved_id",
        "name",
        "namespace",
        "expected_namespace",
        "namespace_match",
        "resolution_path",
        "consider",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


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
    args = parser.parse_args()

    terms, alt_to_primary = parse_obo(args.obo)

    output_dir = args.go_root / "coverage_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "ontology_file": str(args.obo),
        "ontology_primary_terms": len(terms),
        "ontology_alt_ids": len(alt_to_primary),
        "branches": {},
    }

    all_rows = []

    for branch, expected_namespace in EXPECTED_NAMESPACES.items():
        branch_report = {}

        for subset in ("required", "test_only"):
            id_path = args.go_root / branch / f"{subset}_go_ids.txt"
            ids = read_ids(id_path)

            summary, rows = audit_id_set(
                ids,
                expected_namespace,
                terms,
                alt_to_primary,
            )

            branch_report[subset] = summary

            for row in rows:
                row["branch"] = branch.upper()
                row["subset"] = subset
                all_rows.append(row)

        report["branches"][branch.upper()] = branch_report

    report_path = output_dir / "go_coverage_audit.json"
    details_path = output_dir / "go_coverage_details.tsv"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    write_tsv(details_path, all_rows)

    print("=" * 72)
    print("GOR2023 GO COVERAGE AUDIT")
    print("=" * 72)
    print(f"Ontology primary terms: {len(terms):,}")
    print(f"Ontology alt IDs:       {len(alt_to_primary):,}")

    for branch in ("MF", "BP", "CC"):
        print(f"\n{branch}")
        for subset in ("required", "test_only"):
            summary = report["branches"][branch][subset]
            print(
                f"  {subset:10s}: "
                f"{summary['resolved_to_active']:,}/"
                f"{summary['total']:,} resolved "
                f"({summary['coverage_percent']:.4f}%), "
                f"namespace mismatches="
                f"{summary['namespace_mismatch']:,}"
            )
            print(f"    statuses: {summary['status_counts']}")

    print("\nOutputs:")
    print(f"  {report_path}")
    print(f"  {details_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()