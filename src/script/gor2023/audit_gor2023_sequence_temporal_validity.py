#!/usr/bin/env python3

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

DEFAULT_CUTOFF = "2023-06-28"


def parse_date(value):
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    return datetime.strptime(value, "%Y-%m-%d").date()


def save_ids(ids, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for accession in sorted(ids):
            f.write(accession + "\n")


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit temporal validity of UniProt sequences recovered "
            "for the GOR2023 dataset."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cutoff",
        type=str,
        default=DEFAULT_CUTOFF,
        help="Temporal cutoff in YYYY-MM-DD format.",
    )

    args = parser.parse_args()

    cutoff = parse_date(args.cutoff)

    if cutoff is None:
        raise ValueError("Invalid cutoff date.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    safe = set()
    post_cutoff = set()
    missing_date = set()
    secondary_mapped = set()

    records = []

    entry_types = {}

    with args.metadata.open(
            "r",
            encoding="utf-8",
            newline="",
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            requested = row["requested_accession"].strip()
            primary = row["primary_accession"].strip()

            sequence_modified_raw = (
                    row.get("sequence_modified") or ""
            ).strip()

            sequence_modified = parse_date(
                sequence_modified_raw
            )

            entry_type = (
                    row.get("entry_type") or "UNKNOWN"
            ).strip()

            entry_types[entry_type] = (
                    entry_types.get(entry_type, 0) + 1
            )

            if requested != primary:
                secondary_mapped.add(requested)

            if sequence_modified is None:
                status = "MISSING_SEQUENCE_DATE"
                missing_date.add(requested)

            elif sequence_modified <= cutoff:
                status = "TEMPORAL_SAFE"
                safe.add(requested)

            else:
                status = "POST_CUTOFF_SEQUENCE"
                post_cutoff.add(requested)

            records.append({
                "requested_accession": requested,
                "primary_accession": primary,
                "entry_type": entry_type,
                "sequence_length": row.get(
                    "sequence_length", ""
                ),
                "sequence_version": row.get(
                    "sequence_version", ""
                ),
                "sequence_modified": sequence_modified_raw,
                "entry_modified": row.get(
                    "entry_modified", ""
                ),
                "entry_version": row.get(
                    "entry_version", ""
                ),
                "temporal_status": status,
            })

    # Detailed audit TSV
    audit_tsv = args.out_dir / "temporal_validity.tsv"

    fieldnames = [
        "requested_accession",
        "primary_accession",
        "entry_type",
        "sequence_length",
        "sequence_version",
        "sequence_modified",
        "entry_modified",
        "entry_version",
        "temporal_status",
    ]

    with audit_tsv.open(
            "w",
            encoding="utf-8",
            newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()
        writer.writerows(records)

    save_ids(
        safe,
        args.out_dir / "temporal_safe.txt",
    )

    save_ids(
        post_cutoff,
        args.out_dir / "post_cutoff_sequence.txt",
    )

    save_ids(
        missing_date,
        args.out_dir / "missing_sequence_date.txt",
    )

    save_ids(
        secondary_mapped,
        args.out_dir / "secondary_accession_mappings.txt",
    )

    total = len(records)

    summary = {
        "cutoff": args.cutoff,
        "total_recovered": total,
        "temporal_safe": len(safe),
        "post_cutoff_sequence": len(post_cutoff),
        "missing_sequence_date": len(missing_date),
        "secondary_accession_mappings": len(
            secondary_mapped
        ),
        "temporal_safe_fraction": (
            len(safe) / total if total else 0.0
        ),
        "entry_types": entry_types,
    }

    save_json(
        summary,
        args.out_dir / "temporal_validity_summary.json",
    )

    print("=" * 72)
    print("GOR2023 SEQUENCE TEMPORAL VALIDITY AUDIT")
    print("=" * 72)

    print(f"\nCutoff: {args.cutoff}")

    print("\nSequence status:")
    print(f"  total recovered:      {total:,}")
    print(f"  temporal-safe:        {len(safe):,}")
    print(f"  post-cutoff sequence: {len(post_cutoff):,}")
    print(f"  missing date:         {len(missing_date):,}")

    print(
        f"\nTemporal-safe coverage: "
        f"{100 * summary['temporal_safe_fraction']:.4f}%"
    )

    print(
        f"Secondary accession mappings: "
        f"{len(secondary_mapped):,}"
    )

    print("\nEntry types:")

    for entry_type, count in sorted(
            entry_types.items(),
            key=lambda x: -x[1],
    ):
        print(f"  {entry_type}: {count:,}")

    print("\nOutputs:")
    print(f"  audit TSV -> {audit_tsv}")
    print(
        f"  summary   -> "
        f"{args.out_dir / 'temporal_validity_summary.json'}"
    )


if __name__ == "__main__":
    main()