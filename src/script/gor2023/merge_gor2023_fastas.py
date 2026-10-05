#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def read_required_ids(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip()
        }


def parse_fasta(path: Path):
    """
    Returns:
        dict accession -> sequence
    Supports headers:
        >P12345
        >P12345 current_primary=P12345
        >sp|P12345|...
        >tr|A0A123|...
    """
    sequences = {}

    current_id = None
    current_seq = []

    def flush():
        nonlocal current_id, current_seq

        if current_id is None:
            return

        seq = "".join(current_seq).replace(" ", "").upper()

        if current_id in sequences:
            if sequences[current_id] != seq:
                raise ValueError(
                    f"Conflicting duplicate sequence for {current_id} "
                    f"in {path}"
                )
        else:
            sequences[current_id] = seq

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                flush()

                header = line[1:]

                first_token = header.split()[0]

                if "|" in first_token:
                    parts = first_token.split("|")
                    if len(parts) >= 2:
                        current_id = parts[1]
                    else:
                        current_id = first_token
                else:
                    current_id = first_token

                current_seq = []

            else:
                current_seq.append(line)

    flush()

    return sequences


def write_fasta(sequences, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for pid in sorted(sequences):
            seq = sequences[pid]

            f.write(f">{pid}\n")

            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )


def save_ids(ids, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for pid in sorted(ids):
            f.write(pid + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge GOR2023 archived Swiss-Prot and UniProt recovery "
            "FASTA files, preferring archived 2023_03 sequences."
        )
    )

    parser.add_argument(
        "--required_ids",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--archived_fasta",
        type=Path,
        required=True,
        help="Filtered UniProtKB/Swiss-Prot 2023_03 FASTA.",
    )

    parser.add_argument(
        "--recovery_fasta",
        type=Path,
        required=True,
        help="FASTA recovered from current UniProt.",
    )

    parser.add_argument(
        "--out_fasta",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--audit_json",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    required = read_required_ids(args.required_ids)

    print("=" * 72)
    print("GOR2023 FASTA MERGE")
    print("=" * 72)

    print(f"\nRequired accessions: {len(required):,}")

    archived = parse_fasta(args.archived_fasta)
    recovery = parse_fasta(args.recovery_fasta)

    print(f"Archived FASTA sequences: {len(archived):,}")
    print(f"Recovery FASTA sequences: {len(recovery):,}")

    archived_required = set(archived) & required
    recovery_required = set(recovery) & required

    overlap = archived_required & recovery_required

    conflicting_overlap = []

    for pid in overlap:
        if archived[pid] != recovery[pid]:
            conflicting_overlap.append(pid)

    #
    # Merge policy:
    # archived 2023_03 always wins
    #
    merged = {}

    source = {}

    for pid in required:
        if pid in archived:
            merged[pid] = archived[pid]
            source[pid] = "UNIPROTKB_2023_03_ARCHIVED"

        elif pid in recovery:
            merged[pid] = recovery[pid]
            source[pid] = "CURRENT_UNIPROT_RECOVERY"

    missing = required - set(merged)

    unexpected_archived = set(archived) - required
    unexpected_recovery = set(recovery) - required

    print()
    print("MERGE SUMMARY")
    print("-" * 72)

    print(
        f"Archived used:     "
        f"{sum(v == 'UNIPROTKB_2023_03_ARCHIVED' for v in source.values()):,}"
    )

    print(
        f"Recovery used:     "
        f"{sum(v == 'CURRENT_UNIPROT_RECOVERY' for v in source.values()):,}"
    )

    print(f"Input overlap:     {len(overlap):,}")
    print(f"Overlap conflicts: {len(conflicting_overlap):,}")
    print(f"Final sequences:   {len(merged):,}")
    print(f"Missing required:  {len(missing):,}")

    if missing:
        missing_path = (
                args.out_fasta.parent
                / "missing_after_merge.txt"
        )

        save_ids(missing, missing_path)

        raise RuntimeError(
            f"{len(missing):,} required proteins are missing. "
            f"See {missing_path}"
        )

    if len(merged) != len(required):
        raise RuntimeError(
            "Final sequence count does not match required accession count."
        )

    write_fasta(
        merged,
        args.out_fasta,
    )

    audit = {
        "required_accessions": len(required),
        "archived_input_sequences": len(archived),
        "recovery_input_sequences": len(recovery),
        "archived_used": sum(
            v == "UNIPROTKB_2023_03_ARCHIVED"
            for v in source.values()
        ),
        "recovery_used": sum(
            v == "CURRENT_UNIPROT_RECOVERY"
            for v in source.values()
        ),
        "input_overlap": len(overlap),
        "overlap_sequence_conflicts": len(
            conflicting_overlap
        ),
        "overlap_conflict_examples": sorted(
            conflicting_overlap
        )[:50],
        "unexpected_archived_accessions": len(
            unexpected_archived
        ),
        "unexpected_recovery_accessions": len(
            unexpected_recovery
        ),
        "final_sequences": len(merged),
        "missing_required": len(missing),
    }

    save_json(
        audit,
        args.audit_json,
    )

    source_json = (
            args.out_fasta.parent
            / "sequence_source.json"
    )

    save_json(
        source,
        source_json,
    )

    print()
    print("SUCCESS")
    print(f"Final FASTA -> {args.out_fasta}")
    print(f"Audit       -> {args.audit_json}")
    print(f"Sources     -> {source_json}")


if __name__ == "__main__":
    main()