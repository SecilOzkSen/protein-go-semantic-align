#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import defaultdict

BRANCHES = ("mf", "bp", "cc")

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_ids(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return {
            line.strip().split()[0]
            for line in f
            if line.strip()
        }


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_ids(ids, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pid in sorted(ids):
            f.write(pid + "\n")


def parse_uniprot_fasta(path: Path):
    """
    Supports UniProt headers such as:
      >sp|P12345|...
      >tr|A0A123|...
    """

    sequences = {}
    duplicate_ids = defaultdict(list)

    current_pid = None
    current_seq = []

    def flush():
        nonlocal current_pid, current_seq

        if current_pid is None:
            return

        seq = "".join(current_seq).replace(" ", "").upper()

        if current_pid in sequences:
            if sequences[current_pid] != seq:
                duplicate_ids[current_pid].append(seq)
        else:
            sequences[current_pid] = seq

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                flush()

                header = line[1:]

                if "|" in header:
                    parts = header.split("|")
                    if len(parts) >= 2:
                        current_pid = parts[1]
                    else:
                        current_pid = header.split()[0]
                else:
                    current_pid = header.split()[0]

                current_seq = []
            else:
                current_seq.append(line)

    flush()

    return sequences, duplicate_ids


def sequence_status(seq):
    if not seq:
        return "EMPTY"

    invalid = sorted(set(seq) - VALID_AA)

    if invalid:
        return f"INVALID_CHARS:{''.join(invalid)}"

    return "VALID"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed_dir",
        type=Path,
        required=True,
        help="GOR2023 processed directory."
    )

    parser.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="UniProt FASTA file, ideally the selected 2023 release."
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    print("=" * 72)
    print("GOR2023 SEQUENCE COVERAGE AUDIT")
    print("=" * 72)

    branch_sets = {}
    all_required = set()

    for branch in BRANCHES:
        train = load_json(
            args.processed_dir
            / branch
            / "pid_to_go_train_released.json"
        )

        test_ids = read_ids(
            args.processed_dir
            / branch
            / "test_ids.txt"
        )

        difficult_ids = read_ids(
            args.processed_dir
            / branch
            / "difficult_ids.txt"
        )

        train_ids = set(train)

        branch_sets[branch] = {
            "train": train_ids,
            "test": test_ids,
            "difficult": difficult_ids,
        }

        all_required.update(train_ids)
        all_required.update(test_ids)

    print(f"\nUnique proteins requiring sequence: {len(all_required):,}")
    save_ids(
        all_required,
        args.out_dir / "required_accessions.txt"
    )

    print(f"Reading FASTA: {args.fasta}")

    fasta_sequences, duplicate_ids = parse_uniprot_fasta(args.fasta)

    print(f"Sequences indexed from FASTA: {len(fasta_sequences):,}")

    usable_sequences = {}
    invalid_sequences = {}

    for pid in all_required:
        if pid not in fasta_sequences:
            continue

        seq = fasta_sequences[pid]
        status = sequence_status(seq)

        if status == "VALID":
            usable_sequences[pid] = seq
        else:
            invalid_sequences[pid] = {
                "status": status,
                "length": len(seq),
            }

    coverage = {}

    for branch in BRANCHES:
        coverage[branch] = {}

        for split in ("train", "test", "difficult"):
            ids = branch_sets[branch][split]

            found = ids & set(usable_sequences)
            missing = ids - set(usable_sequences)

            coverage[branch][split] = {
                "total": len(ids),
                "found": len(found),
                "missing": len(missing),
                "coverage": (
                    len(found) / len(ids)
                    if ids else 0.0
                ),
                "missing_examples": sorted(missing)[:50],
            }

    # Save normalized FASTA containing only required proteins.
    normalized_fasta = args.out_dir / "protein_sequences.fasta"
    normalized_fasta.parent.mkdir(parents=True, exist_ok=True)

    with normalized_fasta.open("w", encoding="utf-8") as f:
        for pid in sorted(usable_sequences):
            seq = usable_sequences[pid]
            f.write(f">{pid}\n")

            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")

    missing_all = all_required - set(usable_sequences)

    save_ids(
        missing_all,
        args.out_dir / "missing_accessions.txt"
    )

    save_json(
        coverage,
        args.out_dir / "sequence_coverage.json"
    )

    save_json(
        invalid_sequences,
        args.out_dir / "invalid_sequences.json"
    )

    save_json(
        {
            pid: {
                "num_conflicting_duplicates": len(vals)
            }
            for pid, vals in duplicate_ids.items()
        },
        args.out_dir / "duplicate_sequence_report.json"
    )

    print("\n" + "=" * 72)
    print("COVERAGE")
    print("=" * 72)

    for branch in BRANCHES:
        print(f"\n{branch.upper()}")

        for split in ("train", "test", "difficult"):
            s = coverage[branch][split]

            print(
                f"  {split:10s}: "
                f"{s['found']:,}/{s['total']:,} "
                f"({100 * s['coverage']:.4f}%)"
            )

    print("\nGlobal:")
    print(f"  required: {len(all_required):,}")
    print(f"  found:    {len(usable_sequences):,}")
    print(f"  missing:  {len(missing_all):,}")
    print(f"  invalid:  {len(invalid_sequences):,}")
    print(f"  conflicting duplicate IDs: {len(duplicate_ids):,}")

    print(f"\nNormalized FASTA -> {normalized_fasta}")
    print(
        f"Missing IDs      -> "
        f"{args.out_dir / 'missing_accessions.txt'}"
    )


if __name__ == "__main__":
    main()