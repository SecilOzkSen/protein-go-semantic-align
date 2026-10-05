#!/usr/bin/env python3
"""Build audited global and branch-specific temporal-safe GOR2023 FASTAs."""

import argparse
import csv
import json
import pickle
import re
from collections import Counter
from pathlib import Path

BRANCHES = ("mf", "bp", "cc")
EXPECTED = {
    "required": 130109,
    "swissprot_2023_03": 83333,
    "current_temporal_safe": 39932,
    "inactive_historical": 5731,
    "post_cutoff_historical": 1113,
}
ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZUO")


def read_ids(path):
    with path.open(encoding="utf-8") as handle:
        return {
            line.strip()
            for line in handle
            if line.strip() and not line.startswith("#")
        }


def normalize_accession(header):
    token = header.strip().split()[0]
    if token.startswith(">"):
        token = token[1:]

    # Standard UniProt headers: sp|P12345|NAME or tr|A0A...|NAME
    parts = token.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        token = parts[1]

    return token.strip()


def read_fasta(path):
    records = {}
    current_id = None
    chunks = []

    def commit():
        if current_id is None:
            return

        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            raise RuntimeError(f"Empty sequence for {current_id} in {path}")

        invalid = sorted(set(sequence).difference(ALLOWED_AA))
        if invalid:
            raise RuntimeError(
                f"Invalid residues for {current_id} in {path}: {invalid}"
            )

        previous = records.get(current_id)
        if previous is not None and previous != sequence:
            raise RuntimeError(
                f"Conflicting duplicate accession inside {path}: {current_id}"
            )
        records[current_id] = sequence

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                current_id = normalize_accession(line)
                chunks = []
            else:
                if current_id is None:
                    raise RuntimeError(f"Sequence before FASTA header in {path}")
                chunks.append(re.sub(r"\s+", "", line))

    commit()
    return records


def write_fasta(path, accessions, sequences):
    with path.open("w", encoding="utf-8") as handle:
        for accession in sorted(accessions):
            sequence = sequences[accession]
            handle.write(f">{accession}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")


def read_annotation_proteins(path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return set(data)


def write_json(path, obj):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/GOR2023"),
    )
    args = parser.parse_args()

    seq_root = args.root / "sequences"
    output_dir = seq_root / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    required = read_ids(seq_root / "required_accessions.txt")
    temporal_safe_ids = read_ids(
        seq_root / "temporal_audit" / "temporal_safe.txt"
    )
    post_cutoff_ids = read_ids(
        seq_root / "temporal_audit" / "post_cutoff_sequence.txt"
    )

    if len(required) != EXPECTED["required"]:
        raise RuntimeError(
            f"Required accession count mismatch: {len(required)} != "
            f"{EXPECTED['required']}"
        )
    if len(temporal_safe_ids) != EXPECTED["current_temporal_safe"]:
        raise RuntimeError(
            f"Temporal-safe ID count mismatch: {len(temporal_safe_ids)}"
        )
    if len(post_cutoff_ids) != EXPECTED["post_cutoff_historical"]:
        raise RuntimeError(
            f"Post-cutoff ID count mismatch: {len(post_cutoff_ids)}"
        )

    source_paths = {
        "swissprot_2023_03": (
                seq_root / "swissprot_2023_03" / "protein_sequences.fasta"
        ),
        "current_temporal_safe": (
                seq_root
                / "current_uniprot_recovery"
                / "retrieved_current_uniprot.fasta"
        ),
        "inactive_historical": (
                seq_root
                / "inactive_historical_recovery"
                / "historical_recovered.fasta"
        ),
        "post_cutoff_historical": (
                seq_root
                / "post_cutoff_historical_recovery"
                / "historical_recovered.fasta"
        ),
    }

    raw_sources = {
        source: read_fasta(path)
        for source, path in source_paths.items()
    }

    selected_sources = {
        "swissprot_2023_03": {
            accession: sequence
            for accession, sequence in raw_sources["swissprot_2023_03"].items()
            if accession in required
        },
        "current_temporal_safe": {
            accession: sequence
            for accession, sequence in raw_sources["current_temporal_safe"].items()
            if accession in temporal_safe_ids
        },
        "inactive_historical": {
            accession: sequence
            for accession, sequence in raw_sources["inactive_historical"].items()
            if accession in required
        },
        "post_cutoff_historical": {
            accession: sequence
            for accession, sequence in raw_sources[
                "post_cutoff_historical"
            ].items()
            if accession in post_cutoff_ids
        },
    }

    for source, expected_count in EXPECTED.items():
        if source == "required":
            continue
        observed = len(selected_sources[source])
        if observed != expected_count:
            raise RuntimeError(
                f"{source} selected count mismatch: "
                f"{observed} != {expected_count}"
            )

    sequences = {}
    source_by_accession = {}
    overlap_same_sequence = Counter()

    for source in (
            "swissprot_2023_03",
            "current_temporal_safe",
            "inactive_historical",
            "post_cutoff_historical",
    ):
        for accession, sequence in selected_sources[source].items():
            if accession in sequences:
                if sequences[accession] != sequence:
                    raise RuntimeError(
                        f"Conflicting sequences across sources for {accession}: "
                        f"{source_by_accession[accession]} vs {source}"
                    )
                overlap_same_sequence[source] += 1
                continue

            sequences[accession] = sequence
            source_by_accession[accession] = source

    final_ids = set(sequences)
    missing = sorted(required - final_ids)
    unexpected = sorted(final_ids - required)

    if missing or unexpected or len(final_ids) != EXPECTED["required"]:
        raise RuntimeError(
            "Final accession coverage failed: "
            f"final={len(final_ids)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}, "
            f"missing_examples={missing[:20]}, "
            f"unexpected_examples={unexpected[:20]}"
        )

    global_fasta = output_dir / "all_proteins_temporal_safe.fasta"
    write_fasta(global_fasta, final_ids, sequences)

    seq_len_lookup = {
        accession: len(sequence)
        for accession, sequence in sequences.items()
    }
    with (output_dir / "seq_len_lookup.pkl").open("wb") as handle:
        pickle.dump(seq_len_lookup, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with (output_dir / "sequence_sources.tsv").open(
            "w",
            newline="",
            encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["accession", "source", "sequence_length"])
        for accession in sorted(final_ids):
            writer.writerow([
                accession,
                source_by_accession[accession],
                seq_len_lookup[accession],
            ])

    branch_counts = {}
    for branch in BRANCHES:
        train_ids = read_annotation_proteins(
            args.root / branch / "pid_to_go_train_canonical.json"
        )
        test_ids = read_annotation_proteins(
            args.root / branch / "pid_to_go_test_canonical.json"
        )

        overlap = train_ids.intersection(test_ids)
        if overlap:
            raise RuntimeError(
                f"{branch.upper()} train/test protein overlap: "
                f"{sorted(overlap)[:20]}"
            )

        branch_ids = train_ids.union(test_ids)
        branch_missing = sorted(branch_ids - final_ids)
        if branch_missing:
            raise RuntimeError(
                f"{branch.upper()} proteins missing from final FASTA: "
                f"{branch_missing[:20]}"
            )

        write_fasta(
            output_dir / f"{branch}_proteins_temporal_safe.fasta",
            branch_ids,
            sequences,
        )
        with (output_dir / f"{branch}_required_accessions.txt").open(
                "w",
                encoding="utf-8",
        ) as handle:
            for accession in sorted(branch_ids):
                handle.write(accession + "\n")

        branch_counts[branch.upper()] = {
            "train": len(train_ids),
            "test": len(test_ids),
            "total": len(branch_ids),
        }

    source_counts = Counter(source_by_accession.values())
    lengths = list(seq_len_lookup.values())
    summary = {
        "temporal_cutoff": "2023-06-28",
        "required_accessions": len(required),
        "final_accessions": len(final_ids),
        "coverage_fraction": len(final_ids) / len(required),
        "source_counts": dict(sorted(source_counts.items())),
        "same_sequence_overlaps": dict(sorted(overlap_same_sequence.items())),
        "sequence_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "branches": branch_counts,
        "outputs": {
            "global_fasta": str(global_fasta),
            "seq_len_lookup": str(output_dir / "seq_len_lookup.pkl"),
            "source_manifest": str(output_dir / "sequence_sources.tsv"),
        },
    }
    write_json(output_dir / "final_sequence_summary.json", summary)

    print("=" * 72)
    print("GOR2023 FINAL TEMPORAL-SAFE FASTA BUILD")
    print("=" * 72)
    print(f"Global coverage: {len(final_ids):,}/{len(required):,} (100.0000%)")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count:,}")
    print("Branches:")
    for branch, counts in branch_counts.items():
        print(
            f"  {branch}: train={counts['train']:,} | "
            f"test={counts['test']:,} | total={counts['total']:,}"
        )
    print(
        "Sequence length: "
        f"min={summary['sequence_length']['min']:,} | "
        f"max={summary['sequence_length']['max']:,} | "
        f"mean={summary['sequence_length']['mean']:.2f}"
    )
    print(f"Outputs: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
