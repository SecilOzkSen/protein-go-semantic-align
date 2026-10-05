import hashlib
from collections import Counter
from pathlib import Path


PFRESGO_DIR = Path(
    "/workspace/stargo/datasets/pfresgo"
)

FASTA_PATH = (
    PFRESGO_DIR
    / "nrPDB-GO_2019.06.18_sequences.fasta"
)

ANNOT_PATH = PFRESGO_DIR / "annot.tsv"

SPLIT_PATHS = {
    "train": PFRESGO_DIR / "train.txt",
    "valid": PFRESGO_DIR / "valid.txt",
    "test": PFRESGO_DIR / "test.txt",
}


def load_split(path: Path):
    with open(path, encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def sequence_hash(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def load_fasta(path: Path):
    sequences = {}
    duplicate_ids = []
    invalid_sequences = []

    valid_amino_acids = set(
        "ACDEFGHIKLMNPQRSTVWYBXZJUO"
    )

    current_id = None
    current_sequence_parts = []

    def save_current_sequence():
        if current_id is None:
            return

        sequence = "".join(
            current_sequence_parts
        ).strip().upper()

        if current_id in sequences:
            duplicate_ids.append(current_id)
            return

        invalid_chars = sorted(
            set(sequence) - valid_amino_acids
        )

        if invalid_chars:
            invalid_sequences.append(
                {
                    "protein_id": current_id,
                    "invalid_chars": invalid_chars,
                }
            )

        sequences[current_id] = sequence

    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                save_current_sequence()

                header = line[1:].strip()
                current_id = header.split()[0]
                current_sequence_parts = []
            else:
                if current_id is None:
                    raise ValueError(
                        "FASTA sequence encountered before "
                        f"a header in {path}"
                    )

                current_sequence_parts.append(line)

    save_current_sequence()

    return (
        sequences,
        duplicate_ids,
        invalid_sequences,
    )


def load_annotation_ids(path: Path):
    annotation_ids = []

    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if line_number <= 12:
                continue

            columns = line.rstrip("\n").split("\t")

            if line_number == 13:
                continue

            if columns and columns[0]:
                annotation_ids.append(columns[0])

    return annotation_ids


def summarize_lengths(sequences, protein_ids):
    lengths = sorted(
        len(sequences[protein_id])
        for protein_id in protein_ids
        if protein_id in sequences
    )

    if not lengths:
        return {}

    def percentile(percent):
        index = round(
            (len(lengths) - 1) * percent
        )
        return lengths[index]

    return {
        "count": len(lengths),
        "min": lengths[0],
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": lengths[-1],
        "over_512": sum(
            length > 512
            for length in lengths
        ),
        "over_1022": sum(
            length > 1022
            for length in lengths
        ),
        "over_1024": sum(
            length > 1024
            for length in lengths
        ),
    }


def main():
    sequences, duplicate_fasta_ids, invalid_sequences = (
        load_fasta(FASTA_PATH)
    )

    annotation_ids = load_annotation_ids(
        ANNOT_PATH
    )

    splits = {
        name: load_split(path)
        for name, path in SPLIT_PATHS.items()
    }

    print("\nPFresGO protein validation")
    print("=" * 60)
    print(f"FASTA proteins: {len(sequences)}")
    print(
        "Unique annotation protein IDs: "
        f"{len(set(annotation_ids))}"
    )
    print(
        "Annotation rows: "
        f"{len(annotation_ids)}"
    )
    print(
        "Duplicate FASTA IDs: "
        f"{len(duplicate_fasta_ids)}"
    )
    print(
        "Invalid FASTA sequences: "
        f"{len(invalid_sequences)}"
    )

    all_split_ids = []

    for split_name, protein_ids in splits.items():
        all_split_ids.extend(protein_ids)

        missing_fasta = sorted(
            set(protein_ids) - set(sequences)
        )

        missing_annotations = sorted(
            set(protein_ids)
            - set(annotation_ids)
        )

        duplicates = [
            protein_id
            for protein_id, count
            in Counter(protein_ids).items()
            if count > 1
        ]

        length_stats = summarize_lengths(
            sequences,
            protein_ids,
        )

        print(f"\n{split_name.upper()}")
        print(f"  Protein IDs: {len(protein_ids)}")
        print(
            "  Unique protein IDs: "
            f"{len(set(protein_ids))}"
        )
        print(
            "  Duplicate IDs: "
            f"{len(duplicates)}"
        )
        print(
            "  Missing from FASTA: "
            f"{len(missing_fasta)}"
        )
        print(
            "  Missing from annotations: "
            f"{len(missing_annotations)}"
        )

        if length_stats:
            print(
                "  Length min/median/p90/p95/"
                "p99/max: "
                f"{length_stats['min']}/"
                f"{length_stats['median']}/"
                f"{length_stats['p90']}/"
                f"{length_stats['p95']}/"
                f"{length_stats['p99']}/"
                f"{length_stats['max']}"
            )
            print(
                "  Sequences >512: "
                f"{length_stats['over_512']}"
            )
            print(
                "  Sequences >1022: "
                f"{length_stats['over_1022']}"
            )
            print(
                "  Sequences >1024: "
                f"{length_stats['over_1024']}"
            )

        if missing_fasta:
            print(
                "  Missing FASTA examples: "
                f"{missing_fasta[:10]}"
            )

        if missing_annotations:
            print(
                "  Missing annotation examples: "
                f"{missing_annotations[:10]}"
            )

    train_ids = set(splits["train"])
    valid_ids = set(splits["valid"])
    test_ids = set(splits["test"])

    train_valid_overlap = train_ids & valid_ids
    train_test_overlap = train_ids & test_ids
    valid_test_overlap = valid_ids & test_ids

    all_split_set = set(all_split_ids)
    fasta_set = set(sequences)
    annotation_set = set(annotation_ids)

    print("\nGLOBAL")
    print("=" * 60)
    print(
        "Total split rows: "
        f"{len(all_split_ids)}"
    )
    print(
        "Unique split proteins: "
        f"{len(all_split_set)}"
    )
    print(
        "Train-valid ID overlap: "
        f"{len(train_valid_overlap)}"
    )
    print(
        "Train-test ID overlap: "
        f"{len(train_test_overlap)}"
    )
    print(
        "Valid-test ID overlap: "
        f"{len(valid_test_overlap)}"
    )
    print(
        "Split proteins missing from FASTA: "
        f"{len(all_split_set - fasta_set)}"
    )
    print(
        "Split proteins missing annotations: "
        f"{len(all_split_set - annotation_set)}"
    )
    print(
        "FASTA proteins outside splits: "
        f"{len(fasta_set - all_split_set)}"
    )
    print(
        "Annotated proteins outside splits: "
        f"{len(annotation_set - all_split_set)}"
    )

    hashes = Counter(
        sequence_hash(sequence)
        for protein_id, sequence
        in sequences.items()
        if protein_id in all_split_set
    )

    duplicate_sequence_groups = sum(
        count > 1
        for count in hashes.values()
    )

    proteins_in_duplicate_groups = sum(
        count
        for count in hashes.values()
        if count > 1
    )

    print(
        "Duplicate-sequence groups: "
        f"{duplicate_sequence_groups}"
    )
    print(
        "Proteins in duplicate-sequence groups: "
        f"{proteins_in_duplicate_groups}"
    )

    if duplicate_fasta_ids:
        print(
            "Duplicate FASTA ID examples: "
            f"{duplicate_fasta_ids[:10]}"
        )

    if invalid_sequences:
        print(
            "Invalid sequence examples: "
            f"{invalid_sequences[:10]}"
        )

    validation_errors = []

    if duplicate_fasta_ids:
        validation_errors.append(
            "Duplicate protein IDs exist in FASTA."
        )

    if len(annotation_ids) != len(
        set(annotation_ids)
    ):
        validation_errors.append(
            "Duplicate protein IDs exist in annot.tsv."
        )

    if any(
        len(set(ids)) != len(ids)
        for ids in splits.values()
    ):
        validation_errors.append(
            "Duplicate protein IDs exist inside splits."
        )

    if (
        train_valid_overlap
        or train_test_overlap
        or valid_test_overlap
    ):
        validation_errors.append(
            "Protein-ID overlap exists between splits."
        )

    if all_split_set - fasta_set:
        validation_errors.append(
            "Some split proteins are missing from FASTA."
        )

    if all_split_set - annotation_set:
        validation_errors.append(
            "Some split proteins are missing annotations."
        )

    if validation_errors:
        print("\nVALIDATION FAILED")

        for error in validation_errors:
            print(f"  - {error}")

        raise ValueError(
            "PFresGO protein validation failed."
        )

    print("\nValidation completed successfully.")


if __name__ == "__main__":
    main()