import json
from pathlib import Path

import numpy as np


FASTA_PATH = Path(
    "/workspace/stargo/datasets/pfresgo/"
    "nrPDB-GO_2019.06.18_sequences.fasta"
)

EMBEDDING_DIR = Path(
    "/workspace/data_pfresgo/"
    "protein_embeddings/esm1b_residue"
)


def load_fasta_lengths(path: Path):
    lengths = {}

    current_id = None
    current_parts = []

    def flush():
        if current_id is None:
            return

        lengths[current_id] = len(
            "".join(current_parts)
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                flush()
                current_id = line[1:].split()[0]
                current_parts = []
            else:
                current_parts.append(line)

    flush()
    return lengths


def main():
    fasta_lengths = load_fasta_lengths(
        FASTA_PATH
    )

    index_paths = sorted(
        EMBEDDING_DIR.glob(
            "res_esm1b_*.index.tsv"
        )
    )

    if not index_paths:
        raise FileNotFoundError(
            "No ESM-1b index files found."
        )

    all_ids = set()
    duplicate_ids = set()
    length_mismatches = []
    invalid_ranges = []
    metadata_errors = []
    nonfinite_proteins = []
    total_index_rows = 0
    total_indexed_residues = 0

    print("\nPFresGO ESM-1b validation")
    print("=" * 60)

    for index_path in index_paths:
        prefix = index_path.name.replace(
            ".index.tsv",
            "",
        )

        data_path = (
            EMBEDDING_DIR
            / f"{prefix}.data.npy"
        )

        meta_path = (
            EMBEDDING_DIR
            / f"{prefix}.meta.json"
        )

        if not data_path.exists():
            raise FileNotFoundError(data_path)

        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        array = np.load(
            data_path,
            mmap_mode="r",
        )

        with meta_path.open(
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        used_rows = int(
            metadata["used_rows"]
        )

        if array.ndim != 2:
            metadata_errors.append(
                f"{prefix}: ndim={array.ndim}"
            )

        if array.shape[1] != 1280:
            metadata_errors.append(
                f"{prefix}: dim={array.shape[1]}"
            )

        if array.dtype != np.float16:
            metadata_errors.append(
                f"{prefix}: dtype={array.dtype}"
            )

        if used_rows > array.shape[0]:
            metadata_errors.append(
                f"{prefix}: used_rows={used_rows}, "
                f"capacity={array.shape[0]}"
            )

        shard_rows = []
        previous_end = 0

        with index_path.open(
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                columns = (
                    line.rstrip("\n").split("\t")
                )

                if len(columns) != 3:
                    invalid_ranges.append(
                        (
                            prefix,
                            line_number,
                            line,
                        )
                    )
                    continue

                protein_id = columns[0]
                start = int(columns[1])
                end = int(columns[2])

                total_index_rows += 1
                total_indexed_residues += (
                    end - start
                )

                if protein_id in all_ids:
                    duplicate_ids.add(protein_id)

                all_ids.add(protein_id)

                if not (
                    0 <= start < end <= used_rows
                ):
                    invalid_ranges.append(
                        (
                            prefix,
                            protein_id,
                            start,
                            end,
                            used_rows,
                        )
                    )

                if start != previous_end:
                    invalid_ranges.append(
                        (
                            prefix,
                            protein_id,
                            "non_contiguous",
                            previous_end,
                            start,
                        )
                    )

                previous_end = end
                shard_rows.append(
                    (
                        protein_id,
                        start,
                        end,
                    )
                )

                expected_length = (
                    fasta_lengths.get(protein_id)
                )

                actual_length = end - start

                if expected_length is None:
                    length_mismatches.append(
                        (
                            protein_id,
                            "missing_from_fasta",
                        )
                    )
                elif actual_length != expected_length:
                    length_mismatches.append(
                        (
                            protein_id,
                            expected_length,
                            actual_length,
                        )
                    )

        if previous_end != used_rows:
            metadata_errors.append(
                f"{prefix}: final index end "
                f"{previous_end} != used_rows "
                f"{used_rows}"
            )

        # Sample first, middle and last protein.
        if shard_rows:
            sample_indices = sorted(
                {
                    0,
                    len(shard_rows) // 2,
                    len(shard_rows) - 1,
                }
            )

            for sample_index in sample_indices:
                protein_id, start, end = (
                    shard_rows[sample_index]
                )

                embedding = np.asarray(
                    array[start:end]
                )

                if not np.isfinite(
                    embedding
                ).all():
                    nonfinite_proteins.append(
                        protein_id
                    )

        print(
            f"{prefix}: "
            f"capacity={array.shape[0]}, "
            f"used={used_rows}, "
            f"proteins={len(shard_rows)}, "
            f"dtype={array.dtype}, "
            f"dim={array.shape[1]}"
        )

        del array

    fasta_ids = set(fasta_lengths)
    missing_embeddings = (
        fasta_ids - all_ids
    )
    unexpected_embeddings = (
        all_ids - fasta_ids
    )

    print("\nGLOBAL")
    print("=" * 60)
    print(f"FASTA proteins: {len(fasta_ids)}")
    print(f"Index rows: {total_index_rows}")
    print(f"Unique embedding IDs: {len(all_ids)}")
    print(
        "Total indexed residues: "
        f"{total_indexed_residues}"
    )
    print(
        "Missing embeddings: "
        f"{len(missing_embeddings)}"
    )
    print(
        "Unexpected embedding IDs: "
        f"{len(unexpected_embeddings)}"
    )
    print(
        "Duplicate embedding IDs: "
        f"{len(duplicate_ids)}"
    )
    print(
        "Length mismatches: "
        f"{len(length_mismatches)}"
    )
    print(
        "Invalid index ranges: "
        f"{len(invalid_ranges)}"
    )
    print(
        "Metadata errors: "
        f"{len(metadata_errors)}"
    )
    print(
        "Nonfinite sampled proteins: "
        f"{len(nonfinite_proteins)}"
    )

    if missing_embeddings:
        print(
            "Missing examples:",
            sorted(missing_embeddings)[:10],
        )

    if length_mismatches:
        print(
            "Length mismatch examples:",
            length_mismatches[:10],
        )

    if invalid_ranges:
        print(
            "Invalid range examples:",
            invalid_ranges[:10],
        )

    if metadata_errors:
        print(
            "Metadata error examples:",
            metadata_errors[:10],
        )

    errors_exist = any(
        [
            missing_embeddings,
            unexpected_embeddings,
            duplicate_ids,
            length_mismatches,
            invalid_ranges,
            metadata_errors,
            nonfinite_proteins,
        ]
    )

    if errors_exist:
        raise ValueError(
            "PFresGO ESM-1b validation failed."
        )

    print("\nValidation completed successfully.")


if __name__ == "__main__":
    main()