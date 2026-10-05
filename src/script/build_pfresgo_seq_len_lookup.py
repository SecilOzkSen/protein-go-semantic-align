import pickle
from pathlib import Path


EMBEDDING_DIR = Path(
    "/workspace/data_pfresgo/"
    "protein_embeddings/esm1b_residue"
)

OUTPUT_PATH = Path(
    "/workspace/data_pfresgo/"
    "processed/seq_len_lookup.pkl"
)


def main():
    index_paths = sorted(
        EMBEDDING_DIR.glob(
            "res_esm1b_*.index.tsv"
        )
    )

    if not index_paths:
        raise FileNotFoundError(
            f"No residue index files found in "
            f"{EMBEDDING_DIR}"
        )

    seq_len_lookup = {}
    total_index_rows = 0

    for index_path in index_paths:
        with index_path.open(
            "r",
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
                    raise ValueError(
                        f"Malformed index row in "
                        f"{index_path}, line "
                        f"{line_number}: {line!r}"
                    )

                protein_id, start, end = columns
                start = int(start)
                end = int(end)

                if start < 0:
                    raise ValueError(
                        f"{protein_id}: negative start "
                        f"offset {start}"
                    )

                if end <= start:
                    raise ValueError(
                        f"{protein_id}: invalid range "
                        f"[{start}, {end})"
                    )

                sequence_length = end - start

                if protein_id in seq_len_lookup:
                    raise ValueError(
                        f"Duplicate protein ID across "
                        f"embedding shards: {protein_id}"
                    )

                seq_len_lookup[
                    protein_id
                ] = sequence_length

                total_index_rows += 1

    lengths = list(seq_len_lookup.values())

    if not lengths:
        raise ValueError(
            "No protein lengths were extracted."
        )

    expected_protein_count = 36_641

    if len(seq_len_lookup) != expected_protein_count:
        raise ValueError(
            f"Expected {expected_protein_count} "
            f"proteins, found "
            f"{len(seq_len_lookup)}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("wb") as file:
        pickle.dump(
            seq_len_lookup,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("\nPFresGO sequence-length lookup")
    print("=" * 60)
    print(f"Index files: {len(index_paths)}")
    print(f"Index rows: {total_index_rows}")
    print(f"Unique proteins: {len(seq_len_lookup)}")
    print(f"Minimum length: {min(lengths)}")
    print(f"Maximum length: {max(lengths)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nExample entries:")

    for protein_id in sorted(
        seq_len_lookup
    )[:10]:
        print(
            f"  {protein_id}: "
            f"{seq_len_lookup[protein_id]}"
        )

    print("\nLookup created successfully.")


if __name__ == "__main__":
    main()