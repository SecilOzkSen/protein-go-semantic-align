import json
from collections import Counter
from pathlib import Path

import pandas as pd


ANNOT_PATH = Path(
    "/workspace/stargo/datasets/pfresgo/annot.tsv"
)

GO_VOCAB_PATH = Path(
    "/workspace/data_pfresgo/processed/go_vocab.json"
)

ALT_ID_PATH = Path(
    "/workspace/data_pfresgo/processed/go_alt_id_to_primary.json"
)


COLUMN_TO_NAMESPACE = {
    "mf": "molecular_function",
    "bp": "biological_process",
    "cc": "cellular_component",
}


def load_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=12,
        dtype=str,
        keep_default_na=False,
    )

    df.columns = [
        "protein_id",
        "mf",
        "bp",
        "cc",
    ]

    if df["protein_id"].duplicated().any():
        duplicated = df.loc[
            df["protein_id"].duplicated(),
            "protein_id",
        ].tolist()

        raise ValueError(
            "Duplicate protein IDs found in annot.tsv. "
            f"Examples: {duplicated[:10]}"
        )

    return df.set_index("protein_id")


def split_go_ids(value: str):
    if not value:
        return []

    return [
        go_id.strip()
        for go_id in value.split(",")
        if go_id.strip()
    ]


def main():
    with open(GO_VOCAB_PATH, encoding="utf-8") as file:
        go_vocab = json.load(file)

    with open(ALT_ID_PATH, encoding="utf-8") as file:
        alt_id_to_primary = json.load(file)

    annotations = load_annotations(ANNOT_PATH)

    all_raw_ids = set()
    all_canonical_ids = set()

    raw_annotation_count = 0
    canonicalized_count = 0

    missing_ids = set()
    obsolete_ids = set()
    namespace_mismatches = []
    replaced_obsolete_ids = Counter()
    consider_obsolete_ids = Counter()

    branch_stats = {}

    for column, expected_namespace in COLUMN_TO_NAMESPACE.items():
        branch_raw_ids = set()
        branch_canonical_ids = set()
        branch_annotation_count = 0
        proteins_with_annotation = 0

        for protein_id, value in annotations[column].items():
            raw_ids = split_go_ids(value)

            if raw_ids:
                proteins_with_annotation += 1

            for raw_id in raw_ids:
                raw_annotation_count += 1
                branch_annotation_count += 1

                all_raw_ids.add(raw_id)
                branch_raw_ids.add(raw_id)

                canonical_id = alt_id_to_primary.get(
                    raw_id,
                    raw_id,
                )

                if canonical_id != raw_id:
                    canonicalized_count += 1

                all_canonical_ids.add(canonical_id)
                branch_canonical_ids.add(canonical_id)

                term = go_vocab.get(canonical_id)

                if term is None:
                    missing_ids.add(raw_id)
                    continue

                if term.get("is_obsolete", False):
                    obsolete_ids.add(canonical_id)

                    for replacement in term.get(
                        "replaced_by",
                        [],
                    ):
                        replaced_obsolete_ids[
                            replacement
                        ] += 1

                    for candidate in term.get(
                        "consider",
                        [],
                    ):
                        consider_obsolete_ids[
                            candidate
                        ] += 1

                actual_namespace = term.get(
                    "namespace",
                    "",
                )

                if actual_namespace != expected_namespace:
                    namespace_mismatches.append(
                        {
                            "protein_id": protein_id,
                            "column": column,
                            "raw_id": raw_id,
                            "canonical_id": canonical_id,
                            "expected": expected_namespace,
                            "actual": actual_namespace,
                        }
                    )

        branch_stats[column] = {
            "unique_raw_terms": len(branch_raw_ids),
            "unique_canonical_terms": len(
                branch_canonical_ids
            ),
            "annotation_instances": (
                branch_annotation_count
            ),
            "proteins_with_annotation": (
                proteins_with_annotation
            ),
        }

    print("\nPFresGO annotation validation")
    print("=" * 60)
    print(f"Proteins in annot.tsv: {len(annotations)}")
    print(
        "Raw annotation instances: "
        f"{raw_annotation_count}"
    )
    print(f"Unique raw GO IDs: {len(all_raw_ids)}")
    print(
        "Unique canonical GO IDs: "
        f"{len(all_canonical_ids)}"
    )
    print(
        "Alternative-ID occurrences canonicalized: "
        f"{canonicalized_count}"
    )
    print(f"Missing GO IDs: {len(missing_ids)}")
    print(
        "Annotated obsolete GO IDs: "
        f"{len(obsolete_ids)}"
    )
    print(
        "Namespace mismatches: "
        f"{len(namespace_mismatches)}"
    )

    print("\nBranch statistics")
    print("=" * 60)

    for branch, stats in branch_stats.items():
        print(f"\n{branch.upper()}")
        print(
            "  Unique raw terms: "
            f"{stats['unique_raw_terms']}"
        )
        print(
            "  Unique canonical terms: "
            f"{stats['unique_canonical_terms']}"
        )
        print(
            "  Annotation instances: "
            f"{stats['annotation_instances']}"
        )
        print(
            "  Proteins with annotation: "
            f"{stats['proteins_with_annotation']}"
        )

    if missing_ids:
        print("\nMissing GO-ID examples:")
        print(sorted(missing_ids)[:20])

    if obsolete_ids:
        print("\nAnnotated obsolete GO-ID examples:")
        print(sorted(obsolete_ids)[:20])

    if namespace_mismatches:
        print("\nNamespace mismatch examples:")

        for item in namespace_mismatches[:20]:
            print(item)

    if replaced_obsolete_ids:
        print("\nMost common replaced_by targets:")

        for go_id, count in replaced_obsolete_ids.most_common(
            20
        ):
            print(f"  {go_id}: {count}")

    if consider_obsolete_ids:
        print("\nMost common consider targets:")

        for go_id, count in consider_obsolete_ids.most_common(
            20
        ):
            print(f"  {go_id}: {count}")

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} annotation GO IDs could "
            "not be mapped to the ontology."
        )

    if namespace_mismatches:
        raise ValueError(
            f"{len(namespace_mismatches)} annotations "
            "have namespace mismatches."
        )

    print("\nValidation completed successfully.")


if __name__ == "__main__":
    main()