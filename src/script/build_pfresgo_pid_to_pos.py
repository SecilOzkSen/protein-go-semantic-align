"""
Build branch-specific PFresGO protein-to-positive-GO mappings.

Input:
    PFresGO annot.tsv
    Parsed GO vocabulary
    Alternative GO-ID mapping
    Active branch GO-ID lists

Outputs:
    pid_to_positives_bp.json
    pid_to_positives_mf.json
    pid_to_positives_cc.json

GO IDs are stored as canonical integer IDs to remain compatible
with the existing semantic-alignment pipeline:

    GO:0003677 -> 3677

Proteins without an annotation in a branch are retained with an
empty list by default. Training loaders can filter them later.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


DEFAULT_PFRESGO_DIR = Path(
    "/workspace/stargo/datasets/pfresgo"
)

DEFAULT_PROCESSED_DIR = Path(
    "/workspace/data_pfresgo/processed"
)


BRANCH_CONFIG = {
    "bp": {
        "column": "bp",
        "namespace": "biological_process",
        "ids_filename": "go_ids_bp.json",
        "output_filename": "pid_to_positives_bp.json",
    },
    "mf": {
        "column": "mf",
        "namespace": "molecular_function",
        "ids_filename": "go_ids_mf.json",
        "output_filename": "pid_to_positives_mf.json",
    },
    "cc": {
        "column": "cc",
        "namespace": "cellular_component",
        "ids_filename": "go_ids_cc.json",
        "output_filename": "pid_to_positives_cc.json",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build branch-specific PFresGO "
            "protein-to-positive-GO mappings."
        )
    )

    parser.add_argument(
        "--annot-path",
        type=Path,
        default=DEFAULT_PFRESGO_DIR / "annot.tsv",
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )

    parser.add_argument(
        "--branches",
        nargs="+",
        choices=["bp", "mf", "cc"],
        default=["bp", "mf", "cc"],
    )

    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help=(
            "Remove proteins without positive annotations "
            "in the selected branch. Default keeps them "
            "with an empty list."
        ),
    )

    return parser.parse_args()


def normalize_go_id(go_id) -> str:
    """
    Normalize a GO identifier to GO:XXXXXXX.

    Examples:
        GO:0003677 -> GO:0003677
        0003677    -> GO:0003677
        3677       -> GO:0003677
    """
    value = str(go_id).strip()

    if not value:
        raise ValueError("Empty GO identifier.")

    if value.upper().startswith("GO:"):
        value = value.split(":", 1)[1].strip()

    if not value.isdigit():
        raise ValueError(
            f"Invalid GO identifier: {go_id!r}"
        )

    return f"GO:{int(value):07d}"


def go_id_to_int(go_id: str) -> int:
    canonical = normalize_go_id(go_id)
    return int(canonical.split(":", 1)[1])


def split_annotation_string(value: str) -> List[str]:
    if not value:
        return []

    return [
        normalize_go_id(go_id)
        for go_id in value.split(",")
        if go_id.strip()
    ]


def load_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {path}"
        )

    dataframe = pd.read_csv(
        path,
        sep="\t",
        skiprows=12,
        dtype=str,
        keep_default_na=False,
    )

    expected_columns = 4

    if len(dataframe.columns) != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in "
            f"{path}, found {len(dataframe.columns)}."
        )

    dataframe.columns = [
        "protein_id",
        "mf",
        "bp",
        "cc",
    ]

    if dataframe["protein_id"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["protein_id"].duplicated(),
            "protein_id",
        ].tolist()

        raise ValueError(
            "Duplicate protein IDs found in annot.tsv. "
            f"Examples: {duplicates[:10]}"
        )

    return dataframe


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_active_branch_ids(
    processed_dir: Path,
    branch: str,
) -> Set[str]:
    config = BRANCH_CONFIG[branch]

    path = (
        processed_dir
        / config["ids_filename"]
    )

    raw_ids = load_json(path)

    branch_ids = {
        normalize_go_id(go_id)
        for go_id in raw_ids
    }

    if len(branch_ids) != len(raw_ids):
        raise ValueError(
            f"Duplicate GO IDs found in {path}"
        )

    return branch_ids


def load_alt_id_mapping(
    processed_dir: Path,
) -> Dict[str, str]:
    path = (
        processed_dir
        / "go_alt_id_to_primary.json"
    )

    raw_mapping = load_json(path)

    return {
        normalize_go_id(alt_id): normalize_go_id(
            primary_id
        )
        for alt_id, primary_id
        in raw_mapping.items()
    }


def load_go_vocabulary(
    processed_dir: Path,
):
    path = processed_dir / "go_vocab.json"
    raw_vocab = load_json(path)

    return {
        normalize_go_id(go_id): term
        for go_id, term in raw_vocab.items()
    }


def canonicalize_go_id(
    go_id: str,
    alt_id_to_primary: Dict[str, str],
) -> str:
    go_id = normalize_go_id(go_id)

    return alt_id_to_primary.get(
        go_id,
        go_id,
    )


def build_branch_mapping(
    dataframe: pd.DataFrame,
    branch: str,
    active_branch_ids: Set[str],
    go_vocabulary,
    alt_id_to_primary,
    drop_empty: bool,
):
    config = BRANCH_CONFIG[branch]
    annotation_column = config["column"]
    expected_namespace = config["namespace"]

    output: Dict[str, List[int]] = {}

    total_proteins = 0
    kept_proteins = 0
    dropped_empty = 0
    total_annotation_instances = 0
    kept_annotation_instances = 0
    canonicalized_instances = 0

    missing_go_ids = set()
    obsolete_go_ids = set()
    wrong_namespace = set()
    outside_active_branch = set()

    observed_canonical_ids = set()

    for row in dataframe.itertuples(
        index=False
    ):
        protein_id = str(
            row.protein_id
        ).strip()

        raw_value = getattr(
            row,
            annotation_column,
        )

        raw_go_ids = split_annotation_string(
            raw_value
        )

        total_proteins += 1
        mapped_ids = []

        for raw_go_id in raw_go_ids:
            total_annotation_instances += 1

            canonical_go_id = canonicalize_go_id(
                raw_go_id,
                alt_id_to_primary,
            )

            if canonical_go_id != raw_go_id:
                canonicalized_instances += 1

            term = go_vocabulary.get(
                canonical_go_id
            )

            if term is None:
                missing_go_ids.add(
                    raw_go_id
                )
                continue

            if term.get(
                "is_obsolete",
                False,
            ):
                obsolete_go_ids.add(
                    canonical_go_id
                )
                continue

            actual_namespace = term.get(
                "namespace",
                "",
            )

            if actual_namespace != expected_namespace:
                wrong_namespace.add(
                    canonical_go_id
                )
                continue

            if (
                canonical_go_id
                not in active_branch_ids
            ):
                outside_active_branch.add(
                    canonical_go_id
                )
                continue

            mapped_ids.append(
                go_id_to_int(
                    canonical_go_id
                )
            )

            observed_canonical_ids.add(
                canonical_go_id
            )

            kept_annotation_instances += 1

        mapped_ids = sorted(set(mapped_ids))

        if drop_empty and not mapped_ids:
            dropped_empty += 1
            continue

        output[protein_id] = mapped_ids
        kept_proteins += 1

    errors = []

    if missing_go_ids:
        errors.append(
            f"{len(missing_go_ids)} GO IDs are "
            "missing from go_vocab.json."
        )

    if obsolete_go_ids:
        errors.append(
            f"{len(obsolete_go_ids)} annotated GO IDs "
            "are obsolete."
        )

    if wrong_namespace:
        errors.append(
            f"{len(wrong_namespace)} GO IDs have the "
            "wrong namespace."
        )

    if outside_active_branch:
        errors.append(
            f"{len(outside_active_branch)} GO IDs are "
            "outside the active branch vocabulary."
        )

    if errors:
        print(f"\n{branch.upper()} errors")

        for error in errors:
            print(f"  - {error}")

        if missing_go_ids:
            print(
                "  Missing examples:",
                sorted(missing_go_ids)[:10],
            )

        if obsolete_go_ids:
            print(
                "  Obsolete examples:",
                sorted(obsolete_go_ids)[:10],
            )

        if wrong_namespace:
            print(
                "  Namespace examples:",
                sorted(wrong_namespace)[:10],
            )

        if outside_active_branch:
            print(
                "  Outside-branch examples:",
                sorted(
                    outside_active_branch
                )[:10],
            )

        raise ValueError(
            f"PFresGO {branch} mapping failed."
        )

    stats = {
        "branch": branch,
        "namespace": expected_namespace,
        "active_branch_terms": len(
            active_branch_ids
        ),
        "observed_annotation_terms": len(
            observed_canonical_ids
        ),
        "proteins_total": total_proteins,
        "proteins_kept": kept_proteins,
        "proteins_dropped_empty": dropped_empty,
        "annotation_instances_total": (
            total_annotation_instances
        ),
        "annotation_instances_kept": (
            kept_annotation_instances
        ),
        "canonicalized_instances": (
            canonicalized_instances
        ),
        "drop_empty": drop_empty,
    }

    return output, stats


def save_json(data, path: Path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
        )


def main():
    args = parse_args()

    annotations = load_annotations(
        args.annot_path
    )

    go_vocabulary = load_go_vocabulary(
        args.processed_dir
    )

    alt_id_to_primary = load_alt_id_mapping(
        args.processed_dir
    )

    all_stats = {}

    print("\nBuilding PFresGO positive mappings")
    print("=" * 60)
    print(f"Proteins: {len(annotations)}")
    print(f"Drop empty: {args.drop_empty}")

    for branch in args.branches:
        active_branch_ids = (
            load_active_branch_ids(
                args.processed_dir,
                branch,
            )
        )

        mapping, stats = build_branch_mapping(
            dataframe=annotations,
            branch=branch,
            active_branch_ids=(
                active_branch_ids
            ),
            go_vocabulary=go_vocabulary,
            alt_id_to_primary=(
                alt_id_to_primary
            ),
            drop_empty=args.drop_empty,
        )

        output_path = (
            args.processed_dir
            / BRANCH_CONFIG[
                branch
            ]["output_filename"]
        )

        save_json(
            mapping,
            output_path,
        )

        all_stats[branch] = stats

        print(f"\n{branch.upper()}")
        print(
            "  Active branch terms: "
            f"{stats['active_branch_terms']}"
        )
        print(
            "  Observed annotation terms: "
            f"{stats['observed_annotation_terms']}"
        )
        print(
            "  Proteins total: "
            f"{stats['proteins_total']}"
        )
        print(
            "  Proteins kept: "
            f"{stats['proteins_kept']}"
        )
        print(
            "  Proteins dropped empty: "
            f"{stats['proteins_dropped_empty']}"
        )
        print(
            "  Annotation instances: "
            f"{stats['annotation_instances_kept']}"
        )
        print(f"  Output: {output_path}")

    stats_path = (
        args.processed_dir
        / "pid_to_positives_stats.json"
    )

    save_json(
        all_stats,
        stats_path,
    )

    print(
        f"\nSaved statistics: {stats_path}"
    )
    print(
        "\nPFresGO positive mappings "
        "created successfully."
    )


if __name__ == "__main__":
    main()