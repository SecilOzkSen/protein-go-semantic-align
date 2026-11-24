"""Standard-library provenance checks for retained zero-positive GO evaluation.

This module never imports torch or reads checkpoint payloads.  A cohort is a
verified view of the existing development split, not a class-heldout dataset.
Checkpoint metadata is audited without rewriting its original validation policy.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


COHORT_FORMAT = "natural_train_zero_positive_validation_v1"
INPUT_KEYS = (
    "train_ids", "val_ids", "annotations", "candidate_ids", "temporal_ids",
    "temporal_test_ids",
)
EVALUATION_LABEL_POLICY = "all_source_validation_positives_preserved"
LEGACY_LABEL_POLICY = "legacy_exclude_train_unobserved_positives"
REQUIRED_OUTPUTS = (
    "unseen_validation_go_ids.json", "seen_go_ids.json",
    "active_candidate_go_ids.json", "reserved_temporal_go_ids.json",
    "validation_protein_ids.txt", "validation_unseen_positive_protein_ids.txt",
    "validation_labels_full.json", "validation_labels_unseen.json",
    "validation_labels_seen.json", "go_train_validation_counts.tsv",
    "checkpoint_manifest_audit.json", "README.txt",
)
_SPECIAL_CHECKPOINTS = ("best_primary.pt", "best_macro.pt", "last.pt", "checkpoint_final.pt")
_EPOCH_NAME = re.compile(r"checkpoint_epoch([0-9]+)\.pt\Z")


def sha256_file(path: Path) -> str:
    """Hash a file, rejecting a file replaced or modified while it is read."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f"File changed while being hashed: {path}")
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     ensure_ascii=True).encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _go_int(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Invalid GO ID: {value!r}")
    text = str(value).strip()
    _require(re.fullmatch(r"(?:GO:)?[0-9]+", text, re.IGNORECASE), f"Invalid GO ID: {value!r}")
    result = int(text.split(":", 1)[-1])
    _require(result > 0, f"Invalid GO ID: {value!r}")
    return result


def _protein_ids(path):
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
           if line.strip()]
    _require(len(ids) == len(set(ids)), f"Duplicate protein IDs: {path}")
    return ids


def _source_go_ids(path):
    raw = _read_json(path)
    if isinstance(raw, dict):
        raw = next((raw[key] for key in ("ids", "go_ids", "terms") if key in raw), list(raw))
    _require(isinstance(raw, list), f"GO source is not a list or supported mapping: {path}")
    ids = [_go_int(value) for value in raw]
    _require(len(ids) == len(set(ids)), f"Duplicate source GO IDs: {path}")
    return sorted(ids)


def _output_go_ids(path):
    raw = _read_json(path)
    _require(isinstance(raw, list), f"GO artifact is not a list: {path}")
    _require(all(type(value) is int and value > 0 for value in raw),
             f"GO artifact requires positive integer IDs: {path}")
    _require(raw == sorted(set(raw)), f"GO artifact is not unique and ascending: {path}")
    return raw


def _labels(path, protein_ids):
    raw = _read_json(path)
    _require(isinstance(raw, dict) and set(raw) == set(protein_ids),
             f"Label map must contain exactly all validation proteins: {path}")
    result = {}
    for pid in protein_ids:
        values = raw[pid]
        _require(isinstance(values, list) and all(type(g) is int and g > 0 for g in values),
                 f"Invalid GO label list for {pid!r}: {path}")
        _require(values == sorted(set(values)), f"Noncanonical GO label list for {pid!r}: {path}")
        result[pid] = values
    return result


def _verify_output_hashes(directory, hashes):
    _require(isinstance(hashes, dict), "Missing output_sha256_excluding_summary mapping")
    _require(set(REQUIRED_OUTPUTS) <= set(hashes), "Missing required artifact in output hash manifest")
    for name, expected in hashes.items():
        # The producer writes a flat bundle. Never follow a manifest outside it.
        _require(isinstance(name, str) and name not in ("", ".", "..", "summary.json")
                 and Path(name).name == name and "/" not in name and "\\" not in name,
                 f"Invalid output hash entry: {name!r}")
        _require(_digest(expected), f"Invalid output SHA256 for {name}")
        path = directory / name
        _require(path.is_file() and path.resolve().parent == directory.resolve(),
                 f"Missing or external cohort artifact: {path}")
        _require(sha256_file(path) == expected, f"Output SHA256 mismatch: {name}")


def _source_cohort(inputs):
    """Rebuild only development labels; final-test annotation entries are unused."""
    train_ids = _protein_ids(inputs["train_ids"])
    val_ids = _protein_ids(inputs["val_ids"])
    test_ids = _protein_ids(inputs["temporal_test_ids"])
    _require(train_ids and val_ids and test_ids, "Train, validation and final-test ID files must be nonempty")
    train, val, test = set(train_ids), set(val_ids), set(test_ids)
    _require(not (train & val or train & test or val & test), "Source protein splits overlap")
    candidate = set(_source_go_ids(inputs["candidate_ids"]))
    temporal = set(_source_go_ids(inputs["temporal_ids"]))
    _require(candidate and temporal <= candidate, "Invalid source candidate/temporal GO universe")
    active = candidate - temporal
    _require(active, "Source active GO universe is empty")
    raw = _read_json(inputs["annotations"])
    _require(isinstance(raw, dict), "Source annotations must be a mapping")
    labels, duplicates = {}, 0
    for pid in train_ids + val_ids:
        _require(pid in raw and isinstance(raw[pid], list), f"Missing source annotation list: {pid!r}")
        values = [_go_int(g) for g in raw[pid]]
        ys = set(values)
        _require(ys and ys <= active, f"Empty, reserved or out-of-universe development labels: {pid!r}")
        duplicates += len(values) - len(ys)
        labels[pid] = sorted(ys)
    train_counts = Counter(g for p in train_ids for g in labels[p])
    val_counts = Counter(g for p in val_ids for g in labels[p])
    seen = set(train_counts)
    unseen = {g for g in active if train_counts[g] == 0 and val_counts[g] > 0}
    full = {p: labels[p] for p in val_ids}
    masked = {p: sorted(set(full[p]) & unseen) for p in val_ids}
    seen_labels = {p: sorted(set(full[p]) & seen) for p in val_ids}
    positive_ids = [p for p in val_ids if masked[p]]
    counts = {
        "n_train_proteins": len(train_ids), "n_validation_proteins": len(val_ids),
        "n_test_proteins_checked": len(test_ids), "n_candidate_go": len(candidate),
        "n_active_go": len(active), "n_reserved_temporal_go": len(temporal),
        "n_seen_train_go": len(seen), "n_unseen_validation_go": len(unseen),
        "n_unseen_without_validation_positive_go": len(active - seen - unseen),
        "n_validation_proteins_with_unseen": len(positive_ids),
        "n_validation_proteins_without_unseen": len(val_ids) - len(positive_ids),
        "n_validation_proteins_only_unseen": sum(bool(masked[p]) and not seen_labels[p] for p in val_ids),
        "n_training_positive_pairs": sum(train_counts.values()),
        "n_validation_positive_pairs": sum(val_counts.values()),
        "n_validation_unseen_positive_pairs": sum(len(v) for v in masked.values()),
        "n_duplicate_annotation_ids_collapsed": duplicates,
    }
    return {
        "train_ids": train_ids, "protein_ids": val_ids, "positive_ids": positive_ids,
        "candidate": sorted(candidate), "active": sorted(active), "seen": sorted(seen),
        "unseen": sorted(unseen), "temporal": sorted(temporal), "full": full,
        "masked": masked, "seen_labels": seen_labels, "train_counts": train_counts,
        "val_counts": val_counts, "counts": counts,
    }


def _verify_count_table(path, source):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_fields = ["go_id", "go_int", "train_positive_proteins", "validation_positive_proteins", "group"]
        _require(reader.fieldnames == expected_fields, "Invalid GO count table columns")
        rows = list(reader)
    _require(len(rows) == len(source["candidate"]), "GO count table row count mismatch")
    temporal, seen, unseen = (set(source[name]) for name in ("temporal", "seen", "unseen"))
    for row, g in zip(rows, source["candidate"]):
        group = ("reserved_temporal" if g in temporal else "seen" if g in seen else
                 "unseen_validation" if g in unseen else "unseen_without_validation_positive")
        expected = dict(zip(expected_fields, [f"GO:{g:07d}", str(g), str(source["train_counts"][g]),
                                              str(source["val_counts"][g]), group]))
        _require(row == expected, f"GO count table mismatch at GO:{g:07d}")


def load_cohort(directory: Path) -> dict:
    """Verify a prepared cohort and current sources, then return ordered labels.

    Every validation protein remains present, including proteins with no unseen
    positive. An empty unseen cohort fails here, before model/GPU initialization.
    File absence raises OSError; inconsistent provenance or content raises ValueError.
    """
    directory = Path(directory).resolve()
    summary_path = directory / "summary.json"
    signature = sha256_file(summary_path)
    summary = _read_json(summary_path)
    _require(isinstance(summary, dict) and summary.get("format") == COHORT_FORMAT,
             "Unsupported or missing cohort format")
    _require(summary.get("branch") in ("MF", "BP", "CC"), "Invalid cohort ontology branch")
    _require(summary.get("validation_label_policy") == EVALUATION_LABEL_POLICY,
             "Cohort does not preserve all source validation positives")
    _require(summary.get("final_test_annotations_read") is False,
             "Cohort cannot certify final-test annotation exclusion")
    output_hashes = summary.get("output_sha256_excluding_summary")
    _verify_output_hashes(directory, output_hashes)
    input_paths, source_hashes = summary.get("inputs"), summary.get("input_sha256")
    _require(isinstance(input_paths, dict) and isinstance(source_hashes, dict),
             "Missing cohort source paths or input hashes")
    _require(set(INPUT_KEYS) <= set(input_paths) and set(INPUT_KEYS) <= set(source_hashes),
             "Missing required cohort source path or input SHA256")
    _require(set(input_paths) == set(source_hashes), "Source path/hash keys differ")
    inputs = {}
    for key, path in input_paths.items():
        _require(isinstance(path, str) and Path(path).is_absolute(), f"Source path must be absolute: {key}")
        _require(_digest(source_hashes[key]), f"Invalid source SHA256: {key}")
        inputs[key] = Path(path)
        _require(sha256_file(inputs[key]) == source_hashes[key], f"Source SHA256 mismatch: {key}")
    source = _source_cohort(inputs)
    proteins = _protein_ids(directory / "validation_protein_ids.txt")
    _require(proteins == source["protein_ids"], "Validation protein ID order or membership differs from source")
    ids = {}
    names = {"active": "active_candidate_go_ids.json", "seen": "seen_go_ids.json",
             "unseen": "unseen_validation_go_ids.json", "temporal": "reserved_temporal_go_ids.json"}
    for key, filename in names.items():
        ids[key] = _output_go_ids(directory / filename)
        _require(ids[key] == source[key], f"GO membership differs from source: {filename}")
    labels = {}
    for key, filename in (("full", "validation_labels_full.json"),
                          ("masked", "validation_labels_unseen.json"),
                          ("seen_labels", "validation_labels_seen.json")):
        labels[key] = _labels(directory / filename, proteins)
        _require(labels[key] == source[key], f"Validation label partition differs from source: {filename}")
    positive_ids = _protein_ids(directory / "validation_unseen_positive_protein_ids.txt")
    _require(positive_ids == source["positive_ids"], "Unseen-positive protein index/order mismatch")
    counts = summary.get("counts")
    _require(isinstance(counts, dict), "Missing cohort counts")
    for key, value in source["counts"].items():
        _require(type(counts.get(key)) is int and counts[key] == value, f"Cohort count mismatch: {key}")
    _require(summary.get("active_go_ids_sha256") == canonical_hash(ids["active"]), "Active GO ID hash mismatch")
    _require(summary.get("unseen_validation_go_ids_sha256") == canonical_hash(ids["unseen"]),
             "Unseen GO ID hash mismatch")
    _verify_count_table(directory / "go_train_validation_counts.tsv", source)
    supports = [source["val_counts"][g] for g in ids["unseen"]]
    expected_support = {
        "minimum": min(supports) if supports else None,
        "median": statistics.median(supports) if supports else None,
        "maximum": max(supports) if supports else None,
        "terms_with_one_positive": sum(n == 1 for n in supports),
        "terms_with_at_least_five_positives": sum(n >= 5 for n in supports),
    }
    _require(summary.get("unseen_validation_support") == expected_support, "Unseen support summary mismatch")
    expected_status = "unseen_validation_found" if ids["unseen"] else "no_unseen_validation_terms"
    _require(summary.get("status") == expected_status, "Cohort status inconsistent with unseen terms")
    _require(summary.get("unseen_metrics_computable_from_labels") is bool(ids["unseen"]),
             "Cohort metric availability flag mismatch")
    # Recheck fingerprints after parsing, rather than using a mixed live snapshot.
    for key, path in inputs.items():
        _require(sha256_file(path) == source_hashes[key], f"Source changed during cohort validation: {key}")
    _verify_output_hashes(directory, output_hashes)
    _require(sha256_file(summary_path) == signature, "Cohort summary changed during validation")
    _require(ids["unseen"], "No unseen validation GO terms: unseen metrics are not computable")
    return {
        "summary": summary, "summary_path": summary_path, "signature": signature,
        "protein_ids": proteins, "go_ids": ids["active"], "seen_go_ids": ids["seen"],
        "unseen_go_ids": ids["unseen"], "reserved_temporal_go_ids": ids["temporal"],
        "labels_full": labels["full"], "labels_unseen": labels["masked"],
        "labels_seen": labels["seen_labels"], "source_sha256": dict(source_hashes),
        "validation_unseen_positive_protein_ids": positive_ids,
    }


def validate_checkpoint_meta(meta, cohort, expected_ablation=None) -> dict:
    """Require matching checkpoint data identity and return a separate audit.

    This is a metadata audit only. Matching input hashes establish the positive
    supervision cohort; they do not establish which classes appeared as negatives
    in the loss. Neither the checkpoint metadata nor the cohort is modified.
    """
    _require(isinstance(meta, dict), "Checkpoint meta must be a mapping")
    identity = meta.get("data_identity")
    _require(isinstance(identity, dict), "Missing checkpoint meta.data_identity")
    config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    stored = identity.get("input_sha256")
    _require(isinstance(stored, dict), "Missing checkpoint data_identity.input_sha256")
    summary, expected_hashes = cohort["summary"], cohort["source_sha256"]
    missing, mismatches, comparisons = [], [], {}
    for key in INPUT_KEYS:
        expected, observed = expected_hashes[key], stored.get(key)
        comparisons[key] = {"cohort_sha256": expected, "checkpoint_sha256": observed,
                            "match": observed == expected}
        if observed is None:
            missing.append(f"data_identity.input_sha256.{key}")
        elif observed != expected:
            mismatches.append(f"data_identity.input_sha256.{key}")
    branch = summary["branch"]
    if identity.get("branch") is None:
        missing.append("data_identity.branch")
    elif identity["branch"] != branch:
        mismatches.append("data_identity.branch")
    if config.get("branch") is not None and config["branch"] != branch:
        mismatches.append("config.branch")
    expected_active_hash = canonical_hash(cohort["go_ids"])
    if identity.get("active_go_ids_sha256") is None:
        missing.append("data_identity.active_go_ids_sha256")
    elif identity["active_go_ids_sha256"] != expected_active_hash:
        mismatches.append("data_identity.active_go_ids_sha256")
    ablation = config.get("ablation_id")
    if expected_ablation is not None:
        if ablation is None:
            missing.append("config.ablation_id")
        elif ablation != expected_ablation:
            mismatches.append("config.ablation_id")
    counts = summary["counts"]
    expected_counts = {
        "n_train_proteins": counts["n_train_proteins"],
        "n_val_proteins": counts["n_validation_proteins"],
        "n_validation_proteins": counts["n_validation_proteins"],
        "n_active_go": counts["n_active_go"], "n_candidate_go": counts["n_candidate_go"],
        "n_temporal_go": counts["n_reserved_temporal_go"],
        "n_observed_train_go": counts["n_seen_train_go"],
        "n_train_positive_pairs": counts["n_training_positive_pairs"],
        "n_val_source_positive_pairs": counts["n_validation_positive_pairs"],
        "n_train": counts["n_train_proteins"], "n_val": counts["n_validation_proteins"],
        "n_active": counts["n_active_go"], "n_temporal": counts["n_reserved_temporal_go"],
    }
    original_policy = identity.get("validation_label_policy")
    if original_policy == LEGACY_LABEL_POLICY:
        expected_counts["n_val_filtered_positive_pairs"] = counts["n_validation_unseen_positive_pairs"]
        expected_counts["n_val_evaluated_positive_pairs"] = (
            counts["n_validation_positive_pairs"] - counts["n_validation_unseen_positive_pairs"])
    elif original_policy == EVALUATION_LABEL_POLICY:
        expected_counts["n_val_filtered_positive_pairs"] = 0
        expected_counts["n_val_evaluated_positive_pairs"] = counts["n_validation_positive_pairs"]
    count_comparisons = {}
    for key, expected in expected_counts.items():
        if key in identity:
            match = type(identity[key]) is int and identity[key] == expected
            count_comparisons[key] = {"cohort": expected, "checkpoint": identity[key], "match": match}
            if not match:
                mismatches.append(f"data_identity.{key}")
    if missing or mismatches:
        details = ("missing=" + ", ".join(missing) + "; " if missing else "")
        details += "mismatches=" + ", ".join(mismatches) if mismatches else ""
        raise ValueError("Checkpoint metadata incompatible with cohort: " + details.rstrip("; "))
    return {
        "status": "compatible", "scope": "checkpoint_metadata_split_annotation_candidate_identity",
        "cohort_summary_sha256": cohort["signature"], "branch": branch,
        "ablation_id": ablation, "expected_ablation": expected_ablation,
        "input_comparisons": comparisons, "count_comparisons": count_comparisons,
        "active_go_ids_sha256": expected_active_hash, "missing_metadata": [], "mismatches": [],
        "original_data_identity": deepcopy(identity),
        "original_checkpoint_selection": deepcopy(meta.get("selection")),
        "original_validation_label_policy": deepcopy(original_policy),
        "evaluation_validation_label_policy": EVALUATION_LABEL_POLICY,
        "evaluation_policy": {
            "validation_label_policy": EVALUATION_LABEL_POLICY,
            "protein_universe": "all_validation_proteins_in_original_order",
            "ranking_universe": "all_active_candidate_go_ids",
            "metric_target_group": "natural_train_zero_positive_validation",
            "reserved_temporal_go_excluded": True,
        },
        "checkpoint_metadata_mutated": False, "checkpoint_weights_inspected": False,
        "configured_unseen_training_exposure": "unverified",
        "strict_class_holdout_verified": False, "strict_class_nonexposure_verified": False,
        "exposure_note": "Zero positive training count does not establish absence of negative-candidate exposure.",
    }


def _checkpoint_order(path):
    match = _EPOCH_NAME.fullmatch(path.name)
    return (0, int(match.group(1)), path.name) if match else (1, _SPECIAL_CHECKPOINTS.index(path.name), path.name)


def discover_checkpoints(run_dir: Path, selection="all") -> list[dict]:
    """Find retained epoch checkpoints and aliases, deduplicating exact bytes.

    Scan only the run directory itself. Hardlinked aliases share one hash pass;
    independent byte-identical copies are merged by SHA256. A surviving epoch
    filename is preferred as the canonical path, otherwise a named alias is used.
    """
    _require(selection in ("all", "best_primary"), f"Unsupported checkpoint selection: {selection!r}")
    run_dir = Path(run_dir).resolve()
    _require(run_dir.is_dir(), f"Checkpoint run directory does not exist: {run_dir}")
    paths = sorted((p for p in run_dir.iterdir() if p.is_file()
                    and (_EPOCH_NAME.fullmatch(p.name) or p.name in _SPECIAL_CHECKPOINTS)),
                   key=_checkpoint_order)
    _require(paths, f"No supported checkpoints found: {run_dir}")
    if selection == "best_primary":
        _require(any(p.name == "best_primary.pt" for p in paths), f"Missing best_primary.pt: {run_dir}")
    inodes, by_hash = {}, {}
    for path in paths:
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in inodes:
            digest = inodes[inode]
        else:
            digest = sha256_file(path)
            inodes[inode] = digest
        by_hash.setdefault(digest, []).append(path)
    result = []
    for digest, members in by_hash.items():
        members.sort(key=_checkpoint_order)
        aliases = [p.name for p in members]
        if selection == "best_primary" and "best_primary.pt" not in aliases:
            continue
        result.append({
            "path": members[0], "sha256": digest, "aliases": aliases,
            "is_best_primary": "best_primary.pt" in aliases,
            "is_best_macro": "best_macro.pt" in aliases,
            "is_last": "last.pt" in aliases, "is_final": "checkpoint_final.pt" in aliases,
        })
    return sorted(result, key=lambda row: _checkpoint_order(row["path"]))
