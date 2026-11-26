#!/usr/bin/env python3
"""Compute retrieval metrics from saved, audited general-test RAW scores.

CPU and NumPy only. No model loading, inference, score transformation,
true-path propagation, threshold fitting, or changes to the saved inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path
import re
import sys

import numpy as np


VERSION = "1.0.0"
MANIFEST_FORMAT = "mzsgo_fa_general_eval_v1"
DEFAULT_KS = [1, 3, 5, 10, 50, 100, 200, 500, 1000]
REQUIRED_FILES = (
    "scores_raw.npy", "evaluation_data.npz", "protein_ids.json",
    "go_ids.json", "metrics.json",
)
TRAIN_FILE = "checkpoint_train_labels.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_ks(ks):
    require(isinstance(ks, (list, tuple)) and bool(ks), "ks must be a nonempty list of positive integers")
    require(all(isinstance(k, Integral) and not isinstance(k, (bool, np.bool_)) and k > 0 for k in ks),
            "Every K must be a positive integer")
    values = [int(k) for k in ks]
    require(len(values) == len(set(values)), "K values must be distinct")
    return values


def _validate_arrays(scores, labels):
    require(isinstance(scores, np.ndarray) and scores.ndim == 2 and scores.dtype.kind == "f",
            "Scores must be a floating [N,G] NumPy array")
    require(isinstance(labels, np.ndarray) and labels.shape == scores.shape and labels.dtype.kind in "biuf",
            "Labels must be a numeric binary [N,G] array matching scores")
    require(scores.shape[1] > 0, "Candidate universe must not be empty")
    for start in range(0, scores.shape[0], 64):
        require(bool(np.isfinite(scores[start:start + 64]).all()), "Scores contain NaN or infinity")
        block = labels[start:start + 64]
        require(bool(((block == 0) | (block == 1)).all()), "Labels must contain only 0 or 1")


def compute_retrieval_metrics(scores, labels, *, ks, seen_mask=None):
    """Average per-protein target metrics without restricting ranking candidates.

    Empty target groups have null metrics. Stable score ties follow the original
    candidate-column order. K greater than G is effectively capped at G.
    """
    ks = validate_ks(ks)
    _validate_arrays(scores, labels)
    n, g = scores.shape
    masks = {"all": np.ones(g, dtype=bool)}
    if seen_mask is not None:
        seen_mask = np.asarray(seen_mask)
        require(seen_mask.shape == (g,) and seen_mask.dtype.kind == "b",
                "seen_mask must be a boolean vector with one entry per GO candidate")
        masks.update(seen=seen_mask, unseen=~seen_mask)
    term_totals = labels.sum(axis=0, dtype=np.int64)
    term_hits = {k: np.zeros(g, dtype=np.int64) for k in ks}
    groups = {}
    for name, mask in masks.items():
        groups[name] = {
            "n_proteins": 0,
            "n_positive_pairs": int(term_totals[mask].sum()),
            "n_terms_with_positives": int(np.count_nonzero(term_totals[mask])),
            "n_target_candidates": int(mask.sum()),
            "mAP": 0.0, "MRR": 0.0,
            **{f"R@{k}": 0.0 for k in ks},
        }
    inverse = np.empty(g, dtype=np.int64)
    positions = np.arange(1, g + 1, dtype=np.int64)
    for row in range(n):
        # A single full-universe sort is shared by all target groups.
        order = np.argsort(-scores[row], kind="stable")
        inverse[order] = positions
        positives = np.flatnonzero(labels[row])
        if positives.size == 0:
            continue
        for k in ks:
            hits = positives[inverse[positives] <= k]
            term_hits[k][hits] += 1
        for name, mask in masks.items():
            relevant = positives[mask[positives]]
            if relevant.size == 0:
                continue
            ranks = np.sort(inverse[relevant])
            m = ranks.size
            result = groups[name]
            result["n_proteins"] += 1
            result["mAP"] += float(np.mean(np.arange(1, m + 1, dtype=np.float64) / ranks))
            result["MRR"] += 1.0 / float(ranks[0])
            for k in ks:
                result[f"R@{k}"] += int(np.searchsorted(ranks, k, side="right")) / m
    for name, result in groups.items():
        count = result["n_proteins"]
        for key in ["mAP", "MRR"] + [f"R@{k}" for k in ks]:
            result[key] = result[key] / count if count else None
        supported = masks[name] & (term_totals > 0)
        for k in ks:
            result[f"macro_term_recall@{k}"] = (
                float(np.mean(term_hits[k][supported] / term_totals[supported]))
                if bool(supported.any()) else None
            )
    return groups


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key {key!r} in {path.name}")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique)


def _fingerprint(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def sha256_file(path):
    before = _fingerprint(path)
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    require(_fingerprint(path) == before, f"Input changed while hashing: {path}")
    return h.hexdigest()


def _ids(values, label, *, go=False):
    array = np.asarray(values)
    require(array.ndim == 1 and array.dtype.kind in "US", f"{label} must be a one-dimensional string array")
    result = [str(value) for value in array.astype(str)]
    require(result and all(v and v.strip() == v for v in result), f"Empty or whitespace-padded {label}")
    require(len(set(result)) == len(result), f"Duplicate {label}")
    if go:
        require(all(re.fullmatch(r"GO:[0-9]{7,}", v) and int(v[3:]) > 0 for v in result),
                f"{label} must use canonical GO:0000000 identifiers")
        require(all(v == f"GO:{int(v[3:]):07d}" for v in result),
                f"{label} has noncanonical GO padding; identifiers must match the exported GO format")
        require(len({int(v[3:]) for v in result}) == len(result), f"Duplicate numeric GO identifiers in {label}")
    return result


def _sha_string(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            f"Missing or invalid SHA256 for {label}")
    return value


def load_artifacts(raw_dir):
    """Read only the general-test files needed for ranking and target groups."""
    raw_dir = Path(raw_dir).expanduser().resolve()
    require(raw_dir.is_dir(), f"RAW directory not found: {raw_dir}")
    manifest_path = raw_dir / "evaluation_manifest.json"
    require(manifest_path.is_file(), f"Missing evaluation_manifest.json in {raw_dir}")
    manifest_digest = sha256_file(manifest_path)
    manifest = _json(manifest_path)
    require(isinstance(manifest, dict) and manifest.get("format") == MANIFEST_FORMAT,
            "Expected a saved general-test evaluation (mzsgo_fa_general_eval_v1), not temporal or TPR output")
    recorded_files = manifest.get("files")
    require(isinstance(recorded_files, dict), "Manifest lacks its files/SHA256 mapping")
    names = list(REQUIRED_FILES)
    if TRAIN_FILE in recorded_files or (raw_dir / TRAIN_FILE).exists():
        names.append(TRAIN_FILE)
    digests = {"evaluation_manifest.json": manifest_digest}
    fingerprints = {manifest_path: _fingerprint(manifest_path)}
    for name in names:
        path = raw_dir / name
        require(path.is_file(), f"Required input missing: {path}")
        expected = _sha_string(recorded_files.get(name), name)
        actual = sha256_file(path)
        require(actual == expected, f"Input SHA256 mismatch: {name}")
        digests[name] = actual
        fingerprints[path] = _fingerprint(path)
    metadata = _json(raw_dir / "metrics.json")
    require(isinstance(metadata, dict) and metadata.get("mode") == "general",
            "metrics.json must describe mode=general")
    branch = manifest.get("branch")
    require(branch in ("MF", "BP", "CC") and metadata.get("branch") == branch,
            "Manifest/metrics ontology mismatch")
    require(isinstance(metadata.get("model"), str) and bool(metadata["model"].strip()),
            "metrics.json lacks a model name")
    checkpoint_digest = _sha_string(metadata.get("checkpoint_sha256"), "checkpoint")
    require(manifest.get("checkpoint_sha256") == checkpoint_digest,
            "Manifest/metrics checkpoint SHA256 mismatch")
    with np.load(raw_dir / "evaluation_data.npz", allow_pickle=False) as saved:
        require({"labels", "protein_ids", "go_ids"} <= set(saved.files),
                "evaluation_data.npz must contain labels, protein_ids and go_ids; temporal labels are not supported")
        labels = saved["labels"]
        protein_ids = _ids(saved["protein_ids"], "NPZ protein IDs")
        go_ids = _ids(saved["go_ids"], "NPZ GO IDs", go=True)
    require(protein_ids == _ids(_json(raw_dir / "protein_ids.json"), "sidecar protein IDs"),
            "NPZ/sidecar protein ID order mismatch")
    require(go_ids == _ids(_json(raw_dir / "go_ids.json"), "sidecar GO IDs", go=True),
            "NPZ/sidecar GO ID order mismatch")
    scores = np.load(raw_dir / "scores_raw.npy", mmap_mode="r", allow_pickle=False)
    require(isinstance(scores, np.ndarray), "scores_raw.npy must contain one NumPy matrix, not an archive")
    require(scores.dtype == np.float32, "Expected the exporter's float32 scores_raw.npy matrix")
    require(scores.shape == (len(protein_ids), len(go_ids)), "Score dimensions do not match saved identifiers")
    _validate_arrays(scores, labels)
    require(labels.dtype == np.uint8, "Expected the exporter's uint8 binary labels matrix")
    require(bool(np.all(labels.sum(axis=1) > 0)), "General test contains a protein without positive annotations")
    for document, tag in [(manifest, "manifest"), (metadata, "metrics")]:
        for key, expected in [("n_proteins", len(protein_ids)), ("n_candidates", len(go_ids))]:
            require(type(document.get(key)) is int and document[key] == expected,
                    f"{tag} {key} disagrees with the saved matrices")
    for key, expected in [("general_positive_pairs", int(labels.sum())),
                          ("general_unique_positive_terms", int(np.count_nonzero(labels.sum(axis=0))))]:
        if key in metadata:
            require(type(metadata[key]) is int and metadata[key] == expected,
                    f"metrics {key} disagrees with saved labels")
    seen_mask = None
    training_count = None
    if TRAIN_FILE in names:
        annotations = _json(raw_dir / TRAIN_FILE)
        require(isinstance(annotations, dict) and bool(annotations), "Actual training annotation mapping is empty or invalid")
        train_go = set()
        for protein, terms in annotations.items():
            require(isinstance(protein, str) and protein and protein.strip() == protein,
                    "Invalid training protein identifier")
            require(isinstance(terms, list) and bool(terms), f"Invalid or empty training annotations for {protein}")
            train_go.update(_ids(terms, f"training GO IDs for {protein}", go=True))
        require(not set(protein_ids).intersection(annotations), "General-test proteins overlap actual training proteins")
        training_count = len(annotations)
        if "actual_training_proteins" in metadata:
            require(type(metadata["actual_training_proteins"]) is int and metadata["actual_training_proteins"] == training_count,
                    "Actual training annotation count differs from metrics.json")
        seen_mask = np.asarray([term in train_go for term in go_ids], dtype=bool)
    result = {
        "raw_dir": raw_dir, "metadata": metadata, "scores": scores, "labels": labels,
        "protein_ids": protein_ids, "go_ids": go_ids, "seen_mask": seen_mask,
        "training_count": training_count, "input_sha256": digests,
        "fingerprints": fingerprints,
    }
    assert_unchanged(result)
    return result


def assert_unchanged(artifacts):
    for path, expected in artifacts["fingerprints"].items():
        require(_fingerprint(path) == expected, f"Input changed during evaluation: {path}")


def _format(value):
    return "NA" if value is None else f"{value:.8f}"


def make_report(artifacts, groups, ks):
    metadata = artifacts["metadata"]
    return {
        "format": "mzsgo_saved_general_retrieval_v1", "script_version": VERSION,
        "mode": "general", "score_variant": "RAW", "score_file": "scores_raw.npy",
        "branch": metadata["branch"], "model": metadata["model"],
        "n_proteins": len(artifacts["protein_ids"]), "n_candidates": len(artifacts["go_ids"]),
        "ks": ks, "effective_ks": {str(k): min(k, len(artifacts["go_ids"])) for k in ks},
        "checkpoint": metadata.get("checkpoint"), "checkpoint_sha256": metadata["checkpoint_sha256"],
        "selected_validation_epoch": metadata.get("selected_validation_epoch"),
        "checkpoint_selection_metric": metadata.get("checkpoint_selection_metric"),
        "full_train_refit": metadata.get("full_train_refit"),
        "actual_training_proteins": artifacts["training_count"],
        "groups_available": list(groups),
        "seen_unseen_status": "available" if artifacts["seen_mask"] is not None else "training_annotations_not_supplied",
        "definitions": {
            "candidate_ranking": "All saved GO candidates, descending original float32 scores_raw.npy; no score transform or propagation",
            "tie_breaking": "Stable: tied scores retain the original saved GO column order",
            "mAP": "Compute AP per eligible protein as mean(j / full_rank_j), j=1..number of target positives; average proteins",
            "MRR": "Average reciprocal full-universe rank of the first target positive across eligible proteins",
            "R@K": "Average fraction of target positives in the top K full-universe candidates per eligible protein",
            "macro_term_recall@K": "Average per-term recall over target GO terms with at least one test positive",
            "eligible_protein": "At least one test positive in the corresponding target group",
            "seen_unseen": "Target groups use actual checkpoint training positives only; ranking candidates stay unchanged",
            "unseen_scope": "Zero positive training count does not certify absence as a training negative or temporal novelty",
            "empty_groups": "Counts are zero and metrics are null, never zero performance",
        },
        "groups": groups,
        "provenance": {
            "source_raw_dir": str(artifacts["raw_dir"]),
            "input_sha256": artifacts["input_sha256"],
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "numpy_version": np.__version__,
            "all_consumed_files_hash_verified": True,
            "ids_and_labels_preserved": True,
            "inference_repeated": False, "ground_truth_propagated": False,
        },
    }


def serialize_reports(report):
    ks = report["ks"]
    header = ["group", "n_proteins", "n_positive_pairs", "n_terms_with_positives", "mAP", "MRR"] + [f"R@{k}" for k in ks]
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(header)
    rows = []
    for name, values in report["groups"].items():
        row = [name.upper()] + [values[k] for k in header[1:]]
        writer.writerow(["" if v is None else v for v in row])
        rows.append([name.upper()] + [str(values[k]) for k in header[1:4]] +
                    [_format(values[k]) for k in header[4:]])
    lines = [f"{report['model']} | {report['branch']} | general test | original RAW ranking",
             "| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(["", "R@K is protein-averaged target recall, not hit rate or micro recall.",
                  "SEEN/UNSEEN use full-universe ranks and actual checkpoint training positives.",
                  "NA indicates a target group with no eligible positives."])
    return {
        "metrics_retrieval.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        "retrieval_table.csv": csv_buffer.getvalue(),
        "retrieval_table.txt": "\n".join(lines) + "\n",
    }


def execute(args):
    ks = validate_ks(args.ks)
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else raw_dir / "retrieval_raw"
    require(out_dir != raw_dir, "Output directory must differ from the saved RAW input directory")
    if not args.check_only:
        require(not out_dir.exists() or (out_dir.is_dir() and not any(out_dir.iterdir())),
                f"Output directory must be new or empty: {out_dir}")
    artifacts = load_artifacts(raw_dir)
    metadata = artifacts["metadata"]
    print(f"[INPUT] model={metadata['model']} branch={metadata['branch']} proteins={len(artifacts['protein_ids'])} candidates={len(artifacts['go_ids'])} scores=scores_raw.npy", flush=True)
    if artifacts["seen_mask"] is None:
        print("[GROUPS] ALL only; checkpoint_train_labels.json was not supplied or declared", flush=True)
    if args.check_only:
        print("[CHECK] Input hashes, ordering, dimensions and label membership verified; no outputs written", flush=True)
        return None
    groups = compute_retrieval_metrics(artifacts["scores"], artifacts["labels"], ks=ks, seen_mask=artifacts["seen_mask"])
    assert_unchanged(artifacts)
    report = make_report(artifacts, groups, ks)
    payloads = serialize_reports(report)
    require(not out_dir.exists() or (out_dir.is_dir() and not any(out_dir.iterdir())),
            f"Output directory became nonempty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any existing reports even in a concurrent run.
    created = []
    try:
        for name, text in payloads.items():
            path = out_dir / name
            with path.open("x", encoding="utf-8", newline="") as handle:
                created.append(path)
                handle.write(text)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    for name, values in groups.items():
        parts = [f"n={values['n_proteins']}", f"mAP={_format(values['mAP'])}", f"MRR={_format(values['MRR'])}"]
        parts.extend(f"R@{k}={_format(values[f'R@{k}'])}" for k in ks)
        print(f"[{name.upper()}] " + " | ".join(parts), flush=True)
    print(f"[SAVED] {out_dir / 'metrics_retrieval.json'}", flush=True)
    print(f"[SAVED] {out_dir / 'retrieval_table.csv'}", flush=True)
    print(f"[SAVED] {out_dir / 'retrieval_table.txt'}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Saved general-test RAW evaluation folder")
    parser.add_argument("--out-dir", help="New or empty output folder; default: RAW_DIR/retrieval_raw")
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS, help="Distinct positive retrieval budgets")
    parser.add_argument("--check-only", action="store_true", help="Verify inputs without ranking or writing output")
    args = parser.parse_args(argv)
    try:
        execute(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
