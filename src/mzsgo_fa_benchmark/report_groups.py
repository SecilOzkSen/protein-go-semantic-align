"""Report RAW and is_a-TPR general-test groups without changing their labels.

The primary seen/unseen partition uses the checkpoint's actual training
positives. The original train+validation pool is a separate diagnostic
partition, never evidence that this checkpoint was refit on that pool.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

from . import evaluate_mzsgo_generic as generic


RAW_FILES = (
    "evaluation_data.npz", "checkpoint_train_labels.json", "benchmark_train_labels.json",
    "protein_ids.json", "go_ids.json", "scores_01.npy", "metrics.json",
)
METRIC_KEYS = ("Fmax", "precision", "recall", "AUPR", "threshold")
GROUP_KEYS = ("overall", "seen", "unseen")
MANIFEST_FORMAT = "mzsgo_fa_general_eval_v1"
REPORT_FORMAT = "mzsgo_fa_general_group_report_v1"
REPORT_NAME = "metrics_seen_unseen.json"
PRIMARY_BASIS = "checkpoint_actual_training_positive_annotations"
DIAGNOSTIC_BASIS = "original_train_plus_validation_pool_diagnostic_only"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    _require(all(getattr(before, field) == getattr(after, field) for field in fields),
             f"File changed while being hashed: {path}")
    return digest.hexdigest()


def _digest_string(value, label):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             f"Invalid SHA256 for {label}")
    return value


def _json(path):
    def unique_mapping(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_mapping)


def _go(value):
    _require(not isinstance(value, (bool, np.bool_)), "Boolean is not a GO ID")
    normalized = generic.normalize_go(value)
    _require(re.fullmatch(r"GO:[0-9]{7,}", normalized), f"Invalid GO ID: {value!r}")
    return normalized


def _id_array(values, label, *, go=False):
    array = np.asarray(values)
    _require(array.ndim == 1 and array.dtype.kind in "US", f"{label} must be a one-dimensional string array")
    result = [str(value) for value in array.astype(str)]
    _require(result and all(value and value.strip() == value for value in result), f"Empty or padded {label}")
    if go:
        result = [_go(value) for value in result]
    _require(len(result) == len(set(result)), f"Duplicate {label}")
    return result


def _annotation_map(path):
    raw = _json(path)
    _require(isinstance(raw, dict) and raw, f"Expected nonempty protein-to-GO mapping: {path}")
    result = {}
    for protein, terms in raw.items():
        _require(isinstance(protein, str) and protein and protein.strip() == protein,
                 f"Invalid annotation protein ID: {protein!r}")
        _require(isinstance(terms, list) and terms, f"Expected nonempty GO list for {protein}")
        result[protein] = frozenset(_go(term) for term in terms)
    return result


def _finite_metrics(values, label):
    _require(isinstance(values, dict), f"Missing metric mapping: {label}")
    result = {}
    for key in METRIC_KEYS:
        _require(key in values and not isinstance(values[key], bool), f"Missing numeric {label}.{key}")
        try:
            value = float(values[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {label}.{key}") from exc
        _require(math.isfinite(value), f"Nonfinite {label}.{key}")
        result[key] = value
    return result


def _compare_metrics(actual, expected, tolerance, label, keys=METRIC_KEYS):
    expected_values = _finite_metrics(expected, label)
    differences = {key: abs(float(actual[key]) - expected_values[key]) for key in keys}
    _require(all(value <= tolerance for value in differences.values()),
             f"{label} metric parity failed (tolerance {tolerance}): {differences}")
    return differences


def _load_original_util(path, expected_hash):
    _require(sha256_file(path) == expected_hash, "Original MZSGO metric source SHA256 differs from RAW evaluation")
    spec = importlib.util.spec_from_file_location("mzsgo_fa_group_report_original_util", path)
    _require(spec is not None and spec.loader is not None, f"Cannot import original MZSGO utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(callable(getattr(module, "evaluate_annotations", None)), "MZSGO utility has no evaluate_annotations")
    _require(sha256_file(path) == expected_hash, "MZSGO metric source changed during import")
    return module


def _evaluate_subset(labels, scores, util, tag):
    """Cross-check the copied generic function against the supplied original util."""
    calculated = generic.mzsgo_metrics(labels, scores)
    with redirect_stdout(io.StringIO()):
        values = util.evaluate_annotations(labels, scores, tag)
    _require(isinstance(values, (tuple, list)) and len(values) >= 5,
             "Original evaluate_annotations must return Fmax, precision, recall, AUPR, threshold")
    original = _finite_metrics(dict(zip(METRIC_KEYS, values[:5])), f"original {tag}")
    if calculated is None:
        _require(not np.any(labels), f"Unexpected empty generic group: {tag}")
        _require(all(value == 0. for value in original.values()), f"Original empty-group semantics differ: {tag}")
        return None, {"status": "empty_group_null", "max_absolute_difference": 0.}
    differences = _compare_metrics(calculated, original, 1e-5, f"generic/original {tag}")
    # Report the original implementation's values after parity is established.
    original.update(proteins_with_positive=int(np.count_nonzero(labels.sum(axis=1) > 0)),
                    positive_pairs=int(labels.sum()), n_terms=int(labels.shape[1]))
    return original, {"status": "passed", "absolute_differences": differences,
                      "max_absolute_difference": max(differences.values())}


def _partition(labels, scores, go_ids, annotations, util, tag, overall, overall_audit):
    train_go = set().union(*annotations.values())
    supported = labels.sum(axis=0) > 0
    seen = np.asarray([term in train_go and bool(supported[j]) for j, term in enumerate(go_ids)], dtype=bool)
    unseen = np.asarray([term not in train_go and bool(supported[j]) for j, term in enumerate(go_ids)], dtype=bool)
    _require(np.array_equal(seen | unseen, supported) and not np.any(seen & unseen), "Seen/unseen partition is invalid")
    seen_metrics, seen_audit = _evaluate_subset(labels[:, seen], scores[:, seen], util, f"{tag}_seen")
    unseen_metrics, unseen_audit = _evaluate_subset(labels[:, unseen], scores[:, unseen], util, f"{tag}_unseen")
    harmonic = None
    if seen_metrics is not None and unseen_metrics is not None:
        s, u = seen_metrics["AUPR"], unseen_metrics["AUPR"]
        harmonic = float(2 * s * u / (s + u)) if s + u else 0.
    return {
        "n_proteins": int(labels.shape[0]), "n_candidates": int(labels.shape[1]),
        "n_training_annotation_proteins": len(annotations), "n_train_go": len(train_go),
        "n_seen_test_go": int(seen.sum()), "n_unseen_test_go": int(unseen.sum()),
        "n_train_only_go_not_in_candidates": len(train_go - set(go_ids)),
        "overall": overall, "seen": seen_metrics, "unseen": unseen_metrics, "H": harmonic,
        "unseen_go_ids": [go_ids[j] for j in np.flatnonzero(unseen)],
        "metric_parity": {"overall": overall_audit, "seen": seen_audit, "unseen": unseen_audit},
    }


def _scores(path, shape, label):
    scores = np.load(path, mmap_mode="r", allow_pickle=False)
    _require(isinstance(scores, np.ndarray) and scores.shape == shape and scores.dtype.kind == "f",
             f"{label} scores must be a floating [N,G] matrix matching the NPZ")
    for start in range(0, shape[0], 64):
        block = scores[start:start + 64]
        _require(np.isfinite(block).all() and np.all((block >= 0) & (block <= 1)),
                 f"{label} scores must be finite and within [0,1]")
    return scores


def _write_reports_exclusively(outputs):
    """Publish both completed reports, without replacing any existing file."""
    temporary, created = [], []
    try:
        for destination, payload in outputs:
            fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
            temporary.append(Path(name))
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        for (destination, _), source in zip(outputs, temporary):
            # Link a fully written file atomically; unlike replace(), this cannot
            # overwrite an output appearing after the initial preflight.
            os.link(source, destination)
            created.append(destination)
    except BaseException:
        for destination in reversed(created):
            destination.unlink()
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def _print_table(reports, *, diagnostic=False):
    title = ("ORIGINAL TRAIN+VALIDATION POOL: DIAGNOSTIC PARTITION ONLY, NO REFIT"
             if diagnostic else "CHECKPOINT ACTUAL TRAINING POSITIVES: PRIMARY PARTITION")
    print(f"\n[{title}]")
    print("| Scores | Overall Fmax | Overall AUPR | Seen Fmax | Seen AUPR | Unseen Fmax | Unseen AUPR | H |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, report in reports:
        values = report["benchmark_training_pool_partition"] if diagnostic else report
        cells = []
        for group in GROUP_KEYS:
            for metric in ("Fmax", "AUPR"):
                cells.append("NA" if values[group] is None else f"{values[group][metric]:.8f}")
        cells.append("NA" if values["H"] is None else f"{values['H']:.8f}")
        print("| " + " | ".join([label] + cells) + " |")


def execute(args):
    raw_dir, tpr_dir = Path(args.raw_dir).expanduser().resolve(), Path(args.tpr_dir).expanduser().resolve()
    _require(raw_dir != tpr_dir, "RAW and TPR directories must differ")
    raw_output, tpr_output = raw_dir / REPORT_NAME, tpr_dir / REPORT_NAME
    for output in (raw_output, tpr_output):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Report already exists; no files changed: {output}")
    manifest_path = raw_dir / "evaluation_manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = _json(manifest_path)
    _require(isinstance(manifest, dict) and manifest.get("format") == MANIFEST_FORMAT, "Unsupported RAW evaluation manifest format")
    expected_files = manifest.get("files")
    _require(isinstance(expected_files, dict) and set(RAW_FILES) <= set(expected_files),
             "RAW manifest lacks required input file hashes")
    input_hashes = {}
    for filename, expected_hash in expected_files.items():
        _require(isinstance(filename, str) and filename not in ("", ".", "..", "evaluation_manifest.json")
                 and Path(filename).name == filename and "/" not in filename and "\\" not in filename,
                 f"Manifest file key must be a basename: {filename!r}")
        expected_hash = _digest_string(expected_hash, filename)
        actual_hash = sha256_file(raw_dir / filename)
        _require(actual_hash == expected_hash, f"RAW input SHA256 mismatch: {filename}")
        input_hashes[filename] = actual_hash
    previous = _json(raw_dir / "metrics.json")
    branch = manifest.get("branch")
    _require(branch in ("BP", "MF", "CC") and previous.get("branch") == branch,
             "RAW manifest/metrics branch mismatch")
    _require(previous.get("mode") == "general" and previous.get("score_transform") == "linear",
             "Expected general test with the declared linear cosine score transform")
    _require(previous.get("full_train_refit") is False,
             "This report requires an explicit non-refit checkpoint evaluation")
    checkpoint_hash = _digest_string(previous.get("checkpoint_sha256"), "checkpoint")
    if "checkpoint_sha256" in manifest:
        _require(manifest["checkpoint_sha256"] == checkpoint_hash, "RAW manifest/metrics checkpoint SHA256 mismatch")
    with np.load(raw_dir / "evaluation_data.npz", allow_pickle=False) as saved:
        _require(set(("labels", "go_ids", "protein_ids")) <= set(saved.files), "Evaluation NPZ lacks labels or ID arrays")
        labels = saved["labels"].copy()
        go_ids = _id_array(saved["go_ids"], "NPZ GO IDs", go=True)
        protein_ids = _id_array(saved["protein_ids"], "NPZ protein IDs")
    _require(labels.ndim == 2 and labels.dtype.kind in "biuf" and labels.shape == (len(protein_ids), len(go_ids))
             and np.isin(labels, (0, 1)).all(), "NPZ labels must be binary [N,G] and match IDs")
    _require(np.all(labels.sum(axis=1) > 0), "General test includes a protein without positive annotations")
    labels = labels.astype(np.uint8, copy=False)
    _require(_id_array(_json(raw_dir / "protein_ids.json"), "sidecar protein IDs") == protein_ids,
             "NPZ/sidecar protein ID order mismatch")
    _require(_id_array(_json(raw_dir / "go_ids.json"), "sidecar GO IDs", go=True) == go_ids,
             "NPZ/sidecar GO ID order mismatch")
    for document, label in ((manifest, "manifest"), (previous, "metrics")):
        for key, count in (("n_proteins", len(protein_ids)), ("n_candidates", len(go_ids))):
            _require(type(document.get(key)) is int and document[key] == count, f"RAW {label} {key} mismatch")
    actual_train = _annotation_map(raw_dir / "checkpoint_train_labels.json")
    benchmark_train = _annotation_map(raw_dir / "benchmark_train_labels.json")
    test_proteins = set(protein_ids)
    _require(not test_proteins.intersection(actual_train) and not test_proteins.intersection(benchmark_train),
             "General-test proteins overlap training annotations")
    _require(set(actual_train) <= set(benchmark_train), "Checkpoint training proteins are not a subset of the benchmark training pool")
    _require(all(terms == benchmark_train[protein] for protein, terms in actual_train.items()),
             "Checkpoint and benchmark training annotations disagree for a shared protein")
    for key, count in (("actual_training_proteins", len(actual_train)),
                       ("training_pool_proteins", len(benchmark_train)),
                       ("validation_proteins", len(benchmark_train) - len(actual_train))):
        if key in previous:
            _require(type(previous[key]) is int and previous[key] == count,
                     f"RAW metadata {key} disagrees with the annotation files")
    raw = _scores(raw_dir / "scores_01.npy", labels.shape, "RAW")
    tpr = _scores(tpr_dir / "scores_01_tpr.npy", labels.shape, "TPR")
    tpr_path = tpr_dir / "metrics_tpr.json"
    tpr_hash = sha256_file(tpr_dir / "scores_01_tpr.npy")
    tpr_report_hash = sha256_file(tpr_path)
    tpr_metadata = _json(tpr_path)
    _require(tpr_metadata.get("branch") == branch and tpr_metadata.get("n_proteins") == len(protein_ids)
             and tpr_metadata.get("n_candidates") == len(go_ids), "TPR metadata branch/dimensions mismatch")
    _require(tpr_metadata.get("ground_truth_propagated") is False, "TPR report must preserve original gold labels")
    _require(tpr_metadata.get("source_scores_sha256") == input_hashes["scores_01.npy"], "TPR source RAW score SHA256 mismatch")
    source_raw = tpr_metadata.get("source_raw_dir")
    _require(isinstance(source_raw, str) and source_raw and Path(source_raw).expanduser().resolve() == raw_dir,
             "TPR source RAW directory mismatch")
    changed = 0
    for start in range(0, len(protein_ids), 64):
        old, new = raw[start:start + 64], tpr[start:start + 64]
        _require(not np.any(new + 1e-7 < old), "TPR lowered a RAW score")
        changed += int(np.count_nonzero(new > old + 1e-7))
    _require(type(tpr_metadata.get("n_changed_scores")) is int and changed == tpr_metadata["n_changed_scores"],
             "TPR changed-score count mismatch")
    metric_source_hash = _digest_string(previous.get("metric_source_sha256"), "original MZSGO metric source")
    metric_source_path = Path(args.mzsgo_repo).expanduser().resolve() / "utils" / "util.py"
    util = _load_original_util(metric_source_path, metric_source_hash)
    generic_path = Path(generic.__file__).resolve()
    generic_hash = sha256_file(generic_path)

    raw_overall, raw_parity = _evaluate_subset(labels, raw, util, "general_raw_recheck")
    tpr_overall, tpr_parity = _evaluate_subset(labels, tpr, util, "general_tpr_recheck")
    checks = {
        "raw_vs_original_general": _compare_metrics(raw_overall, previous.get("general_original_mzsgo"), 1e-6, "RAW/original general", keys=("Fmax", "AUPR")),
        "raw_vs_tpr_raw_recheck": _compare_metrics(raw_overall, tpr_metadata.get("raw"), 1e-6, "RAW/TPR raw recheck", keys=("Fmax", "AUPR")),
        "tpr_vs_recorded_propagated": _compare_metrics(tpr_overall, tpr_metadata.get("propagated"), 1e-6, "TPR/recorded propagated", keys=("Fmax", "AUPR")),
    }
    reports = []
    for variant, scores, overall, audit in (("RAW", raw, raw_overall, raw_parity), ("TPR", tpr, tpr_overall, tpr_parity)):
        primary = _partition(labels, scores, go_ids, actual_train, util, f"{variant}_checkpoint_training", overall, audit)
        diagnostic = _partition(labels, scores, go_ids, benchmark_train, util, f"{variant}_benchmark_pool", overall, audit)
        diagnostic.update(training_basis=DIAGNOSTIC_BASIS, diagnostic_only=True, checkpoint_refit_on_this_pool=False)
        report = {
            "format": REPORT_FORMAT, "mode": "general", "branch": branch, "scores": variant,
            "checkpoint_sha256": checkpoint_hash, **primary,
            "training_basis": PRIMARY_BASIS, "checkpoint_refit_on_train_plus_validation": False,
            "benchmark_training_pool_partition": diagnostic,
            "basis_note": "Primary seen/unseen groups use only GO positives from proteins actually used to train this checkpoint. The original train+validation pool is a separate diagnostic partition. This checkpoint was not refit on that pool. Gold labels and protein/GO order are identical for RAW and TPR.",
            "metric_note": "Original MZSGO protein-centric 101-threshold AUPR/Fmax; group columns retain test-positive terms only, then proteins without group positives are excluded by the metric. Fmax is a threshold-swept benchmark statistic, not a selected deployment threshold.",
            "ground_truth_propagated": False,
            "score_transform": "linear; use saved [0,1] scores without refitting or another transform",
            "propagation_note": "TPR uses the supplied full is_a ancestor propagation; equivalence to MZSGO's separate subset-adjacency propagated metric is not assumed.",
            "provenance": {
                "raw_dir": str(raw_dir), "tpr_dir": str(tpr_dir),
                "evaluation_manifest_sha256": manifest_hash, "raw_files_sha256": input_hashes,
                "tpr_scores_sha256": tpr_hash, "tpr_report_sha256": tpr_report_hash,
                "original_metric_source": str(metric_source_path), "original_metric_source_sha256": metric_source_hash,
                "generic_metric_source": str(generic_path), "generic_metric_source_sha256": generic_hash,
                "n_changed_scores": changed, "metric_rechecks": checks,
            },
        }
        reports.append((variant, report))
    # No output is published if inputs changed while metrics were being computed.
    for filename, digest in input_hashes.items():
        _require(sha256_file(raw_dir / filename) == digest, f"RAW input changed during reporting: {filename}")
    for path, digest in ((manifest_path, manifest_hash), (tpr_dir / "scores_01_tpr.npy", tpr_hash),
                         (tpr_path, tpr_report_hash), (metric_source_path, metric_source_hash),
                         (generic_path, generic_hash)):
        _require(sha256_file(path) == digest, f"Input/source changed during reporting: {path}")
    _write_reports_exclusively([(raw_output, reports[0][1]), (tpr_output, reports[1][1])])
    _print_table(reports)
    _print_table(reports, diagnostic=True)
    print(f"\n[SAVED] {raw_output}\n[SAVED] {tpr_output}")
    return {"raw": reports[0][1], "tpr": reports[1][1]}


def parser():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--raw-dir", type=Path, required=True)
    arguments.add_argument("--tpr-dir", type=Path, required=True)
    arguments.add_argument("--mzsgo-repo", type=Path, default=Path("/workspace/MZSGO"))
    return arguments


def main(argv=None):
    try:
        execute(parser().parse_args(argv))
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
