"""Score the fixed B3-FA checkpoint on the existing MZSGO general test cohort.

The model, its source identity, batching, precision and frozen training inputs
come from the checkpoint. This adapter changes only the evaluation cohort and
uses the pre-existing expanded general-test GO bank. It never trains, refits,
re-encodes text, chooses a checkpoint on test, or applies frequency weights to
predictions. True-path propagation is a separate step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .fa_scoring import inspect_checkpoint, prepare_base_resources, score_test
from .evaluate_general_true_path import load_util, format_metrics
from .evaluate_mzsgo_generic import mzsgo_metrics


EXPECTED_DIMS = {"BP": (5712, 20639), "CC": (5764, 2803), "MF": (5584, 7007)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gid(value):
    s = str(value).strip()
    n = int(s.split(":", 1)[1] if s.upper().startswith("GO:") else s)
    require(n > 0, f"Invalid GO identifier: {value}")
    return n


def go_name(value):
    return f"GO:{gid(value):07d}"


def read_annotations(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(obj, dict) and bool(obj), f"Empty or invalid annotations: {path}")
    result = {}
    for protein, values in obj.items():
        require(isinstance(protein, str) and protein and isinstance(values, list),
                f"Expected protein -> GO list in {path}")
        result[protein] = sorted({gid(x) for x in values})
        require(bool(result[protein]), f"Protein without annotations: {protein}")
    return result


def check_bank_tensor(bank, key, shape, binary=False):
    value = bank.get(key)
    require(torch.is_tensor(value) and tuple(value.shape) == tuple(shape),
            f"Expanded bank tensor {key} must have shape {shape}")
    require(bool(torch.isfinite(value).all()), f"Nonfinite bank tensor: {key}")
    if binary:
        require(bool(((value == 0) | (value == 1)).all()), f"Invalid presence mask: {key}")
    else:
        require(value.is_floating_point(), f"Expected floating embeddings: {key}")


def load_general_bank(bundle, resources, path):
    """Establish provenance and identical scoring inputs for original GO terms."""
    path = Path(path)
    require(path.is_file(), f"Expanded general-test GO bank missing: {path}")
    bank = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(bank, dict) and bank.get("format") == "mzsgo_frozen_segment_parent_v1",
            "Unexpected general-test frozen GO bank format")
    metadata = bank.get("metadata", {})
    hashes = resources["input_sha256"]
    for key, original in [("canonical_jsonl_sha256", "canonical_text"),
                          ("candidate_ids_sha256", "candidate_ids"),
                          ("temporal_ids_sha256", "temporal_ids")]:
        require(metadata.get(key) == hashes[original], f"General bank metadata mismatch: {key}")
    require(metadata.get("base_bank_sha256") == sha256(resources["base_bank_path"]),
            "Expanded bank was not derived from this checkpoint's original frozen bank")
    require(metadata.get("model") == bundle["cfg"]["encoder_name"] and metadata.get("normalized") is False,
            "General bank encoder or normalization differs from the training bank")
    require(metadata.get("segment_max_len") == bundle["cfg"]["go_segment_max_len"],
            "General bank GO token length differs from the training bank")
    branch = bundle["identity"]["branch"].upper()
    purpose = str(metadata.get("purpose", "")).lower()
    require("general" in purpose and branch.lower() in purpose,
            f"Expanded bank lacks general-{branch} provenance: {purpose!r}")
    require(list(bank.get("segment_names", [])) == ["name", "definition", "is_a"],
            "Expanded bank segment order differs from the frozen bank contract")
    bank_ids = [gid(x) for x in bank.get("go_ids", [])]
    require(bank_ids and len(set(bank_ids)) == len(bank_ids), "Empty or duplicate general-bank IDs")
    base = resources["base_bank"]
    base_ids = [gid(x) for x in base["go_ids"]]
    require(set(base_ids) < set(bank_ids), "General bank must expand the original candidate universe")
    dim = int(bundle["architecture"]["d_g"])
    n = len(bank_ids)
    parents = bank.get("parent_embs")
    require(torch.is_tensor(parents) and parents.ndim == 3 and
            parents.shape[0] == n and parents.shape[2] == dim,
            "Expanded bank parent_embs must have shape [G,P,D]")
    for key, shape, binary in [("segment_embs", (n, 3, dim), False),
                               ("segment_present", (n, 3), True),
                               ("parent_embs", tuple(parents.shape), False),
                               ("parent_present", tuple(parents.shape[:2]), True)]:
        check_bank_tensor(bank, key, shape, binary)
    index = {g: i for i, g in enumerate(bank_ids)}
    rows = torch.tensor([index[g] for g in base_ids], dtype=torch.long)
    # Parent banks may gain padding slots when the GO universe expands.
    # FA has parent_strength=0 and never passes these tensors to its scorer.
    # Require exactness for all segment scoring inputs, including their masks.
    for key in ("segment_embs", "segment_present"):
        shared = bank[key].index_select(0, rows)
        left = shared.bool() if "present" in key else shared.float()
        right = base[key].bool() if "present" in key else base[key].float()
        require(torch.equal(left, right),
                f"Expanded bank changed original training GO features: {key}")
    temporal = set(resources["temporal_go_ids"])
    require(temporal <= set(bank_ids), "General bank omitted reserved temporal IDs")
    evaluation_ids = sorted(set(bank_ids) - temporal)
    require(len(evaluation_ids) == EXPECTED_DIMS[branch][1],
            f"General {branch} candidate count {len(evaluation_ids)} differs from expected {EXPECTED_DIMS[branch][1]}")
    return bank, evaluation_ids, {
        "path": str(path.resolve()), "sha256": sha256(path),
        "base_bank_sha256": sha256(resources["base_bank_path"]),
        "original_scoring_inputs_exactly_equal_after_training_cast": True,
        "parent_features_used_for_scoring": False,
        "n_original_rows": len(base_ids),
        "n_general_bank_rows": len(bank_ids), "temporal_terms_excluded": len(temporal),
    }


def prepare_cohort(bundle, resources, bank_path, annotation_path):
    branch = bundle["identity"]["branch"].upper()
    labels = read_annotations(annotation_path)
    pids = sorted(labels)
    require(len(pids) == EXPECTED_DIMS[branch][0],
            f"General {branch} protein count {len(pids)} differs from expected {EXPECTED_DIMS[branch][0]}")
    train = set(resources["train_ids"])
    valid = set(resources["val_ids"])
    require(not set(pids) & (train | valid), "General test proteins overlap training or validation proteins")
    bank, ids, audit = load_general_bank(bundle, resources, bank_path)
    index = {g: j for j, g in enumerate(ids)}
    missing = sorted({g for v in labels.values() for g in v} - set(ids))
    require(not missing, f"Test ground truth missing from scored GO universe: {missing[:20]}")
    y = np.zeros((len(pids), len(ids)), dtype=np.uint8)
    for i, p in enumerate(pids):
        y[i, [index[g] for g in labels[p]]] = 1
    require(bool(np.all(y.sum(axis=1) > 0)), "Empty test annotation row")
    full_annotations = read_annotations(resources["input_paths"]["annotations"])
    pool = train | valid
    require(pool <= set(full_annotations), "Training-pool annotation entries missing")
    active = set(resources["active_go_ids"])
    pool_annotations = {p: sorted(set(full_annotations[p]) & active) for p in sorted(pool)}
    require(all(pool_annotations.values()), "Training-pool protein lost every effective positive")
    actual = {p: sorted(map(gid, values)) for p, values in resources["train_annotations"].items()}
    require(set(actual) == train, "Actual training annotations do not match checkpoint training IDs")
    require(all(actual[p] == pool_annotations[p] for p in train), "Actual and pool training labels disagree")
    resources["evaluation_annotations"] = labels
    return pids, ids, y, bank, actual, pool_annotations, audit


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--branch", required=True, choices=sorted(EXPECTED_DIMS))
    p.add_argument("--repo-root", default="/workspace/protein-go-semantic-align")
    p.add_argument("--root", default="/workspace/MZSGO_temporal")
    p.add_argument("--mzsgo-repo", default="/workspace/MZSGO")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["general"], default="general")
    p.add_argument("--score-transform", choices=["linear"], default="linear")
    args = p.parse_args(argv)
    out = Path(args.out_dir).resolve()
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())),
            f"Output directory must be new or empty: {out}")
    bundle = inspect_checkpoint(args.checkpoint, args.repo_root, args.branch)
    resources = prepare_base_resources(bundle, args.root)
    meta = bundle["meta"]
    selection = meta.get("selection", {})
    require(selection.get("best_primary_epoch") == meta.get("epoch"),
            "Use the validation-selected best_primary checkpoint, not a test-chosen epoch")
    metric_name = bundle["cfg"].get("monitor_metric", "protein_wise_map")
    recorded = meta.get("validation_metrics", {}).get(metric_name)
    require(recorded is not None and selection.get("best_primary") is not None and
            abs(float(recorded) - float(selection["best_primary"])) <= 1e-10,
            "Checkpoint does not match its recorded primary validation winner")
    root = Path(args.root).resolve()
    branch = args.branch.lower()
    annotation_path = root / "processed" / branch / f"pid_to_go_general_test_{branch}.json"
    bank_path = root / "go_cache" / branch / "frozen_segment_parent_bank_general_test.pt"
    pids, ids, y, bank, actual_labels, pool_labels, bank_audit = prepare_cohort(
        bundle, resources, bank_path, annotation_path)
    util_path = Path(args.mzsgo_repo).resolve() / "utils" / "util.py"
    require(util_path.is_file(), f"Original MZSGO metric implementation missing: {util_path}")
    util = load_util(args.mzsgo_repo)
    metric_source_hash = sha256(util_path)
    print(f"[CHECKPOINT] {args.checkpoint} | ablation=B3-FA branch={args.branch} epoch={meta['epoch']}", flush=True)
    print(f"[GENERAL TEST] proteins={len(pids)} GO={len(ids)} | validation-selected fixed checkpoint", flush=True)
    print(f"[TRAINING] actual_train={len(actual_labels)} validation={len(resources['val_ids'])}; no train+val refit", flush=True)

    def progress(item):
        if item["batch"] % 10 == 0 or item["batch"] == item["n_batches"]:
            print(f"[SCORE] {item['proteins']}/{item['n_proteins']} proteins, batch {item['batch']}/{item['n_batches']}", flush=True)

    scores = score_test(bundle, resources, pids, bank, ids, args.device, progress=progress)
    require(isinstance(scores, np.ndarray) and scores.shape == y.shape and scores.dtype == np.float32,
            "Scorer returned an unexpected matrix")
    require(bool(np.isfinite(scores).all()), "Nonfinite raw scores")
    scores01 = np.clip((scores + np.float32(1)) / np.float32(2), 0, 1).astype(np.float32)
    original = format_metrics(util.evaluate_annotations(torch.from_numpy(y), torch.from_numpy(scores01), "general"))
    generic = mzsgo_metrics(y, scores01)
    require(generic is not None, "Empty overall metric cohort")
    difference = {key: abs(original[key] - generic[key]) for key in ("Fmax", "AUPR")}
    require(max(difference.values()) <= 1e-5,
            f"Original and generic MZSGO metrics disagree; check installed util semantics: {difference}")
    require(sha256(util_path) == metric_source_hash, "Metric implementation changed during evaluation")
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "Output directory became nonempty while scoring")
    np.save(out / "scores_raw.npy", scores, allow_pickle=False)
    np.save(out / "scores_01.npy", scores01, allow_pickle=False)
    names = [go_name(x) for x in ids]
    write_json(out / "protein_ids.json", pids)
    write_json(out / "go_ids.json", names)
    np.savez_compressed(out / "evaluation_data.npz", labels=y,
                        protein_ids=np.asarray(pids, dtype=str), go_ids=np.asarray(names, dtype=str))
    for filename, mapping in [("checkpoint_train_labels.json", actual_labels),
                              ("benchmark_train_labels.json", pool_labels)]:
        write_json(out / filename, {pid: [go_name(g) for g in values] for pid, values in mapping.items()})
    result = {
        "mode": "general", "branch": args.branch, "model": "B3-FA",
        "n_proteins": len(pids), "n_candidates": len(ids), "score_transform": "linear",
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "selected_validation_epoch": meta["epoch"], "checkpoint_selection_metric": metric_name,
        "annotation_path": str(annotation_path), "annotation_sha256": sha256(annotation_path),
        "bank_sha256": bank_audit["sha256"], "bank_audit": bank_audit,
        "metric_source": str(util_path), "metric_source_sha256": metric_source_hash,
        "general_original_mzsgo": original, "overall_generic_crosscheck": generic,
        "overall_crosscheck_abs_difference": difference,
        "general_positive_pairs": int(y.sum()), "general_unique_positive_terms": int((y.sum(0) > 0).sum()),
        "actual_training_proteins": len(actual_labels), "training_pool_proteins": len(pool_labels),
        "validation_proteins": len(resources["val_ids"]), "full_train_refit": False,
        "train_membership_primary": "actual checkpoint training proteins and effective positives",
        "train_membership_secondary": "original benchmark train+validation pool, partition diagnostic only",
        "source_audit": bundle["source_audit"], "training_input_audit": resources["audit"],
        "scoring_runtime": bundle.get("scoring_runtime", {}),
        "note": "General test only. Fixed linear score mapping, no learned calibration. No inference-time frequency weighting. Fmax is the benchmark threshold-sweep maximum, not a deployed operating threshold. No true-path propagation has been applied to this RAW output.",
    }
    write_json(out / "metrics.json", result)
    manifest = {"format": "mzsgo_fa_general_eval_v1", "branch": args.branch,
                "n_proteins": len(pids), "n_candidates": len(ids),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "files": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out / "evaluation_manifest.json", manifest)
    print(json.dumps({"RAW": original, "saved": str(out), "next": "evaluate_general_true_path, then report_groups"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
