"""Evaluate the fixed B3-FA checkpoint on the supplied temporal GO cohort.

This preserves the two scopes in the supplied MZSGO temporal evaluator:
protein-centric Fmax/AUPR on all reserved temporal GO columns, and retrieval
of temporal positives against the complete original candidate universe.
No true-path propagation, new model fitting, or test checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .fa_scoring import inspect_checkpoint, prepare_base_resources, score_test
from .evaluate_general_true_path import load_util, format_metrics
from .evaluate_mzsgo_generic import mzsgo_metrics
from .evaluate_test import require, sha256, gid, go_name, read_annotations, write_json


def prepare_temporal_cohort(bundle, resources, annotation_path):
    """Retain every annotated temporal protein and every reserved GO column."""
    labels = read_annotations(annotation_path)
    pids = sorted(labels)
    candidates = sorted(map(gid, resources["candidate_go_ids"]))
    temporal = sorted(map(gid, resources["temporal_go_ids"]))
    active = set(map(gid, resources["active_go_ids"]))
    require(candidates and len(candidates) == len(set(candidates)), "Invalid temporal candidate universe")
    require(temporal and len(temporal) == len(set(temporal)), "Empty or duplicate reserved temporal GO IDs")
    require(not active & set(temporal) and set(candidates) == active | set(temporal),
            "Reserved temporal GO IDs must be excluded from the recorded training loss candidate universe")
    train, valid = set(resources["train_ids"]), set(resources["val_ids"])
    require(not set(pids) & (train | valid), "Temporal test proteins overlap training or validation")
    recorded_test = set(resources["temporal_test_ids"])
    require(recorded_test and set(pids) <= recorded_test,
            "Temporal annotation protein IDs are outside the checkpoint-audited test ID file")
    missing = sorted({g for values in labels.values() for g in values} - set(temporal))
    require(not missing, f"Temporal ground truth contains non-temporal GO labels: {missing[:20]}")
    trained_positives = {gid(g) for values in resources["train_annotations"].values() for g in values}
    require(not trained_positives & set(temporal), "Reserved temporal GO term was a training positive")
    bank = resources["base_bank"]
    require(set(map(gid, bank["go_ids"])) == set(candidates),
            "Temporal evaluation must use the original, unexpanded candidate bank")
    metadata = bank.get("metadata", {})
    for bank_key, input_key in (("canonical_jsonl_sha256", "canonical_text"),
                                ("candidate_ids_sha256", "candidate_ids"),
                                ("temporal_ids_sha256", "temporal_ids")):
        require(metadata.get(bank_key) == resources["input_sha256"][input_key],
                f"Temporal bank metadata mismatch: {bank_key}")
    temporal_index = {g: j for j, g in enumerate(temporal)}
    full_index = {g: j for j, g in enumerate(candidates)}
    columns = np.asarray([full_index[g] for g in temporal], dtype=np.int64)
    y = np.zeros((len(pids), len(temporal)), dtype=np.uint8)
    for row, pid in enumerate(pids):
        y[row, [temporal_index[g] for g in labels[pid]]] = 1
    require(bool(np.all(y.sum(axis=1) > 0)), "Temporal protein has no positive label")
    resources["evaluation_annotations"] = labels
    audit = {
        "annotation_path": str(Path(annotation_path).resolve()),
        "annotation_sha256": sha256(annotation_path),
        "n_annotation_proteins": len(pids),
        "n_recorded_test_ids": len(recorded_test),
        "annotation_proteins_subset_of_recorded_test_ids": True,
        "n_recorded_ids_without_temporal_annotation": len(recorded_test - set(pids)),
        "train_validation_protein_overlap": 0,
        "temporal_training_positive_overlap": 0,
        "temporal_active_loss_candidate_overlap": 0,
        "n_candidates": len(candidates), "n_temporal_go": len(temporal),
        "n_temporal_go_with_test_positives": int(np.count_nonzero(y.sum(axis=0))),
        "n_temporal_positive_pairs": int(y.sum()),
        "all_reserved_temporal_columns_retained": True,
        "expanded_general_test_bank_used": False,
    }
    return pids, candidates, temporal, y, columns, audit


def retrieval_metrics(scores, temporal_labels, temporal_columns, ks):
    """Same stable full-universe ranks as the supplied temporal evaluator.

    Process one protein at a time to avoid two full N-by-G integer matrices.
    Only temporal annotations are counted as positives in these diagnostics.
    """
    n, g = scores.shape
    require(temporal_labels.shape[0] == n and temporal_labels.shape[1] == len(temporal_columns),
            "Temporal retrieval label/column shape mismatch")
    require(ks and all(type(k) is int and k > 0 for k in ks), "Retrieval K values must be positive integers")
    ranks_all, aps, reciprocal_ranks = [], [], []
    recalls = {k: [] for k in ks}
    term_hits = {k: np.zeros(temporal_labels.shape[1], dtype=np.int64) for k in ks}
    term_count = temporal_labels.sum(axis=0)
    for row in range(n):
        positives = np.flatnonzero(temporal_labels[row])
        require(len(positives) > 0, "Temporal retrieval protein has no positives")
        order = np.argsort(-scores[row], kind="stable")
        inverse = np.empty(g, dtype=np.int64)
        inverse[order] = np.arange(g)
        ranks = np.sort(inverse[temporal_columns[positives]] + 1)
        ranks_all.extend(ranks.tolist())
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
        reciprocal_ranks.append(1. / float(ranks[0]))
        for k in ks:
            hits = inverse[temporal_columns[positives]] < min(k, g)
            recalls[k].append(float(hits.mean()))
            term_hits[k][positives] += hits.astype(np.int64)
    result = {"temporal_positive_mAP": float(np.mean(aps)),
              "temporal_positive_MRR": float(np.mean(reciprocal_ranks)),
              "temporal_positive_rank_median": float(np.median(ranks_all)),
              "temporal_positive_rank_p90": float(np.quantile(ranks_all, .9))}
    supported = term_count > 0
    for k in ks:
        result[f"temporal_recall@{k}"] = float(np.mean(recalls[k]))
        result[f"macro_temporal_term_recall@{k}"] = float(np.mean(term_hits[k][supported] / term_count[supported]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--branch", choices=["BP", "CC", "MF"], required=True)
    parser.add_argument("--repo-root", default="/workspace/protein-go-semantic-align")
    parser.add_argument("--root", default="/workspace/MZSGO_temporal")
    parser.add_argument("--mzsgo-repo", default="/workspace/MZSGO")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    args = parser.parse_args(argv)
    require(all(k > 0 for k in args.ks) and len(args.ks) == len(set(args.ks)),
            "Retrieval K values must be positive and unique")
    out = Path(args.out_dir).resolve()
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())),
            f"Output directory must be new or empty: {out}")
    bundle = inspect_checkpoint(args.checkpoint, args.repo_root, args.branch)
    resources = prepare_base_resources(bundle, args.root)
    meta, cfg = bundle["meta"], bundle["cfg"]
    selection = meta.get("selection", {})
    metric_name = cfg.get("monitor_metric", "protein_wise_map")
    recorded = meta.get("validation_metrics", {}).get(metric_name)
    require(selection.get("best_primary_epoch") == meta.get("epoch") and recorded is not None
            and selection.get("best_primary") is not None
            and abs(float(recorded) - float(selection["best_primary"])) <= 1e-10,
            "Use the fixed validation-selected best_primary checkpoint")
    branch = args.branch.lower()
    annotation_path = Path(args.root).resolve() / "processed" / branch / f"pid_to_go_temporal_{branch}.json"
    pids, candidates, temporal, y, columns, audit = prepare_temporal_cohort(bundle, resources, annotation_path)
    util_path = Path(args.mzsgo_repo).resolve() / "utils" / "util.py"
    require(util_path.is_file(), f"Original MZSGO metric implementation missing: {util_path}")
    util_hash = sha256(util_path)
    util = load_util(args.mzsgo_repo)
    print(f"[CHECKPOINT] {args.checkpoint} | ablation=B3-FA branch={args.branch} epoch={meta['epoch']}", flush=True)
    print(f"[TEMPORAL] proteins={len(pids)} scored_candidates={len(candidates)} temporal_metric_columns={len(temporal)}", flush=True)
    print("[PROTOCOL] Fmax/AUPR: reserved temporal columns only; retrieval: complete original candidate universe; RAW scores, no true path", flush=True)

    def progress(item):
        if item["batch"] % 10 == 0 or item["batch"] == item["n_batches"]:
            print(f"[SCORE] {item['proteins']}/{item['n_proteins']} proteins, batch {item['batch']}/{item['n_batches']}", flush=True)

    raw = score_test(bundle, resources, pids, resources["base_bank"], candidates, args.device, progress=progress)
    require(isinstance(raw, np.ndarray) and raw.shape == (len(pids), len(candidates)) and raw.dtype == np.float32,
            "Unexpected temporal scorer output")
    require(bool(np.isfinite(raw).all()), "Nonfinite temporal raw scores")
    score01 = np.clip((raw + np.float32(1)) / np.float32(2), 0, 1).astype(np.float32)
    original = format_metrics(util.evaluate_annotations(torch.from_numpy(y),
                                                        torch.from_numpy(score01[:, columns]), "temporal_unseen"))
    generic = mzsgo_metrics(y, score01[:, columns])
    require(generic is not None, "Empty temporal benchmark metric cohort")
    differences = {key: abs(original[key] - generic[key]) for key in ("Fmax", "precision", "recall", "AUPR", "threshold")}
    require(all(np.isfinite(value) and value <= 1e-5 for value in differences.values()),
            f"Original/generic temporal metric mismatch: {differences}")
    retrieval = retrieval_metrics(raw, y, columns, args.ks)
    require(sha256(util_path) == util_hash, "Metric implementation changed during temporal evaluation")
    require(sha256(annotation_path) == audit["annotation_sha256"], "Temporal annotation file changed during evaluation")
    require(sha256(args.checkpoint) == bundle["checkpoint_sha256"], "Checkpoint changed during temporal evaluation")
    result = {
        "mode": "temporal", "branch": args.branch, "model": "B3-FA",
        "n_proteins": len(pids), "n_candidates": len(candidates), "n_temporal_go": len(temporal),
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": bundle["checkpoint_sha256"],
        "selected_validation_epoch": meta["epoch"], "checkpoint_selection_metric": metric_name,
        "bank_path": str(Path(resources["base_bank_path"]).resolve()),
        "bank_sha256": resources["input_sha256"]["frozen_bank"],
        "annotation_sha256": audit["annotation_sha256"],
        "score_transform": "linear", "true_path_applied": False,
        "metric_source": str(util_path), "metric_source_sha256": util_hash,
        "metric_scope": "all reserved temporal GO columns only, including columns with zero test positives",
        "retrieval_scope": "temporal positive labels ranked against the complete original candidate universe",
        "unseen_original_mzsgo": original, "temporal_generic_crosscheck": generic,
        "temporal_crosscheck_abs_difference": differences, "full_universe_retrieval": retrieval,
        "temporal_cohort_audit": audit, "source_audit": bundle["source_audit"],
        "training_input_audit": resources["audit"], "scoring_runtime": bundle.get("scoring_runtime", {}),
        "full_train_refit": False, "training_proteins": len(resources["train_ids"]),
        "validation_proteins": len(resources["val_ids"]),
        "note": "This is the supplied temporal cohort, not general-test unseen or unseen validation. No seen/unseen H is defined for the temporal-only target set. Fmax uses the benchmark test threshold sweep; its threshold is not a validation-selected deployment threshold. No frequency reweighting or true-path propagation is applied to prediction scores.",
    }
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "Output directory became nonempty while scoring")
    np.save(out / "scores_raw.npy", raw, allow_pickle=False)
    np.save(out / "scores_01.npy", score01, allow_pickle=False)
    write_json(out / "protein_ids.json", pids)
    write_json(out / "go_ids.json", [go_name(g) for g in candidates])
    write_json(out / "temporal_go_ids.json", [go_name(g) for g in temporal])
    np.savez_compressed(out / "evaluation_data.npz", labels_temporal=y,
                        protein_ids=np.asarray(pids, dtype=str),
                        full_go_ids=np.asarray([go_name(g) for g in candidates], dtype=str),
                        temporal_go_ids=np.asarray([go_name(g) for g in temporal], dtype=str),
                        temporal_columns=columns)
    write_json(out / "metrics.json", result)
    write_json(out / "evaluation_manifest.json", {
        "format": "mzsgo_fa_temporal_eval_v1", "branch": args.branch,
        "n_proteins": len(pids), "n_candidates": len(candidates), "n_temporal_go": len(temporal),
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "files": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    })
    print(json.dumps({"TEMPORAL_UNSEEN": original, "FULL_UNIVERSE_RETRIEVAL": retrieval,
                      "saved": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
