r"""Evaluate all retained checkpoints from the seven MF ablations.

Use from the existing protein-go-semantic-align repository:
    python -m src.script.mzsgo.evaluate_zero_shot_checkpoints \
        --root /workspace/MZSGO_temporal --branch MF --device cuda:0

Default selection is ALL retained epoch files plus best/last/final aliases,
deduplicated by file content within each run. Deleted training epochs cannot be
recovered. Pass --checkpoint-selection best_primary for one seen-mAP-selected
checkpoint from EACH of the seven ablations, not only B2-2.

Every run uses its own hash-verified inference implementation. There is no
generic SmoothMIL fallback for B3-1. All validation proteins and the full active
GO candidate universe are scored before metric groups are applied. Original
evaluation batch sizes and CUDA bfloat16 policy are preserved.

Each invocation recomputes all requested checkpoints, including prior completed
results. This avoids silently reusing metrics after embedding files or shared
repository code change. A failure is recorded and remaining checkpoints are
attempted. The final exit status is nonzero if any requested item failed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import gc
import importlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from .zero_shot_eval_io import (load_cohort, validate_checkpoint_meta,
                                discover_checkpoints, sha256_file, canonical_hash)
from .zero_shot_eval_metrics import evaluate_rankings
from .zero_shot_eval_model import resolve_loader, read_checkpoint_meta, score_checkpoint


VERSION = "seven_ablation_natural_zero_positive_evaluation_v1"
DEFAULT_RUNS = {
    "B0": "b0_mean_mean_infonce_parent_off",
    "B1": "b1_global_segments_fixed_infonce_parent_off",
    "B1-1": "b1_1_global_segments_learned_infonce_parent_off",
    "B2": "b2_local_smoothmil_fixed_infonce_parent_off",
    "B3-1": "b3_1_local_mean_learned_infonce_parent_off",
    "B2-2": "b2_2_local_smoothmil_single_text_infonce_parent_off",
    "B3": "b3_local_smoothmil_learned_infonce_parent_off",
}
EXPECTED_ARCH = {
    "B0": ("global_mean", "single_text", "fixed"),
    "B1": ("global_mean", "segments", "fixed"),
    "B1-1": ("global_mean", "segments", "learned"),
    "B2": ("local", "segments", "fixed"),
    "B3-1": ("local", "segments", "learned"),
    "B2-2": ("local", "single_text", "fixed"),
    "B3": ("local", "segments", "learned"),
}
ENGINE_FILES = ("evaluate_zero_shot_checkpoints.py", "zero_shot_eval_io.py",
                "zero_shot_eval_model.py", "zero_shot_eval_metrics.py")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_tsv(path, rows, first_columns=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(first_columns)
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: "" if v is None else v for k, v in row.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_assignments(values, description):
    mapping = {}
    for value in values:
        label, separator, target = value.partition("=")
        if not separator or label not in DEFAULT_RUNS or not target:
            raise ValueError(f"{description} expects ABLATION=VALUE, e.g. B0=/path/to/run")
        if label in mapping:
            raise ValueError(f"Duplicate {description} for {label}")
        mapping[label] = target
    return mapping


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/workspace/MZSGO_temporal"))
    p.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Existing code repository containing src/")
    p.add_argument("--branch", type=str.upper, choices=("MF", "BP", "CC"), default="MF")
    p.add_argument("--cohort-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--ablations", nargs="+", choices=list(DEFAULT_RUNS), default=list(DEFAULT_RUNS))
    p.add_argument("--run-dir", action="append", default=[], metavar="ABLATION=PATH")
    p.add_argument("--model-module", action="append", default=[], metavar="ABLATION=MODULE",
                   help="Optional explicit inference module; its source hashes must still match")
    p.add_argument("--checkpoint-selection", choices=("all", "best_primary"), default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ks", nargs="+", type=int, default=[50, 100, 200, 500, 1000])
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10, help="Scoring progress cadence in batches")
    p.add_argument("--save-scores", action="store_true", help="Also save the full [all proteins, all active GO] score matrix")
    p.add_argument("--dry-run", action="store_true", help="Verify cohort and discover runs/loaders/checkpoint files without GPU scoring")
    return p


def validate_ablation(meta, ablation):
    architecture = meta["architecture"]
    config = meta["config"]
    observed = tuple(architecture.get(k) for k in ("protein_mode", "go_representation", "segment_aggregation"))
    if observed != EXPECTED_ARCH[ablation]:
        raise ValueError(f"{ablation} architecture differs: expected {EXPECTED_ARCH[ablation]}, found {observed}")
    if ablation == "B3-1" and architecture.get("window_reduction") != "masked_mean":
        raise ValueError("B3-1 requires its recorded masked_mean window reducer")
    if ablation != "B3-1" and architecture.get("window_reduction") == "masked_mean":
        raise ValueError(f"Unexpected mean-window reducer in {ablation}")
    if float(architecture.get("parent_strength", -1)) != 0 or float(config.get("pbr_lambda", -1)) != 0:
        raise ValueError("The requested seven-run ablation comparison requires parent adjustment and PBR OFF")
    if bool(config.get("frequency_aware_infonce", False)):
        raise ValueError("B2-2-FA is not one of the seven requested completed ablations")


def build_labels(cohort):
    index = {g: j for j, g in enumerate(cohort["go_ids"])}
    truth = np.zeros((len(cohort["protein_ids"]), len(index)), dtype=bool)
    for i, pid in enumerate(cohort["protein_ids"]):
        for g in cohort["labels_full"][pid]:
            truth[i, index[g]] = True
    return truth


def epoch_id(meta):
    value = meta.get("epoch")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or not meta.get("epoch_complete"):
        raise ValueError("Expected a completed checkpoint with a nonnegative zero-based epoch")
    return value


def flatten_result(result, job, entry, result_directory):
    epoch = result["checkpoint"]["epoch_zero_based"]
    row = {
        "ablation": job["ablation"], "epoch": epoch, "completed_epochs": epoch + 1,
        "checkpoint": str(entry["path"]), "checkpoint_sha256": entry["sha256"],
        "aliases": ",".join(entry["aliases"]),
        "is_best_primary": bool(entry["is_best_primary"]),
        "is_best_macro_recall": bool(entry["is_best_macro"]),
        "is_last": bool(entry["is_last"]), "is_final": bool(entry["is_final"]),
        "result_directory": str(result_directory),
        "seen_map_recorded": result["seen_validation_parity"]["recorded_protein_wise_map"],
        "seen_map_delta": result["seen_validation_parity"]["difference"],
        "seen_map_parity_status": result["seen_validation_parity"]["status"],
    }
    for group in ("unseen", "seen", "all"):
        for name, value in result["metrics"][group].items():
            row[f"{group}/{name}"] = value
    return row


def write_combined(output, rows, failures, inventory, cohort, args, finished=False):
    ordering = {name: i for i, name in enumerate(DEFAULT_RUNS)}
    rows = sorted(rows, key=lambda row: (ordering[row["ablation"]], row["epoch"], row["checkpoint_sha256"]))
    first = ("ablation", "epoch", "completed_epochs", "is_best_primary", "is_best_macro_recall",
             "unseen/protein_wise_map", "unseen/macro_term_ap", "unseen/protein_recall@500",
             "unseen/micro_recall@500", "unseen/macro_term_recall@500", "seen/protein_wise_map")
    atomic_tsv(output / "all_checkpoints.tsv", rows, first)
    atomic_tsv(output / "best_primary_checkpoints.tsv", [r for r in rows if r["is_best_primary"]], first)
    atomic_json(output / "failures.json", failures)
    atomic_json(output / "evaluation_summary.json", {
        "format": VERSION, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "requested_ablations": args.ablations,
        "checkpoint_selection": args.checkpoint_selection,
        "completed_unique_checkpoints": len(rows), "failed_items": len(failures),
        "expected_unique_checkpoints": len(inventory),
        "status": (("completed_with_errors" if failures else "completed") if finished else "in_progress"),
        "cohort_signature": cohort["signature"], "cohort_counts": cohort["summary"]["counts"],
        "go_candidate_count": len(cohort["go_ids"]), "protein_count": len(cohort["protein_ids"]),
        "unseen_go_count": len(cohort["unseen_go_ids"]),
        "available_checkpoint_files_only": True,
        "checkpoint_inventory": inventory,
        "selection_note": "No architecture or checkpoint winner is selected automatically. best_primary tags refer to original seen-validation mAP selection.",
        "interpretation": "Natural train-zero-positive validation. This does not certify that unseen terms were excluded from the training denominator; final temporal GO terms are excluded here.",
        "metric_note": "macro_term_ap is non-interpolated term AP over ALL validation proteins, not trapezoidal AUPR. Rankings use ALL active candidates.",
    })


def execute(args):
    if len(args.ablations) != len(set(args.ablations)):
        raise ValueError("Do not repeat an ablation label")
    if not args.ks or any(k < 1 for k in args.ks) or args.torch_threads < 1 or args.log_every < 1:
        raise ValueError("K values, thread count and logging cadence must be positive")
    args.ks = sorted(set(args.ks))
    args.root = args.root.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    if not (args.repo_root / "src").is_dir():
        raise ValueError("--repo-root must contain the existing project's src directory")
    # Make an explicitly supplied repository importable without changing files.
    if str(args.repo_root) not in sys.path:
        sys.path.insert(0, str(args.repo_root))
        importlib.invalidate_caches()
    cohort_path = (args.cohort_dir or args.root / "analysis" / f"zero_shot_validation_{args.branch.lower()}").expanduser().resolve()
    output = (args.output_dir or args.root / "analysis" / f"zero_shot_checkpoint_evaluation_{args.branch.lower()}").expanduser().resolve()
    cohort = load_cohort(cohort_path)
    if cohort["summary"]["branch"] != args.branch:
        raise ValueError("Cohort branch differs from --branch")
    if output == cohort_path or output in cohort_path.parents or cohort_path in output.parents:
        raise ValueError("Evaluation output must be separate from the prepared cohort directory")
    overrides = parse_assignments(args.run_dir, "--run-dir")
    modules = parse_assignments(args.model_module, "--model-module")
    if (set(overrides) | set(modules)) - set(args.ablations):
        raise ValueError("Overrides must refer to a requested --ablations entry")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only cpu and cuda are supported")
    if not args.dry_run and device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Use the training environment or --device cpu explicitly.")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".evaluation.lock").open("a+b")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process is using this output directory; use separate --output-dir paths for parallel subsets") from exc
        return execute_locked(args, cohort, output, overrides, modules, device)
    finally:
        lock.close()


def execute_locked(args, cohort, output, overrides, modules, device):
    jobs, inventory, failures, rows = [], [], [], []
    engine = {name: sha256_file(Path(__file__).with_name(name)) for name in ENGINE_FILES}
    engine_contract = {
        "version": VERSION, "cohort_signature": cohort["signature"], "engine_sha256": engine,
        "ks": args.ks, "device": str(device), "torch": str(torch.__version__),
        "numpy": str(np.__version__), "save_scores": bool(args.save_scores),
    }
    existing_contract = output / "evaluation_contract.json"
    if existing_contract.exists():
        old = json.loads(existing_contract.read_text(encoding="utf-8"))
        if old != engine_contract:
            raise ValueError("Output contains results from a different cohort/code/metric/device contract. Choose a new --output-dir.")
    elif any(p.name != ".evaluation.lock" for p in output.iterdir()):
        raise ValueError("Output directory contains unrelated files without an evaluation contract")
    else:
        atomic_json(existing_contract, engine_contract)
    if not args.dry_run:
        write_combined(output, rows, failures, inventory, cohort, args)
    print(f"[COHORT] proteins={len(cohort['protein_ids'])} active_go={len(cohort['go_ids'])} unseen_go={len(cohort['unseen_go_ids'])}", flush=True)
    print(f"[SCOPE] ablations={','.join(args.ablations)} checkpoint_selection={args.checkpoint_selection}", flush=True)
    for ablation in args.ablations:
        directory = Path(overrides.get(ablation, args.root / "outputs" / args.branch.lower() / DEFAULT_RUNS[ablation])).expanduser().resolve()
        try:
            if not directory.is_dir():
                raise ValueError(f"Run directory does not exist: {directory}")
            manifest_path = directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            loader, loader_audit = resolve_loader(manifest, args.repo_root, modules.get(ablation))
            entries = discover_checkpoints(directory, args.checkpoint_selection)
            job = {"ablation": ablation, "directory": directory, "loader": loader,
                   "loader_audit": loader_audit, "manifest": manifest,
                   "manifest_sha256": sha256_file(manifest_path), "entries": entries}
            jobs.append(job)
            for entry in entries:
                inventory.append({"ablation": ablation, "path": str(entry["path"]),
                                  "sha256": entry["sha256"], "aliases": entry["aliases"], "loader": loader})
            print(f"[DISCOVER] {ablation}: unique_checkpoints={len(entries)} loader={loader}", flush=True)
        except Exception as exc:
            failures.append({"ablation": ablation, "stage": "discovery", "error": str(exc)})
            print(f"[ERROR] {ablation} discovery: {exc}", flush=True)
    atomic_json(output / "checkpoint_inventory.json", inventory)
    if args.dry_run:
        atomic_json(output / "dry_run.json", {"inventory": inventory, "failures": failures,
            "checkpoint_tensors_loaded": False, "per_checkpoint_identity_verified": False})
        print(f"[DRY RUN] unique_checkpoints={len(inventory)} failures={len(failures)}", flush=True)
        return 1 if failures else 0
    truth = build_labels(cohort)
    for job in jobs:
        for entry in job["entries"]:
            ablation = job["ablation"]
            directory = output / ablation / entry["sha256"][:16]
            result_path = directory / "result.json"
            contract = {**engine_contract, "checkpoint_sha256": entry["sha256"],
                        "run_manifest_sha256": job["manifest_sha256"], "loader_audit": job["loader_audit"]}
            try:
                began = time.perf_counter()
                print(f"[EVAL] {ablation} {entry['path'].name} aliases={','.join(entry['aliases'])}", flush=True)
                meta = read_checkpoint_meta(job["loader"], entry["path"])
                audit = validate_checkpoint_meta(meta, cohort, expected_ablation=ablation)
                validate_ablation(meta, ablation)
                epoch = epoch_id(meta)
                if meta["architecture"] != job["manifest"].get("architecture"):
                    raise ValueError("Checkpoint architecture differs from this run's manifest")
                scores, loaded_meta, runtime = score_checkpoint(
                    entry["path"], job["loader"], cohort, device, args.repo_root,
                    expected_meta=meta, log_every=args.log_every,
                )
                if loaded_meta != meta:
                    raise ValueError("Checkpoint metadata changed between inspection and scoring")
                metrics, term_rows, protein_rows = evaluate_rankings(
                    scores, truth, cohort["go_ids"], cohort["seen_go_ids"],
                    cohort["unseen_go_ids"], cohort["protein_ids"], ks=args.ks,
                )
                recorded = meta.get("validation_metrics", {}).get("protein_wise_map")
                actual = metrics["seen"]["protein_wise_map"]
                delta = float(actual - recorded) if recorded is not None and actual is not None else None
                parity_status = ("not_available" if delta is None else
                                 "matched_within_1e-5" if abs(delta) <= 1e-5 else "review_score_or_metric_parity")
                if delta is not None and not math.isfinite(delta):
                    raise ValueError("Nonfinite seen-validation parity difference")
                if sha256_file(entry["path"]) != entry["sha256"]:
                    raise ValueError("Checkpoint bytes changed during evaluation")
                directory.mkdir(parents=True, exist_ok=True)
                atomic_tsv(directory / "term_metrics.tsv", term_rows, ("go_id", "train_status", "n_positive_pairs", "ap"))
                atomic_tsv(directory / "protein_metrics.tsv", protein_rows, ("protein_id",))
                payloads = ["term_metrics.tsv", "protein_metrics.tsv"]
                if args.save_scores:
                    score_path = directory / "scores.npz"
                    fd, temporary = tempfile.mkstemp(prefix=".scores.", suffix=".npz", dir=directory)
                    os.close(fd)
                    try:
                        np.savez_compressed(temporary, scores=scores, go_ids=np.asarray(cohort["go_ids"], dtype=np.int64),
                                            protein_ids=np.asarray(cohort["protein_ids"], dtype=str))
                        os.replace(temporary, score_path)
                    finally:
                        if os.path.exists(temporary): os.unlink(temporary)
                    payloads.append("scores.npz")
                result = {
                    "format": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "contract": contract, "ablation": ablation,
                    "checkpoint": {"path_at_evaluation": str(entry["path"]), "sha256": entry["sha256"],
                                   "epoch_zero_based": epoch, "completed_epochs": epoch + 1,
                                   "aliases_at_evaluation": entry["aliases"], "global_step": meta.get("global_step")},
                    "architecture": meta["architecture"], "checkpoint_identity_audit": audit,
                    "runtime": runtime, "metrics": metrics,
                    "seen_validation_parity": {"recorded_protein_wise_map": recorded,
                                               "reevaluated_protein_wise_map": actual,
                                               "difference": delta, "status": parity_status},
                    "elapsed_seconds": time.perf_counter() - began,
                    "output_sha256": {name: sha256_file(directory / name) for name in payloads},
                }
                atomic_json(result_path, result)
                del scores, term_rows, protein_rows
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
                row = flatten_result(result, job, entry, directory)
                rows.append(row)
                print(f"[RESULT] {ablation} epoch={row['epoch']} "
                      f"unseen_mAP={row.get('unseen/protein_wise_map')} "
                      f"unseen_macro_AP={row.get('unseen/macro_term_ap')} "
                      f"unseen_R500={row.get('unseen/protein_recall@500')} "
                      f"seen_mAP={row.get('seen/protein_wise_map')} "
                      f"seen_parity={row['seen_map_parity_status']}", flush=True)
            except Exception as exc:
                failures.append({"ablation": ablation, "checkpoint": str(entry["path"]),
                                 "checkpoint_sha256": entry["sha256"], "stage": "evaluation",
                                 "error": str(exc), "traceback": traceback.format_exc()})
                print(f"[ERROR] {ablation} {entry['path'].name}: {exc}", flush=True)
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
            write_combined(output, rows, failures, inventory, cohort, args)
    write_combined(output, rows, failures, inventory, cohort, args, finished=True)
    print(f"[FINISHED] completed_unique_checkpoints={len(rows)} failures={len(failures)}", flush=True)
    print(f"all_checkpoints={output / 'all_checkpoints.tsv'}", flush=True)
    print(f"original_seen_selected_checkpoints={output / 'best_primary_checkpoints.tsv'}", flush=True)
    return 1 if failures else 0


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
