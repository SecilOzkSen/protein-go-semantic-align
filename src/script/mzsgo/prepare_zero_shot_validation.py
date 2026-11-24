#!/usr/bin/env python3
r"""Find natural train-zero-positive GO terms in the existing validation split.

Python 3.9+; standard library only. No GPU, torch or model weights are needed.

Quick start from the MZSGO repository root:
    python prepare_zero_shot_validation.py --root /workspace/MZSGO_temporal --branch MF

Optional provenance check against one or more existing runs:
    python prepare_zero_shot_validation.py --root /workspace/MZSGO_temporal --branch MF \
        --output-dir /workspace/MZSGO_temporal/analysis/zero_shot_validation_mf_audit \
        --run-manifest /path/to/run/run_manifest.json

Definition:
    active = candidate_GO_ids - reserved_temporal_GO_ids
    U = {g in active: train_positive_proteins(g) == 0
                     and validation_positive_proteins(g) > 0}

This script discovers U without changing the original protein split, training
labels or checkpoints. It never manufactures a heldout split from trained GO
terms. Test annotations are not read; only test protein IDs are read to check
that train/validation proteins do not overlap the final test.

Important interpretation:
    U has no positive TRAIN examples. That does not prove that its classes were
    absent from the training loss. In the inspected B2-2 full-GO InfoNCE code,
    active GO terms also appear in the softmax denominator. Such terms can be
    exposed as negative candidates. Optional run-manifest audits distinguish
    verified data identity from configured training-candidate exposure. Actual
    checkpoint states and training history are not loaded here.

Evaluation contract for the NEXT, separate checkpoint evaluator:
    - Score every validation protein against active_candidate_go_ids.json.
    - Keep seen + unseen candidates together; do not rank only U.
    - For unseen GO macro AP, retain ALL validation proteins, including those
      whose unseen label list is empty. Removing them removes negative examples.
    - Use validation_unseen_positive_protein_ids.txt only as a recall/mAP cohort
      index, not as the full protein universe for term AP.
    - Do not route the new labels through the legacy dataset option that drops
      train-unobserved positives. Do not bypass checkpoint data-identity guards.

Outputs are written into a new or empty directory, by default
    ROOT/analysis/zero_shot_validation_BRANCH
and include summary.json, GO lists, full/unseen/seen validation labels, protein
lists, a TSV count table, a manifest audit and README.txt. GO lists/label maps
use integer GO IDs, matching the training code. The TSV also includes GO: IDs.
All three label maps retain all validation protein keys and empty lists.

Exit codes:
    0: data extraction completed, including a valid EMPTY unseen cohort.
    2: invalid/missing input, unsafe overlap, or nonempty output directory.
    3: extraction completed, but a supplied run manifest is mismatched or lacks
       enough metadata to verify the split. Inspect the audit before evaluation.

If U is empty, unseen metrics are NOT COMPUTABLE, not zero. The script reports
this and emits empty unseen lists. Any new class-heldout training experiment is
a separate decision, not an automatic fallback.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile


VERSION = "natural_train_zero_positive_validation_v1"

# Exact source snapshots inspected when this script was prepared. Unknown
# versions remain unverified rather than being guessed from architecture names.
KNOWN_FULL_GO_SOURCES = {
    "inspected_b2_2_single_text": {
        "main.py": "7857733b166e0ea7d0763b6385591eba527522d0eb7af75748795759889465c6",
        "data.py": "c95c7830adbd87a12f253444cbd4ce411da666569d51ca0a07c86e7688b5fa20",
        "trainer.py": "2ea1f29f37ea0fd075cc679736e7f119a08fda44f092b6e094cbb14ca58ceb22",
    },
    "inspected_b2_2_frequency_infonce": {
        "main.py": "a835c69792188c7fccb14a296eb059211ace896a14d93e71658b9c999221f3d4",
        "data.py": "c95c7830adbd87a12f253444cbd4ce411da666569d51ca0a07c86e7688b5fa20",
        "trainer.py": "f4c343d461f5059328892b43b371870570dcab23afc1f913712c067f62e187bf",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     ensure_ascii=True).encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=_unique_object)


def go_int(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"GO ID must be a positive integer or GO: string, got {value!r}")
    text = str(value).strip()
    if not re.fullmatch(r"(?:GO:)?[0-9]+", text, flags=re.IGNORECASE):
        raise ValueError(f"Invalid GO ID: {value!r}")
    result = int(text.split(":", 1)[-1])
    if result <= 0:
        raise ValueError(f"Invalid GO ID: {value!r}")
    return result


def read_go_ids(path):
    raw = read_json(path)
    if isinstance(raw, dict):
        for key in ("ids", "go_ids", "terms"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raw = list(raw)  # Existing data loader also supports GO-keyed maps.
    if not isinstance(raw, list):
        raise ValueError(f"GO ID file must contain a list, supported wrapper or GO-keyed map: {path}")
    ids = [go_int(value) for value in raw]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate GO IDs after normalization: {path}")
    return ids


def read_protein_ids(path):
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
           if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate protein IDs: {path}")
    return ids


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/workspace/MZSGO_temporal"))
    parser.add_argument("--branch", type=str.upper, choices=("MF", "BP", "CC"), default="MF")
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--train-ids", type=Path)
    parser.add_argument("--val-ids", type=Path)
    parser.add_argument("--annotations", type=Path, help="Unfiltered train+validation pid-to-GO JSON")
    parser.add_argument("--candidate-go-ids", type=Path)
    parser.add_argument("--temporal-go-ids", type=Path, help="Reserved final temporal GO IDs")
    parser.add_argument("--test-ids", type=Path, help="Final test PROTEIN IDs, only for overlap protection")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-manifest", type=Path, action="append", default=[],
                        help="Optional existing run_manifest.json; repeat for multiple runs")
    return parser


def resolve_paths(args):
    root = args.root.expanduser().resolve()
    branch = args.branch.lower()
    split = root / f"mzsgo_graphs_{branch}"
    if branch == "bp" and not split.is_dir():
        split = root / "mzsgo_graphs"
    split = (args.split_dir or split).expanduser().resolve()
    proc = root / "processed" / branch
    values = {
        "train_ids": args.train_ids or split / "train_homology30_v2.txt",
        "val_ids": args.val_ids or split / "valid_homology30_v2.txt",
        "annotations": args.annotations or proc / f"pid_to_go_train_{branch}.json",
        "candidate_ids": args.candidate_go_ids or proc / f"candidate_go_ids_{branch}.json",
        "temporal_ids": args.temporal_go_ids or proc / f"temporal_go_ids_{branch}.json",
        "temporal_test_ids": args.test_ids or proc / "test.txt",
    }
    inputs = {key: path.expanduser().resolve() for key, path in values.items()}
    output = (args.output_dir or root / "analysis" / f"zero_shot_validation_{branch}").expanduser().resolve()
    manifests = [p.expanduser().resolve() for p in args.run_manifest]
    missing = [str(p) for p in list(inputs.values()) + manifests if not p.is_file()]
    if missing:
        raise ValueError("Missing input file(s):\n  " + "\n  ".join(missing))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output is not an empty directory: {output}. Choose a new --output-dir.")
    return inputs, output, manifests


def extract_cohort(inputs):
    train_ids = read_protein_ids(inputs["train_ids"])
    val_ids = read_protein_ids(inputs["val_ids"])
    test_ids = read_protein_ids(inputs["temporal_test_ids"])
    if not train_ids or not val_ids:
        raise ValueError("Train and validation protein lists must both be nonempty")
    if not test_ids:
        raise ValueError("Final test protein ID file is empty; protein-disjointness cannot be verified")
    train, val, test = set(train_ids), set(val_ids), set(test_ids)
    for label, overlap in (("train/validation", train & val), ("train/test", train & test),
                           ("validation/test", val & test)):
        if overlap:
            raise ValueError(f"Protein overlap in {label}: {sorted(overlap)[:10]}")

    candidate = set(read_go_ids(inputs["candidate_ids"]))
    temporal = set(read_go_ids(inputs["temporal_ids"]))
    if not candidate or not temporal <= candidate:
        raise ValueError("Candidate GO IDs must be nonempty and contain all reserved temporal IDs")
    active = candidate - temporal
    if not active:
        raise ValueError("No active non-temporal GO candidates remain")
    raw = read_json(inputs["annotations"])
    if not isinstance(raw, dict):
        raise ValueError("Annotations must be a JSON mapping from protein ID to GO list")
    labels, duplicate_labels = {}, 0
    # Only development proteins are inspected. Extra keys, including possible
    # test entries, are not used for counts, cohort membership or selection.
    for pid in train_ids + val_ids:
        if pid not in raw or not isinstance(raw[pid], list):
            raise ValueError(f"Missing/list-valued annotation entry required for protein {pid!r}")
        normalized = [go_int(g) for g in raw[pid]]
        ys = set(normalized)
        duplicate_labels += len(normalized) - len(ys)
        if not ys:
            raise ValueError(f"Empty unfiltered annotation list for development protein {pid!r}")
        if ys & temporal:
            raise ValueError(f"Reserved temporal GO appears in development annotations for {pid!r}: "
                             f"{sorted(ys & temporal)[:10]}; inspect the source split")
        if not ys <= candidate:
            raise ValueError(f"Out-of-universe GO annotations for {pid!r}: {sorted(ys - candidate)[:10]}")
        labels[pid] = sorted(ys)
    train_counts = Counter(g for pid in train_ids for g in labels[pid])
    val_counts = Counter(g for pid in val_ids for g in labels[pid])
    seen = set(train_counts)
    unseen = {g for g in active if train_counts[g] == 0 and val_counts[g] > 0}
    full_labels = {pid: labels[pid] for pid in val_ids}
    unseen_labels = {pid: sorted(set(labels[pid]) & unseen) for pid in val_ids}
    seen_labels = {pid: sorted(set(labels[pid]) & seen) for pid in val_ids}
    unseen_proteins = [pid for pid in val_ids if unseen_labels[pid]]
    # Validate a complete, lossless partition of validation truth.
    for pid in val_ids:
        if set(seen_labels[pid]) & set(unseen_labels[pid]):
            raise RuntimeError("Internal error: seen/unseen validation labels overlap")
        if sorted(seen_labels[pid] + unseen_labels[pid]) != full_labels[pid]:
            raise RuntimeError("Internal error: validation truth was not fully preserved")
    counts = {
        "n_train_proteins": len(train_ids), "n_validation_proteins": len(val_ids),
        "n_test_proteins_checked": len(test_ids), "n_candidate_go": len(candidate),
        "n_active_go": len(active), "n_reserved_temporal_go": len(temporal),
        "n_seen_train_go": len(seen), "n_unseen_validation_go": len(unseen),
        "n_unseen_without_validation_positive_go": len(active - seen - unseen),
        "n_validation_proteins_with_unseen": len(unseen_proteins),
        "n_validation_proteins_without_unseen": len(val_ids) - len(unseen_proteins),
        "n_validation_proteins_only_unseen": sum(bool(unseen_labels[p]) and not seen_labels[p] for p in val_ids),
        "n_training_positive_pairs": sum(train_counts.values()),
        "n_validation_positive_pairs": sum(val_counts.values()),
        "n_validation_unseen_positive_pairs": sum(len(v) for v in unseen_labels.values()),
        "n_duplicate_annotation_ids_collapsed": duplicate_labels,
    }
    return {"train_ids": train_ids, "val_ids": val_ids, "candidate": sorted(candidate),
            "temporal": sorted(temporal), "active": sorted(active), "seen": sorted(seen),
            "unseen": sorted(unseen), "unseen_proteins": unseen_proteins,
            "full_labels": full_labels, "unseen_labels": unseen_labels, "seen_labels": seen_labels,
            "train_counts": train_counts, "val_counts": val_counts, "counts": counts}


def audit_manifest(path, *, branch, hashes, cohort):
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Run manifest must be a JSON object: {path}")
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    source = raw.get("source_sha256") if isinstance(raw.get("source_sha256"), dict) else {}
    stored = data.get("input_sha256") if isinstance(data.get("input_sha256"), dict) else {}
    comparisons, mismatches, missing = {}, [], []
    for key, expected in hashes.items():
        observed = stored.get(key)
        comparisons[key] = {"current_sha256": expected, "manifest_sha256": observed,
                            "match": None if observed is None else observed == expected}
        if observed is None:
            missing.append(f"data.input_sha256.{key}")
        elif observed != expected:
            mismatches.append(f"data.input_sha256.{key}")
    active_hash = canonical_hash(cohort["active"])
    if data.get("active_go_ids_sha256") is None:
        missing.append("data.active_go_ids_sha256")
    elif data["active_go_ids_sha256"] != active_hash:
        mismatches.append("data.active_go_ids_sha256")
    data_branch = data.get("branch", config.get("branch"))
    if data_branch is None:
        missing.append("data.branch/config.branch")
    elif str(data_branch).upper() != branch:
        mismatches.append("branch")
    n = cohort["counts"]
    expected_counts = {
        "n_train_proteins": n["n_train_proteins"], "n_val_proteins": n["n_validation_proteins"],
        "n_active_go": n["n_active_go"], "n_temporal_go": n["n_reserved_temporal_go"],
        "n_observed_train_go": n["n_seen_train_go"], "n_train_positive_pairs": n["n_training_positive_pairs"],
        "n_val_source_positive_pairs": n["n_validation_positive_pairs"],
    }
    if data.get("validation_label_policy") == "legacy_exclude_train_unobserved_positives":
        expected_counts["n_val_filtered_positive_pairs"] = n["n_validation_unseen_positive_pairs"]
        expected_counts["n_val_evaluated_positive_pairs"] = (
            n["n_validation_positive_pairs"] - n["n_validation_unseen_positive_pairs"])
    for key, value in expected_counts.items():
        if key in data and data[key] != value:
            mismatches.append(f"data.{key}")
    status = "mismatch" if mismatches else ("unverified" if missing else "compatible")
    family = next((name for name, known in KNOWN_FULL_GO_SOURCES.items()
                   if all(source.get(key) == value for key, value in known.items())), None)
    try:
        coefficient = float(config["lambda_con"])
        full_go_enabled = math.isfinite(coefficient) and coefficient > 0
    except (KeyError, TypeError, ValueError):
        full_go_enabled = False
    configured_exposure = "unverified"
    if not cohort["unseen"]:
        configured_exposure = "not_applicable_empty_cohort"
    elif status == "compatible" and family and full_go_enabled:
        configured_exposure = "full_go_softmax_denominator_without_positive_targets"
    return {
        "manifest_path": str(path), "manifest_sha256": sha256_file(path), "status": status,
        "scope": "split_annotation_and_candidate_identity_only",
        "checkpoint_weights_inspected": False,
        "ablation_id": config.get("ablation_id"), "mismatches": mismatches,
        "missing_metadata": missing, "input_comparisons": comparisons,
        "active_go_ids_sha256": active_hash,
        "original_validation_label_policy": data.get("validation_label_policy"),
        "recognized_source_family": family,
        "configured_unseen_training_exposure": configured_exposure,
        "strict_class_holdout_verified": False,
        "strict_class_holdout_note": (
            "Not class-unexposed: inspected configured loss includes these GO scores as negative candidates."
            if configured_exposure == "full_go_softmax_denominator_without_positive_targets"
            else "Not certified by this audit. Zero positive count alone does not establish class non-exposure."
        ),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_ids(path, ids):
    path.write_text("".join(f"{pid}\n" for pid in ids), encoding="utf-8")


def write_bundle(output, *, inputs, branch, hashes, cohort, audits):
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".zero_shot_validation_", dir=output.parent))
    try:
        payloads = {
            "unseen_validation_go_ids.json": cohort["unseen"],
            "seen_go_ids.json": cohort["seen"],
            "active_candidate_go_ids.json": cohort["active"],
            "reserved_temporal_go_ids.json": cohort["temporal"],
            "validation_labels_full.json": cohort["full_labels"],
            "validation_labels_unseen.json": cohort["unseen_labels"],
            "validation_labels_seen.json": cohort["seen_labels"],
            "checkpoint_manifest_audit.json": audits,
        }
        for name, value in payloads.items():
            write_json(staging / name, value)
        write_ids(staging / "validation_protein_ids.txt", cohort["val_ids"])
        write_ids(staging / "validation_unseen_positive_protein_ids.txt", cohort["unseen_proteins"])
        seen, unseen, temporal = set(cohort["seen"]), set(cohort["unseen"]), set(cohort["temporal"])
        with (staging / "go_train_validation_counts.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["go_id", "go_int", "train_positive_proteins", "validation_positive_proteins", "group"])
            for g in cohort["candidate"]:
                group = ("reserved_temporal" if g in temporal else "seen" if g in seen else
                         "unseen_validation" if g in unseen else "unseen_without_validation_positive")
                writer.writerow([f"GO:{g:07d}", g, cohort["train_counts"][g], cohort["val_counts"][g], group])
        supports = [cohort["val_counts"][g] for g in cohort["unseen"]]
        counts = cohort["counts"]
        status = "unseen_validation_found" if unseen else "no_unseen_validation_terms"
        bad_audits = [a for a in audits if a["status"] != "compatible"]
        readme = f"""Natural train-zero-positive validation cohort ({branch})

Status: {status}
Unseen GO terms: {len(unseen)}
Validation proteins: {len(cohort['val_ids'])}
Validation proteins with >=1 unseen positive: {len(cohort['unseen_proteins'])}
Unseen validation positive pairs: {counts['n_validation_unseen_positive_pairs']}

This is a view of the existing protein split, not a newly trained class-heldout
experiment. Original files and checkpoint weights were not changed.

NEXT EVALUATOR:
1. Score ALL validation_protein_ids.txt against ALL active_candidate_go_ids.json.
2. Keep validation_labels_full.json as complete ground truth. Use the unseen GO
   IDs / validation_labels_unseen.json only to select the metric's target group.
3. Keep proteins with empty unseen lists when computing unseen term macro AP.
   They supply negative examples. Do not use only the unseen-positive protein
   list as the AP evaluation universe.
4. For per-protein unseen recall/mAP, the unseen-positive protein index identifies
   proteins with a defined unseen target denominator. Rank in the shared active
   seen+unseen GO universe, not an unseen-only candidate list.
5. A new evaluator must retain unseen positives. The old dataset option
   exclude_zero_shot_positives=True removes them. Do not change a checkpoint's
   recorded identity to get around resume/eval-only identity checks.
6. All checkpoints must match the source split, annotations and active GO IDs.
   Optional --run-manifest checks these hashes, not trained checkpoint tensors.

EXPOSURE:
Zero positive TRAIN examples is verified by these input annotations. It does
not certify that a class never appeared in the loss. The inspected B2-2
full-GO InfoNCE implementation places active GO terms in the denominator,
including GO terms with zero positive TRAIN examples. Its metric values should
be described as natural train-zero-positive validation, with configured negative
candidate exposure stated separately from positive supervision.

EMPTY COHORT:
If unseen GO terms=0, unseen metrics are NOT COMPUTABLE. Do not report a zero
score, mine the final test, or silently create a random heldout set.

Final test annotations were not used. Reserved temporal GO IDs are excluded.
Test protein IDs were read only to verify protein-disjoint development data.
"""
        (staging / "README.txt").write_text(readme, encoding="utf-8")
        outputs = {p.name: sha256_file(p) for p in sorted(staging.iterdir()) if p.is_file()}
        summary = {
            "format": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "branch": branch, "status": status,
            "definition": "GO in candidate-temporal with train_positive_protein_count=0 and validation_positive_protein_count>0",
            "counts": counts,
            "unseen_validation_support": {
                "minimum": min(supports) if supports else None,
                "median": statistics.median(supports) if supports else None,
                "maximum": max(supports) if supports else None,
                "terms_with_one_positive": sum(n == 1 for n in supports),
                "terms_with_at_least_five_positives": sum(n >= 5 for n in supports),
            },
            "inputs": {key: str(path) for key, path in inputs.items()},
            "input_sha256": hashes,
            "active_go_ids_sha256": canonical_hash(cohort["active"]),
            "unseen_validation_go_ids_sha256": canonical_hash(cohort["unseen"]),
            "validation_label_policy": "all_source_validation_positives_preserved",
            "final_test_annotations_read": False,
            "training_files_changed": False, "checkpoint_weights_loaded": False,
            "random_class_holdout_created": False,
            "unseen_metrics_computable_from_labels": bool(unseen),
            "empty_cohort_metric_value": None,
            "training_exposure": {
                "zero_positive_train_count_verified": True,
                "strict_class_nonexposure_verified": False,
                "reference_protocol": "active=candidate-temporal; full-GO InfoNCE denominator includes all active GO terms",
                "interpretation": "No positive examples; configured negative-candidate exposure must be reported separately.",
            },
            "manifest_audit": {
                "supplied": len(audits), "compatible": sum(a["status"] == "compatible" for a in audits),
                "mismatch": sum(a["status"] == "mismatch" for a in audits),
                "unverified": sum(a["status"] == "unverified" for a in audits),
                "all_supplied_manifests_compatible": (not bad_audits) if audits else None,
            },
            "output_sha256_excluding_summary": outputs,
        }
        write_json(staging / "summary.json", summary)
        # A result bundle appears only once it is complete. No existing nonempty
        # result directory or source dataset is silently overwritten.
        if output.exists():
            if not output.is_dir() or any(output.iterdir()):
                raise ValueError(f"Output became nonempty while preparing results: {output}")
            output.rmdir()  # Empty directory only.
        os.rename(staging, output)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        inputs, output, manifest_paths = resolve_paths(args)
        # Fingerprint before and after reading: abort if a live process changes
        # any source file during extraction.
        hashes = {key: sha256_file(path) for key, path in inputs.items()}
        cohort = extract_cohort(inputs)
        audits = [audit_manifest(path, branch=args.branch, hashes=hashes, cohort=cohort)
                  for path in manifest_paths]
        if hashes != {key: sha256_file(path) for key, path in inputs.items()}:
            raise ValueError("Input files changed during extraction; rerun with stable source files")
        summary = write_bundle(output, inputs=inputs, branch=args.branch, hashes=hashes,
                               cohort=cohort, audits=audits)
        n = summary["counts"]
        print(f"[ZERO-SHOT VALIDATION AUDIT] branch={args.branch}")
        print(f"status={summary['status']}")
        print(f"train_proteins={n['n_train_proteins']} validation_proteins={n['n_validation_proteins']}")
        print(f"active_go={n['n_active_go']} reserved_temporal_go={n['n_reserved_temporal_go']}")
        print(f"unseen_validation_go={n['n_unseen_validation_go']}")
        print(f"validation_proteins_with_unseen={n['n_validation_proteins_with_unseen']}")
        print(f"unseen_validation_positive_pairs={n['n_validation_unseen_positive_pairs']}")
        print("definition=train_positive_count=0 AND validation_positive_count>0; temporal GO excluded")
        print("exposure_note=zero_positive_count_does_not_certify_absence_from_training_loss")
        if not cohort["unseen"]:
            print("NO ELIGIBLE GO TERMS: unseen metrics are NOT COMPUTABLE, not zero.")
        if audits:
            print("manifest_audit=" + json.dumps(summary["manifest_audit"], sort_keys=True))
        else:
            print("checkpoint_identity=not_inspected; no --run-manifest supplied")
        print(f"output_dir={output}")
        print(f"summary={output / 'summary.json'}")
        if any(a["status"] != "compatible" for a in audits):
            print("MANIFEST REVIEW REQUIRED: inspect checkpoint_manifest_audit.json before checkpoint comparison.")
            return 3
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
