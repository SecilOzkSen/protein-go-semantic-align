"""Capture actual FA window scores while replaying the original test batches.

Depends on the previously installed mzsgo_fa_benchmark package. Model source,
checkpoint, evaluation bank and original batch companions are preserved. Hooks
observe tensors entering/leaving SmoothMIL; they do not replace the scorer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def go_int(value):
    return int(str(value).replace("GO:", ""))


def go_name(value):
    return f"GO:{go_int(value):07d}"


def stable_rank(values, column):
    values = np.asarray(values)
    return int(np.count_nonzero(values > values[column]) +
               np.count_nonzero(values[:column] == values[column]) + 1)


def replay_indices(n_proteins, batch_size, target_rows):
    """Complete original batches, ordered so the partial final batch stays last."""
    require(n_proteins > 0 and batch_size > 0, "Invalid original batch dimensions")
    require(target_rows and all(0 <= x < n_proteins for x in target_rows),
            "Case protein row outside the original evaluation")
    batches = sorted({x // batch_size for x in target_rows})
    return [row for block in batches
            for row in range(block * batch_size, min((block + 1) * batch_size, n_proteins))]


def window_starts(padded_length, window_length, stride):
    starts = list(range(0, max(1, padded_length - window_length + 1), stride))
    tail = max(0, padded_length - window_length)
    if tail not in starts:
        starts.append(tail)
    return sorted(set(starts))


def canonical_texts(path, selected_gos):
    """Display source text only; embeddings and presence come from the frozen bank."""
    selected_gos = set(map(go_int, selected_gos))
    result = {}
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            gid = go_int(obj["go_id"])
            if gid not in selected_gos:
                continue
            require(gid not in result, f"Duplicate canonical text for {go_name(gid)}")
            segments = obj.get("segments") or {}
            fields = {}
            for key in ("name", "definition"):
                value = segments.get(key) or obj.get(key) or ""
                fields[key] = " ".join(str(value).split())
            result[gid] = fields
    # Expanded general-test GO texts may not be present in the original JSONL.
    # An ID-only caption is honest; fetching another GO version would not be.
    return result


class LocalEvidenceCapture:
    """Observe exact intermediate tensors of a model without changing its output."""

    def __init__(self, model, ordered_pids, cases, n_go):
        self.model = model
        self.pids = list(ordered_pids)
        self.cases = list(cases)
        self.n_go = n_go
        self.offset = 0
        self.context = None
        self.results = {}
        self.original_score = model.score_fine_grained
        self.original_encode = model.encode_local_windows
        self.instance_values = {name: model.__dict__.get(name) for name in
                                ("score_fine_grained", "encode_local_windows")}
        self.had_instance_value = {name: name in model.__dict__ for name in self.instance_values}
        self.handle = None

    def __enter__(self):
        self.model.score_fine_grained = self._score
        self.model.encode_local_windows = self._encode
        self.handle = self.model.mil.register_forward_hook(self._mil)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        for name in self.instance_values:
            if self.had_instance_value[name]:
                setattr(self.model, name, self.instance_values[name])
            else:
                delattr(self.model, name)

    def _encode(self, H, mask):
        zw, valid = self.original_encode(H, mask)
        ctx = self.context
        require(ctx is not None and ctx["encoded"] is False, "Unexpected local window call")
        ctx["encoded"] = True
        starts = window_starts(H.shape[1], self.model.window_length, self.model.window_stride)
        require(len(starts) == zw.shape[1], "Window geometry differs from the training scorer")
        m = mask.detach().bool().cpu().numpy()
        v = valid.detach().bool().cpu().numpy()
        for case in ctx["cases"]:
            row = ctx["ids"].index(case["protein_id"])
            length = int(m[row].sum())
            require(bool(m[row, :length].all()) and not bool(m[row, length:].any()),
                    "Protein padding mask is not a contiguous sequence")
            records = []
            for j, start in enumerate(starts):
                if not v[row, j]:
                    continue
                end = min(start + self.model.window_length, length)
                require(end > start, "Invalid nonempty protein window")
                records.append({"index": j, "start": start + 1, "end": end,
                                "valid_residue_count": int(m[row, start:end].sum())})
            ctx["evidence"][case["case_id"]] = {
                "protein_length": length, "padded_batch_length": int(H.shape[1]),
                "geometry": records, "raw": [None, None], "mil": [None, None],
            }
        return zw, valid

    def _mil(self, module, inputs, output):
        scores, valid = inputs
        ctx = self.context
        require(ctx is not None and ctx["encoded"], "Unexpected SmoothMIL call order")
        call = ctx["mil_calls"]
        segment = call % 2
        first_go = (call // 2) * self.model.go_chunk_size
        require(first_go < self.n_go and scores.shape[2] == min(
            self.model.go_chunk_size, self.n_go - first_go), "Unexpected GO chunk shape")
        for case in ctx["cases"]:
            column = case["go_column"]
            if first_go <= column < first_go + scores.shape[2]:
                row = ctx["ids"].index(case["protein_id"])
                record = ctx["evidence"][case["case_id"]]
                indices = [w["index"] for w in record["geometry"]]
                record["raw"][segment] = scores[row, indices, column - first_go].detach().float().cpu().numpy()
                record["mil"][segment] = float(output[row, column - first_go].detach().float().cpu())
        ctx["mil_calls"] += 1

    def _score(self, H, mask, *args, **kwargs):
        ids = self.pids[self.offset:self.offset + H.shape[0]]
        require(len(ids) == H.shape[0], "Replay batch exceeds recorded protein IDs")
        cases = [case for case in self.cases if case["protein_id"] in ids]
        self.context = {"ids": ids, "cases": cases, "evidence": {},
                        "mil_calls": 0, "encoded": False}
        output = self.original_score(H, mask, *args, **kwargs)
        ctx = self.context
        require(ctx["mil_calls"] == 2 * math.ceil(self.n_go / self.model.go_chunk_size),
                "Unexpected number of SmoothMIL calls; parent evidence must be disabled")
        info = self.model.last_go_segment_info
        weights = info["segment_weights"].detach().float().cpu().numpy()
        present = info["segment_present"].detach().bool().cpu().numpy()
        for case in cases:
            row, column = ids.index(case["protein_id"]), case["go_column"]
            record = ctx["evidence"][case["case_id"]]
            require(all(x is not None for x in record["raw"] + record["mil"]),
                    "A selected GO segment was not captured")
            raw = np.stack(record["raw"], axis=1).astype(np.float64)
            z = raw / self.model.mil.tau
            z -= z.max(axis=0, keepdims=True)
            responsibility = np.exp(z)
            responsibility /= responsibility.sum(axis=0, keepdims=True)
            windows = []
            for j, geometry in enumerate(record["geometry"]):
                window = {k: v for k, v in geometry.items() if k != "index"}
                window.update(similarities=raw[j].tolist(), mil_responsibility=responsibility[j].tolist())
                windows.append(window)
            final_score = float(output[row, column].detach().float().cpu())
            reconstructed = float(np.dot(weights[column].astype(np.float64), record["mil"]))
            require(abs(reconstructed - final_score) <= 1e-6,
                    "Captured segment scores do not reconstruct the FA output")
            self.results[case["case_id"]] = {
                "protein_length": record["protein_length"],
                "padded_batch_length": record["padded_batch_length"],
                "segment_names": ["name", "definition"],
                "segment_present": present[column].tolist(),
                "segment_weights": weights[column].tolist(),
                "segment_scores": record["mil"], "windows": windows,
                "score_replayed": final_score, "score_reconstructed": reconstructed,
                "reconstruction_abs_difference": abs(reconstructed - final_score),
            }
        self.offset += H.shape[0]
        self.context = None
        return output


def capture_replay(bundle, resources, ordered_pids, bank, go_ids, cases, device, scorer):
    """Use the unmodified benchmark scorer and attach temporary observation hooks."""
    module = bundle["inference_module"]
    original_loader = module.load_aligner
    observers = []

    def loader(*args, **kwargs):
        model, meta = original_loader(*args, **kwargs)
        observer = LocalEvidenceCapture(model, ordered_pids, cases, len(go_ids))
        observer.__enter__()
        observers.append(observer)
        return model, meta

    module.load_aligner = loader
    try:
        values = scorer(bundle, resources, ordered_pids, bank, go_ids, device,
                        progress=lambda x: print(f"[REPLAY] batch {x['batch']}/{x['n_batches']}", flush=True))
        require(len(observers) == 1 and observers[0].offset == len(ordered_pids),
                "Incomplete original-batch replay")
        captured = observers[0].results
        require(set(captured) == {c["case_id"] for c in cases}, "Not all requested cases were captured")
        return values, captured
    finally:
        module.load_aligner = original_loader
        for observer in observers:
            observer.__exit__(None, None, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-root", default="/workspace/protein-go-semantic-align")
    parser.add_argument("--root", default="/workspace/MZSGO_temporal")
    parser.add_argument("--checkpoint", help="Optional relocated copy; its SHA256 must match the evaluation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    from .select_cases import load_evaluation
    try:
        from mzsgo_fa_benchmark.fa_scoring import inspect_checkpoint, prepare_base_resources, score_test
        from mzsgo_fa_benchmark.evaluate_test import prepare_cohort
    except ImportError as exc:
        raise RuntimeError("Install the previously supplied mzsgo_fa_benchmark package alongside this addon") from exc

    out = Path(args.out_dir).resolve()
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "Output directory must be new or empty")
    selection_path = Path(args.selection).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    require(selection.get("format") == "mzsgo_fa_case_selection_v1", "Unsupported case selection format")
    data = load_evaluation(selection["eval_dir"])
    require(data["source_files_sha256"] == selection["source_files_sha256"] and
            data["source_manifest_sha256"] == selection["source_manifest_sha256"],
            "Evaluation artifacts changed after case selection")
    metrics, cases = data["metrics"], selection["cases"]
    require(cases and len(cases) <= 2, "Select one or two cases for this compact figure")
    require(len({c["case_id"] for c in cases}) == len(cases), "Duplicate case identifiers")
    columns = list(map(int, data["label_columns"]))
    for case in cases:
        r, c = int(case["protein_row"]), int(case["go_column"])
        require(0 <= r < len(data["protein_ids"]) and 0 <= c < len(data["go_ids"]), "Case index out of range")
        require(case["protein_id"] == data["protein_ids"][r] and
                go_name(case["go_id"]) == go_name(data["go_ids"][c]), "Case ID/index mismatch")
        require(c in columns and data["labels"][r, columns.index(c)] == 1 and case.get("label") == 1,
                "Case must be an annotated positive in this evaluation")
        require(case["rank"] == stable_rank(data["scores"][r], c) and
                abs(case["score_raw"] - float(data["scores"][r, c])) <= 1e-8,
                "Case rank or raw score changed")
    checkpoint = args.checkpoint or metrics["checkpoint"]
    require(sha256(checkpoint) == metrics["checkpoint_sha256"] == selection["checkpoint_sha256"],
            "Case study must use the exact evaluated checkpoint")
    branch = metrics["branch"]
    bundle = inspect_checkpoint(checkpoint, args.repo_root, branch)
    resources = prepare_base_resources(bundle, args.root)
    cfg, meta = bundle["cfg"], bundle["meta"]
    require(meta["epoch"] == metrics["selected_validation_epoch"], "Evaluated checkpoint epoch differs")
    metric = cfg.get("monitor_metric", "protein_wise_map")
    selected = meta.get("selection", {})
    recorded = meta.get("validation_metrics", {}).get(metric)
    require(selected.get("best_primary_epoch") == meta["epoch"] and recorded is not None and
            selected.get("best_primary") is not None and abs(float(recorded) - float(selected["best_primary"])) <= 1e-10,
            "Use the original validation-selected best_primary checkpoint")
    root = Path(args.root).resolve()
    suffix = branch.lower()
    if metrics["mode"] == "general":
        annotation_path = root / "processed" / suffix / f"pid_to_go_general_test_{suffix}.json"
        bank_path = root / "go_cache" / suffix / "frozen_segment_parent_bank_general_test.pt"
        pids, gids, labels, bank, _, _, bank_audit = prepare_cohort(bundle, resources, bank_path, annotation_path)
        require(bank_audit["sha256"] == metrics["bank_sha256"], "General bank differs from the evaluated bank")
    elif metrics["mode"] == "temporal":
        from mzsgo_fa_benchmark.evaluate_temporal import prepare_temporal_cohort
        annotation_path = root / "processed" / suffix / f"pid_to_go_temporal_{suffix}.json"
        pids, gids, temporal, labels, temporal_columns, audit = prepare_temporal_cohort(bundle, resources, annotation_path)
        bank = resources["base_bank"]
        require(sha256(resources["base_bank_path"]) == metrics["bank_sha256"], "Temporal bank differs from evaluation")
        require(list(temporal_columns) == columns, "Temporal label columns changed")
    else:
        raise ValueError("Only RAW general or temporal evaluations are supported")
    require(sha256(annotation_path) == metrics["annotation_sha256"], "Evaluation annotations changed")
    require(pids == data["protein_ids"] and list(map(go_name, gids)) == list(map(go_name, data["go_ids"])),
            "Original benchmark protein/GO ordering could not be reproduced")
    require(np.array_equal(labels, data["labels"]), "Evaluation gold labels differ from the saved matrix")
    runtime = metrics.get("scoring_runtime", {})
    expected_autocast = "bfloat16" if cfg["fp16"] and str(args.device).startswith("cuda") else "disabled"
    require(runtime.get("eval_batch_size") == cfg["eval_batch_size"] and
            runtime.get("autocast") == expected_autocast,
            "Replay must use the original evaluation batch size and numerical precision")
    indices = replay_indices(len(pids), cfg["eval_batch_size"], [c["protein_row"] for c in cases])
    replay_pids = [pids[i] for i in indices]
    print(f"[CASE STUDY] {len(cases)} pairs; replaying {len(replay_pids)} proteins in their original batches", flush=True)
    replayed, captured = capture_replay(bundle, resources, replay_pids, bank, gids, cases, args.device, score_test)
    expected = np.asarray(data["scores"])[indices]
    max_diff = float(np.max(np.abs(replayed - expected)))
    require(max_diff <= 1e-5,
            f"Replay differs from stored RAW scores (max abs {max_diff:.8g}); no figure exported. Use original scoring environment.")
    display_texts = canonical_texts(resources["input_paths"]["canonical_text"], [c["go_id"] for c in cases])
    exported = []
    for case in cases:
        evidence = captured[case["case_id"]]
        row = replay_pids.index(case["protein_id"])
        require(stable_rank(replayed[row], case["go_column"]) == case["rank"],
                "Selected GO rank changed during replay; no figure exported")
        texts = display_texts.get(go_int(case["go_id"]), {})
        exported.append({**case, **evidence,
                         "go_name": texts.get("name", ""), "go_definition": texts.get("definition", ""),
                         "go_display_text_source": str(resources["input_paths"]["canonical_text"]) if texts else None,
                         "annotations": [], "annotation_status": "not_yet_independently_verified",
                         "original_batch_number": case["protein_row"] // cfg["eval_batch_size"] + 1,
                         "parity": {"max_abs_score_difference": max_diff, "threshold": 1e-5,
                                    "full_candidate_score_row_verified": True, "target_rank_unchanged": True}})
    result = {
        "format": "mzsgo_fa_local_evidence_v1", "model": "B3-FA", "branch": branch,
        "evaluation_mode": metrics["mode"], "n_candidates": len(gids), "n_evaluation_proteins": len(pids),
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256(checkpoint),
        "selection_file": str(selection_path), "selection_sha256": sha256(selection_path),
        "selection_policy": selection["selection_policy"], "source_files_sha256": selection["source_files_sha256"],
        "window_length": cfg["window_length"], "window_stride": cfg["window_stride"], "mil_tau": cfg["mil_tau"],
        "gate_conditioning": "GO text only; shared across proteins for the same GO",
        "score_order": "window cosine -> SmoothMIL separately per segment -> learned GO-only segment weights",
        "parent_strength": 0.0, "frequency_weights_at_inference": False, "true_path_applied": False,
        "scoring_runtime": bundle["scoring_runtime"], "source_audit": bundle["source_audit"],
        "replayed_original_batch_count": len({i // cfg["eval_batch_size"] for i in indices}),
        "cases": exported,
        "interpretation": "Illustrative model-window evidence, not residue-level attribution or biological causal validation. MIL responsibilities are derived local score sensitivities, not probabilities that a region is functional. Contextual residue embeddings may contain information from outside each window.",
    }
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "Output directory became nonempty")
    (out / "evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "window_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["case_id", "protein_id", "go_id", "start", "end", "valid_residue_count", "segment",
                  "segment_present", "cosine_similarity", "mil_responsibility", "go_segment_weight"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in exported:
            for window in case["windows"]:
                for j, segment in enumerate(case["segment_names"]):
                    writer.writerow({"case_id": case["case_id"], "protein_id": case["protein_id"], "go_id": case["go_id"],
                                     "start": window["start"], "end": window["end"], "valid_residue_count": window["valid_residue_count"],
                                     "segment": segment, "segment_present": case["segment_present"][j],
                                     "cosine_similarity": window["similarities"][j], "mil_responsibility": window["mil_responsibility"][j],
                                     "go_segment_weight": case["segment_weights"][j]})
    print(f"[PARITY] original RAW scores matched, max_abs_difference={max_diff:.8g}")
    print(f"[SAVED] {out / 'evidence.json'}")


if __name__ == "__main__":
    main()
