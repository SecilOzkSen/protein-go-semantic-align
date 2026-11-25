"""Source-verified B3-FA scoring on a separately prepared benchmark cohort.

This module reads existing checkpoints and frozen features only. It does not
encode GO text, apply frequency weights at inference, fit a model, or decide
the benchmark's protein/GO universe. The caller validates the expanded bank
against the base bank and supplies the complete, ordered test cohort.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys


PACKAGE = "src.ablations.b3_frequency_infonce"
CRITICAL_SOURCE_FILES = ("aligner.py", "inference.py", "config.py", "checkpoints.py")
INPUT_CONFIG_KEYS = {
    "train_ids": "train_ids_path", "val_ids": "val_ids_path",
    "annotations": "pid2pos", "candidate_ids": "branch_go_ids_path",
    "temporal_ids": "temporal_go_ids_path", "canonical_text": "go_text_path",
    "original_train_ids": "original_train_ids_path",
    "original_valid_ids": "original_valid_ids_path",
    "temporal_test_ids": "temporal_test_ids_path", "frozen_bank": "mzsgo_frozen_bank",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _source_check(bundle):
    for filename, entry in bundle["source_audit"]["files"].items():
        _require(_sha256(entry["path"]) == entry["expected_sha256"],
                 f"B3-FA source changed since training: {filename}")


def _input_path(bundle, root, config_key):
    value = bundle["cfg"].get(config_key)
    _require(isinstance(value, str) and value, f"Missing saved path: {config_key}")
    path = Path(value).expanduser()
    saved_root = Path(bundle["cfg"]["root"]).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    if path.is_absolute():
        try:
            return root / path.relative_to(saved_root)
        except ValueError:
            return path
    return root / path


def inspect_checkpoint(checkpoint_path, repo_root, expected_branch):
    """Read a FA checkpoint through its exact package and adjacent manifest.

    The public checkpoint reader validates the package-specific format. Four
    scoring-critical files and the data helper are checked against the training
    run manifest before any package code is imported. No checkpoint tensors are
    retained in the returned bundle.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    _require(checkpoint_path.is_file(), f"Checkpoint not found: {checkpoint_path}")
    _require(manifest_path.is_file(), f"Matching run_manifest.json must accompany the checkpoint: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    _require(isinstance(manifest, dict), "Run manifest must be a JSON object")
    stored = manifest.get("source_sha256")
    _require(isinstance(stored, dict), "Run manifest has no source_sha256 mapping")
    package_dir = repo_root.joinpath(*PACKAGE.split("."))
    checked_files = {}
    for filename in (*CRITICAL_SOURCE_FILES, "data.py"):
        expected = stored.get(filename)
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                 f"Manifest lacks source SHA256 for {filename}")
        path = package_dir / filename
        _require(path.is_file(), f"Install the original B3-FA package first: {path}")
        actual = _sha256(path)
        _require(actual == expected, f"B3-FA source SHA256 differs from training: {filename}")
        checked_files[filename] = {"path": str(path), "expected_sha256": expected,
                                   "actual_sha256": actual}
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    importlib.invalidate_caches()
    modules = {}
    for filename in (*CRITICAL_SOURCE_FILES, "data.py"):
        module = importlib.import_module(PACKAGE + "." + filename[:-3])
        _require(Path(module.__file__).resolve() == (package_dir / filename).resolve(),
                 f"Imported B3-FA {filename} from another repository")
        modules[filename[:-3]] = module
    checkpoint_sha = _sha256(checkpoint_path)
    payload = modules["checkpoints"].read_checkpoint(checkpoint_path)
    try:
        _require(isinstance(payload.get("meta"), dict), "Checkpoint metadata missing")
        meta = deepcopy(payload["meta"])
    finally:
        del payload
    _require(_sha256(checkpoint_path) == checkpoint_sha, "Checkpoint changed during inspection")
    for key in ("config", "architecture", "data_identity", "training_contract"):
        _require(isinstance(meta.get(key), dict), f"Checkpoint {key} missing")
    cfg, arch, identity = meta["config"], meta["architecture"], meta["data_identity"]
    branch = str(expected_branch).upper()
    _require(branch in ("MF", "BP", "CC"), f"Invalid branch: {branch}")
    _require(cfg.get("branch") == branch and identity.get("branch") == branch,
             "Checkpoint branch differs from requested benchmark branch")
    _require(cfg.get("ablation_id") == "B3-FA" and cfg.get("frequency_aware_infonce") is True,
             "This benchmark adapter requires a B3-FA frequency-aware InfoNCE checkpoint")
    _require(meta.get("inference_entrypoint") == PACKAGE + ".inference.load_aligner",
             "Checkpoint inference entrypoint is not the original B3-FA package")
    _require(meta.get("epoch_complete") is True, "Checkpoint is not a completed epoch")
    _require(arch.get("architecture_version") == "mzsgo_mf_ablation_v1",
             "Unsupported B3-FA architecture version")
    expected = {"protein_mode": "local", "go_representation": "segments",
                "segment_aggregation": "learned", "parent_strength": 0.0}
    for key, value in expected.items():
        _require(cfg.get(key) == value and arch.get(key) == value,
                 f"B3-FA scoring contract differs: {key}")
    for key in ("window_length", "window_stride", "mil_tau", "go_chunk_size"):
        _require(cfg.get(key) == arch.get(key), f"Saved config/architecture disagree on {key}")
    _require(arch.get("normalize") is True and arch.get("gate_conditioning") == "go_text"
             and arch.get("aggregation_level") == "segment_scores_after_window_reduction"
             and arch.get("window_layout") == "legacy_padded_batch"
             and arch.get("score_segments") == ["name", "definition"],
             "B3-FA gate, score normalization or window layout differs")
    _require(cfg.get("pbr_lambda") == 0.0, "PBR must remain disabled")
    _require(type(cfg.get("eval_batch_size")) is int and cfg["eval_batch_size"] > 0,
             "Invalid saved eval_batch_size")
    _require(type(cfg.get("fp16")) is bool, "Saved fp16 must be boolean")
    _require(isinstance(cfg.get("root"), str) and cfg["root"], "Saved dataset root missing")
    for key in ("max_len", "overlap"):
        _require(type(cfg.get(key)) is int and cfg[key] >= (1 if key == "max_len" else 0),
                 f"Invalid saved ESM store {key}")
    _require(cfg["overlap"] < cfg["max_len"], "Invalid saved ESM overlap")
    _require(manifest.get("architecture") == arch, "Run manifest/checkpoint architecture differs")
    _require(manifest.get("data") == identity, "Run manifest/checkpoint training data identity differs")
    _require(isinstance(manifest.get("config"), dict), "Run manifest config missing")
    contract = modules["checkpoints"].training_contract
    _require(contract(cfg) == meta["training_contract"] == contract(manifest["config"]),
             "Run manifest/checkpoint training contract differs")
    _require(isinstance(identity.get("input_sha256"), dict), "Checkpoint input SHA256 mapping missing")
    _require(isinstance(identity.get("frequency_infonce"), dict)
             and identity["frequency_infonce"].get("method") == "train_positive_exposure_mean_v1",
             "Checkpoint TRAIN-only frequency weighting provenance missing")
    source_audit = {"status": "training_source_sha256_matched", "files": checked_files,
                    "critical_files": list(CRITICAL_SOURCE_FILES), "hash_bypass": False,
                    "module": PACKAGE + ".inference"}
    return {"meta": meta, "manifest": manifest, "cfg": cfg, "architecture": arch,
            "identity": identity, "package_data": modules["data"],
            "inference_module": modules["inference"], "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha, "repo_root": repo_root,
            "manifest_path": manifest_path, "source_audit": source_audit}


def prepare_base_resources(bundle, root):
    """Verify recorded TRAIN inputs and recover actual effective TRAIN labels.

    Validation labels are never added to the training membership or counts.
    The original dataset excludes train-unobserved labels; that exclusion
    cannot remove a positive belonging to an actual training protein.
    """
    import torch

    _source_check(bundle)
    root = Path(root).expanduser().resolve()
    data, cfg, identity = bundle["package_data"], bundle["cfg"], bundle["identity"]
    paths, hashes = {}, {}
    for label, config_key in INPUT_CONFIG_KEYS.items():
        path = _input_path(bundle, root, config_key)
        _require(path.is_file(), f"Recorded training input is missing: {path}")
        expected = identity["input_sha256"].get(label)
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                 f"Checkpoint lacks input SHA256 for {label}")
        actual = data.file_sha256(path)
        _require(actual == expected, f"Training input SHA256 differs from checkpoint: {label} ({path})")
        paths[label], hashes[label] = path, actual
    train_ids, val_ids = data.read_protein_ids(paths["train_ids"]), data.read_protein_ids(paths["val_ids"])
    train, val = set(train_ids), set(val_ids)
    _require(train and val and not (train & val), "Invalid recorded train/validation split")
    original_train = set(data.read_protein_ids(paths["original_train_ids"]))
    original_val = set(data.read_protein_ids(paths["original_valid_ids"]))
    temporal_test_ids = data.read_protein_ids(paths["temporal_test_ids"])
    _require(not (original_train & original_val) and train | val == original_train | original_val,
             "Recorded split does not cover the original training pool")
    _require(not ((train | val) & set(temporal_test_ids)), "Training pool overlaps recorded temporal test IDs")
    candidate = sorted(data.read_go_ids(paths["candidate_ids"]))
    temporal = sorted(data.read_go_ids(paths["temporal_ids"]))
    _require(candidate and set(temporal) <= set(candidate), "Invalid recorded candidate/temporal GO sets")
    active = sorted(set(candidate) - set(temporal))
    _require(active and _json_hash(active) == identity.get("active_go_ids_sha256"),
             "Recorded active GO IDs differ from checkpoint")
    raw_annotations = json.loads(paths["annotations"].read_text(encoding="utf-8-sig"))
    _require(isinstance(raw_annotations, dict), "Expected protein-to-GO training annotation object")
    annotations, counts = {}, Counter()
    active_set = set(active)
    for pid in train_ids:
        raw = raw_annotations.get(pid)
        _require(isinstance(raw, list), f"Training GO list missing: {pid}")
        positives = sorted({data.go_int(g) for g in raw})
        _require(positives and set(positives) <= active_set,
                 f"Empty or out-of-active-universe training labels: {pid}")
        annotations[pid] = positives
        counts.update(positives)
    expected_counts = {"n_train_proteins": len(train_ids), "n_val_proteins": len(val_ids),
                       "n_active_go": len(active), "n_temporal_go": len(temporal),
                       "n_observed_train_go": len(counts), "n_train_positive_pairs": sum(counts.values())}
    for key, value in expected_counts.items():
        _require(identity.get(key) == value, f"Recorded training identity count differs: {key}")
    frequency = identity["frequency_infonce"]
    _require(frequency.get("active_go_ids_sha256") == _json_hash(active),
             "Frequency-weight GO order differs from active training order")
    _require(frequency.get("train_counts_sha256") == _json_hash(
        {"go_ids": active, "counts": [counts.get(g, 0) for g in active]}),
        "Effective TRAIN positive counts differ from saved frequency weighting")
    positives_hash = hashlib.sha256()
    for pid in train_ids:
        positives_hash.update(json.dumps(annotations[pid], separators=(",", ":")).encode("utf-8"))
        positives_hash.update(b"\n")
    _require(frequency.get("train_positive_sets_sha256") == positives_hash.hexdigest(),
             "Effective TRAIN positive sets/order differ from saved frequency weighting")
    base_bank = torch.load(paths["frozen_bank"], map_location="cpu", weights_only=False)
    _require(isinstance(base_bank, dict) and base_bank.get("format") == "mzsgo_frozen_segment_parent_v1",
             "Unexpected frozen base bank format")
    bank_ids = [data.go_int(g) for g in base_bank.get("go_ids", [])]
    _require(len(bank_ids) == len(set(bank_ids)) and set(bank_ids) == set(candidate),
             "Base bank does not contain the exact training candidate universe")
    metadata = base_bank.get("metadata", {})
    _require(metadata.get("canonical_jsonl_sha256") == hashes["canonical_text"],
             "Base bank canonical text provenance differs from checkpoint")
    _require(metadata.get("model") == cfg["encoder_name"] and metadata.get("normalized") is False
             and metadata.get("segment_max_len") == cfg["go_segment_max_len"],
             "Base bank encoder/tokenization/normalization differs from checkpoint")
    _require(list(base_bank.get("segment_names", [])) == cfg["enabled_segments"],
             "Base bank segment layout differs from checkpoint")
    embed_dir = _input_path(bundle, root, "embed_dir_res")
    _require(embed_dir.is_dir(), f"Recorded residue embedding directory is missing: {embed_dir}")
    return {"root": root, "train_ids": train_ids, "val_ids": val_ids,
            "active_go_ids": active, "candidate_go_ids": candidate, "temporal_go_ids": temporal,
            "temporal_test_ids": temporal_test_ids, "train_annotations": annotations,
            "train_counts": dict(counts), "train_seen_go_ids": sorted(counts),
            "base_bank": base_bank, "base_bank_path": paths["frozen_bank"],
            "embed_dir": embed_dir, "input_paths": paths, "input_sha256": hashes,
            "audit": {"training_inputs_sha256_matched": True,
                      "training_identity_counts": expected_counts,
                      "effective_train_positive_sets_sha256": positives_hash.hexdigest(),
                      "seen_definition": "positive in actual checkpoint TRAIN proteins only",
                      "validation_annotations_in_seen_definition": False,
                      "frequency_weights_applied_at_inference": False}}


def score_test(bundle, resources, ordered_pids, evaluation_bank, evaluation_go_ids,
               device, progress=None):
    """Score the supplied test cohort with the saved B3-FA forward contract.

    ``progress``, when supplied, receives a JSON-native dict after each batch.
    Raw float32 scores are returned in the exact supplied protein and GO order.
    No validation-size equality is imposed on this independent test cohort.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    _source_check(bundle)
    _require(_sha256(bundle["checkpoint_path"]) == bundle["checkpoint_sha256"],
             "Checkpoint changed after inspection")
    cfg, arch = bundle["cfg"], bundle["architecture"]
    pids = list(ordered_pids)
    gids = [bundle["package_data"].go_int(g) for g in evaluation_go_ids]
    _require(pids and all(isinstance(p, str) and p for p in pids) and len(pids) == len(set(pids)),
             "Test protein IDs must be nonempty, unique strings")
    _require(gids and len(gids) == len(set(gids)), "Test GO IDs must be nonempty and unique")
    test_annotations = resources.get("evaluation_annotations")
    _require(isinstance(test_annotations, dict),
             "resources['evaluation_annotations'] must contain the actual test GO labels for the collator")
    labels = {}
    for pid in pids:
        raw_labels = test_annotations.get(pid)
        _require(isinstance(raw_labels, (list, tuple)) and raw_labels,
                 f"Test annotation list missing: {pid}")
        positive_ids = [bundle["package_data"].go_int(g) for g in raw_labels]
        _require(len(positive_ids) == len(set(positive_ids)) and set(positive_ids) <= set(gids),
                 f"Duplicate or out-of-universe test annotations: {pid}")
        labels[pid] = positive_ids
    _require(isinstance(evaluation_bank, dict)
             and evaluation_bank.get("format") == "mzsgo_frozen_segment_parent_v1",
             "Expected a frozen segmented evaluation bank")
    names = list(evaluation_bank.get("segment_names", []))
    _require(names == cfg["enabled_segments"] and "name" in names and "definition" in names,
             "Evaluation bank segment layout differs from checkpoint")
    metadata = evaluation_bank.get("metadata", {})
    _require(metadata.get("model") == cfg["encoder_name"] and metadata.get("normalized") is False
             and metadata.get("segment_max_len") == cfg["go_segment_max_len"],
             "Evaluation bank encoder/tokenization/normalization differs from checkpoint")
    bank_ids = [bundle["package_data"].go_int(g) for g in evaluation_bank.get("go_ids", [])]
    _require(len(bank_ids) == len(set(bank_ids)), "Duplicate evaluation bank GO IDs")
    lookup = {g: row for row, g in enumerate(bank_ids)}
    _require(set(gids) <= set(lookup), "Evaluation GO IDs are missing from frozen bank")
    segments, present = evaluation_bank.get("segment_embs"), evaluation_bank.get("segment_present")
    _require(torch.is_tensor(segments) and segments.shape == (len(bank_ids), len(names), arch["d_g"])
             and segments.is_floating_point() and bool(torch.isfinite(segments).all()),
             "Invalid evaluation segment embeddings")
    _require(torch.is_tensor(present) and present.shape == segments.shape[:2]
             and bool(((present == 0) | (present == 1)).all()), "Invalid evaluation segment presence mask")
    index = torch.tensor([lookup[g] for g in gids], dtype=torch.long, device=segments.device)
    selected = segments.index_select(0, index).float()
    selected_present = present.index_select(0, index.to(present.device)).bool()
    ni, di = names.index("name"), names.index("definition")
    _require(bool(selected_present[:, ni].all()), "Evaluation GO term has no name channel")
    device = torch.device(device)
    _require(device.type in ("cpu", "cuda"), "Only CPU/CUDA scoring is supported")
    _require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    use_bf16 = cfg["fp16"] and device.type == "cuda"
    if use_bf16:
        with torch.cuda.device(device):
            _require(torch.cuda.is_bf16_supported(), "Checkpoint requires CUDA bfloat16 support")
    model, loaded_meta = bundle["inference_module"].load_aligner(
        bundle["checkpoint_path"], device=device, expected_branch=cfg["branch"], expected_ablation="B3-FA")
    _require(loaded_meta == bundle["meta"] and model.architecture_metadata() == arch,
             "Loaded checkpoint/model differs from inspected metadata")
    _require(not model.training, "FA inference loader did not select eval mode")
    from src.datasets.residue_store import ESMResidueStore
    from src.training.collate import ContrastiveEmbCollator

    store = ESMResidueStore(embed_dir=resources["embed_dir"], max_len=cfg["max_len"],
                            overlap=cfg["overlap"], prefer_fp16=cfg["fp16"])

    class TestDataset(Dataset):
        def __len__(self):
            return len(pids)

        def __getitem__(self, index):
            pid = pids[index]
            embeddings = store.get(pid)
            _require(torch.is_tensor(embeddings) and embeddings.ndim == 2
                     and embeddings.shape[0] > 0 and embeddings.shape[1] == arch["d_h"]
                     and embeddings.is_floating_point() and bool(torch.isfinite(embeddings).all()),
                     f"Invalid residue embeddings for test protein {pid}")
            # The original collator requires real, nonempty positive lists;
            # they are not consumed by the scoring rule below.
            return {"protein_id": pid, "prot_emb": embeddings,
                    "pos_go_ids": torch.tensor(labels[pid], dtype=torch.long)}

    generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 1)
    batches = DataLoader(TestDataset(), batch_size=cfg["eval_batch_size"],
                         shuffle=False, drop_last=False, num_workers=0, pin_memory=False,
                         generator=generator, collate_fn=ContrastiveEmbCollator())
    name = selected[:, ni].contiguous().to(device)
    definition = selected[:, di].contiguous().to(device)
    definition_present = selected_present[:, di].contiguous().to(device)
    scores_out = np.empty((len(pids), len(gids)), dtype=np.float32)
    old_precision = torch.get_float32_matmul_precision()
    old_tf32, old_cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    offset = 0
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                    dtype=torch.bfloat16, enabled=use_bf16):
            weights = model.aggregation_weights(name, definition, definition_present)
            for batch_number, batch in enumerate(batches, 1):
                ids = batch["protein_ids"]
                end = offset + len(ids)
                _require(ids == pids[offset:end], "Test protein order changed in the collator")
                H = batch["prot_emb_pad"].to(device, non_blocking=True).float()
                mask = batch["prot_attn_mask"].to(device, non_blocking=True).bool()
                scores = model.score_fine_grained(H, mask, name, definition, definition_present,
                                                 None, None, None, segment_weights=weights)
                _require(tuple(scores.shape) == (len(ids), len(gids)), "Unexpected FA score shape")
                values = scores.detach().float().cpu().numpy()
                _require(np.isfinite(values).all(), "Nonfinite FA benchmark scores")
                scores_out[offset:end] = values
                offset = end
                if progress is not None:
                    progress({"batch": batch_number, "n_batches": len(batches),
                              "proteins": offset, "n_proteins": len(pids)})
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    _require(offset == len(pids), "Incomplete FA benchmark scoring")
    bundle["scoring_runtime"] = {
        "device": str(device), "torch": str(torch.__version__), "numpy": str(np.__version__),
        "eval_batch_size": cfg["eval_batch_size"], "n_batches": len(batches),
        "autocast": "bfloat16" if use_bf16 else "disabled",
        "window_layout": arch["window_layout"], "protein_order_sha256": _json_hash(pids),
        "evaluation_go_ids_sha256": _json_hash(gids), "shuffle": False, "drop_last": False,
        "parent_strength": 0.0, "frequency_score_adjustment": False,
        "encoder_initialized": False, "bank_created": False,
    }
    return scores_out
