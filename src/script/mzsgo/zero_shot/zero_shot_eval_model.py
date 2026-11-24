"""Score retained checkpoints with their source-verified original implementations.

Only cached protein and GO embeddings are read. This module never constructs a
text encoder, generates a GO cache, retrains a model, or changes its prediction
rule. Full validation order and saved batch size are part of the local scorer's
contract because its window layout depends on batch padding.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import logging
import math
from pathlib import Path
import re
import sys
import time
from types import ModuleType


CRITICAL_SOURCE_FILES = ("aligner.py", "inference.py", "config.py", "checkpoints.py")
_RESOLVED_LOADERS = {}
_LOG = logging.getLogger("zero_shot_eval.scoring")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _file_sha256(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    _require(all(getattr(before, k) == getattr(after, k) for k in fields),
             f"File changed while being hashed: {path}")
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     ensure_ascii=True).encode("utf-8")).hexdigest()


def _importable_component(value):
    """Allow source folders such as b1-1 through importlib's string API.

    Dotted import statements require identifiers; importlib also supports
    literal hyphens in package directories. Keep path separators, empty parts
    and traversal out of module names. File hashes still establish identity.
    """
    return (value.isidentifier()
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", value) is not None)


def _module_name(value):
    if isinstance(value, ModuleType):
        value = value.__name__
    _require(isinstance(value, str) and value, "Loader must be an inference module name")
    if value.endswith(".load_aligner"):
        value = value[:-len(".load_aligner")]
    _require(value.startswith("src.") and value.endswith(".inference")
             and all(_importable_component(part) for part in value.split(".")),
             f"Invalid inference module: {value!r}")
    return value


def resolve_loader(run_manifest, repo_root, explicit_module=None):
    """Find the original scorer by all four required source SHA256 values.

    The returned module name ends with ``.inference``. Relocating a package
    inside ``src`` is supported only when its critical file contents match the
    recorded manifest exactly. An explicit module never bypasses these hashes.
    This function does not import torch or load any checkpoint.
    """
    if isinstance(run_manifest, (str, Path)):
        run_manifest = json.loads(Path(run_manifest).read_text(encoding="utf-8-sig"))
    _require(isinstance(run_manifest, dict), "Run manifest must be a mapping")
    stored = run_manifest.get("source_sha256")
    _require(isinstance(stored, dict), "Run manifest has no source_sha256 mapping")
    expected = {}
    for filename in CRITICAL_SOURCE_FILES:
        digest = stored.get(filename)
        _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                 f"Run manifest lacks a valid source SHA256 for {filename}")
        expected[filename] = digest

    root = Path(repo_root).resolve()
    _require((root / "src").is_dir(), f"Repository src directory not found: {root}")
    explicit = _module_name(explicit_module) if explicit_module is not None else None
    declared_value = (run_manifest.get("inference_entrypoint")
                      or run_manifest.get("config", {}).get("inference_entrypoint"))
    declared = _module_name(declared_value) if declared_value else None
    # Original training packages may be nested beneath an extra folder, and
    # exact historical scorers may be bundled privately with this evaluator.
    # Discover all importable inference modules beneath src; identity still
    # requires all four original file hashes, never just an architecture name.
    candidate_paths = set((root / "src").rglob("inference.py"))
    if explicit is not None:
        # An explicit module is a path preference, never a hash bypass.
        explicit_path = root.joinpath(*explicit.split(".")).with_suffix(".py")
        if explicit_path.is_file():
            candidate_paths.add(explicit_path)
    candidates = sorted(candidate_paths)
    matches, rejected = [], {}
    for path in candidates:
        name = ".".join(path.relative_to(root).with_suffix("").parts)
        if not all(_importable_component(part) for part in name.split(".")):
            continue
        if explicit is not None and name != explicit:
            continue
        actual = {}
        for filename in CRITICAL_SOURCE_FILES:
            source = path.parent / filename
            if source.is_file():
                actual[filename] = _file_sha256(source)
        differences = [key for key in expected if actual.get(key) != expected[key]]
        if differences:
            rejected[name] = differences
        else:
            matches.append((name, path, actual))
    if not matches:
        detail = "; ".join(f"{name}: {', '.join(files)}" for name, files in rejected.items())
        ablation = run_manifest.get("config", {}).get("ablation_id", "unknown")
        raise ValueError(f"No inference package matches {ablation}'s recorded source hashes. "
                         f"Searched recursively under {root / 'src'}"
                         + (f" for {explicit}" if explicit else "")
                         + (f". Mismatched files: {detail}" if detail else ""))
    matches.sort(key=lambda item: (item[0] != declared, item[0]))
    name, path, actual = matches[0]
    record = {"repo_root": root, "path": path.resolve(), "hashes": expected}
    prior = _RESOLVED_LOADERS.get(name)
    _require(prior is None or prior == record,
             f"Loader name {name} was already resolved to different source in this process")
    _RESOLVED_LOADERS[name] = record
    # Import only when metadata/scoring is requested. Namespace packages pick up
    # this path on import; a conflicting already-imported module fails below.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    audit = {
        "status": "critical_source_sha256_matched", "module": name,
        "repository_root": str(root), "inference_file": str(path.resolve()),
        "critical_files": {
            filename: {"path": str((path.parent / filename).resolve()),
                       "expected_sha256": expected[filename], "actual_sha256": actual[filename]}
            for filename in CRITICAL_SOURCE_FILES
        },
        "declared_entrypoint": declared_value, "explicit_module": explicit,
        "identical_matching_modules": [item[0] for item in matches],
        "hash_bypass": False,
        "discovery_scope": "all importable inference.py files recursively under repository src",
        "bundled_reference_scorer": ".reference_scoring." in name,
    }
    return name, audit


def _import_loader(loader_module):
    name = _module_name(loader_module)
    record = _RESOLVED_LOADERS.get(name)
    _require(record is not None, f"Call resolve_loader before importing {name}")
    for filename, expected in record["hashes"].items():
        _require(_file_sha256(record["path"].parent / filename) == expected,
                 f"Resolved scorer source changed: {filename}")
    module = importlib.import_module(name)
    _require(Path(module.__file__).resolve() == record["path"],
             f"Imported inference module comes from a different repository: {module.__file__}")
    for filename in CRITICAL_SOURCE_FILES:
        sibling_name = name.rsplit(".", 1)[0] + "." + filename[:-3]
        sibling = importlib.import_module(sibling_name)
        _require(Path(sibling.__file__).resolve() == (record["path"].parent / filename).resolve(),
                 f"Imported {filename} comes from a different repository")
    _require(callable(getattr(module, "load_aligner", None)), "Inference module has no load_aligner")
    return module


def read_checkpoint_meta(loader_module, checkpoint_path):
    """Use the source-verified package's CPU reader and release all tensors."""
    module = _import_loader(loader_module)
    checkpoints = importlib.import_module(module.__package__ + ".checkpoints")
    payload = checkpoints.read_checkpoint(Path(checkpoint_path))
    try:
        _require(isinstance(payload.get("meta"), dict), "Checkpoint has no metadata mapping")
        return deepcopy(payload["meta"])
    finally:
        del payload


def _input_path(value, config, repo_root, label):
    _require(isinstance(value, (str, Path)) and str(value), f"Missing saved path: {label}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(repo_root) / path
    if not path.exists() and config.get("root"):
        # A repository relocation is explicit and deterministic. No basename or
        # directory search is used, and file content is checked after resolution.
        old_root = Path(config["root"]).expanduser()
        if old_root.is_absolute():
            try:
                path = Path(repo_root) / path.relative_to(old_root)
            except ValueError:
                pass
    _require(path.exists(), f"Required cached input not found ({label}): {path}")
    return path.resolve()


def _floating_tensor(torch, value, shape, label):
    _require(torch.is_tensor(value) and tuple(value.shape) == tuple(shape),
             f"Invalid {label} tensor shape, expected {tuple(shape)}")
    _require(value.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64),
             f"{label} must contain floating point embeddings")
    _require(bool(torch.isfinite(value).all()), f"Nonfinite {label} embeddings")


def _presence_tensor(torch, value, shape, label):
    _require(torch.is_tensor(value) and tuple(value.shape) == tuple(shape),
             f"Invalid {label} presence-mask shape")
    _require(value.dtype == torch.bool or value.dtype in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
        f"{label} presence mask must be boolean or binary integer")
    _require(bool(((value == 0) | (value == 1)).all()), f"{label} presence mask is not binary")


def _load_go_bank(meta, go_ids, repo_root, device):
    """Load and validate cached features without creating a text encoder."""
    import torch

    config, architecture, identity = meta["config"], meta["architecture"], meta["data_identity"]
    representation = architecture["go_representation"]
    key = "single_text_bank_path" if representation == "single_text" else "mzsgo_frozen_bank"
    path = _input_path(config.get(key), config, repo_root, key)
    _require(path.is_file(), f"GO cache is not a file: {path}")
    canonical = _input_path(config.get("go_text_path") or config.get("canonical_go_jsonl"),
                            config, repo_root, "go_text_path")
    canonical_sha = _file_sha256(canonical)
    stored_hashes = identity.get("input_sha256", {})
    _require(canonical_sha == stored_hashes.get("canonical_text"),
             "Canonical GO text SHA256 differs from checkpoint")
    bank_sha = _file_sha256(path)
    expected_bank_sha = stored_hashes.get("frozen_bank") if representation == "segments" else None
    if expected_bank_sha is not None:
        _require(bank_sha == expected_bank_sha, "Frozen segmented GO bank SHA256 differs from checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(_file_sha256(path) == bank_sha, "GO cache changed while being loaded")
    _require(isinstance(payload, dict), "GO cache payload must be a mapping")
    metadata = payload.get("metadata")
    _require(isinstance(metadata, dict), "GO cache metadata missing")
    _require(metadata.get("canonical_jsonl_sha256") == canonical_sha,
             "GO cache canonical text SHA256 mismatch")
    _require(metadata.get("model") == config.get("encoder_name") and config.get("encoder_name"),
             "GO cache encoder identity differs from checkpoint")
    _require(metadata.get("normalized") is False, "Expected unnormalized frozen GO embeddings")
    raw_ids = payload.get("go_ids")
    _require(isinstance(raw_ids, (list, tuple)) and raw_ids, "GO cache IDs missing")
    _require(all(type(g) is int and g > 0 for g in raw_ids), "GO cache IDs must be positive integers")
    full_ids = list(raw_ids)
    _require(len(full_ids) == len(set(full_ids)), "Duplicate GO IDs in cache")
    lookup = {g: row for row, g in enumerate(full_ids)}
    missing = sorted(set(go_ids) - set(lookup))
    _require(not missing, f"Active GO IDs missing from cached bank: {missing[:20]}")
    index = torch.tensor([lookup[g] for g in go_ids], dtype=torch.long)
    n, dim = len(full_ids), int(architecture["d_g"])
    bank = {}
    if representation == "single_text":
        _require(payload.get("format") == "mzsgo_frozen_single_text_v1",
                 "Single-text scorer requires its independent single-text GO cache")
        _require(metadata == identity.get("single_text_bank_metadata"),
                 "Single-text GO cache metadata differs from checkpoint data identity")
        _require(metadata.get("representation") == "single_text"
                 and metadata.get("text_fields") == ["name", "definition"]
                 and metadata.get("parent_text_included") is False
                 and metadata.get("pooling") == "mean",
                 "Single-text GO cache representation or pooling policy mismatch")
        _require(metadata.get("go_single_text_max_len") == config.get("go_single_text_max_len"),
                 "Single-text GO token length differs from checkpoint")
        vectors = payload.get("text_embs")
        _floating_tensor(torch, vectors, (n, dim), "single-text")
        bank["text"] = vectors.index_select(0, index).float().contiguous().to(device)
        input_dtype = str(vectors.dtype)
    else:
        _require(payload.get("format") == "mzsgo_frozen_segment_parent_v1",
                 "Unexpected frozen segmented GO bank format")
        _require(metadata.get("segment_max_len") == config.get("go_segment_max_len"),
                 "GO segment token length differs from checkpoint")
        names = payload.get("segment_names")
        _require(isinstance(names, (list, tuple)) and len(names) == len(set(names))
                 and "name" in names and "definition" in names,
                 "GO cache requires unique name/definition segment channels")
        _require(list(names) == config.get("enabled_segments"),
                 "Frozen GO segment layout differs from checkpoint configuration")
        segments, present = payload.get("segment_embs"), payload.get("segment_present")
        parents, parents_present = payload.get("parent_embs"), payload.get("parent_present")
        _floating_tensor(torch, segments, (n, len(names), dim), "segment")
        _presence_tensor(torch, present, (n, len(names)), "segment")
        _require(torch.is_tensor(parents) and parents.ndim == 3
                 and parents.shape[0] == n and parents.shape[2] == dim,
                 "Invalid parent cache dimensions")
        _floating_tensor(torch, parents, tuple(parents.shape), "parent")
        _presence_tensor(torch, parents_present, tuple(parents.shape[:2]), "parent")
        selected = segments.index_select(0, index).float().contiguous().to(device)
        selected_present = present.index_select(0, index).bool().contiguous().to(device)
        ni, di = list(names).index("name"), list(names).index("definition")
        _require(bool(selected_present.any(dim=1).all()), "Active GO term with no present segment")
        bank.update(name=selected[:, ni], definition=selected[:, di],
                    definition_present=selected_present[:, di])
        input_dtype = str(segments.dtype)
    audit = {
        "path": str(path), "format": payload["format"], "sha256": bank_sha,
        "checkpoint_bank_sha256": expected_bank_sha,
        "checkpoint_bank_sha256_matched": True if expected_bank_sha is not None else None,
        "canonical_text_path": str(canonical), "canonical_text_sha256": canonical_sha,
        "metadata": deepcopy(metadata), "input_dtype": input_dtype,
        "scoring_dtype": "torch.float32", "cache_go_count": n,
        "active_go_count": len(go_ids), "active_go_ids_sha256": _canonical_hash(go_ids),
        "encoder_initialized": False, "cache_created": False,
        "single_text_embedding_bytes_fingerprinted_at_training": False if representation == "single_text" else None,
    }
    return bank, audit


def _check_scoring_contract(meta, cohort):
    _require(isinstance(meta, dict), "Expected checkpoint metadata must be a mapping")
    for key in ("config", "architecture", "data_identity"):
        _require(isinstance(meta.get(key), dict), f"Checkpoint {key} missing")
    cfg, arch, identity = meta["config"], meta["architecture"], meta["data_identity"]
    _require(meta.get("epoch_complete") is True, "Checkpoint must represent a completed epoch")
    branch = cohort["summary"]["branch"]
    _require(identity.get("branch") == branch and cfg.get("branch") == branch,
             "Checkpoint branch differs from cohort")
    _require(isinstance(cfg.get("ablation_id"), str) and cfg["ablation_id"], "Missing checkpoint ablation ID")
    _require(arch.get("go_representation") in ("single_text", "segments"),
             "Unsupported GO representation")
    for key in ("go_representation", "protein_mode", "segment_aggregation"):
        _require(cfg.get(key) == arch.get(key), f"Saved config/architecture disagree on {key}")
    _require(float(arch.get("parent_strength", math.nan)) == 0.0
             and float(cfg.get("parent_strength", math.nan)) == 0.0,
             "This evaluator supports parent-disabled ablations only")
    _require(float(cfg.get("pbr_lambda", math.nan)) == 0.0,
             "This evaluator is scoped to the completed PBR-disabled ablations")
    _require(type(cfg.get("eval_batch_size")) is int and cfg["eval_batch_size"] > 0,
             "Saved eval_batch_size must be a positive integer")
    _require(type(cfg.get("fp16")) is bool, "Saved fp16 flag must be boolean")
    protein_ids, go_ids, labels = cohort["protein_ids"], cohort["go_ids"], cohort["labels_full"]
    _require(protein_ids and len(protein_ids) == len(set(protein_ids)), "Invalid validation protein order")
    _require(go_ids and go_ids == sorted(set(go_ids)), "Active GO IDs must be unique and ascending")
    _require(set(labels) == set(protein_ids), "Full label map must contain every validation protein")
    active = set(go_ids)
    for pid in protein_ids:
        _require(labels[pid] and len(labels[pid]) == len(set(labels[pid]))
                 and set(labels[pid]) <= active,
                 f"Invalid full active validation labels for {pid}")
    _require(identity.get("active_go_ids_sha256") == _canonical_hash(go_ids),
             "Checkpoint active GO universe differs from cohort")
    _require(identity.get("n_val_proteins") == len(protein_ids),
             "Checkpoint validation protein count differs from full cohort")
    _require(identity.get("n_active_go") == len(go_ids),
             "Checkpoint active GO count differs from cohort")


def score_checkpoint(checkpoint_path, loader_module, cohort, device, repo_root,
                     expected_meta, log_every=10):
    """Return raw float32 cosine scores in exact full-validation/active-GO order.

    CPU and CUDA are supported. CUDA reproduces the original bf16 autocast flag,
    including the learned segment gate. No temperature, sigmoid, threshold,
    parent adjustment, or loss is applied to the returned ranking scores.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    _check_scoring_contract(expected_meta, cohort)
    module = _import_loader(loader_module)
    record = _RESOLVED_LOADERS[_module_name(loader_module)]
    _require(Path(repo_root).resolve() == record["repo_root"], "Scoring repository differs from resolved loader")
    device = torch.device(device)
    _require(device.type in ("cpu", "cuda"), "Only CPU/CUDA scorer precision contracts are supported")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA device requested but unavailable")
    _require(type(log_every) is int and log_every >= 0, "log_every must be a nonnegative integer")
    cfg, arch = expected_meta["config"], expected_meta["architecture"]
    use_bf16 = cfg["fp16"] and device.type == "cuda"
    if use_bf16:
        with torch.cuda.device(device):
            _require(torch.cuda.is_bf16_supported(), "Saved run requires CUDA bfloat16 support")
    started = time.perf_counter()
    model, loaded_meta = module.load_aligner(
        checkpoint_path, device=device, expected_branch=cohort["summary"]["branch"],
        expected_ablation=cfg["ablation_id"],
    )
    _require(loaded_meta == expected_meta, "Checkpoint metadata changed between audit and model load")
    _require(model.architecture_metadata() == arch, "Loaded scorer architecture differs from checkpoint")
    _require(not model.training, "Inference loader did not put model in evaluation mode")
    bank, bank_audit = _load_go_bank(loaded_meta, cohort["go_ids"], repo_root, device)

    # These are the original repository classes. Importing the encoder/data
    # preparation pipeline is unnecessary and can otherwise trigger downloads.
    from src.datasets.residue_store import ESMResidueStore
    from src.training.collate import ContrastiveEmbCollator

    embed_dir = _input_path(cfg.get("embed_dir_res"), cfg, repo_root, "embed_dir_res")
    _require(embed_dir.is_dir(), f"Residue embedding directory not found: {embed_dir}")
    for key in ("max_len", "overlap"):
        _require(type(cfg.get(key)) is int and cfg[key] >= (1 if key == "max_len" else 0),
                 f"Invalid saved ESM store option {key}")
    _require(cfg["overlap"] < cfg["max_len"], "ESM store overlap must be smaller than max_len")
    store = ESMResidueStore(embed_dir=embed_dir, max_len=cfg["max_len"],
                            overlap=cfg["overlap"], prefer_fp16=cfg["fp16"])
    protein_ids = list(cohort["protein_ids"])

    class FullValidationDataset(Dataset):
        def __len__(self):
            return len(protein_ids)

        def __getitem__(self, index):
            pid = protein_ids[index]
            embeddings = store.get(pid)
            _require(torch.is_tensor(embeddings) and embeddings.ndim == 2
                     and embeddings.shape[0] > 0 and embeddings.shape[1] == int(arch["d_h"]),
                     f"Invalid residue embedding dimensions for {pid}")
            _require(embeddings.is_floating_point() and bool(torch.isfinite(embeddings).all()),
                     f"Invalid residue embedding values for {pid}")
            return {"protein_id": pid, "prot_emb": embeddings,
                    "pos_go_ids": torch.tensor(cohort["labels_full"][pid], dtype=torch.long)}

    generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 1)
    batches = DataLoader(FullValidationDataset(), batch_size=cfg["eval_batch_size"],
                         shuffle=False, drop_last=False, num_workers=0,
                         pin_memory=False, generator=generator, collate_fn=ContrastiveEmbCollator())
    result = np.empty((len(protein_ids), len(cohort["go_ids"])), dtype=np.float32)
    previous_precision = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    position = 0
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                    dtype=torch.bfloat16, enabled=use_bf16):
            # eval_epoch computes these once under the same autocast context.
            weights = (None if arch["go_representation"] == "single_text" else
                       model.aggregation_weights(bank["name"], bank["definition"],
                                                 bank["definition_present"]))
            for batch_number, batch in enumerate(batches, start=1):
                ids = batch["protein_ids"]
                end = position + len(ids)
                _require(ids == protein_ids[position:end], "Validation batch order changed")
                H = batch["prot_emb_pad"].to(device, non_blocking=True).float()
                mask = batch["prot_attn_mask"].to(device, non_blocking=True).bool()
                if arch["go_representation"] == "single_text":
                    scores = model.score_single_text(H, mask, bank["text"])
                else:
                    scores = model.score_fine_grained(
                        H, mask, bank["name"], bank["definition"], bank["definition_present"],
                        None, None, None, segment_weights=weights,
                    )
                _require(tuple(scores.shape) == (len(ids), len(cohort["go_ids"])),
                         "Scorer returned the wrong protein/GO dimensions")
                values = scores.detach().float().cpu().numpy()
                _require(np.isfinite(values).all(), "Scorer returned nonfinite values")
                result[position:end] = values
                position = end
                if log_every and (batch_number % log_every == 0 or position == len(protein_ids)):
                    _LOG.info("%s scored %d/%d validation proteins, batch %d/%d",
                              cfg["ablation_id"], position, len(protein_ids), batch_number, len(batches))
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    _require(position == len(protein_ids), "Incomplete validation scoring")
    runtime = {
        "loader_module": module.__name__, "device": str(device),
        "torch": torch.__version__, "numpy": np.__version__,
        "n_validation_proteins": len(protein_ids), "n_active_go": len(cohort["go_ids"]),
        "protein_ids_order_sha256": _canonical_hash(protein_ids),
        "active_go_ids_sha256": _canonical_hash(cohort["go_ids"]),
        "eval_batch_size": cfg["eval_batch_size"], "n_batches": len(batches),
        "shuffle": False, "drop_last": False, "num_workers": 0, "pin_memory": False,
        "all_validation_proteins_scored": True, "labels": "all_source_validation_positives_preserved",
        "scores": "raw_original_scorer_float32", "autocast_enabled": use_bf16,
        "autocast_dtype": "torch.bfloat16" if use_bf16 else None,
        "float32_matmul_precision": "high", "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True, "window_layout": arch.get("window_layout"),
        "window_reduction": arch.get("window_reduction"),
        "go_representation": arch["go_representation"], "parent_strength": 0.0,
        "segment_weights_computed_under_scoring_autocast": arch["go_representation"] == "segments",
        "residue_store": {"class": "src.datasets.residue_store.ESMResidueStore",
                          "embed_dir": str(embed_dir), "max_len": cfg["max_len"],
                          "overlap": cfg["overlap"], "prefer_fp16": cfg["fp16"]},
        "go_bank": bank_audit, "elapsed_seconds": time.perf_counter() - started,
        "encoder_initialized": False, "network_used": False,
    }
    return result, deepcopy(loaded_meta), runtime
