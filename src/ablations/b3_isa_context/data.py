"""MZSGO split and bank loading shared by all three ontology branches."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def go_int(value):
    text = str(value).strip()
    value = int(text.split(":", 1)[1] if text.upper().startswith("GO:") else text)
    if value < 1:
        raise ValueError(f"Invalid GO ID: {text}")
    return value


def read_protein_ids(path):
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate protein IDs in {path}")
    return ids


def read_go_ids(path):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("ids", data.get("go_ids", data.get("terms", data)))
    ids = [go_int(x) for x in data]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate GO IDs in {path}")
    return ids


def _tensor_sha256(*tensors):
    """Fingerprint actual tokenizer output, including masks and tensor shapes."""
    digest = hashlib.sha256()
    for tensor in tensors:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def load_or_build_single_text_bank(
        *, encoder, text_store, go_ids, canonical_path, model_name,
        max_length, cache_path, device, batch_size=256, expected_dim=None):
    """Encode one joined name+definition sequence per GO, never segment means.

    The canonical loader supplies markerless present name/definition fields in
    that order, joined by a newline. The entire sequence is tokenized once,
    truncated at ``max_length`` (including special tokens), and pooled by the
    same frozen BioMedBERT masked-mean encoder used for the segmented banks.
    This conservative 128-token default is a different tokenization policy from
    two independent 64-token segments, and is recorded rather than hidden.

    Cache metadata fingerprints the actual strings and token IDs as well as
    the source/model/pooling policy. An incompatible existing cache is rejected,
    never overwritten. A per-path file lock and atomic rename protect first-run
    creation by concurrent jobs on the Linux/macOS training environments.
    """
    if bool(getattr(text_store, "is_go_segmented", False)):
        raise ValueError("The single-text bank requires an unsegmented GoTextStore")
    if getattr(encoder, "pooling_strategy", "mean") != "mean":
        raise ValueError("The single-text ablation requires frozen mean GO pooling")
    go_ids = [int(g) for g in go_ids]
    if not go_ids or len(go_ids) != len(set(go_ids)):
        raise ValueError("Single-text bank GO IDs must be nonempty and unique")
    if int(max_length) < 1 or int(batch_size) < 1:
        raise ValueError("Single-text token length and encoding batch size must be positive")
    if int(getattr(text_store, "max_len", max_length)) != int(max_length):
        raise ValueError("Single-text store token length differs from its bank policy")
    texts = []
    for gid in go_ids:
        text = text_store.id2text.get(gid)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"Missing combined name+definition text for GO:{gid:07d}")
        texts.append([gid, text])
    toks = text_store.batch(go_ids)
    if any(key.startswith("seg_") for key in toks):
        raise RuntimeError("Segment tokens must not enter the single-text bank")
    ids, mask = toks["input_ids"].cpu(), toks["attention_mask"].cpu()
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape != (len(go_ids), int(max_length)):
        raise RuntimeError("Single-text tokens must have shape [GO, single_text_max_len]")
    if (mask.sum(1) == 0).any():
        raise RuntimeError("Single-text token bank contains an empty attention mask")
    model_config = getattr(getattr(encoder, "model", None), "config", None)
    tokenizer = text_store.tokenizer
    metadata = {
        "canonical_jsonl_sha256": file_sha256(canonical_path),
        "model": str(model_name),
        "model_revision": getattr(model_config, "_commit_hash", None),
        "representation": "single_text",
        "text_policy": "canonical_markerless_name_newline_definition_v1",
        "text_fields": ["name", "definition"],
        "parent_text_included": False,
        "pooling": "mean", "normalized": False,
        "go_single_text_max_len": int(max_length),
        "tokenizer": str(getattr(tokenizer, "name_or_path", model_name)),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "right")),
        "padding_side": str(getattr(tokenizer, "padding_side", "right")),
        "combined_text_sha256": hashlib.sha256(
            json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "token_sha256": _tensor_sha256(ids, mask),
    }

    def validate(payload):
        if not isinstance(payload, dict) or payload.get("format") != "mzsgo_frozen_single_text_v1":
            raise RuntimeError("Expected a true single-text GO cache, not a segmented bank")
        if payload.get("metadata") != metadata:
            raise RuntimeError("Single-text GO cache metadata mismatch; use a new cache path/output directory")
        if payload.get("go_ids") != go_ids:
            raise RuntimeError("Single-text GO cache candidate IDs/order mismatch")
        vectors = payload.get("text_embs")
        if not torch.is_tensor(vectors) or vectors.ndim != 2 or vectors.shape[0] != len(go_ids):
            raise RuntimeError("Single-text GO bank must contain one vector per GO")
        if expected_dim is not None and vectors.shape[1] != int(expected_dim):
            raise RuntimeError("Single-text GO embedding dimension differs from the reference bank")
        if not torch.isfinite(vectors).all():
            raise RuntimeError("Nonfinite single-text GO embeddings")
        return payload

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with cache_path.with_suffix(cache_path.suffix + ".lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache_path.exists():
            return validate(torch.load(cache_path, map_location="cpu", weights_only=False))
        encoder.requires_grad_(False)
        encoder.eval()
        chunks = []
        for start in range(0, len(go_ids), int(batch_size)):
            stop = min(start + int(batch_size), len(go_ids))
            out = encoder(input_ids=ids[start:stop].to(device),
                          attention_mask=mask[start:stop].to(device), output_mode="pooled")
            if isinstance(out, tuple):
                out = out[0]
            elif isinstance(out, dict):
                out = out.get("pooled", out.get("pooler_output"))
            if not torch.is_tensor(out) or out.ndim != 2 or out.shape[0] != stop - start:
                raise RuntimeError("Unexpected frozen single-text GO encoder output")
            chunks.append(out.detach().float().cpu())
        payload = validate({"format": "mzsgo_frozen_single_text_v1", "go_ids": go_ids,
                            "text_embs": torch.cat(chunks, dim=0).contiguous(), "metadata": metadata})
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+b", dir=cache_path.parent,
                                             prefix=cache_path.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, cache_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        logging.getLogger("data").info("[GO-SINGLE-TEXT-BANK] created %s shape=%s",
                                       cache_path, tuple(payload["text_embs"].shape))
        return payload


def prepare_data(a, device):
    # These are established repository data interfaces, not a second branch pipeline.
    from src.configs.data_classes import FewZeroConfig
    from src.datasets import ESMResidueStore, GoTextStore
    from src.datasets.protein_dataset import ProteinEmbDataset
    from src.encoders import BioMedBERTEncoder
    from src.training.collate import ContrastiveEmbCollator
    from src.utils.helpers import load_go_texts_canonical

    log = logging.getLogger("data")
    representation = str(getattr(a, "go_representation", "segments"))
    if representation not in {"single_text", "segments"}:
        raise ValueError("go_representation must be single_text or segments")
    reference_bank = Path(a.mzsgo_frozen_bank) if getattr(a, "mzsgo_frozen_bank", None) else None
    inputs = {
        "train_ids": a.train_ids_path, "val_ids": a.val_ids_path,
        "annotations": a.pid2pos, "candidate_ids": a.branch_go_ids_path,
        "temporal_ids": a.temporal_go_ids_path, "canonical_text": a.go_text_path,
        "original_train_ids": a.original_train_ids_path,
        "original_valid_ids": a.original_valid_ids_path, "temporal_test_ids": a.temporal_test_ids_path,
    }
    # B0 can be prepared without a segmented cache. When the reference exists,
    # validate its universe/dimension, but never use its segment vectors to score.
    if representation == "segments" or (reference_bank is not None and reference_bank.is_file()):
        inputs["frozen_bank"] = reference_bank
    missing = [str(p) for p in inputs.values() if not Path(p).is_file()]
    if not a.embed_dir_res.is_dir():
        missing.append(str(a.embed_dir_res))
    if missing:
        raise FileNotFoundError("Missing experiment inputs: " + ", ".join(missing))
    train_ids = read_protein_ids(a.train_ids_path)
    val_ids = read_protein_ids(a.val_ids_path)
    train, val = set(train_ids), set(val_ids)
    if not train or not val or train & val:
        raise RuntimeError("Training and validation sets must be nonempty and disjoint")
    original_train = set(read_protein_ids(a.original_train_ids_path))
    original_val = set(read_protein_ids(a.original_valid_ids_path))
    if original_train & original_val:
        raise RuntimeError("Original train/valid ID files overlap")
    if train | val != original_train | original_val:
        raise RuntimeError("Configured train/val split does not cover the original training pool")
    temporal_test = set(read_protein_ids(a.temporal_test_ids_path))
    if (train | val) & temporal_test:
        raise RuntimeError("Train/validation overlaps temporal test protein IDs")

    candidate = set(read_go_ids(a.branch_go_ids_path))
    temporal = set(read_go_ids(a.temporal_go_ids_path))
    if not candidate or not temporal.issubset(candidate):
        raise RuntimeError("Temporal GO IDs must be a subset of nonempty candidate IDs")
    active = sorted(candidate - temporal)
    if not active:
        raise RuntimeError("No non-temporal GO candidates remain")
    active_set = set(active)
    with a.pid2pos.open(encoding="utf-8") as f:
        raw_labels = json.load(f)
    pid2pos = {}
    for pid in train | val:
        if pid not in raw_labels:
            raise RuntimeError(f"Missing annotation entry: {pid}")
        if not isinstance(raw_labels[pid], list):
            raise ValueError(f"Expected a GO list for {pid}")
        positives = sorted({go_int(x) for x in raw_labels[pid]})
        if not positives or not set(positives) <= active_set:
            raise RuntimeError(f"Empty or out-of-universe annotations for {pid}")
        pid2pos[pid] = positives
    counts = Counter()
    for pid in train_ids:
        counts.update(pid2pos[pid])
    seen = sorted(counts)
    rare = sorted(g for g, n in counts.items() if n < a.rare_lt)
    # Preserve the original few/zero config, including temporal IDs outside active.
    zero = sorted(candidate - set(counts))
    zero_set = set(zero)

    hashes = {name: file_sha256(path) for name, path in inputs.items()}
    dg, bank_ids = None, sorted(candidate)
    if "frozen_bank" in inputs:
        bank = torch.load(reference_bank, map_location="cpu", weights_only=False)
        if bank.get("format") != "mzsgo_frozen_segment_parent_v1":
            raise RuntimeError("Unexpected frozen GO bank format")
        bank_ids = [int(g) for g in bank["go_ids"]]
        if len(bank_ids) != len(set(bank_ids)) or set(bank_ids) != candidate:
            raise RuntimeError("Training bank GO IDs differ from the candidate universe")
        if bank.get("metadata", {}).get("canonical_jsonl_sha256") != hashes["canonical_text"]:
            raise RuntimeError("Frozen bank canonical text SHA256 mismatch")
        if list(bank.get("segment_names", [])) != a.enabled_segments:
            raise RuntimeError("Frozen bank segment order must be name, definition, is_a")
        dg = int(bank["segment_embs"].shape[-1])
        del bank
    log.info("[DATA] %s train=%d val=%d active_go=%d temporal_go=%d train_observed_go=%d",
             a.branch, len(train_ids), len(val_ids), len(active), len(temporal), len(seen))

    # Retain the same frozen encoder construction/tokenizer and bank validation as
    # the baseline. No GO text encoder weights receive gradients.
    encoder = BioMedBERTEncoder(
        model_name=a.encoder_name, device=device, max_length=512,
        attention_pooling_strategy=a.go_encoder_inner_pooling, attn_hidden=128,
        attn_dropout=0.1, gradient_checkpointing=False,
    )
    encoder.requires_grad_(False)
    encoder.eval()
    if representation == "single_text" and a.go_encoder_inner_pooling != "mean":
        raise ValueError("B0 single-text representation requires mean GO encoder pooling")
    text_fields = ["name", "definition"] if representation == "single_text" else a.enabled_segments
    id2text, id2segments, id2present = load_go_texts_canonical(
        str(a.go_text_path), phase=a.phase, return_segments=True,
        enabled_segments=text_fields,
    )
    if representation == "single_text":
        missing_names = [gid for gid in bank_ids if not id2present.get(gid, {}).get("name", False)]
        if missing_names:
            raise RuntimeError(f"Single-text GO terms without a canonical name: {missing_names[:20]}")
        text_store = GoTextStore(
            {a.phase: id2text}, encoder.tokenizer, phase=a.phase, lazy=True,
            max_len=int(getattr(a, "go_single_text_max_len", 128)), is_segmented=False,
            segment_names=["text"],
        )
    else:
        text_store = GoTextStore(
            {a.phase: id2text}, encoder.tokenizer, phase=a.phase, lazy=True,
            max_len=a.go_text_store_max_len, is_segmented=True,
            segment_max_len=a.go_segment_max_len,
            full_id2segments={a.phase: id2segments}, full_id2seg_present={a.phase: id2present},
            segment_names=list(a.enabled_segments),
        )
    text_store.materialize_tokens_once(batch_size=512, show_progress=True)
    single_metadata = None
    if representation == "single_text":
        cache_path = getattr(a, "single_text_bank_path", None)
        if cache_path is None:
            cache_path = Path(a.output_dir) / "go_single_text_bank.pt"
        cache_path = Path(cache_path)
        if reference_bank is not None and cache_path.resolve() == reference_bank.resolve():
            raise ValueError("Single-text cache must not overwrite the shared segmented bank")
        payload = load_or_build_single_text_bank(
            encoder=encoder, text_store=text_store, go_ids=bank_ids,
            canonical_path=a.go_text_path, model_name=a.encoder_name,
            max_length=int(getattr(a, "go_single_text_max_len", 128)),
            cache_path=cache_path, device=device,
            batch_size=int(getattr(a, "eval_go_bs", 256)), expected_dim=dg,
        )
        dg = int(payload["text_embs"].shape[1])
        single_metadata = payload["metadata"]
        go_cache = SimpleNamespace(
            embs=payload["text_embs"], row2id=bank_ids,
            id2row={g: i for i, g in enumerate(bank_ids)},
            representation="single_text", metadata=single_metadata, cache_path=cache_path,
        )
    else:
        go_cache = SimpleNamespace(embs=torch.empty((0, dg)), row2id=bank_ids,
                                  id2row={g: i for i, g in enumerate(bank_ids)})
    residue_store = ESMResidueStore(embed_dir=a.embed_dir_res, max_len=a.max_len,
                                    overlap=a.overlap, prefer_fp16=a.fp16)
    missing_proteins = [p for p in train_ids + val_ids if not residue_store.has(p)]
    if missing_proteins:
        raise RuntimeError(f"Missing {len(missing_proteins)} ESM-2 embeddings, e.g. {missing_proteins[:10]}")
    fewzero = FewZeroConfig(zero_shot_terms=zero_set, few_shot_terms=set(rare), fs_target_ratio=0.0)
    # The supplied baseline dataset removes train-unobserved positives, including
    # validation-only GO terms. Keep that protocol explicit for this ablation.
    # Changing it requires re-evaluating both fixed and learned models identically.
    common = dict(pid2pos=pid2pos, go_text_store=text_store, fewzero=fewzero,
                  dag_parents=None, store=residue_store, exclude_zero_shot_positives=True)
    train_ds = ProteinEmbDataset(protein_ids=train_ids, **common)
    val_ds = ProteinEmbDataset(protein_ids=val_ids, **common)
    if len(train_ds) != len(train_ids) or len(val_ds) != len(val_ids):
        raise RuntimeError("Dataset silently removed configured proteins")
    for ds, ids in ((train_ds, train_ids), (val_ds, val_ids)):
        if list(ds.pids) != ids:
            raise RuntimeError("Dataset changed the specified protein ordering")
        for pid in ids:
            expected = set(pid2pos[pid]) - zero_set
            if set(ds.pid2pos[pid]) != expected:
                raise RuntimeError(f"Dataset unexpectedly changed GO annotations for {pid}")
    val_pairs_source = sum(len(pid2pos[p]) for p in val_ids)
    val_pairs_used = sum(len(val_ds.pid2pos[p]) for p in val_ids)
    log.info("[VALIDATION-PROTOCOL] legacy train-observed positives: source_pairs=%d used_pairs=%d filtered_pairs=%d",
             val_pairs_source, val_pairs_used, val_pairs_source - val_pairs_used)

    # Gate dropout must not change the order of proteins in later epochs.
    # These generators are included in epoch-end resume checkpoints.
    generators = {"train": torch.Generator().manual_seed(a.seed),
                  "val": torch.Generator().manual_seed(a.seed + 1)}
    collate = ContrastiveEmbCollator()
    train_loader = DataLoader(
        train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
        # Restarting workers each epoch makes epoch-boundary resume reproducible.
        persistent_workers=False, generator=generators["train"],
        pin_memory=device.type == "cuda", collate_fn=collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=a.eval_batch_size, shuffle=False, num_workers=0,
        generator=generators["val"], pin_memory=False, collate_fn=collate, drop_last=False,
    )
    dh = int(train_ds[0]["prot_emb"].shape[1])
    identity = {
        "branch": a.branch, "input_sha256": hashes,
        "active_go_ids_sha256": hashlib.sha256(json.dumps(active, separators=(",", ":")).encode()).hexdigest(),
        "n_train_proteins": len(train_ids), "n_val_proteins": len(val_ids),
        "n_active_go": len(active), "n_temporal_go": len(temporal),
        "n_observed_train_go": len(seen), "n_train_positive_pairs": sum(counts.values()),
        "split_shuffle_policy": "dedicated_train_generator",
        "validation_label_policy": "legacy_exclude_train_unobserved_positives",
        "n_val_source_positive_pairs": val_pairs_source, "n_val_evaluated_positive_pairs": val_pairs_used,
        "n_val_filtered_positive_pairs": val_pairs_source - val_pairs_used,
    }
    if representation == "single_text":
        identity.update({"go_representation": "single_text", "go_text_fields": ["name", "definition"],
                         "score_segments": [], "single_text_bank_metadata": single_metadata})
    return SimpleNamespace(
        go_cache=go_cache, encoder=encoder, text_store=text_store, residue_store=residue_store,
        train_ds=train_ds, val_ds=val_ds, train_loader=train_loader, val_loader=val_loader,
        generators=generators, active_ids=active, train_counts=dict(counts), seen=seen,
        rare=rare, zero=zero, d_h=dh, d_g=dg, identity=identity,
    )
