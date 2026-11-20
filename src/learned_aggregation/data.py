"""MZSGO split and bank loading shared by all three ontology branches."""
from __future__ import annotations

import hashlib
import json
import logging
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


def prepare_data(a, device):
    # These are established repository data interfaces, not a second branch pipeline.
    from src.configs.data_classes import FewZeroConfig
    from src.datasets import ESMResidueStore, GoTextStore
    from src.datasets.protein_dataset import ProteinEmbDataset
    from src.encoders import BioMedBERTEncoder
    from src.training.collate import ContrastiveEmbCollator
    from src.utils.helpers import load_go_texts_canonical

    log = logging.getLogger("data")
    inputs = {
        "train_ids": a.train_ids_path, "val_ids": a.val_ids_path,
        "annotations": a.pid2pos, "candidate_ids": a.branch_go_ids_path,
        "temporal_ids": a.temporal_go_ids_path, "canonical_text": a.go_text_path,
        "frozen_bank": a.mzsgo_frozen_bank, "original_train_ids": a.original_train_ids_path,
        "original_valid_ids": a.original_valid_ids_path, "temporal_test_ids": a.temporal_test_ids_path,
    }
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
    bank = torch.load(a.mzsgo_frozen_bank, map_location="cpu", weights_only=False)
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
    go_cache = SimpleNamespace(embs=torch.empty((0, dg)), row2id=bank_ids,
                               id2row={g: i for i, g in enumerate(bank_ids)})
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
    id2text, id2segments, id2present = load_go_texts_canonical(
        str(a.go_text_path), phase=a.phase, return_segments=True,
        enabled_segments=a.enabled_segments,
    )
    text_store = GoTextStore(
        {a.phase: id2text}, encoder.tokenizer, phase=a.phase, lazy=True,
        max_len=a.go_text_store_max_len, is_segmented=True,
        segment_max_len=a.go_segment_max_len,
        full_id2segments={a.phase: id2segments}, full_id2seg_present={a.phase: id2present},
        segment_names=list(a.enabled_segments),
    )
    text_store.materialize_tokens_once(batch_size=512, show_progress=True)
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
    return SimpleNamespace(
        go_cache=go_cache, encoder=encoder, text_store=text_store, residue_store=residue_store,
        train_ds=train_ds, val_ds=val_ds, train_loader=train_loader, val_loader=val_loader,
        generators=generators, active_ids=active, train_counts=dict(counts), seen=seen,
        rare=rare, zero=zero, d_h=dh, d_g=dg, identity=identity,
    )
