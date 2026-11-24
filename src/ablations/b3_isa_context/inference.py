"""Reconstruct the actual scorer, rather than infer architecture from tensor names.

This is a model loader, not a replacement for the benchmark's general/temporal
split preparation, score transform, ground truth or metric definitions.
"""
from __future__ import annotations

import torch
from torch import nn

from .aligner import ProteinGoAligner
from .checkpoints import read_checkpoint
from .config import PACKAGE_VARIANT
from .isa_context import ISA_CONTEXT_POLICY


class _FrozenBankOnlyEncoder(nn.Module):
    """Scoring uses supplied frozen embeddings; live text encoding is unavailable."""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def forward(self, *args, **kwargs):
        raise RuntimeError("Use the matching precomputed frozen GO bank for this inference loader")


def load_aligner(checkpoint_path, *, device="cpu", expected_branch=None, expected_ablation=None, go_encoder=None):
    """Return (model, metadata); use the scorer matching go_representation.

    B3-ISA uses score_fine_grained with the frozen name, definition and is_a
    bank rows. Pass isa_context_raw and isa_context_present explicitly to both
    aggregation_weights and score_fine_grained. Missing context inputs fail.

    If go_encoder is omitted, only its frozen, unused weights are omitted from
    loading. All scoring tensors, including the learned gate, load strictly.
    No new encoder download is needed to score an existing frozen bank.
    """
    checkpoint = read_checkpoint(checkpoint_path)
    meta = checkpoint["meta"]
    architecture = meta["architecture"]
    expected = expected_ablation if expected_ablation is not None else PACKAGE_VARIANT
    if expected is not None and meta["config"].get("ablation_id") != expected:
        raise RuntimeError(f"Checkpoint ablation differs from requested {expected}")
    if architecture.get("architecture_version") != "mzsgo_mf_b3_isa_context_v1":
        raise RuntimeError("Unsupported scorer architecture")
    if architecture.get("isa_context_enabled") is not True or architecture.get("isa_context_policy") != ISA_CONTEXT_POLICY:
        raise RuntimeError("Checkpoint lacks the canonical cached is_a semantic context contract")
    if expected_branch is not None and str(expected_branch).upper() != meta["data_identity"]["branch"]:
        raise RuntimeError("Checkpoint branch differs from the requested evaluation branch")
    missing_encoder = go_encoder is None
    encoder = go_encoder if go_encoder is not None else _FrozenBankOnlyEncoder(meta["config"]["encoder_name"])
    model = ProteinGoAligner(
        d_h=architecture["d_h"], d_g=architecture["d_g"], d_z=architecture["d_z"],
        go_encoder=encoder, normalize=architecture["normalize"],
        protein_mode=architecture["protein_mode"], segment_aggregation=architecture["segment_aggregation"],
        go_representation=architecture.get("go_representation", "segments"),
        window_length=architecture["window_length"], window_stride=architecture["window_stride"],
        mil_tau=architecture["mil_tau"], parent_strength=architecture["parent_strength"],
        go_chunk_size=architecture["go_chunk_size"],
        isa_context_enabled=architecture["isa_context_enabled"],
    )
    if model.architecture_metadata() != architecture:
        raise RuntimeError("Reconstructed model architecture differs from checkpoint")
    state = checkpoint["model"]
    if missing_encoder:
        state = {k: v for k, v in state.items() if not k.startswith("go_encoder.")}
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, meta
