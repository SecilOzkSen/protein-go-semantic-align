"""Reconstruct the actual scorer, rather than infer architecture from tensor names.

This is a model loader, not a replacement for the benchmark's general/temporal
split preparation, score transform, ground truth or metric definitions.
"""
from __future__ import annotations

import torch
from torch import nn

from .aligner import ProteinGoAligner
from .checkpoints import read_checkpoint


class _FrozenBankOnlyEncoder(nn.Module):
    """Scoring uses supplied frozen embeddings; live text encoding is unavailable."""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def forward(self, *args, **kwargs):
        raise RuntimeError("Use the matching precomputed frozen GO bank for this inference loader")


def load_aligner(checkpoint_path, *, device="cpu", expected_branch=None, go_encoder=None):
    """Return (model, metadata); score with model.score_fine_grained(...).

    If go_encoder is omitted, only its frozen, unused weights are omitted from
    loading. All scoring tensors, including the learned gate, load strictly.
    No new encoder download is needed to score an existing frozen bank.
    """
    checkpoint = read_checkpoint(checkpoint_path)
    meta = checkpoint["meta"]
    architecture = meta["architecture"]
    if architecture.get("architecture_version") != "mzsgo_learned_aggregation_v1":
        raise RuntimeError("Unsupported scorer architecture")
    if expected_branch is not None and str(expected_branch).upper() != meta["data_identity"]["branch"]:
        raise RuntimeError("Checkpoint branch differs from the requested evaluation branch")
    missing_encoder = go_encoder is None
    encoder = go_encoder if go_encoder is not None else _FrozenBankOnlyEncoder(meta["config"]["encoder_name"])
    model = ProteinGoAligner(
        d_h=architecture["d_h"], d_g=architecture["d_g"], d_z=architecture["d_z"],
        go_encoder=encoder, normalize=architecture["normalize"],
        protein_mode=architecture["protein_mode"], segment_aggregation=architecture["segment_aggregation"],
        window_length=architecture["window_length"], window_stride=architecture["window_stride"],
        mil_tau=architecture["mil_tau"], parent_strength=architecture["parent_strength"],
        go_chunk_size=architecture["go_chunk_size"],
    )
    if model.architecture_metadata() != architecture:
        raise RuntimeError("Reconstructed model architecture differs from checkpoint")
    state = checkpoint["model"]
    if missing_encoder:
        state = {k: v for k, v in state.items() if not k.startswith("go_encoder.")}
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, meta
