from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.encoders import BioMedBERTEncoder
from src.models.projection import ProjectionHead
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ProteinGoGateContext:
    """GO-only gate preparation; never an actual protein/GO weight matrix."""

    hidden: torch.Tensor
    dropout_mask: torch.Tensor
    present: torch.Tensor
    owner: int
    training: bool
    inputs: tuple
    parameter_versions: tuple


class AttentionPool1D(nn.Module):
    """Learn one residue-attention distribution for a protein."""

    def __init__(self, d_in: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.score = nn.Linear(d_in, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ):
        if x.ndim != 3:
            raise RuntimeError(
                f"AttentionPool1D expects x [B,T,D], got {tuple(x.shape)}"
            )

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask != 0
            if mask.shape != x.shape[:2]:
                raise RuntimeError(
                    f"Protein mask {tuple(mask.shape)} does not match "
                    f"protein tensor {tuple(x.shape[:2])}."
                )
            if (mask.sum(dim=1) == 0).any():
                raise RuntimeError(
                    "AttentionPool1D received a protein with zero valid residues."
                )

        logits = self.score(self.dropout(self.ln(x))).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e4)

        alpha = torch.softmax(logits.float(), dim=1).to(x.dtype)
        pooled = torch.sum(x * alpha.unsqueeze(-1), dim=1)
        return pooled, alpha


class GatedMeanAttnPool1D(nn.Module):
    """
    Single-vector protein pooling used by Retriever v2.

    The representation is a learned mixture of:
        1. masked mean pooling
        2. learned residue-attention pooling

        h = (1 - gate) * h_mean + gate * h_attn

    Retained for state-dict and initialization compatibility. The current
    local/global-mean scorers do not train or use this attention pool.
    """

    def __init__(self, d_in: int, dropout: float = 0.05):
        super().__init__()

        self.attn_pool = AttentionPool1D(
            d_in=d_in,
            dropout=dropout,
        )

        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_in),
            nn.Linear(2 * d_in, d_in // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 4, 1),
        )

        # Start close to mean pooling.
        nn.init.constant_(self.gate[-1].bias, -2.0)

    @staticmethod
    def masked_mean(
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            eps: float = 1e-6,
    ) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)

        if mask.dtype != torch.bool:
            mask = mask != 0

        if (mask.sum(dim=1) == 0).any():
            raise RuntimeError(
                "GatedMeanAttnPool1D received a protein with zero valid residues."
            )

        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(eps)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ):
        h_mean = self.masked_mean(x, mask)
        h_attn, alpha = self.attn_pool(x, mask)

        gate_in = torch.cat([h_mean, h_attn], dim=-1)
        gate = torch.sigmoid(self.gate(gate_in))

        pooled = (1.0 - gate) * h_mean + gate * h_attn

        info = {
            "protein_attn_alpha": alpha,
            "protein_attn_gate": gate,
        }
        return pooled, info


class SmoothMIL(nn.Module):
    """Normalized log-sum-exp across valid protein windows."""

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = float(tau)
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("mil_tau must be finite and positive")

    def forward(self, scores, valid):
        # scores [B,K,G], valid [B,K]
        if not valid.any(dim=1).all():
            raise RuntimeError("Protein without valid local windows")
        x = scores.float().masked_fill(~valid[:, :, None], -1e4)
        return self.tau * (torch.logsumexp(x / self.tau, dim=1) - valid.sum(1).float().log()[:, None])


class ProteinGoAligner(nn.Module):
    """B3's local name/definition scorer with protein-conditioned aggregation.

    Local mode scores projected protein windows against each text segment,
    reduces windows by SmoothMIL, then mixes the segment scores. Global-mean
    mode instead scores one projected mean protein vector. B3-PC adds a masked
    frozen-residue mean to the existing GO gate before its GELU. The protein
    summary is used only by the gate, never as an additional score branch.
    Parent evidence is disabled in the B3-PC release.
    Single-text mode instead scores one independently encoded combined GO text
    vector against a projected global mean protein vector, with no segments,
    gate or parent evidence.

    Module names and creation order match the supplied local baseline so that
    identical seeds/shared states reproduce its initialization. Legacy pooling
    and fusion modules are retained only for state-dict compatibility.
    """

    def __init__(
        self,
        d_h: int,
        d_g: Optional[int] = None,
        d_z: int = 768,
        go_encoder: Optional[BioMedBERTEncoder] = None,
        normalize: bool = True,
        protein_pool_type: str = "mean_attn_gate",
        go_pool_type: str = "mean",
        go_segment_representation_mode: str = "segments_only",
        *,
        protein_mode: str = "local",
        segment_aggregation: str = "learned",
        go_representation: str = "segments",
        window_length: int = 128,
        window_stride: int = 64,
        mil_tau: float = 0.1,
        parent_strength: float = 0.15,
        go_chunk_size: int = 512,
        protein_conditioning: bool = True,
    ):
        super().__init__()
        if protein_mode not in {"local", "global_mean"}:
            raise ValueError("protein_mode must be local or global_mean")
        if segment_aggregation not in {"fixed", "learned"}:
            raise ValueError("segment_aggregation must be fixed or learned")
        if go_representation not in {"segments", "single_text"}:
            raise ValueError("go_representation must be segments or single_text")
        if protein_pool_type not in {"mean_attn_gate", "mean"}:
            raise ValueError("Legacy protein_pool_type must be mean_attn_gate or mean")
        if go_pool_type != "mean" or go_segment_representation_mode != "segments_only":
            raise ValueError("Expected go_pool_type=mean and segments_only GO representation")
        if go_encoder is not None and d_g is None:
            d_g = int(go_encoder.model.config.hidden_size)
        if d_g is None:
            raise ValueError("d_g must be supplied when go_encoder is None")
        if min(int(d_h), int(d_g), int(d_z)) < 4:
            raise ValueError("Projection dimensions must be at least 4")
        if min(int(window_length), int(window_stride), int(go_chunk_size)) < 1:
            raise ValueError("Window length, stride and GO chunk size must be positive")
        if not math.isfinite(float(parent_strength)) or float(parent_strength) < 0:
            raise ValueError("parent_strength must be finite and nonnegative")
        if not isinstance(protein_conditioning, bool):
            raise ValueError("protein_conditioning must be a boolean")
        if protein_conditioning and (
                protein_mode != "local" or segment_aggregation != "learned"
                or go_representation != "segments" or float(parent_strength) != 0.0
                or not normalize):
            raise ValueError("B3 protein conditioning requires local, learned, segments, normalized scoring and parent_strength=0")
        if go_representation == "single_text":
            if protein_mode != "global_mean" or segment_aggregation != "fixed":
                raise ValueError("single_text requires protein_mode=global_mean and segment_aggregation=fixed")
            if float(parent_strength) != 0.0:
                raise ValueError("single_text requires parent_strength=0")
            if not normalize:
                raise ValueError("single_text requires normalize=True for cosine scoring")

        self.d_h, self.d_g, self.d_z = int(d_h), int(d_g), int(d_z)
        self.normalize = bool(normalize)
        self.go_encoder = go_encoder
        self.protein_mode = protein_mode
        self.segment_aggregation = segment_aggregation
        self.go_representation = go_representation
        self.protein_conditioning = protein_conditioning
        # Kept for old TrainingContext consumers; metadata identifies real scoring.
        self.protein_pool_type = protein_pool_type
        self.go_pool_type = go_pool_type
        self.go_segment_representation_mode = go_segment_representation_mode

        # Do not reorder: preserves shared initialization with the original model.
        self.protein_mean_attn_gate_pool = GatedMeanAttnPool1D(d_in=d_h, dropout=0.05)
        self.proj_p = ProjectionHead(d_in=d_h, d_out=d_z)
        self.proj_g = ProjectionHead(d_in=d_g, d_out=d_z)
        self.go_segment_gate = nn.Sequential(
            nn.LayerNorm(d_g),
            nn.Linear(d_g, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.go_segment_gate[-1].weight)
        nn.init.zeros_(self.go_segment_gate[-1].bias)
        self.proj_local = ProjectionHead(d_in=d_h, d_out=d_z)
        self.mil = SmoothMIL(tau=mil_tau)
        self.fusion_gate = nn.Linear(3, 1)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -1.38629436)

        for module in (self.protein_mean_attn_gate_pool, self.fusion_gate):
            module.requires_grad_(False)
        self.proj_p.requires_grad_(protein_mode == "global_mean")
        self.proj_local.requires_grad_(protein_mode == "local")
        self.go_segment_gate.requires_grad_(segment_aggregation == "learned")
        if self.go_encoder is not None:
            self.go_encoder.requires_grad_(False)
            self.go_encoder.eval()

        self.window_length = int(window_length)
        self.window_stride = int(window_stride)
        self.parent_strength = float(parent_strength)
        self.go_chunk_size = int(go_chunk_size)
        self.last_h_diagnostics = {}
        self.last_pool_info = {}
        self.last_go_segment_info = {}
        # Append after EVERY original B3 module and preserve the construction RNG
        # state. Original tensors and subsequent dropout draws therefore match B3.
        with torch.random.fork_rng(devices=[]):
            self.protein_gate_projection = nn.Linear(self.d_h, 128, bias=False)
        nn.init.zeros_(self.protein_gate_projection.weight)
        self.protein_gate_projection.requires_grad_(protein_conditioning)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.go_encoder is not None:
            self.go_encoder.eval()
        return self

    def architecture_metadata(self):
        """Persist this alongside weights: state-dict keys do not identify scorer."""
        single_text = self.go_representation == "single_text"
        return {
            "architecture_version": "mzsgo_mf_b3_protein_conditioned_v1",
            "protein_mode": self.protein_mode,
            "go_representation": self.go_representation,
            "segment_aggregation": self.segment_aggregation,
            "gate_conditioning": "protein_and_go_text" if self.protein_conditioning else "disabled" if single_text else (
                "go_text" if self.segment_aggregation == "learned" else "fixed"
            ),
            "aggregation_level": "single_text_cosine" if single_text else (
                "segment_scores_after_window_reduction" if self.protein_mode == "local"
                else "segment_scores_from_global_mean"
            ),
            "score_segments": [] if single_text else ["name", "definition"],
            "parent_role": "fixed_consistency_penalty" if self.parent_strength > 0 else "disabled",
            "window_layout": "legacy_padded_batch" if self.protein_mode == "local" else "not_applicable",
            "window_length": self.window_length,
            "window_stride": self.window_stride,
            "mil_tau": self.mil.tau,
            "parent_strength": self.parent_strength,
            "go_chunk_size": self.go_chunk_size,
            "normalize": self.normalize,
            "d_h": self.d_h,
            "d_g": self.d_g,
            "d_z": self.d_z,
            "protein_conditioning": self.protein_conditioning,
            "protein_gate_summary": "masked_mean_frozen_residue_features",
            "protein_gate_injection": "additive_pre_gelu",
            "protein_gate_hidden": 128,
            "protein_gate_dropout": "shared_across_proteins",
            "protein_gate_initialization": "zero_bias_free_projection_after_base_modules_rng_preserved",
            "protein_gate_training_memory": "checkpointed_go_chunks",
        }

    @staticmethod
    def _norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
        n = torch.linalg.vector_norm(x.float(), dim=dim, keepdim=True).clamp_min(eps)
        return x / n.to(x.dtype)

    def _validate_protein_inputs(self, H, mask):
        if H.ndim != 3 or H.shape[-1] != self.d_h:
            raise ValueError(f"Expected protein [B,T,{self.d_h}], got {tuple(H.shape)}")
        if H.shape[0] == 0 or H.shape[1] == 0:
            raise ValueError("Empty protein batch or zero residue length")
        if mask is not None:
            if mask.shape != H.shape[:2]:
                raise ValueError("Protein mask shape does not match embeddings")
            if not mask.bool().any(dim=1).all():
                raise ValueError("Protein with no valid residues")

    def encode_protein_for_scoring(self, H, mask=None):
        """Return the global-mean vector; local scoring has no single vector."""
        if self.protein_mode != "global_mean":
            raise RuntimeError(
                "Local scoring uses multiple windows. Use encode_local_windows or "
                "score_fine_grained; a frozen global vector is not its representation."
            )
        self._validate_protein_inputs(H, mask)
        pooled = GatedMeanAttnPool1D.masked_mean(H, mask)
        z = self.proj_p(pooled)
        return self._norm(z, dim=-1) if self.normalize else z

    def project_go(self, G):
        if G.ndim not in {2, 3}:
            raise ValueError(f"Expected GO [G,D] or [G,P,D], got {tuple(G.shape)}")
        z = self.proj_g(G)
        if self.normalize:
            z = self._norm(z, dim=-1)
        return torch.nan_to_num(z)

    def score_single_text(self, H, mask, G_text):
        """Return [B,G] cosine scores for the explicit single-text baseline.

        G_text must contain one frozen embedding per GO term, encoded from the
        combined name/definition text by the data pipeline. This method never
        constructs a combined representation by averaging segment embeddings.
        Both protein and GO projections remain trainable.
        """
        if self.go_representation != "single_text":
            raise ValueError("score_single_text requires go_representation=single_text")
        if G_text.ndim != 2 or G_text.shape[-1] != self.d_g or G_text.shape[0] == 0:
            raise ValueError(f"Expected nonempty single GO text [G,{self.d_g}], got {tuple(G_text.shape)}")
        zp = self.encode_protein_for_scoring(H, mask).float()
        zg = self.project_go(G_text).float()
        scores = zp @ zg.T
        self.last_h_diagnostics = {
            "h/local_only": 0.0,
            "h/parent_adjustment_mean_abs": 0.0,
            "h/window_count_mean": 1.0,
        }
        self.last_go_segment_info = {
            "go_representation": "single_text",
            "segment_names": [],
            "aggregation_type": "disabled",
            "gate_conditioning": "disabled",
            "aggregation_level": "single_text_cosine",
            "parent_role": "disabled",
        }
        return scores

    def masked_protein_summary(self, H, mask=None):
        """Frozen-residue mean used only by the gate, never as a score branch."""
        self._validate_protein_inputs(H, mask)
        if mask is None:
            return H.float().mean(dim=1)
        valid = mask.bool()
        values = H.float().masked_fill(~valid[..., None], 0.0)
        return values.sum(dim=1) / valid.sum(dim=1)[:, None].float()

    @staticmethod
    def _gate_input_identity(value):
        # Inference-mode tensors deliberately have no mutation version counter.
        version = None if torch.is_inference(value) else value._version
        return (value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                str(value.device), str(value.dtype), version)

    def _gate_parameter_versions(self):
        return tuple(p._version for layer in (self.go_segment_gate[0], self.go_segment_gate[1])
                     for p in layer.parameters())

    def prepare_go_gate_context(self, name_raw, definition_raw, definition_present):
        """Prepare [G,2,128] GO features once; eval may reuse this across batches.

        Dropout remains one shared GO/segment/hidden mask across proteins. This
        preserves B3's gate noise when the new protein projection is zero.
        Actual pair weights are produced only after a protein summary is supplied.
        """
        if not self.protein_conditioning:
            raise ValueError("GO gate context is specific to the protein-conditioned scorer")
        if name_raw.ndim != 2 or name_raw.shape != definition_raw.shape:
            raise ValueError("Name and definition embeddings must have equal [G,D] shapes")
        if name_raw.shape[1] != self.d_g or name_raw.shape[0] == 0:
            raise ValueError("Invalid GO embedding dimension or empty GO universe")
        if definition_present.shape != name_raw.shape[:1]:
            raise ValueError("Definition presence mask must have shape [G]")
        present = torch.stack((torch.ones_like(definition_present, dtype=torch.bool),
                               definition_present.bool()), dim=1)
        raw = torch.stack((name_raw, definition_raw), dim=1)
        hidden = self.go_segment_gate[1](self.go_segment_gate[0](raw))
        dropout_mask = self.go_segment_gate[3](torch.ones_like(hidden))
        return ProteinGoGateContext(
            hidden=hidden, dropout_mask=dropout_mask, present=present,
            owner=id(self), training=self.training,
            inputs=tuple(self._gate_input_identity(x) for x in (name_raw, definition_raw, definition_present)),
            parameter_versions=self._gate_parameter_versions(),
        )

    def _conditioned_gate_chunk(self, go_hidden, protein_hidden, dropout_mask, present):
        # The protein contribution is before the nonlinearity. A shared scalar
        # added after the final head would cancel in the two-segment softmax.
        hidden = self.go_segment_gate[2](go_hidden[None] + protein_hidden[:, None, None, :])
        hidden = hidden * dropout_mask[None]
        logits = self.go_segment_gate[4](hidden).squeeze(-1).float()
        logits = logits.masked_fill(~present[None], -torch.inf)
        return torch.softmax(logits, dim=-1)

    def aggregation_weights(self, name_raw, definition_raw, definition_present, *,
                            protein_summary=None, go_gate_context=None):
        """Return actual [B,G,2] protein-conditioned weights in name/definition order.

        A protein summary is mandatory for B3-PC. Legacy GO-only consumers fail
        here instead of silently evaluating a different gate. Disabled conditioning
        is a constructor-only parity control and retains the original [G,2] path.

        A GO's name must be present. Missing definitions get exactly zero
        weight and the name gets one. Softmax masks only these two channels;
        parent information is never a third aggregation channel.
        """
        if self.protein_conditioning:
            if protein_summary is None:
                raise ValueError("Protein-conditioned aggregation requires protein_summary; use score_fine_grained for automatic masked pooling")
            if protein_summary.ndim != 2 or protein_summary.shape[1] != self.d_h or protein_summary.shape[0] == 0:
                raise ValueError(f"protein_summary must have shape [B,{self.d_h}]")
            context = (self.prepare_go_gate_context(name_raw, definition_raw, definition_present)
                       if go_gate_context is None else go_gate_context)
            if not isinstance(context, ProteinGoGateContext) or context.owner != id(self):
                raise ValueError("GO gate context belongs to a different scorer")
            identities = tuple(self._gate_input_identity(x) for x in (name_raw, definition_raw, definition_present))
            if context.training != self.training or context.inputs != identities:
                raise ValueError("GO gate context mode or frozen input identity differs")
            if context.parameter_versions != self._gate_parameter_versions():
                raise ValueError("GO gate parameters changed since context preparation")
            if protein_summary.device != context.hidden.device:
                raise ValueError("Protein summary and GO gate features must share a device")
            protein_hidden = self.protein_gate_projection(protein_summary)
            chunks = []
            for st in range(0, name_raw.shape[0], self.go_chunk_size):
                ed = min(name_raw.shape[0], st + self.go_chunk_size)
                inputs = (context.hidden[st:ed], protein_hidden,
                          context.dropout_mask[st:ed], context.present[st:ed])
                if self.training and torch.is_grad_enabled():
                    # Only the added B x chunk x segment x hidden interaction is
                    # recomputed; never retain B x full-G x hidden activations.
                    weights = checkpoint(self._conditioned_gate_chunk, *inputs,
                                         use_reentrant=False, preserve_rng_state=False)
                else:
                    weights = self._conditioned_gate_chunk(*inputs)
                chunks.append(weights)
            weights = torch.cat(chunks, dim=1)
            self._record_segment_weights(weights, context.present)
            return weights
        if protein_summary is not None or go_gate_context is not None:
            raise ValueError("Disabled protein conditioning does not accept protein gate inputs")
        if self.go_representation != "segments":
            raise ValueError("Single-text scoring has no GO segment weights; use score_single_text")
        if name_raw.ndim != 2 or name_raw.shape != definition_raw.shape:
            raise ValueError("Name and definition embeddings must have equal [G,D] shapes")
        if name_raw.shape[1] != self.d_g or name_raw.shape[0] == 0:
            raise ValueError("Invalid GO embedding dimension or empty GO universe")
        if definition_present.shape != name_raw.shape[:1]:
            raise ValueError("Definition presence mask must have shape [G]")
        present = torch.stack(
            (torch.ones_like(definition_present, dtype=torch.bool), definition_present.bool()),
            dim=1,
        )
        if self.segment_aggregation == "fixed":
            weights = present.float() / present.sum(1, keepdim=True)
        else:
            raw = torch.stack((name_raw, definition_raw), dim=1)
            logits = self.go_segment_gate(raw).squeeze(-1).float()
            logits = logits.masked_fill(~present, -torch.inf)
            weights = torch.softmax(logits, dim=1)
        self._record_segment_weights(weights, present)
        return weights

    def _record_segment_weights(self, weights, present):
        self.last_go_segment_info = {
            "go_representation": "segments",
            "segment_weights": weights.detach(),
            "segment_present": present.detach(),
            "segment_names": ["name", "definition"],
            "aggregation_type": self.segment_aggregation,
            "gate_conditioning": "protein_and_go_text" if self.protein_conditioning else "go_text" if self.segment_aggregation == "learned" else "fixed",
            "weight_axes": ["protein", "go", "segment"] if self.protein_conditioning else ["go", "segment"],
            "aggregation_level": "score",
            "parent_role": "separate_consistency_penalty" if self.parent_strength > 0 else "disabled",
        }

    def encode_local_windows(self, H, mask):
        """Return [B,K,Dz] and [B,K], preserving the original window layout.

        Window starts are based on padded batch length, as in the fixed
        baseline. Changing this policy would be a separate intervention;
        keep batching/padding identical in matched comparisons.
        """
        self._validate_protein_inputs(H, mask)
        if mask is None:
            mask = torch.ones(H.shape[:2], dtype=torch.bool, device=H.device)
        else:
            mask = mask.bool()
        B, T, _ = H.shape
        length = self.window_length
        stride = self.window_stride
        starts = list(range(0, max(1, T - length + 1), stride))
        tail = max(0, T - length)
        if tail not in starts:
            starts.append(tail)
        starts = sorted(set(starts))
        windows, valid = [], []
        for st in starts:
            chunk = H[:, st:st + length]
            m = mask[:, st:st + length]
            ok = m.any(dim=1)
            pooled = (chunk.float() * m[..., None]).sum(1) / m.sum(1).clamp_min(1)[:, None]
            z = F.normalize(self.proj_local(pooled).float(), dim=-1)
            windows.append(z)
            valid.append(ok)
        return torch.stack(windows, 1), torch.stack(valid, 1)

    def score_fine_grained(
        self, H, mask, name_raw, definition_raw, definition_present,
        parent_raw=None, parent_present=None, global_go_raw=None, *, segment_weights=None,
        go_gate_context=None,
    ):
        """Return [B,G] using actual fixed/learned post-MIL score weights.

        global_go_raw is accepted for the old call signature but never used.
        B3-PC computes actual pair weights internally unless supplied as [B,G,2].
        Evaluation may cache prepare_go_gate_context(...) under the same autocast
        as this call; it never caches GO-only weights for reuse across proteins.
        Parent inputs are unused and may be None when parent_strength is zero.
        """
        if self.go_representation != "segments":
            raise ValueError("single_text does not use segment scoring; call score_single_text")
        self._validate_protein_inputs(H, mask)
        if segment_weights is None:
            if self.protein_conditioning:
                summary = self.masked_protein_summary(H, mask)
                weights = self.aggregation_weights(name_raw, definition_raw, definition_present,
                                                   protein_summary=summary, go_gate_context=go_gate_context)
            else:
                if go_gate_context is not None:
                    raise ValueError("Disabled protein conditioning does not accept a GO gate context")
                weights = self.aggregation_weights(name_raw, definition_raw, definition_present)
        else:
            if go_gate_context is not None:
                raise ValueError("Pass actual pair weights or a GO gate context, not both")
            weights = segment_weights
            expected_shape = ((H.shape[0], name_raw.shape[0], 2) if self.protein_conditioning
                              else (name_raw.shape[0], 2))
            if weights.shape != expected_shape:
                raise ValueError(f"segment_weights must have shape {expected_shape}")
            if weights.device != name_raw.device:
                raise ValueError("segment_weights and GO embeddings must be on the same device")
            present = torch.stack(
                (torch.ones_like(definition_present, dtype=torch.bool), definition_present.bool()), dim=1,
            )
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("segment_weights must be finite and nonnegative")
            if not torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-5, rtol=1e-5):
                raise ValueError("segment_weights rows must sum to one")
            if (weights.masked_select(~present[None] if self.protein_conditioning else ~present) != 0).any():
                raise ValueError("Missing definitions must have zero aggregation weight")
            if self.segment_aggregation == "fixed":
                expected = present.float() / present.sum(1, keepdim=True)
                if not torch.equal(weights.float(), expected):
                    raise ValueError("Fixed aggregation requires uniform available-segment weights")
            elif self.training and torch.is_grad_enabled() and not weights.requires_grad:
                raise ValueError("Learned training weights must retain their gate gradient graph")
            self._record_segment_weights(weights, present)

        G = name_raw.shape[0]
        if self.parent_strength > 0:
            if parent_raw is None or parent_present is None:
                raise ValueError("Parent inputs are required when parent_strength is positive")
            if parent_raw.ndim != 3 or parent_raw.shape[0] != G or parent_raw.shape[2] != self.d_g:
                raise ValueError("parent_raw must have shape [G,P,D]")
            if parent_present.shape != parent_raw.shape[:2]:
                raise ValueError("parent_present must have shape [G,P]")
        gn = self.project_go(name_raw).float()
        gd = self.project_go(definition_raw).float()
        if self.protein_mode == "local":
            zw, valid = self.encode_local_windows(H, mask)
            B = zw.shape[0]
        else:
            zp = self.encode_protein_for_scoring(H, mask).float()
            B = zp.shape[0]
        chunks, penalties = [], []
        for st in range(0, G, self.go_chunk_size):
            ed = min(G, st + self.go_chunk_size)
            if self.protein_mode == "local":
                sn = self.mil(torch.einsum("bkd,gd->bkg", zw, gn[st:ed]), valid)
                sd = self.mil(torch.einsum("bkd,gd->bkg", zw, gd[st:ed]), valid)
            else:
                sn = zp @ gn[st:ed].T
                sd = zp @ gd[st:ed].T
            if self.segment_aggregation == "fixed":
                # Exact original arithmetic order for fixed baseline parity.
                dp = definition_present[st:ed].bool()[None, :]
                child = torch.where(dp, 0.5 * (sn + sd), sn)
            else:
                if self.protein_conditioning:
                    w = weights[:, st:ed]
                    child = w[:, :, 0] * sn + w[:, :, 1] * sd
                else:
                    w = weights[st:ed]
                    child = w[:, 0][None, :] * sn + w[:, 1][None, :] * sd
            if self.parent_strength > 0:
                pr = parent_raw[st:ed]
                pm = parent_present[st:ed].bool()
                if pm.any():
                    pg = self.project_go(pr.reshape(-1, pr.shape[-1])).float()
                    pg = pg.reshape(ed - st, pr.shape[1], -1)
                    if self.protein_mode == "local":
                        ps = torch.einsum("bkd,cpd->bkcp", zw, pg)
                        C, P = pm.shape
                        flat = ps.permute(0, 3, 1, 2).reshape(B * P, zw.shape[1], C)
                        pv = valid[:, None, :].expand(B, P, valid.shape[1]).reshape(B * P, -1)
                        parent_scores = self.mil(flat, pv).reshape(B, P, C).permute(0, 2, 1)
                    else:
                        parent_scores = torch.einsum("bd,cpd->bcp", zp, pg)
                    parent_scores = parent_scores.masked_fill(~pm[None, :, :], 0)
                    parent_avg = parent_scores.sum(-1) / pm.sum(-1).clamp_min(1)[None, :]
                    adjustment = -self.parent_strength * pm.any(-1)[None, :] * torch.relu(child - parent_avg)
                else:
                    adjustment = torch.zeros_like(child)
            else:
                adjustment = torch.zeros_like(child)
            chunks.append(child + adjustment)
            penalties.append(adjustment)
        scores = torch.cat(chunks, dim=1)
        self.last_h_diagnostics = {
            "h/local_only": float(self.protein_mode == "local"),
            "h/parent_adjustment_mean_abs": float(torch.cat(penalties, 1).detach().abs().mean().item()),
            "h/window_count_mean": float(valid.sum(1).float().mean().item()) if self.protein_mode == "local" else 1.0,
        }
        return scores

    def forward(self, H, G=None, mask=None, **segment_inputs):
        """Dispatch only to the scorer declared by go_representation."""
        if self.go_representation == "single_text":
            if G is None:
                raise ValueError("single_text forward requires the combined GO text tensor G")
            if segment_inputs:
                raise ValueError("single_text forward does not accept segment or parent inputs")
            return self.score_single_text(H, mask, G)
        if G is not None:
            raise ValueError(
                "A pooled GO tensor bypasses learned aggregation. Call "
                "score_fine_grained with separate name/definition/parent inputs."
            )
        return self.score_fine_grained(H, mask, **segment_inputs)
