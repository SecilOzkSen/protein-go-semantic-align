from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import math

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.encoders import BioMedBERTEncoder
from src.models.projection import ProjectionHead
import torch.nn.functional as F

from .isa_context import ISA_CONTEXT_POLICY


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
    """B3 scorer with the additional cached canonical is_a semantic channel.

    Local mode scores projected protein windows against each text segment,
    reduces windows by SmoothMIL, then mixes the segment scores. Global-mean
    mode instead scores one projected mean protein vector. A shared gate on
    frozen GO text features provides GO-conditioned, not protein-conditioned,
    weights. The existing is_a text is a third score channel using that same
    gate and GO projection. Parent consistency penalty remains separately OFF
    in B3-ISA. No new trainable module is introduced.
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
        isa_context_enabled: bool = True,
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
        if not isinstance(isa_context_enabled, bool):
            raise ValueError("isa_context_enabled must be a boolean")
        if isa_context_enabled and (go_representation != "segments"
                                    or segment_aggregation != "learned"
                                    or float(parent_strength) != 0.0):
            raise ValueError("B3-ISA requires segmented learned aggregation and parent_strength=0")
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
        self.isa_context_enabled = isa_context_enabled
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.go_encoder is not None:
            self.go_encoder.eval()
        return self

    def architecture_metadata(self):
        """Persist this alongside weights: state-dict keys do not identify scorer."""
        single_text = self.go_representation == "single_text"
        return {
            "architecture_version": "mzsgo_mf_b3_isa_context_v1",
            "protein_mode": self.protein_mode,
            "go_representation": self.go_representation,
            "segment_aggregation": self.segment_aggregation,
            "gate_conditioning": "disabled" if single_text else (
                "go_text" if self.segment_aggregation == "learned" else "fixed"
            ),
            "aggregation_level": "single_text_cosine" if single_text else (
                "segment_scores_after_window_reduction" if self.protein_mode == "local"
                else "segment_scores_from_global_mean"
            ),
            "score_segments": [] if single_text else (
                ["name", "definition", "is_a"] if self.isa_context_enabled else ["name", "definition"]
            ),
            "isa_context_enabled": self.isa_context_enabled,
            "isa_context_policy": ISA_CONTEXT_POLICY if self.isa_context_enabled else "disabled",
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

    def _context_inputs(self, name_raw, isa_context_raw, isa_context_present):
        if not self.isa_context_enabled:
            return None, None
        if isa_context_raw is None or isa_context_present is None:
            raise ValueError("B3-ISA requires explicit isa_context_raw and isa_context_present; missing context cannot silently use B3")
        if isa_context_raw.shape != name_raw.shape or isa_context_raw.device != name_raw.device:
            raise ValueError("isa_context_raw must match name_raw [G,D] shape and device")
        if isa_context_present.shape != name_raw.shape[:1] or isa_context_present.device != name_raw.device:
            raise ValueError("isa_context_present must have shape [G] on the GO device")
        present = isa_context_present.bool()
        if not torch.isfinite(isa_context_raw[present]).all():
            raise ValueError("Present is_a context must be finite")
        # Absent cached rows carry no semantic evidence, including in gradients.
        return isa_context_raw.masked_fill(~present[:, None], 0), present

    def aggregation_weights(self, name_raw, definition_raw, definition_present, *,
                            isa_context_raw=None, isa_context_present=None):
        """Return the original two weights, or three with a masked is_a channel.

        The name/definition gate call retains its original tensor layout. The
        shared MLP is called separately for present is_a rows. If every context
        row is masked, this added call is skipped, preserving B3 dropout RNG,
        scores and gradients exactly.
        """
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
        context, context_present = self._context_inputs(name_raw, isa_context_raw, isa_context_present)
        if self.segment_aggregation == "fixed":
            weights = present.float() / present.sum(1, keepdim=True)
        else:
            raw = torch.stack((name_raw, definition_raw), dim=1)
            logits = self.go_segment_gate(raw).squeeze(-1).float()
            logits = logits.masked_fill(~present, -torch.inf)
            weights = torch.softmax(logits, dim=1)
            if self.isa_context_enabled:
                base_weights = torch.cat((weights, torch.zeros_like(weights[:, :1])), dim=1)
                if context_present.any():
                    # No new module or parameter initialization is introduced.
                    context_logits = self.go_segment_gate(context[context_present]).squeeze(-1).float()
                    full_context_logits = logits.new_full((logits.shape[0],), -torch.inf)
                    positions = torch.nonzero(context_present, as_tuple=False).flatten()
                    full_context_logits = full_context_logits.index_copy(0, positions, context_logits)
                    three_logits = torch.cat((logits, full_context_logits[:, None]), dim=1)
                    three_weights = torch.softmax(three_logits, dim=1)
                    weights = torch.where(context_present[:, None], three_weights, base_weights)
                else:
                    weights = base_weights
                present = torch.cat((present, context_present[:, None]), dim=1)
        self._record_segment_weights(weights, present)
        return weights

    def _record_segment_weights(self, weights, present):
        self.last_go_segment_info = {
            "go_representation": "segments",
            "segment_weights": weights.detach(),
            "segment_present": present.detach(),
            "segment_names": ["name", "definition", "is_a"] if self.isa_context_enabled else ["name", "definition"],
            "aggregation_type": self.segment_aggregation,
            "gate_conditioning": "go_text" if self.segment_aggregation == "learned" else "fixed",
            "aggregation_level": "score",
            "parent_role": "separate_consistency_penalty" if self.parent_strength > 0 else "disabled",
            "isa_context_policy": ISA_CONTEXT_POLICY if self.isa_context_enabled else "disabled",
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
        isa_context_raw=None, isa_context_present=None,
    ):
        """Return [B,G] using actual fixed/learned post-MIL score weights.

        global_go_raw is accepted for the old call signature but never used.
        Supplying segment_weights avoids repeating the GO-only gate in eval;
        pass aggregation_weights(...) without detach during training.
        Parent-penalty inputs remain unused at parent_strength zero. The is_a
        semantic channel is an explicit, separate input and is required when
        isa_context_enabled is true, even if all its presence values are false.
        """
        if self.go_representation != "segments":
            raise ValueError("single_text does not use segment scoring; call score_single_text")
        self._validate_protein_inputs(H, mask)
        context, context_present = self._context_inputs(name_raw, isa_context_raw, isa_context_present)
        if segment_weights is None:
            weights = self.aggregation_weights(name_raw, definition_raw, definition_present,
                                               isa_context_raw=context, isa_context_present=context_present)
        else:
            weights = segment_weights
            channels = 3 if self.isa_context_enabled else 2
            if weights.shape != (name_raw.shape[0], channels):
                raise ValueError(f"segment_weights must have shape [G,{channels}]")
            if weights.device != name_raw.device:
                raise ValueError("segment_weights and GO embeddings must be on the same device")
            present = torch.stack(
                (torch.ones_like(definition_present, dtype=torch.bool), definition_present.bool()), dim=1,
            )
            if self.isa_context_enabled:
                present = torch.cat((present, context_present[:, None]), dim=1)
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("segment_weights must be finite and nonnegative")
            if not torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-5, rtol=1e-5):
                raise ValueError("segment_weights rows must sum to one")
            if (weights.masked_select(~present) != 0).any():
                raise ValueError("Unavailable text channels must have zero aggregation weight")
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
        has_context = self.isa_context_enabled and bool(context_present.any())
        gc = self.project_go(context).float() if has_context else None
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
                w = weights[st:ed]
                child = w[:, 0][None, :] * sn + w[:, 1][None, :] * sd
                if has_context:
                    if self.protein_mode == "local":
                        sc = self.mil(torch.einsum("bkd,gd->bkg", zw, gc[st:ed]), valid)
                    else:
                        sc = zp @ gc[st:ed].T
                    child = child + w[:, 2][None, :] * sc
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
