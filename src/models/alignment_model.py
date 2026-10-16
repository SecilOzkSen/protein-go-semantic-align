from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.encoders import BioMedBERTEncoder
from src.models.projection import ProjectionHead
import torch.nn.functional as F


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

    This is intentionally the only protein representation in the retriever.
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

    def forward(self, scores, valid):
        # scores [B,K,G], valid [B,K]
        if not valid.any(dim=1).all():
            raise RuntimeError("Protein without valid local windows")
        x = scores.float().masked_fill(~valid[:, :, None], -1e4)
        return self.tau * (torch.logsumexp(x / self.tau, dim=1) - valid.sum(1).float().log()[:, None])


class ProteinGoAligner(nn.Module):
    """
    Retriever v2 alignment model.

    Protein side
    ------------
    residue embeddings [B,T,Dh]
        -> GatedMeanAttnPool1D
        -> proj_p
        -> L2 normalization
        -> one protein vector [B,Dz]

    GO side
    -------
    BioMedBERT segment embeddings are produced/cached by the trainer.
    This module owns the trainable segment gate:
        segment embeddings [G,S,Dg]
        -> go_segment_gate
        -> weighted pooled GO [G,Dg]
        -> proj_g
        -> L2 normalization

    No slots, local evidence branch, expert fusion, token alignment,
    multivector scoring, or GO full-text/segment mixing is present.
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
    ):
        super().__init__()

        if protein_pool_type != "mean_attn_gate":
            raise ValueError(
                "Retriever v2 supports only "
                "protein_pool_type='mean_attn_gate'."
            )

        if go_pool_type != "mean":
            raise ValueError(
                "Retriever v2 supports only go_pool_type='mean'."
            )

        if go_segment_representation_mode != "segments_only":
            raise ValueError(
                "Retriever v2 supports only "
                "go_segment_representation_mode='segments_only'."
            )

        self.normalize = bool(normalize)
        self.go_encoder = go_encoder
        self.protein_pool_type = protein_pool_type
        self.go_pool_type = go_pool_type
        self.go_segment_representation_mode = (
            go_segment_representation_mode
        )

        if self.go_encoder is not None and d_g is None:
            d_g = int(self.go_encoder.model.config.hidden_size)

        if d_g is None:
            raise ValueError(
                "d_g must be supplied when go_encoder is None."
            )

        # Keep these module names identical to the previous model so the useful
        # representation/projection weights can be warm-started.
        self.protein_mean_attn_gate_pool = GatedMeanAttnPool1D(
            d_in=d_h,
            dropout=0.05,
        )

        self.proj_p = ProjectionHead(
            d_in=d_h,
            d_out=d_z,
        )

        self.proj_g = ProjectionHead(
            d_in=d_g,
            d_out=d_z,
        )

        # The gate acts independently on each available segment embedding.
        # It therefore does NOT need a hard-coded number/order of segments.
        # The actual segment set comes from GoTextStore, e.g.
        # name + definition + is_a.
        self.go_segment_gate = nn.Sequential(
            nn.LayerNorm(d_g),
            nn.Linear(d_g, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 1),
        )

        # Uniform weighting at initialization.
        nn.init.zeros_(self.go_segment_gate[-1].weight)
        nn.init.zeros_(self.go_segment_gate[-1].bias)

        self.proj_local = ProjectionHead(d_in=d_h, d_out=d_z)
        self.mil = SmoothMIL(tau=0.1)
        # Parent contribution is bounded and cannot independently produce a high score.
        self.parent_strength = 0.15
        # Gate sees only scalar score evidence, not arbitrary protein/GO IDs.
        self.fusion_gate = nn.Linear(3, 1)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -1.38629436)  # alpha = 0.2
        self.window_length = 128
        self.window_stride = 64
        self.go_chunk_size = 128
        self.last_h_diagnostics = {}

        self.last_pool_info = {}
        self.last_go_segment_info = {}

    @staticmethod
    def _norm(
            x: torch.Tensor,
            dim: int = -1,
            eps: float = 1e-6,
    ) -> torch.Tensor:
        n = torch.linalg.vector_norm(
            x.float(),
            dim=dim,
            keepdim=True,
        ).clamp_min(eps)

        return x / n.to(x.dtype)

    def encode_protein_for_scoring(
            self,
            H: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode one protein into exactly one normalized retrieval vector.

        Returns:
            [B,Dz]
        """
        if H.ndim != 3:
            raise RuntimeError(
                f"Protein tensor must be [B,T,Dh], got {tuple(H.shape)}"
            )

        if mask is not None and mask.dtype != torch.bool:
            mask = mask != 0

        h_pool, pool_info = self.protein_mean_attn_gate_pool(
            H,
            mask,
        )
        self.last_pool_info = pool_info

        z = self.proj_p(h_pool)

        if self.normalize:
            z = self._norm(z, dim=-1)

        z = torch.nan_to_num(z)

        if z.ndim != 2:
            raise RuntimeError(
                f"Protein representation must be [B,Dz], got {tuple(z.shape)}"
            )

        return z

    def pool_go_segments(
            self,
            segment_embs: torch.Tensor,
            segment_present: torch.Tensor,
    ):
        """
        Trainable aggregation of frozen GO segment embeddings.

        segment_embs:
            [G,S,Dg]

        segment_present:
            [G,S] bool

        Returns:
            pooled:  [G,Dg]
            weights: [G,S]
        """
        if segment_embs.ndim != 3:
            raise RuntimeError(
                "segment_embs must be [G,S,Dg], got "
                f"{tuple(segment_embs.shape)}"
            )

        if segment_present.ndim != 2:
            raise RuntimeError(
                "segment_present must be [G,S], got "
                f"{tuple(segment_present.shape)}"
            )

        if segment_present.shape != segment_embs.shape[:2]:
            raise RuntimeError(
                "GO segment mask does not match segment embeddings: "
                f"{tuple(segment_present.shape)} vs "
                f"{tuple(segment_embs.shape[:2])}"
            )

        if segment_present.dtype != torch.bool:
            segment_present = segment_present != 0

        if (segment_present.sum(dim=1) == 0).any():
            raise RuntimeError(
                "At least one GO term has zero present segments."
            )

        logits = self.go_segment_gate(
            segment_embs
        ).squeeze(-1).float()

        logits = logits.masked_fill(
            ~segment_present,
            -1e9,
        )

        weights = torch.softmax(
            logits,
            dim=-1,
        ).to(segment_embs.dtype)

        pooled = torch.einsum(
            "gs,gsd->gd",
            weights,
            segment_embs,
        )

        pooled = torch.nan_to_num(pooled)

        self.last_go_segment_info = {
            "segment_weights": weights.detach(),
            "segment_present": segment_present.detach(),
            "representation_mode": "segments_only",
        }

        return pooled, weights

    def project_go(
            self,
            G: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project raw pooled GO vectors into the shared retrieval space.

        Accepts:
            [G,Dg] or [B,G,Dg]
        """
        if G.ndim not in {2, 3}:
            raise RuntimeError(
                f"GO tensor must be [G,Dg] or [B,G,Dg], got {tuple(G.shape)}"
            )

        z = self.proj_g(G)

        if self.normalize:
            z = self._norm(z, dim=-1)

        return torch.nan_to_num(z)

    def score_projected(
            self,
            Zp: torch.Tensor,
            Zg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score already projected normalized protein and GO vectors.

        Zp:
            [B,Dz]

        Zg:
            [G,Dz] or [B,G,Dz]
        """
        if Zp.ndim != 2:
            raise RuntimeError(
                f"Zp must be [B,Dz], got {tuple(Zp.shape)}"
            )

        if Zg.ndim == 2:
            return Zp.float() @ Zg.float().T
        if Zg.ndim == 3:
            return torch.einsum("bd,bgd->bg", Zp.float(), Zg.float())
        raise ValueError("GO tensor must be 2D or 3D")

    def encode_local_windows(self, H, mask):
        """Return [B,K,Dz] and [B,K] valid windows, including the final tail."""
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

    def score_fine_grained(self, H, mask, name_raw, definition_raw,
                           definition_present, parent_raw, parent_present,
                           global_go_raw):
        """One score per protein and GO, with GO chunking and differentiable MIL.

        name_raw, definition_raw [G,Dg]; parent_raw [G,P,Dg].
        """
        zp = self.encode_protein_for_scoring(H, mask)
        zw, valid = self.encode_local_windows(H, mask)
        zg = self.project_go(global_go_raw).float()
        base = self.score_projected(zp, zg)
        gn = self.project_go(name_raw).float()
        gd = self.project_go(definition_raw).float()
        B, G = base.shape
        chunks, local_chunks, parent_chunks = [], [], []
        for st in range(0, G, self.go_chunk_size):
            ed = min(G, st + self.go_chunk_size)
            sn = self.mil(torch.einsum("bkd,gd->bkg", zw, gn[st:ed]), valid)
            sd = self.mil(torch.einsum("bkd,gd->bkg", zw, gd[st:ed]), valid)
            dp = definition_present[st:ed].bool()[None, :]
            child = torch.where(dp, 0.5 * (sn + sd), sn)
            pr = parent_raw[st:ed]
            pm = parent_present[st:ed].bool()
            if pm.any():
                projected = self.project_go(pr.reshape(-1, pr.shape[-1])).float()
                projected = projected.reshape(ed - st, pr.shape[1], -1)
                # [B,K,C,P], MIL over windows, then mean across present parents.
                parent_scores = torch.einsum("bkd,cpd->bkcp", zw, projected)
                C, P = pm.shape
                parent_flat = parent_scores.permute(0, 3, 1, 2).reshape(B * P, zw.shape[1], C)
                parent_valid = valid[:, None, :].expand(B, P, valid.shape[1]).reshape(B * P, -1)
                parent_mil = self.mil(parent_flat, parent_valid).reshape(B, P, C).permute(0, 2, 1)
                parent_mil = parent_mil.masked_fill(~pm[None, :, :], 0)
                parent_avg = parent_mil.sum(-1) / pm.sum(-1).clamp_min(1)[None, :]
                # Only nonnegative parent-child agreement can boost evidence;
                # parent evidence alone cannot exceed the child score.
                # Capped at zero for parent-only boost: parents instead penalize disagreement.
                parent_adjustment = -self.parent_strength * pm.any(-1)[None, :] * torch.relu(child - parent_avg)
            else:
                parent_adjustment = torch.zeros_like(child)
            local = child + parent_adjustment
            features = torch.stack((base[:, st:ed], local, local - base[:, st:ed]), -1)
            alpha = torch.sigmoid(self.fusion_gate(features)).squeeze(-1)
            chunks.append((1 - alpha) * base[:, st:ed] + alpha * local)
            local_chunks.append(local)
            parent_chunks.append(parent_adjustment)
        result = torch.cat(chunks, 1)
        self.last_h_diagnostics = {
            "h/local_global_mean_abs_gap": float((torch.cat(local_chunks, 1).detach() - base.detach()).abs().mean().item()),
            "h/parent_adjustment_mean_abs": float(torch.cat(parent_chunks, 1).detach().abs().mean().item()),
            "h/window_count_mean": float(valid.sum(1).float().mean().item()),
        }
        return result

    def forward(
            self,
            H: torch.Tensor,
            G: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            **_,
    ) -> torch.Tensor:
        """
        Compatibility scorer for pooled GO vectors.

        H:
            [B,T,Dh]

        G:
            [G,Dg] or [B,G,Dg]
        """
        Zp = self.encode_protein_for_scoring(
            H,
            mask,
        )
        Zg = self.project_go(G)
        return self.score_projected(
            Zp,
            Zg,
        )
