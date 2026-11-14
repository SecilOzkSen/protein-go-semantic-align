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

        # Experiment H-Local: global modules retained for API compatibility only.
        # They receive no gradients and are excluded from the optimizer.
        for module in (self.protein_mean_attn_gate_pool, self.proj_p,
                       self.go_segment_gate):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.proj_local = ProjectionHead(d_in=d_h, d_out=d_z)
        # Nonlinear low-rank segment interaction, independently factorized for GEMM.
        self.interaction_p = nn.Sequential(nn.Linear(d_z, 128), nn.GELU())
        self.interaction_g = nn.Sequential(nn.Linear(d_z, 128), nn.GELU())
        self.interaction_mix_logit = nn.Parameter(torch.tensor(-2.944439))  # sigmoid = 0.05
        # Shared affine calibration of the ranking score for BCE.
        self.bce_log_scale = nn.Parameter(torch.tensor(1.609438))  # scale ~5
        self.bce_bias = nn.Parameter(torch.tensor(-5.0))
        self.mil = SmoothMIL(tau=0.1)
        # Parent contribution is bounded and cannot independently produce a high score.
        self.parent_strength = 0.15
        # Gate sees only scalar score evidence, not arbitrary protein/GO IDs.
        self.fusion_gate = nn.Linear(3, 1)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -1.38629436)  # alpha = 0.2
        for parameter in self.fusion_gate.parameters():
            parameter.requires_grad_(False)
        self.window_length = 128
        self.window_stride = 64
        self.go_chunk_size = int(__import__("os").environ.get("RETRIEVER_GO_CHUNK", "1024"))
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
        """Vectorized exact fixed 128/64 windows plus final tail window."""
        if mask is None:
            mask = torch.ones(H.shape[:2], dtype=torch.bool, device=H.device)
        else:
            mask = mask.bool()
        B, T, D = H.shape
        L, S = self.window_length, self.window_stride
        if T < L:
            H = F.pad(H, (0, 0, 0, L - T))
            mask = F.pad(mask, (0, L - T), value=False)
            T = L
        starts = list(range(0, T - L + 1, S))
        tail = T - L
        if starts[-1] != tail:
            starts.append(tail)
        # Prefix sums avoid materializing [B,K,128,1280] windows.
        masked = H.float() * mask.unsqueeze(-1).float()
        prefix = F.pad(masked.cumsum(1), (0, 0, 1, 0))
        counts = F.pad(mask.to(torch.int32).cumsum(1), (1, 0))
        st = torch.tensor(starts, device=H.device)
        ed = st + L
        sums = prefix[:, ed] - prefix[:, st]
        n = counts[:, ed] - counts[:, st]
        valid = n > 0
        pooled = sums / n.clamp_min(1).unsqueeze(-1).to(sums.dtype)
        z = self.proj_local(pooled)
        return F.normalize(z.float(), dim=-1), valid

    def score_fine_grained(self, H, mask, name_raw, definition_raw,
                           definition_present, parent_raw, parent_present,
                           global_go_raw):
        """Full universe with once-per-batch projections and batched GEMMs."""
        zw, valid = self.encode_local_windows(H, mask)
        gn = self.project_go(name_raw)
        gd = self.project_go(definition_raw)
        up = self.interaction_p(zw)
        un = self.interaction_g(gn)
        ud = self.interaction_g(gd)
        mix = torch.sigmoid(self.interaction_mix_logit)
        # Combine name and definition along GO axis: one GEMM per scoring path.
        P, K, _ = zw.shape
        flat = zw.reshape(P * K, -1)
        uflat = up.reshape(P * K, -1)
        chunks = []
        # Chunk by GO term so each chunk contains name and definition together.
        for st in range(0, gn.shape[0], self.go_chunk_size):
            ed = min(st + self.go_chunk_size, gn.shape[0])
            g = torch.cat((gn[st:ed], gd[st:ed]), dim=0)
            ug = torch.cat((un[st:ed], ud[st:ed]), dim=0)
            cos = (flat @ g.T).reshape(P, K, -1)
            inter = (uflat @ ug.T).reshape(P, K, -1) * (128.0 ** -0.5)
            s = (1.0 - mix) * cos + mix * inter
            agg = self.mil(s, valid)
            n = ed - st
            sn, sd = agg[:, :n], agg[:, n:]
            present = definition_present[st:ed].bool()[None, :]
            chunks.append(torch.where(present, 0.5 * (sn + sd), sn))
        result = torch.cat(chunks, dim=1)
        self.last_h_diagnostics = {"h/local_only": 1.0,
                                   "h/parent_adjustment_mean_abs": 0.0}
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
