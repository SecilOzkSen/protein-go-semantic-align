import torch
import torch.nn as nn
from typing import Optional

from src.models.projection import ProjectionHead
from src.encoders import BioMedBERTEncoder


class SharedInteractionMLP(nn.Module):
    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, zp: torch.Tensor, zg: torch.Tensor) -> torch.Tensor:
        """
        zp: [B, K, D]
        zg: [B, K, D]
        """
        h = torch.cat([zp, zg, zp * zg, torch.abs(zp - zg)], dim=-1)  # [B, K, 4D]
        logits = self.net(h).squeeze(-1)  # [B, K]
        return logits


class AttentionPool1D(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Linear(d_in, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                      # [B, T, D]
        mask: Optional[torch.Tensor] = None  # [B, T], bool, True=valid
    ):
        s = self.score(self.dropout(x)).squeeze(-1)   # [B, T]

        if mask is not None:
            s = s.masked_fill(~mask, -1e9)

        a = torch.softmax(s, dim=1)                   # [B, T]
        pooled = torch.sum(x * a.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, a


class ProteinGoAligner(nn.Module):
    def __init__(
        self,
        d_h: int,
        d_g: Optional[int] = None,
        d_z: int = 768,
        go_encoder: Optional[BioMedBERTEncoder] = None,
        normalize: bool = True,
        use_score_head: bool = False,
        protein_pool_type: str = "mean",   # "mean" | "attn"
    ):
        super().__init__()

        if protein_pool_type not in {"mean", "attn"}:
            raise ValueError(f"Unsupported protein_pool_type: {protein_pool_type}")

        self.normalize = bool(normalize)
        self.go_encoder = go_encoder
        self.protein_pool_type = protein_pool_type

        self.use_score_head = use_score_head
        self.score_head = (
            SharedInteractionMLP(d=d_z, hidden=256, dropout=0.1)
            if self.use_score_head else None
        )

        if self.go_encoder is not None and d_g is None:
            d_g = int(self.go_encoder.model.config.hidden_size)
        if d_g is None:
            raise ValueError("d_g must be provided if go_encoder is None.")

        self.protein_attn_pool = AttentionPool1D(d_h, dropout=0.05)

        self.proj_p = ProjectionHead(d_in=d_h, d_out=d_z)
        self.proj_g = ProjectionHead(d_in=d_g, d_out=d_z)

        self.protein_ln = nn.LayerNorm(d_h)
        self.go_ln = nn.LayerNorm(d_g)

    @staticmethod
    def _norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        n = torch.linalg.vector_norm(x.to(torch.float32), dim=dim, keepdim=True).clamp_min(eps)
        return x / n.to(x.dtype)

    def forward(
        self,
        H: torch.Tensor,                     # [B, T, Dh]
        G: torch.Tensor,                     # [B, K, Dg]
        mask: Optional[torch.Tensor],        # [B, T], bool, True=valid
        return_alpha: bool = False,
        return_logits: bool = True,
        **kwargs
    ):
        if mask is not None and mask.dtype != torch.bool:
            mask = mask != 0

        if G.dim() != 3:
            raise ValueError("G must be [B, K, Dg].")

        B, T, Dh = H.shape
        _, K, Dg = G.shape

        alpha_info = {}

        # Protein pooling
        if self.protein_pool_type == "mean":
            if mask is not None:
                w = mask.to(H.dtype).unsqueeze(-1)          # [B, T, 1]
                denom = w.sum(dim=1).clamp_min(1.0)         # [B, 1]
                h_pool = (H * w).sum(dim=1) / denom         # [B, Dh]
            else:
                h_pool = H.mean(dim=1)                      # [B, Dh]

        else:  # "attn"
            h_pool, attn = self.protein_attn_pool(H, mask)  # [B, Dh], [B, T]
            if return_alpha:
                alpha_info["protein_attn"] = attn

        h_pool = self.protein_ln(h_pool)                    # [B, Dh]
        Z = h_pool.unsqueeze(1).expand(B, K, Dh)            # [B, K, Dh]

        # GO side
        G = self.go_ln(G)                                   # [B, K, Dg]

        # Projection
        Zp = self.proj_p(Z)                                 # [B, K, Dz]
        Gz = self.proj_g(G)                                 # [B, K, Dz]

        if self.normalize:
            Zp = self._norm(Zp, dim=-1)
            Gz = self._norm(Gz, dim=-1)

        scores = (Zp * Gz).sum(dim=-1)                      # [B, K]

        if return_logits and self.score_head is not None:
            logits = self.score_head(Zp, Gz)
            if return_alpha:
                return (scores, logits), alpha_info
            return scores, logits

        if return_alpha:
            return scores, alpha_info
        return scores