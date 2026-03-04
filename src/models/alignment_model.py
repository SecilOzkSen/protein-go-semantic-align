import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from src.models.projection import ProjectionHead
from src.encoders import BioMedBERTEncoder
from src.models.go_token_align_pooler import GoTokenAlignPooler

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
        # zp: [B,D], zg: [B,K,D]
        B, K, D = zg.shape
        zp_exp = zp.unsqueeze(1).expand(B, K, D)
        h = torch.cat([zp_exp, zg, zp_exp * zg, (zp_exp - zg).abs()], dim=-1)
        return self.net(h).squeeze(-1)  # [B,K]


class ProteinGoAligner(nn.Module):
    def __init__(
        self,
        d_h: int,
        d_g: Optional[int] = None,
        d_z: int = 768,
        go_encoder: Optional[BioMedBERTEncoder] = None,
        normalize: bool = True,
        mean_pool: bool = False,
        att_d: int = 256,
        use_score_head: bool = True,
    ):
        super().__init__()
        self.normalize = bool(normalize)
        self.go_encoder = go_encoder

        self.use_score_head = use_score_head # ya da direkt param
        self.score_head = SharedInteractionMLP(d=d_z, hidden=256, dropout=0.1) if self.use_score_head else None

        if self.go_encoder is not None and d_g is None:
            d_g = int(self.go_encoder.model.config.hidden_size)
        if d_g is None:
            raise ValueError("d_g must be provided if go_encoder is None.")

        self.mean_pool = bool(mean_pool)
        self.proj_p = ProjectionHead(d_in=d_h, d_out=d_z)
        self.proj_g = ProjectionHead(d_in=d_g, d_out=d_z)
        self.protein_ln = nn.LayerNorm(d_h)
        self.go_ln = nn.LayerNorm(d_g)

        self.pooler = None if self.mean_pool else GoTokenAlignPooler(d_h=d_h, d_g=d_g, d_att=att_d, dropout=0.05)

    @staticmethod
    def _norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        n = torch.linalg.vector_norm(x.to(torch.float32), dim=dim, keepdim=True).clamp_min(eps)
        return x / n.to(x.dtype)

    def forward(
            self,
            H: torch.Tensor,  # [B,T,Dh]
            G: torch.Tensor,  # [B,K,Dg]
            mask: Optional[torch.Tensor],  # [B,T] bool True=valid
            return_alpha: bool = False,
            return_logits: bool = True,
            **kwargs
    ):
        if mask is not None and mask.dtype != torch.bool:
            mask = mask != 0

        if G.dim() != 3:
            raise ValueError("G must be [B,K,Dg].")

        B, T, Dh = H.shape
        _, K, Dg = G.shape

        alpha_info = {}

        if self.mean_pool:
            # masked mean pool: [B,Dh]
            if mask is not None:
                w = mask.to(H.dtype).unsqueeze(-1)  # [B,T,1]
                denom = w.sum(dim=1).clamp_min(1.0)  # [B,1]
                h_pool = (H * w).sum(dim=1) / denom  # [B,Dh]
            else:
                h_pool = H.mean(dim=1)  # [B,Dh]

            h_pool = self.protein_ln(h_pool)  # LN AFTER pool
            Z = h_pool.unsqueeze(1).expand(B, K, Dh)  # [B,K,Dh]
        else:
            Z, alpha_info = self.pooler(H, G, mask, return_alpha=return_alpha)  # [B,K,Dh]
            Z = self.protein_ln(Z)  # LN on [B,K,Dh]

        G = self.go_ln(G)  # [B,K,Dg]

        Zp = self.proj_p(Z)  # [B,K,Dz]
        Gz = self.proj_g(G)  # [B,K,Dz]

        if self.normalize:
            Zp = self._norm(Zp, dim=-1)
            Gz = self._norm(Gz, dim=-1)

        scores = (Zp * Gz).sum(dim=-1)  # [B,K]

        if return_logits and self.score_head is not None:
            logits = self.score_head(Zp, Gz)  # [B,K]
            return scores, logits if not return_alpha else (scores, logits), alpha_info

        if return_alpha:
            return scores, alpha_info
        return scores
