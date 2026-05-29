from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemExpInteractionMLP(nn.Module):
    """
    P3a semantic-expansion candidate reranker.

    Candidate-level features:
      protein_z, go_z, |protein-go|, protein*go, cosine
      normalized P3a/SemExp candidate score
      normalized rank feature
      GO metadata: namespace one-hot + log_count + IC
      SemExp metadata: direct/parent/child/sibling/text flags, seed score/rank,
                       relation distance, text-neighbor sim, expanded flag
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.10,
        go_meta_dim: int = 5,
        semexp_dim: int = 10,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.go_meta_dim = int(go_meta_dim)
        self.semexp_dim = int(semexp_dim)

        self.prot_proj = nn.Linear(self.dim, self.hidden_dim)
        self.go_proj = nn.Linear(self.dim, self.hidden_dim)

        # prot, go, |diff|, product, cosine, score, rank, GO meta, SemExp meta
        in_dim = self.hidden_dim * 4 + 1 + 1 + 1 + self.go_meta_dim + self.semexp_dim

        self.scorer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(
        self,
        protein_z: torch.Tensor,          # [B,D]
        go_z: torch.Tensor,               # [B,K,D]
        retriever_score: torch.Tensor,    # [B,K]
        rank_feature: torch.Tensor,       # [B,K]
        go_meta: torch.Tensor,            # [B,K,M]
        semexp_feat: torch.Tensor,        # [B,K,S]
    ) -> torch.Tensor:
        B, K, _ = go_z.shape

        p = self.prot_proj(protein_z)       # [B,H]
        g = self.go_proj(go_z)              # [B,K,H]
        p_rep = p.unsqueeze(1).expand(B, K, self.hidden_dim)

        diff = torch.abs(p_rep - g)
        prod = p_rep * g
        cos = F.cosine_similarity(p_rep, g, dim=-1).unsqueeze(-1)

        x = torch.cat(
            [
                p_rep,
                g,
                diff,
                prod,
                cos,
                retriever_score.unsqueeze(-1),
                rank_feature.unsqueeze(-1),
                go_meta,
                semexp_feat,
            ],
            dim=-1,
        )
        return self.scorer(x).squeeze(-1)


class SemExpScoreOnlyReranker(nn.Module):
    """Calibration-only baseline for P3a semantic-expanded candidates."""
    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.10,
        go_meta_dim: int = 5,
        semexp_dim: int = 10,
    ):
        super().__init__()
        in_dim = 2 + int(go_meta_dim) + int(semexp_dim)  # score, rank, meta
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )

    def forward(
        self,
        protein_z: torch.Tensor,
        go_z: torch.Tensor,
        retriever_score: torch.Tensor,
        rank_feature: torch.Tensor,
        go_meta: torch.Tensor,
        semexp_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [retriever_score.unsqueeze(-1), rank_feature.unsqueeze(-1), go_meta, semexp_feat],
            dim=-1,
        )
        return self.net(x).squeeze(-1)
