from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RerankerBatch:
    protein_z: torch.Tensor      # [B, D]
    go_z: torch.Tensor           # [B, K, D]
    retriever_score: torch.Tensor  # [B, K]
    rank_feature: torch.Tensor     # [B, K]
    label: Optional[torch.Tensor] = None  # [B, K]


class CandidateInteractionMLP(nn.Module):
    """
    Lightweight candidate-level reranker for frozen retriever outputs.

    This model does not re-encode protein sequences or GO text. It consumes the
    frozen retriever embeddings and first-stage retrieval features.

    Features per candidate:
      - protein query embedding z_p
      - GO candidate embedding z_g
      - |z_p - z_g|
      - z_p * z_g
      - cosine(z_p, z_g)
      - retriever score
      - rank feature

    Output:
      - candidate logits [B, K]
    """

    def __init__(
        self,
        dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.10,
        use_embeddings: bool = True,
        use_score: bool = True,
        use_rank: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.use_embeddings = bool(use_embeddings)
        self.use_score = bool(use_score)
        self.use_rank = bool(use_rank)

        in_dim = 0
        if self.use_embeddings:
            # z_p, z_g, |z_p-z_g|, z_p*z_g, cosine
            in_dim += 4 * self.dim + 1
        if self.use_score:
            in_dim += 1
        if self.use_rank:
            in_dim += 1

        if in_dim <= 0:
            raise ValueError("At least one feature group must be enabled.")

        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            act,
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        protein_z: torch.Tensor,       # [B,D]
        go_z: torch.Tensor,            # [B,K,D]
        retriever_score: torch.Tensor, # [B,K]
        rank_feature: torch.Tensor,    # [B,K]
    ) -> torch.Tensor:
        if protein_z.dim() != 2:
            raise RuntimeError(f"protein_z must be [B,D], got {tuple(protein_z.shape)}")
        if go_z.dim() != 3:
            raise RuntimeError(f"go_z must be [B,K,D], got {tuple(go_z.shape)}")

        B, K, D = go_z.shape
        if protein_z.size(0) != B or protein_z.size(1) != D:
            raise RuntimeError(
                f"Shape mismatch: protein_z={tuple(protein_z.shape)} go_z={tuple(go_z.shape)}"
            )

        z_p = F.normalize(protein_z.float(), dim=-1)
        z_g = F.normalize(go_z.float(), dim=-1)
        z_p_rep = z_p.unsqueeze(1).expand(B, K, D)

        feats = []

        if self.use_embeddings:
            diff = torch.abs(z_p_rep - z_g)
            prod = z_p_rep * z_g
            cos = F.cosine_similarity(z_p_rep, z_g, dim=-1).unsqueeze(-1)
            feats.extend([z_p_rep, z_g, diff, prod, cos])

        if self.use_score:
            feats.append(retriever_score.float().unsqueeze(-1))

        if self.use_rank:
            feats.append(rank_feature.float().unsqueeze(-1))

        x = torch.cat(feats, dim=-1)  # [B,K,F]
        logits = self.net(x).squeeze(-1)  # [B,K]
        return logits


class ScoreOnlyReranker(nn.Module):
    """
    Very small calibration baseline using only retriever score and rank.
    Useful as R0 baseline.
    """

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        protein_z: torch.Tensor,
        go_z: torch.Tensor,
        retriever_score: torch.Tensor,
        rank_feature: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack([retriever_score.float(), rank_feature.float()], dim=-1)
        return self.net(x).squeeze(-1)
