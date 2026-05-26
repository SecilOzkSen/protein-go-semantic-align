from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceAwareCandidateInteractionMLP(nn.Module):
    """
    Source-aware reranker for union candidate dumps.

    Candidate features:
      protein_z, go_z, |p-g|, p*g, cosine
      union/retriever score, union rank
      source flags and per-source score/rank features
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.10,
        use_embeddings: bool = True,
        use_union_score: bool = True,
        use_union_rank: bool = True,
        use_source_features: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.use_embeddings = bool(use_embeddings)
        self.use_union_score = bool(use_union_score)
        self.use_union_rank = bool(use_union_rank)
        self.use_source_features = bool(use_source_features)

        if self.use_embeddings:
            self.prot_ln = nn.LayerNorm(self.dim)
            self.go_ln = nn.LayerNorm(self.dim)
            self.prot_proj = nn.Linear(self.dim, self.hidden_dim)
            self.go_proj = nn.Linear(self.dim, self.hidden_dim)
            emb_feat_dim = self.hidden_dim * 4 + 1
        else:
            emb_feat_dim = 0

        scalar_dim = 0
        if self.use_union_score:
            scalar_dim += 1
        if self.use_union_rank:
            scalar_dim += 1
        if self.use_source_features:
            # in_a, in_b, both, score_a, score_b, rank_a, rank_b
            scalar_dim += 7

        in_dim = emb_feat_dim + scalar_dim
        if in_dim <= 0:
            raise ValueError("At least one feature group must be enabled.")

        self.scorer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, max(64, self.hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(max(64, self.hidden_dim // 2), 1),
        )

    def forward(
        self,
        protein_z: torch.Tensor,          # [B,D]
        go_z: torch.Tensor,               # [B,K,D]
        retriever_score: Optional[torch.Tensor] = None,  # [B,K]
        rank_feature: Optional[torch.Tensor] = None,     # [B,K]
        source_score_a: Optional[torch.Tensor] = None,
        source_score_b: Optional[torch.Tensor] = None,
        source_rank_a: Optional[torch.Tensor] = None,
        source_rank_b: Optional[torch.Tensor] = None,
        source_in_a: Optional[torch.Tensor] = None,
        source_in_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, K, D = go_z.shape
        feats = []

        if self.use_embeddings:
            p = self.prot_ln(protein_z.float())
            g = self.go_ln(go_z.float())
            p_rep = p.unsqueeze(1).expand(B, K, D).contiguous()
            p_h = self.prot_proj(p_rep)
            g_h = self.go_proj(g)
            diff = torch.abs(p_h - g_h)
            prod = p_h * g_h
            cos = F.cosine_similarity(p_h, g_h, dim=-1).unsqueeze(-1)
            feats.append(torch.cat([p_h, g_h, diff, prod, cos], dim=-1))

        scalars = []
        if self.use_union_score:
            if retriever_score is None:
                raise RuntimeError("retriever_score is required when use_union_score=True")
            scalars.append(retriever_score.float().unsqueeze(-1))
        if self.use_union_rank:
            if rank_feature is None:
                raise RuntimeError("rank_feature is required when use_union_rank=True")
            scalars.append(rank_feature.float().unsqueeze(-1))
        if self.use_source_features:
            for name, x in [
                ("source_in_a", source_in_a),
                ("source_in_b", source_in_b),
                ("source_score_a", source_score_a),
                ("source_score_b", source_score_b),
                ("source_rank_a", source_rank_a),
                ("source_rank_b", source_rank_b),
            ]:
                if x is None:
                    raise RuntimeError(f"{name} is required when use_source_features=True")
            in_a = source_in_a.float()
            in_b = source_in_b.float()
            both = in_a * in_b
            scalars.extend([
                in_a.unsqueeze(-1),
                in_b.unsqueeze(-1),
                both.unsqueeze(-1),
                source_score_a.float().unsqueeze(-1),
                source_score_b.float().unsqueeze(-1),
                source_rank_a.float().unsqueeze(-1),
                source_rank_b.float().unsqueeze(-1),
            ])

        if scalars:
            feats.append(torch.cat(scalars, dim=-1))

        x = torch.cat(feats, dim=-1)
        logits = self.scorer(x).squeeze(-1)
        return logits


class SourceAwareScoreOnlyReranker(SourceAwareCandidateInteractionMLP):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__(
            dim=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_embeddings=False,
            use_union_score=True,
            use_union_rank=True,
            use_source_features=True,
        )
