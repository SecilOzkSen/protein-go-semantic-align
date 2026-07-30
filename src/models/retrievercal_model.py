from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedStats(nn.Module):
    """Compute compact per-protein score-distribution summaries."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, score_z: torch.Tensor, rank_feature: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        score_z: [B,K]
        rank_feature: [B,K]
        valid: [B,K] bool
        returns [B,10]
        """
        valid = valid.bool()
        x = torch.where(valid, score_z, torch.zeros_like(score_z))
        r = torch.where(valid, rank_feature, torch.zeros_like(rank_feature))
        w = valid.to(x.dtype)
        n = w.sum(dim=1).clamp_min(1.0)

        mean = (x * w).sum(dim=1) / n
        var = (((x - mean[:, None]) ** 2) * w).sum(dim=1) / n
        std = torch.sqrt(var.clamp_min(0.0) + self.eps)

        neg_large = torch.full_like(x, -1e6)
        x_masked = torch.where(valid, x, neg_large)
        top1 = x_masked.max(dim=1).values
        k5 = min(5, x.size(1))
        top5 = torch.topk(x_masked, k=k5, dim=1).values.mean(dim=1)
        k10 = min(10, x.size(1))
        top10 = torch.topk(x_masked, k=k10, dim=1).values.mean(dim=1)

        sorted_top = torch.topk(x_masked, k=min(20, x.size(1)), dim=1).values
        gap_1_2 = sorted_top[:, 0] - sorted_top[:, 1] if sorted_top.size(1) >= 2 else torch.zeros_like(top1)
        gap_5_10 = (
            sorted_top[:, min(4, sorted_top.size(1) - 1)] - sorted_top[:, min(9, sorted_top.size(1) - 1)]
            if sorted_top.size(1) >= 2
            else torch.zeros_like(top1)
        )

        rank_mean = (r * w).sum(dim=1) / n
        valid_frac = w.mean(dim=1)

        return torch.stack(
            [mean, std, top1, top5, top10, gap_1_2, gap_5_10, rank_mean, valid_frac, torch.log1p(n)],
            dim=-1,
        )


class RetrieverCal(nn.Module):
    """Minimal score/rank calibration decoder."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.stats = MaskedStats()
        self.candidate_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.threshold_head = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.candidate_mlp[-1].weight)
        nn.init.zeros_(self.candidate_mlp[-1].bias)
        nn.init.zeros_(self.threshold_head[-1].weight)
        nn.init.zeros_(self.threshold_head[-1].bias)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.rank_scale = nn.Parameter(torch.tensor(-0.5))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, score_z: torch.Tensor, rank_feature: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.bool()
        score_z = torch.nan_to_num(score_z, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        rank_feature = torch.nan_to_num(rank_feature, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.5)
        score_z = torch.where(valid, score_z, torch.zeros_like(score_z))
        rank_feature = torch.where(valid, rank_feature, torch.zeros_like(rank_feature))
        x = torch.stack([score_z, rank_feature], dim=-1)
        residual = self.candidate_mlp(x).squeeze(-1)
        base = self.score_scale * score_z + self.rank_scale * rank_feature + self.bias + residual
        summary = self.stats(score_z, rank_feature, valid)
        tau = self.threshold_head(summary).squeeze(-1)
        logits = base - tau[:, None]
        return torch.where(valid, logits, torch.full_like(logits, -1e9))


class ScoreSetCal(nn.Module):
    """Small candidate-set calibration Transformer over score/rank only."""

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.10,
        max_k: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_k = int(max_k)
        self.stats = MaskedStats()
        self.input_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_emb = nn.Embedding(max_k, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.threshold_head = nn.Sequential(
            nn.Linear(10 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.threshold_head[-1].weight)
        nn.init.zeros_(self.threshold_head[-1].bias)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.rank_scale = nn.Parameter(torch.tensor(-0.5))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, score_z: torch.Tensor, rank_feature: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.bool()
        B, K = score_z.shape
        if K > self.max_k:
            raise ValueError(f"K={K} exceeds max_k={self.max_k}")
        score_z = torch.nan_to_num(score_z, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        rank_feature = torch.nan_to_num(rank_feature, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.5)
        score_z = torch.where(valid, score_z, torch.zeros_like(score_z))
        rank_feature = torch.where(valid, rank_feature, torch.zeros_like(rank_feature))
        inp = torch.stack([score_z, rank_feature], dim=-1)
        h = self.input_proj(inp)
        pos = torch.arange(K, device=score_z.device).unsqueeze(0).expand(B, K)
        h = h + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=~valid)
        delta = self.delta_head(h).squeeze(-1)
        base = self.score_scale * score_z + self.rank_scale * rank_feature + self.bias
        w = valid.to(h.dtype).unsqueeze(-1)
        h_summary = (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        stats = self.stats(score_z, rank_feature, valid)
        tau = self.threshold_head(torch.cat([stats, h_summary], dim=-1)).squeeze(-1)
        logits = base + delta - tau[:, None]
        return torch.where(valid, logits, torch.full_like(logits, -1e9))


class EmbCal(nn.Module):
    """
    Candidate-independent embedding calibration model.

    Uses score/rank plus pooled protein and GO embeddings, but no candidate-set Transformer.
    This is the cheap diagnostic between ScoreSetCal and full HierCross.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 128, dropout: float = 0.10, proj_dim: Optional[int] = None):
        super().__init__()
        self.stats = MaskedStats()
        self.emb_dim = int(emb_dim)
        self.proj_dim = int(proj_dim or hidden_dim)
        self.protein_proj = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, self.proj_dim), nn.GELU())
        self.go_proj = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, self.proj_dim), nn.GELU())
        feat_dim = 4 * self.proj_dim + 3  # p, g, |p-g|, p*g, score/rank/cos
        self.candidate_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.threshold_head = nn.Sequential(
            nn.Linear(10 + self.proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Do not zero candidate head, embeddings must be allowed to score immediately.
        nn.init.zeros_(self.threshold_head[-1].weight)
        nn.init.zeros_(self.threshold_head[-1].bias)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.rank_scale = nn.Parameter(torch.tensor(-0.5))
        self.cos_scale = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        score_z: torch.Tensor,
        rank_feature: torch.Tensor,
        valid: torch.Tensor,
        protein_z: torch.Tensor,
        go_z: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid.bool()
        score_z = torch.nan_to_num(score_z, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        rank_feature = torch.nan_to_num(rank_feature, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.5)
        score_z = torch.where(valid, score_z, torch.zeros_like(score_z))
        rank_feature = torch.where(valid, rank_feature, torch.zeros_like(rank_feature))

        p = self.protein_proj(protein_z.float())       # [B,H]
        g = self.go_proj(go_z.float())                 # [B,K,H]
        B, K, H = g.shape
        p_rep = p.unsqueeze(1).expand(B, K, H)
        cos = F.cosine_similarity(p_rep, g, dim=-1)
        x = torch.cat(
            [p_rep, g, torch.abs(p_rep - g), p_rep * g, score_z.unsqueeze(-1), rank_feature.unsqueeze(-1), cos.unsqueeze(-1)],
            dim=-1,
        )
        residual = self.candidate_mlp(x).squeeze(-1)
        base = self.score_scale * score_z + self.rank_scale * rank_feature + self.cos_scale * cos + self.bias
        summary = self.stats(score_z, rank_feature, valid)
        tau = self.threshold_head(torch.cat([summary, p], dim=-1)).squeeze(-1)
        logits = base + residual - tau[:, None]
        return torch.where(valid, logits, torch.full_like(logits, -1e9))


class EmbSetCal(nn.Module):
    """
    Embedding-aware candidate-set calibrator.

    Candidate features:
      - projected protein embedding
      - projected GO embedding
      - absolute protein–GO difference
      - element-wise protein–GO product
      - normalized retriever score
      - projected protein–GO cosine similarity

    No rank feature and no positional embedding are used.
    Therefore candidate order is not treated as retriever rank.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.10,
        max_k: int = 1024,
        proj_dim: Optional[int] = None,
    ):
        super().__init__()

        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.proj_dim = int(proj_dim or hidden_dim)
        self.max_k = int(max_k)

        self.protein_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, self.proj_dim),
            nn.GELU(),
        )
        self.go_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, self.proj_dim),
            nn.GELU(),
        )

        self.pos_emb = nn.Embedding(max_k, hidden_dim)

        # p, g, |p-g|, p*g, retriever score, cosine
        token_feat_dim = 4 * self.proj_dim + 1

        self.input_proj = nn.Sequential(
            nn.Linear(token_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_layers,
        )

        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Protein-specific calibration threshold.
        # Summary contains:
        # score mean, std, max, valid fraction, log candidate count
        # + Transformer summary + projected protein
        summary_dim = 5 + hidden_dim + self.proj_dim

        self.threshold_head = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.threshold_head[-1].weight)
        nn.init.zeros_(self.threshold_head[-1].bias)

        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.cos_scale = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _score_summary(
            score_z: torch.Tensor,
            valid: torch.Tensor,
            eps: float = 1e-6,
    ) -> torch.Tensor:
        w = valid.to(score_z.dtype)
        x = torch.where(valid, score_z, torch.zeros_like(score_z))
        n = w.sum(dim=1).clamp_min(1.0)

        mean = (x * w).sum(dim=1) / n
        var = (((x - mean[:, None]) ** 2) * w).sum(dim=1) / n
        std = torch.sqrt(var.clamp_min(0.0) + eps)

        masked = torch.where(
            valid,
            score_z,
            torch.full_like(score_z, -1e9),
        )
        max_score = masked.max(dim=1).values
        max_score = torch.where(
            valid.any(dim=1),
            max_score,
            torch.zeros_like(max_score),
        )

        valid_fraction = w.mean(dim=1)
        log_count = torch.log1p(n)

        return torch.stack(
            [mean, std, max_score, valid_fraction, log_count],
            dim=-1,
        )

    def forward(
        self,
        score_z: torch.Tensor,
        valid: torch.Tensor,
        protein_z: torch.Tensor,
        go_z: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid.bool()

        batch_size, candidate_count = score_z.shape
        if candidate_count > self.max_k:
            raise ValueError(
                f"K={candidate_count} exceeds max_k={self.max_k}"
            )

        score_z = torch.nan_to_num(
            score_z,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)

        score_z = torch.where(
            valid,
            score_z,
            torch.zeros_like(score_z),
        )

        p = self.protein_proj(protein_z.float())  # [B,P]
        g = self.go_proj(go_z.float())            # [B,K,P]

        _, _, proj_dim = g.shape
        p_rep = p.unsqueeze(1).expand(
            batch_size,
            candidate_count,
            proj_dim,
        )

    #    cosine = F.cosine_similarity(
    #        p_rep,
    #        g,
    #        dim=-1,
    #        eps=1e-8,
    #    )

        token_x = torch.cat(
            [
                p_rep,
                g,
                torch.abs(p_rep - g),
                p_rep * g,
                score_z.unsqueeze(-1),
            ],
            dim=-1,
        )

        h = self.input_proj(token_x)

        pos_idx = torch.arange(
            candidate_count,
            device=h.device,
        ).unsqueeze(0).expand(batch_size, candidate_count)

        h = h + self.pos_emb(pos_idx)
        h = self.encoder(
            h,
            src_key_padding_mask=~valid,
        )

        # Prevent invalid query positions from contaminating summaries.
        h = torch.where(
            valid.unsqueeze(-1),
            h,
            torch.zeros_like(h),
        )

        delta = self.delta_head(h).squeeze(-1)
        delta = torch.where(
            valid,
            delta,
            torch.zeros_like(delta),
        )

        base = (
                self.score_scale * score_z
                + self.bias
        )

        weights = valid.to(h.dtype).unsqueeze(-1)
        h_summary = (
            (h * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        )

        score_summary = self._score_summary(
            score_z,
            valid,
        )

        tau_input = torch.cat(
            [score_summary, h_summary, p],
            dim=-1,
        )
        tau = self.threshold_head(tau_input).squeeze(-1)

        logits = base + delta - tau[:, None]

        return torch.where(
            valid,
            logits,
            torch.full_like(logits, -1e9),
        )
