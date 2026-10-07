from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class PredictionHeadConfig:
    dim: int = 768
    adapter_rank: int = 32
    hidden_dim: int = 256
    dropout: float = 0.1


class LowRankResidualAdapter(nn.Module):
    """Low-rank residual adapter: z -> z + B(phi(A(z))).

    The up projection is zero-initialized, so adapter(z) == z at
    initialization.
    """

    def __init__(self, dim: int, rank: int, activation: nn.Module | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if rank > dim:
            raise ValueError(f"rank must be <= dim, got rank={rank}, dim={dim}")

        self.dim = int(dim)
        self.rank = int(rank)
        self.down = nn.Linear(dim, rank, bias=False)
        self.activation = activation if activation is not None else nn.GELU()
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.up(self.activation(self.down(z)))


class ProteinGOPredictionHead(nn.Module):
    """Ontology-size-independent shared protein-GO prediction head.

    protein_z:        [B, D]
    go_z:             [G, D]
    retriever_scores: [B, G]
    output logits:    [B, G]

    No learnable parameter depends on G.
    """

    def __init__(self, cfg: PredictionHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.protein_adapter = LowRankResidualAdapter(cfg.dim, cfg.adapter_rank)
        self.go_adapter = LowRankResidualAdapter(cfg.dim, cfg.adapter_rank)

        pair_dim = cfg.dim + 1  # interaction D + retriever similarity 1
        self.predictor = nn.Sequential(
            nn.Linear(pair_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(
            self,
            protein_z: torch.Tensor,
            go_z: torch.Tensor,
            retriever_scores: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(protein_z, go_z, retriever_scores)

        h_p = self.protein_adapter(protein_z)  # [B, D]
        h_g = self.go_adapter(go_z)  # [G, D]

        interaction = h_p.unsqueeze(1) * h_g.unsqueeze(0)  # [B, G, D]
        similarity = retriever_scores.unsqueeze(-1)  # [B, G, 1]
        pair_features = torch.cat([interaction, similarity], dim=-1)

        return self.predictor(pair_features).squeeze(-1)  # [B, G]

    def _validate_inputs(
            self,
            protein_z: torch.Tensor,
            go_z: torch.Tensor,
            retriever_scores: torch.Tensor,
    ) -> None:
        if protein_z.ndim != 2:
            raise ValueError(f"protein_z must be [B, D], got {tuple(protein_z.shape)}")
        if go_z.ndim != 2:
            raise ValueError(f"go_z must be [G, D], got {tuple(go_z.shape)}")
        if retriever_scores.ndim != 2:
            raise ValueError(
                f"retriever_scores must be [B, G], got {tuple(retriever_scores.shape)}"
            )

        batch_size, protein_dim = protein_z.shape
        num_go, go_dim = go_z.shape

        if protein_dim != self.cfg.dim:
            raise ValueError(f"protein_z D={protein_dim}, expected {self.cfg.dim}")
        if go_dim != self.cfg.dim:
            raise ValueError(f"go_z D={go_dim}, expected {self.cfg.dim}")

        expected = (batch_size, num_go)
        if tuple(retriever_scores.shape) != expected:
            raise ValueError(
                f"retriever_scores shape mismatch: got {tuple(retriever_scores.shape)}, "
                f"expected {expected}"
            )
