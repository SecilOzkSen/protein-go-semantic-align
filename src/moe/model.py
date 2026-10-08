"""Experiment D: two calibrated experts with pair-specific probability gating."""
from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class MoEConfig:
    hidden_dim: int = 32
    dropout: float = 0.1
    initial_retriever_weight: float = 0.9
    eps: float = 1e-6


class NeighbourGuidedMoE(nn.Module):
    def __init__(self, cfg: MoEConfig | None = None):
        super().__init__()
        self.cfg = cfg or MoEConfig()
        self.retriever = nn.Sequential(nn.Linear(1, self.cfg.hidden_dim), nn.ReLU(), nn.Linear(self.cfg.hidden_dim, 1))
        self.neighbour = nn.Sequential(nn.Linear(4, self.cfg.hidden_dim), nn.ReLU(), nn.Dropout(self.cfg.dropout), nn.Linear(self.cfg.hidden_dim, 1))
        self.gate = nn.Sequential(nn.Linear(6, self.cfg.hidden_dim), nn.ReLU(), nn.Dropout(self.cfg.dropout), nn.Linear(self.cfg.hidden_dim, 1))
        w = self.cfg.initial_retriever_weight
        if not 0 < w < 1: raise ValueError('initial_retriever_weight must be in (0,1)')
        with torch.no_grad():
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.fill_(math.log(w / (1 - w)))

    def forward(self, retriever_scores, direct_support, semantic_support, neighbour_similarity, return_details=False):
        x = [t.float() for t in (retriever_scores, direct_support, semantic_support, neighbour_similarity)]
        if any(t.shape != x[0].shape for t in x): raise ValueError('Input shapes must match [B,G]')
        if any(not torch.isfinite(t).all() for t in x): raise ValueError('Nonfinite MoE features')
        s, d, e, n = x
        pr = torch.sigmoid(self.retriever(s.unsqueeze(-1)).squeeze(-1))
        pn = torch.sigmoid(self.neighbour(torch.stack([s, d, e, n], dim=-1)).squeeze(-1))
        w = torch.sigmoid(self.gate(torch.stack([pr, pn, s, d, e, n], dim=-1)).squeeze(-1))
        p = w * pr + (1 - w) * pn
        logits = torch.logit(p.clamp(self.cfg.eps, 1 - self.cfg.eps))
        if return_details: return {'logits': logits, 'probability': p, 'retriever_probability': pr, 'neighbour_probability': pn, 'retriever_weight': w}
        return logits
