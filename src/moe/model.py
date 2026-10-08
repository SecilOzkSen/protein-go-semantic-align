"""Experiment D: ontology-size-independent two-expert neighbour MoE.

Input tensors all [batch, num_GO]. C probabilities are frozen teacher predictions.
Neighbour support and semantic support must be computed without label leakage.
"""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class NeighbourMoEConfig:
    hidden_dim: int = 32
    dropout: float = 0.1
    initial_global_weight: float = 0.95
    eps: float = 1e-6


class NeighbourGuidedMoE(nn.Module):
    def __init__(self, cfg: NeighbourMoEConfig = NeighbourMoEConfig()):
        super().__init__()
        if not 0 < cfg.initial_global_weight < 1:
            raise ValueError('initial_global_weight must be between 0 and 1')
        self.cfg = cfg
        # Neighbour expert calibrates direct and semantic support.
        self.neighbour = nn.Sequential(
            nn.Linear(4, cfg.hidden_dim), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, 1)
        )
        # Gate sees both expert opinions and evidence reliability.
        self.gate = nn.Sequential(
            nn.Linear(6, cfg.hidden_dim), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, 1)
        )
        # Conservative initial preference, not a frozen gate.
        with torch.no_grad():
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.fill_(torch.logit(torch.tensor(cfg.initial_global_weight)).item())

    def forward(self, c_logits, retriever_scores, direct_support,
                semantic_support, neighbour_similarity, return_details=False):
        tensors = (c_logits, retriever_scores, direct_support,
                   semantic_support, neighbour_similarity)
        if any(t.shape != c_logits.shape for t in tensors):
            raise ValueError('All inputs must have identical [B,G] shapes')
        if any(not torch.isfinite(t).all() for t in tensors):
            raise ValueError('Non-finite input detected')
        # Inputs are features, not learnable retriever / teacher paths.
        c_logits, retriever_scores, direct_support, semantic_support, neighbour_similarity = [
            t.detach().float() for t in tensors
        ]
        c_prob = torch.sigmoid(c_logits)
        evidence = torch.stack((direct_support, semantic_support,
                                retriever_scores, neighbour_similarity), dim=-1)
        n_logits = self.neighbour(evidence).squeeze(-1)
        n_prob = torch.sigmoid(n_logits)
        gate_features = torch.stack((c_prob, n_prob, direct_support,
                                     semantic_support, retriever_scores,
                                     neighbour_similarity), dim=-1)
        global_weight = torch.sigmoid(self.gate(gate_features).squeeze(-1))
        prob = global_weight * c_prob + (1.0 - global_weight) * n_prob
        # ASL consumes logits. Clamp only to prevent logit infinities.
        final_logits = torch.logit(prob.clamp(self.cfg.eps, 1 - self.cfg.eps))
        if return_details:
            return {'logits': final_logits, 'probability': prob,
                    'global_weight': global_weight, 'neighbour_probability': n_prob}
        return final_logits
