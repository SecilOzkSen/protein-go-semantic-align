from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class AsymmetricLossConfig:
    gamma_pos: float = 0.0
    gamma_neg: float = 4.0
    clip: float = 0.05
    eps: float = 1e-8


class AsymmetricLoss(nn.Module):
    """
    Multi-label Asymmetric Loss (ASL).

    logits:  [B, G]
    targets: [B, G], binary {0,1}

    Important:
        Focusing is applied to the probability of the TRUE class:

            p_t = p       for positive labels
            p_t = p_neg   for negative labels

        weight = (1 - p_t) ** gamma

    This is NOT (1 - p_pos - p_neg) ** gamma.
    """

    def __init__(self, cfg: AsymmetricLossConfig):
        super().__init__()
        if cfg.gamma_pos < 0 or cfg.gamma_neg < 0:
            raise ValueError("gamma_pos and gamma_neg must be >= 0")
        if not 0.0 <= cfg.clip < 1.0:
            raise ValueError("clip must satisfy 0 <= clip < 1")
        if cfg.eps <= 0:
            raise ValueError("eps must be > 0")
        self.cfg = cfg

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            *,
            return_diagnostics: bool = False,
    ):
        self._validate(logits, targets)

        # fp32 probability math is safer under autocast.
        probs = torch.sigmoid(logits.float())
        targets_f = targets.float()
        anti_targets = 1.0 - targets_f

        p_pos = probs
        p_neg = 1.0 - probs

        # ASL asymmetric clipping for negatives.
        if self.cfg.clip > 0.0:
            p_neg = (p_neg + self.cfg.clip).clamp(max=1.0)

        log_pos = torch.log(p_pos.clamp(min=self.cfg.eps))
        log_neg = torch.log(p_neg.clamp(min=self.cfg.eps))

        # Probability assigned to the correct class.
        p_t = p_pos * targets_f + p_neg * anti_targets

        gamma = (
                self.cfg.gamma_pos * targets_f
                + self.cfg.gamma_neg * anti_targets
        )
        weight = torch.pow(1.0 - p_t, gamma)

        pos_loss_matrix = -(targets_f * log_pos * weight)
        neg_loss_matrix = -(anti_targets * log_neg * weight)
        loss_matrix = pos_loss_matrix + neg_loss_matrix

        # Mean over all protein-GO pairs.
        loss = loss_matrix.mean()

        if not return_diagnostics:
            return loss

        with torch.no_grad():
            pos_mask = targets_f > 0.5
            neg_mask = ~pos_mask
            num_pos = int(pos_mask.sum().item())
            num_neg = int(neg_mask.sum().item())

            zero = torch.zeros((), device=logits.device)

            pos_loss_mean = (
                pos_loss_matrix[pos_mask].mean() if num_pos else zero
            )
            neg_loss_mean = (
                neg_loss_matrix[neg_mask].mean() if num_neg else zero
            )
            pos_prob_mean = probs[pos_mask].mean() if num_pos else zero
            neg_prob_mean = probs[neg_mask].mean() if num_neg else zero

            if num_neg:
                neg_probs = probs[neg_mask]

                # With clipping, p <= clip makes p_neg == 1 and therefore
                # the negative loss exactly zero.
                suppressed = (
                    (neg_probs <= self.cfg.clip).float().mean()
                    if self.cfg.clip > 0
                    else zero
                )

                easy = (neg_probs < 0.10).float().mean()
                medium = (
                    ((neg_probs >= 0.10) & (neg_probs < 0.50))
                    .float()
                    .mean()
                )
                hard = (neg_probs >= 0.50).float().mean()
            else:
                suppressed = easy = medium = hard = zero

            diagnostics: Dict[str, float] = {
                "loss": float(loss.detach().item()),
                "positive_loss_mean": float(pos_loss_mean.item()),
                "negative_loss_mean": float(neg_loss_mean.item()),
                "positive_probability_mean": float(pos_prob_mean.item()),
                "negative_probability_mean": float(neg_prob_mean.item()),
                "probability_gap": float(
                    (pos_prob_mean - neg_prob_mean).item()
                ),
                "suppressed_negative_fraction": float(suppressed.item()),
                "easy_negative_fraction": float(easy.item()),
                "medium_negative_fraction": float(medium.item()),
                "hard_negative_fraction": float(hard.item()),
                "num_positive_pairs": num_pos,
                "num_negative_pairs": num_neg,
            }

        return loss, diagnostics

    @staticmethod
    def _validate(logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                f"logits/targets must be [B,G], got "
                f"{tuple(logits.shape)} and {tuple(targets.shape)}"
            )
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits/targets shape mismatch: "
                f"{tuple(logits.shape)} vs {tuple(targets.shape)}"
            )
        if not torch.isfinite(logits).all():
            raise ValueError("logits contain NaN or Inf")
        if not torch.isfinite(targets).all():
            raise ValueError("targets contain NaN or Inf")
        if torch.any((targets != 0) & (targets != 1)):
            raise ValueError("targets must be binary {0,1}")


def asl_preflight_configs() -> Tuple[AsymmetricLossConfig, ...]:
    return (
        AsymmetricLossConfig(gamma_pos=0.0, gamma_neg=2.0, clip=0.00),
        AsymmetricLossConfig(gamma_pos=0.0, gamma_neg=4.0, clip=0.00),
        AsymmetricLossConfig(gamma_pos=0.0, gamma_neg=2.0, clip=0.05),
        AsymmetricLossConfig(gamma_pos=0.0, gamma_neg=4.0, clip=0.05),
    )
