from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AsymmetricLossConfig:
    """
    Configuration for multi-label Asymmetric Loss (ASL).

    gamma_pos:
        Focusing strength for positive labels.
        We start from 0.0 so positives are not down-weighted.

    gamma_neg:
        Focusing strength for negative labels.
        Larger values suppress easy negatives more aggressively.

    clip:
        Asymmetric probability shifting for negatives.
        p_neg = clamp((1 - p) + clip, max=1).

        A positive clip value makes sufficiently easy negatives contribute
        exactly zero loss.

    eps:
        Numerical stability for logarithms.
    """

    gamma_pos: float = 0.0
    gamma_neg: float = 4.0
    clip: float = 0.05
    eps: float = 1e-8


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for full-GO multi-label prediction.

    Expected shapes
    ---------------
    logits:
        [B, G]

    targets:
        [B, G], binary {0, 1}

    The implementation keeps positive and negative contributions separate
    so Experiment B preflight can inspect whether negative suppression is
    behaving as intended.

    Reference form
    --------------
        p      = sigmoid(logit)
        p_pos  = p
        p_neg  = 1 - p

        if clip > 0:
            p_neg = clamp(p_neg + clip, max=1)

        CE = y * log(p_pos) + (1-y) * log(p_neg)

        asymmetric_weight =
            (1 - p_pos - p_neg) ^
            (gamma_pos * y + gamma_neg * (1-y))

        loss = - asymmetric_weight * CE

    With clipping, very easy negatives can receive exactly zero loss.
    """

    def __init__(self, cfg: AsymmetricLossConfig):
        super().__init__()

        if cfg.gamma_pos < 0:
            raise ValueError("gamma_pos must be >= 0")
        if cfg.gamma_neg < 0:
            raise ValueError("gamma_neg must be >= 0")
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

        targets = targets.to(dtype=logits.dtype)

        # Compute sigmoid in fp32 for numerical stability under autocast.
        probs = torch.sigmoid(logits.float())
        targets_f = targets.float()

        p_pos = probs
        p_neg_raw = 1.0 - probs

        if self.cfg.clip > 0.0:
            p_neg = (p_neg_raw + self.cfg.clip).clamp(max=1.0)
        else:
            p_neg = p_neg_raw

        log_pos = torch.log(p_pos.clamp(min=self.cfg.eps))
        log_neg = torch.log(p_neg.clamp(min=self.cfg.eps))

        pos_ce = targets_f * log_pos
        neg_ce = (1.0 - targets_f) * log_neg

        # ASL focusing term.
        #
        # For positives:
        #   1 - p_pos = 1 - p
        #
        # For negatives after clipping:
        #   1 - p_neg
        #
        one_sided_gamma = (
                self.cfg.gamma_pos * targets_f
                + self.cfg.gamma_neg * (1.0 - targets_f)
        )

        one_sided_w = torch.pow(
            1.0 - p_pos - p_neg,
            one_sided_gamma,
        )

        pos_loss_matrix = -(one_sided_w * pos_ce)
        neg_loss_matrix = -(one_sided_w * neg_ce)
        loss_matrix = pos_loss_matrix + neg_loss_matrix

        # Mean over all protein-GO pairs. This is the optimized scalar.
        loss = loss_matrix.mean()

        if not return_diagnostics:
            return loss

        with torch.no_grad():
            pos_mask = targets_f > 0.5
            neg_mask = ~pos_mask

            num_pos = int(pos_mask.sum().item())
            num_neg = int(neg_mask.sum().item())

            pos_loss = (
                pos_loss_matrix[pos_mask].mean()
                if num_pos > 0
                else torch.zeros((), device=logits.device)
            )

            neg_loss = (
                neg_loss_matrix[neg_mask].mean()
                if num_neg > 0
                else torch.zeros((), device=logits.device)
            )

            pos_prob_mean = (
                probs[pos_mask].mean()
                if num_pos > 0
                else torch.zeros((), device=logits.device)
            )

            neg_prob_mean = (
                probs[neg_mask].mean()
                if num_neg > 0
                else torch.zeros((), device=logits.device)
            )

            # Under ASL clipping, an easy negative is completely suppressed
            # when p_neg reaches 1 after clipping:
            #
            #   (1 - p) + clip >= 1
            #   p <= clip
            #
            if num_neg > 0 and self.cfg.clip > 0:
                suppressed_fraction = (
                    (probs[neg_mask] <= self.cfg.clip)
                    .float()
                    .mean()
                )
            else:
                suppressed_fraction = torch.zeros(
                    (), device=logits.device
                )

            # Useful preflight buckets.
            # These are diagnostics only, not part of the loss.
            if num_neg > 0:
                neg_probs = probs[neg_mask]

                easy_neg_fraction = (
                    (neg_probs < 0.10).float().mean()
                )
                medium_neg_fraction = (
                    ((neg_probs >= 0.10) & (neg_probs < 0.50))
                    .float()
                    .mean()
                )
                hard_neg_fraction = (
                    (neg_probs >= 0.50).float().mean()
                )
            else:
                zero = torch.zeros((), device=logits.device)
                easy_neg_fraction = zero
                medium_neg_fraction = zero
                hard_neg_fraction = zero

            diagnostics: Dict[str, float] = {
                "loss": float(loss.detach().item()),
                "positive_loss_mean": float(pos_loss.item()),
                "negative_loss_mean": float(neg_loss.item()),
                "positive_probability_mean": float(pos_prob_mean.item()),
                "negative_probability_mean": float(neg_prob_mean.item()),
                "probability_gap": float(
                    (pos_prob_mean - neg_prob_mean).item()
                ),
                "suppressed_negative_fraction": float(
                    suppressed_fraction.item()
                ),
                "easy_negative_fraction": float(
                    easy_neg_fraction.item()
                ),
                "medium_negative_fraction": float(
                    medium_neg_fraction.item()
                ),
                "hard_negative_fraction": float(
                    hard_neg_fraction.item()
                ),
                "num_positive_pairs": num_pos,
                "num_negative_pairs": num_neg,
            }

        return loss, diagnostics

    @staticmethod
    def _validate(
            logits: torch.Tensor,
            targets: torch.Tensor,
    ) -> None:
        if logits.ndim != 2:
            raise ValueError(
                f"logits must be [B, G], got {tuple(logits.shape)}"
            )

        if targets.ndim != 2:
            raise ValueError(
                f"targets must be [B, G], got {tuple(targets.shape)}"
            )

        if logits.shape != targets.shape:
            raise ValueError(
                "logits/targets shape mismatch: "
                f"{tuple(logits.shape)} vs {tuple(targets.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise ValueError("logits contain NaN or Inf")

        if not torch.isfinite(targets).all():
            raise ValueError("targets contain NaN or Inf")

        if torch.any((targets != 0) & (targets != 1)):
            raise ValueError("targets must be binary {0, 1}")


def asl_preflight_configs() -> Tuple[AsymmetricLossConfig, ...]:
    """
    Small mechanistic preflight grid.

    gamma_pos remains fixed at zero so positive examples retain their
    full contribution. We vary only negative focusing and clipping.

    This is deliberately not a broad hyperparameter sweep.
    """
    return (
        AsymmetricLossConfig(
            gamma_pos=0.0,
            gamma_neg=2.0,
            clip=0.00,
        ),
        AsymmetricLossConfig(
            gamma_pos=0.0,
            gamma_neg=4.0,
            clip=0.00,
        ),
        AsymmetricLossConfig(
            gamma_pos=0.0,
            gamma_neg=2.0,
            clip=0.05,
        ),
        AsymmetricLossConfig(
            gamma_pos=0.0,
            gamma_neg=4.0,
            clip=0.05,
        ),
    )
