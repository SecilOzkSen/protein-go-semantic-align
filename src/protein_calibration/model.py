"""Single shared MLP, cardinality-guided adaptive affine decision scores."""
import torch
from torch import nn
import torch.nn.functional as F


def score_features(scores):
    x = scores.float()
    q = torch.quantile(x, torch.tensor([.1, .25, .5, .75, .9, .95, .99], device=x.device), dim=1).T
    top = torch.topk(x, min(10, x.shape[1]), dim=1).values
    return torch.cat((x.mean(1, keepdim=True), x.std(1, keepdim=True, unbiased=False), q,
                      (top[:, :1] - top[:, -1:])), dim=1)


class CardinalityGuidedCalibrator(nn.Module):
    """All outputs are functions of scores only; labels are training targets only.

    c_hat = expm1(log_card). Decision boundary explicitly depends on log_card.
    z_pg = positive_scale_p * (score_pg - boundary_p).
    """
    def __init__(self, hidden=32, reference_cardinality=15., min_card_gain=.01):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(10, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.card_head = nn.Linear(hidden, 1)
        self.boundary_head = nn.Linear(hidden, 1)
        self.scale_head = nn.Linear(hidden, 1)
        self.global_boundary = nn.Parameter(torch.tensor(.30))
        self.card_gain_raw = nn.Parameter(torch.tensor(-2.0))
        self.register_buffer('reference_log_card', torch.log1p(torch.tensor(float(reference_cardinality))))
        self.min_card_gain = float(min_card_gain)
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.zeros_(self.boundary_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.card_head.weight)
        nn.init.constant_(self.card_head.bias, float(self.reference_log_card))

    def forward(self, scores):
        h = self.trunk(score_features(scores))
        log_card = F.softplus(self.card_head(h))  # log(1 + cardinality) >= 0
        card_gain = F.softplus(self.card_gain_raw) + self.min_card_gain
        # Larger predicted cardinality lowers the decision boundary, all else equal.
        boundary = self.global_boundary + .20 * torch.tanh(self.boundary_head(h)) - card_gain * (log_card - self.reference_log_card)
        scale = torch.exp(.7 * torch.tanh(self.scale_head(h)))  # positive, bounded
        adjusted = scale * (scores.float() - boundary)
        return {'scores': adjusted, 'log_card': log_card, 'cardinality': torch.expm1(log_card),
                'boundary': boundary, 'scale': scale, 'card_gain': card_gain}


def soft_f1_loss(adjusted_scores, labels, temperature=.10):
    probabilities = torch.sigmoid(adjusted_scores / temperature)
    positives = labels.float()
    tp = (probabilities * positives).sum(1)
    f1 = 2 * tp / (probabilities.sum(1) + positives.sum(1)).clamp_min(1e-8)
    return 1 - f1.mean()
