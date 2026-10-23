"""V2: one shared MLP with score statistics and frozen ESM-1b mean-pooled protein features."""
import torch
from torch import nn
import torch.nn.functional as F


def score_features(scores):
    x = scores.float()
    q = torch.quantile(x, torch.tensor([.1, .25, .5, .75, .9, .95, .99], device=x.device), dim=1).T
    top = torch.topk(x, min(10, x.shape[1]), dim=1).values
    return torch.cat((x.mean(1, keepdim=True), x.std(1, keepdim=True, unbiased=False), q,
                      top[:, :1] - top[:, -1:]), dim=1)


class CardinalityGuidedCalibratorV2(nn.Module):
    def __init__(self, protein_dim, hidden=32, protein_hidden=32, reference_cardinality=15., min_card_gain=.01):
        super().__init__()
        self.register_buffer('protein_mean', torch.zeros(protein_dim))
        self.register_buffer('protein_std', torch.ones(protein_dim))
        self.protein_encoder = nn.Sequential(nn.Linear(protein_dim, protein_hidden), nn.LayerNorm(protein_hidden), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(10 + protein_hidden, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.card_head = nn.Linear(hidden, 1)
        self.boundary_head = nn.Linear(hidden, 1)
        self.scale_head = nn.Linear(hidden, 1)
        self.global_boundary = nn.Parameter(torch.tensor(.30))
        self.card_gain_raw = nn.Parameter(torch.tensor(-2.0))
        self.register_buffer('reference_log_card', torch.log1p(torch.tensor(float(reference_cardinality))))
        self.min_card_gain = float(min_card_gain)
        for layer in (self.boundary_head, self.scale_head, self.card_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.card_head.bias, float(self.reference_log_card))

    @torch.no_grad()
    def set_protein_normalization(self, embeddings):
        x = torch.as_tensor(embeddings, dtype=torch.float32, device=self.protein_mean.device)
        if x.ndim != 2 or x.shape[1] != self.protein_mean.numel():
            raise ValueError('Protein embedding shape mismatch')
        self.protein_mean.copy_(x.mean(0))
        self.protein_std.copy_(x.std(0, unbiased=False).clamp_min(1e-3))

    def forward(self, scores, protein_embeddings):
        if scores.shape[0] != protein_embeddings.shape[0]:
            raise ValueError('Score and protein batch sizes differ')
        protein = (protein_embeddings.float() - self.protein_mean) / self.protein_std
        h = self.trunk(torch.cat([score_features(scores), self.protein_encoder(protein)], dim=1))
        log_card = F.softplus(self.card_head(h))
        card_gain = F.softplus(self.card_gain_raw) + self.min_card_gain
        boundary = self.global_boundary + .20 * torch.tanh(self.boundary_head(h)) - card_gain * (log_card - self.reference_log_card)
        scale = torch.exp(.7 * torch.tanh(self.scale_head(h)))
        adjusted = scale * (scores.float() - boundary)
        return {'scores': adjusted, 'log_card': log_card, 'cardinality': torch.expm1(log_card),
                'boundary': boundary, 'scale': scale, 'card_gain': card_gain}


def soft_f1_loss(adjusted_scores, labels, temperature=.10):
    q = torch.sigmoid(adjusted_scores / temperature)
    y = labels.float()
    tp = (q * y).sum(1)
    f1 = 2 * tp / (q.sum(1) + y.sum(1)).clamp_min(1e-8)
    return 1 - f1.mean()
