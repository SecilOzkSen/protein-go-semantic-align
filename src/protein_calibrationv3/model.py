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


class ProteinCalibrationV3(nn.Module):
    def __init__(self, protein_dim, hidden=32, protein_hidden=32, ):
        super().__init__()
        self.register_buffer('protein_mean', torch.zeros(protein_dim))
        self.register_buffer('protein_std', torch.ones(protein_dim))
        self.protein_encoder = nn.Sequential(nn.Linear(protein_dim, protein_hidden), nn.LayerNorm(protein_hidden), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(10 + protein_hidden, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.boundary_head = nn.Linear(hidden, 1)
        self.scale_head = nn.Linear(hidden, 1)
        self.global_boundary = nn.Parameter(torch.tensor(.30))
        for layer in (self.boundary_head, self.scale_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

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
        boundary = self.global_boundary + .20 * torch.tanh(self.boundary_head(h))
        scale = torch.exp(.7 * torch.tanh(self.scale_head(h)))
        return {'scores': scale * (scores.float() - boundary), 'boundary': boundary, 'scale': scale}



def cafa_soft_f_loss(scores, labels, temperature=.1):
    """Batch approximation to CAFA protein-centric F: macro P and macro R then harmonic mean."""
    q = torch.sigmoid(scores / temperature)
    y = labels.float()
    tp = (q*y).sum(1)
    precision = (tp / q.sum(1).clamp_min(1e-8)).mean()
    actual = y.sum(1)
    valid = actual > 0
    if not bool(valid.any()): raise ValueError('No annotated proteins in batch')
    recall = (tp[valid] / actual[valid]).mean()
    return 1 - 2*precision*recall/(precision+recall).clamp_min(1e-8)
