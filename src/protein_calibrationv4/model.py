"""Frozen-score, cardinality-free residual V3. Outputs probabilities."""
import torch
from torch import nn
import torch.nn.functional as F


def score_features(scores):
    x = scores.float()
    q = torch.quantile(x, torch.tensor([.1,.25,.5,.75,.9,.95,.99], device=x.device), dim=1).T
    top = torch.topk(x, min(10, x.shape[1]), dim=1).values
    return torch.cat((x.mean(1,keepdim=True),x.std(1,keepdim=True,unbiased=False),q,top[:,:1]-top[:,-1:]),dim=1)

class ResidualV3(nn.Module):
    def __init__(self, protein_dim, hidden=32, protein_hidden=32):
        super().__init__()
        self.register_buffer('protein_mean',torch.zeros(protein_dim))
        self.register_buffer('protein_std',torch.ones(protein_dim))
        self.protein_encoder=nn.Sequential(nn.Linear(protein_dim,protein_hidden),nn.LayerNorm(protein_hidden),nn.GELU())
        self.trunk=nn.Sequential(nn.Linear(10+protein_hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.boundary_head=nn.Linear(hidden,1)
        self.scale_head=nn.Linear(hidden,1)
        self.global_boundary=nn.Parameter(torch.tensor(.30))
        self.alpha_logit=nn.Parameter(torch.tensor(-1.38629436)) # alpha = 0.2 initially
        for layer in (self.boundary_head,self.scale_head):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    @torch.no_grad()
    def set_protein_normalization(self,embeddings):
        x=torch.as_tensor(embeddings,dtype=torch.float32,device=self.protein_mean.device)
        self.protein_mean.copy_(x.mean(0))
        self.protein_std.copy_(x.std(0,unbiased=False).clamp_min(1e-3))
    def forward(self,scores,protein_embeddings):
        protein=(protein_embeddings.float()-self.protein_mean)/self.protein_std
        h=self.trunk(torch.cat([score_features(scores),self.protein_encoder(protein)],dim=1))
        boundary=self.global_boundary+.20*torch.tanh(self.boundary_head(h))
        scale=torch.exp(.7*torch.tanh(self.scale_head(h)))
        raw=torch.sigmoid(scores.float())
        corrected=torch.sigmoid(scale*(scores.float()-boundary))
        alpha=torch.sigmoid(self.alpha_logit)
        prob=(1-alpha)*raw+alpha*corrected
        return {'prob':prob,'alpha':alpha,'boundary':boundary,'scale':scale}


def ancestor_max(scores, ancestor_edges):
    """Differentiable max propagation over original scores (transitive ancestor edges)."""
    out=scores.clone()
    for child,parents in enumerate(ancestor_edges):
        if parents.numel():
            out[:,parents]=torch.maximum(out[:,parents],scores[:,child:child+1])
    return out


def pfresgo_soft_f(prob, labels, edges, root_idx=None, temperature=.08,
                   thresholds=None, sweep_temperature=.05):
    """Differentiable approximation of PFresGO's max over global thresholds.

    The SAME threshold is applied to every protein/GO. GO ancestors are
    propagated before thresholding. Precision uses soft predicted-positive
    protein weights; recall uses all proteins, as in the original evaluator.
    """
    pp = ancestor_max(prob, edges)
    yy = ancestor_max(labels.float(), edges)
    if root_idx is not None:
        keep = torch.ones(pp.shape[1], device=pp.device, dtype=torch.bool)
        keep[root_idx] = False
        pp, yy = pp[:, keep], yy[:, keep]
    if thresholds is None:
        thresholds = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    true = yy.sum(dim=1)
    eligible = (true > 0).float()
    fs = []
    for threshold in thresholds:
        q = torch.sigmoid((pp - float(threshold)) / temperature)
        tp = (q * yy).sum(dim=1)
        pred = q.sum(dim=1)
        active = 1 - torch.exp(-pred)
        weight = active * eligible
        precision = ((tp / pred.clamp_min(1e-8)) * weight).sum() / weight.sum().clamp_min(1e-8)
        recall = ((tp / true.clamp_min(1)) * eligible).sum() / prob.shape[0]
        fs.append(2 * precision * recall / (precision + recall).clamp_min(1e-8))
    fs = torch.stack(fs)
    # Soft selection over thresholds, differentiable and bounded by max(F).
    weights = torch.softmax(fs / sweep_temperature, dim=0)
    return 1 - (weights * fs).sum()
