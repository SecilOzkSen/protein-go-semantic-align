
import numpy as np
import torch
from torch import nn

def features(scores):
    """No GO labels, only per-protein distribution statistics."""
    x=np.asarray(scores,dtype=np.float32)
    q=np.quantile(x,[.1,.5,.9,.99],axis=1).T
    return np.column_stack([x.mean(1),x.std(1),q]).astype(np.float32)

class OffsetHead(nn.Module):
    """One shared head, protein-specific offset. Ranking within protein is invariant."""
    def __init__(self,n_features=6,hidden=16,max_offset=.5):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_features,hidden),nn.Tanh(),
                               nn.Linear(hidden,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_offset=float(max_offset)

    def forward(self,feat):
        return self.max_offset*torch.tanh(self.net(feat)).squeeze(-1)

def normalize_features(f,mean,std):
    return (f-mean)/np.maximum(std,1e-6)
