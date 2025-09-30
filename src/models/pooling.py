import torch
import torch.nn as nn

class GoSpecificWattiPooling(nn.Module):
    """
    Minimal GO-specific attention pooling.
    """
    def __init__(self, d_h, d_g, d_proj=256, dropout=0.0):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)

    def forward(self, H, G, mask=None, return_alpha=False):
        # H:(B,L,Dh), G:(B,T,Dg)
        K = self.Wk(H)                         # (B,L,P)
        Q = self.Wq(G)                         # (B,T,P)
        logits = torch.einsum('btd,bld->btl', Q, K) * self.scale
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(1), float('-inf'))
        alpha = torch.softmax(logits, dim=-1)  # (B,T,L)
        Z = torch.einsum('btl,bld->btd', alpha, H)
        return (Z, alpha) if return_alpha else Z
