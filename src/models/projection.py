import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_out, bias=False)
        self.fc2 = nn.Linear(d_out, d_out, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_in]
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])              # [N, d_in]
        x = self.ln(x)
        y1 = self.fc1(x)                               # [N, d_out]  (skip bu uzayda)
        h = self.act(y1)
        h = self.fc2(h)                                # [N, d_out]
        h.add_(y1).mul_(0.5)                           # in-place residual, ekstra buffer yok
        return h.view(*orig_shape[:-1], h.size(-1))    # [..., d_out]
