import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    Simple, stable projection head for mapping embeddings to a shared space.
    Design:
      - Linear -> GELU -> Dropout -> Linear
      - Optional LayerNorm on output for stability
      - Residual (optional) if d_in == d_out
    """
    def __init__(self, d_in: int, d_out: int, hidden: int = None, p_drop: float = 0.1, use_layernorm: bool = True):
        super().__init__()
        hidden = hidden or max(d_in, d_out)
        self.fc1 = nn.Linear(d_in, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, d_out, bias=True)
        self.drop = nn.Dropout(p_drop)
        self.use_ln = use_layernorm
        self.ln = nn.LayerNorm(d_out) if use_layernorm else nn.Identity()
        self.use_residual = (d_in == d_out)

        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.ln(h)
        if self.use_residual:
            h = 0.5 * (h + x)
        return h
