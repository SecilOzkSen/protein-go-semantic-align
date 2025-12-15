import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class GoTokenAlignPooler(nn.Module):
    """
    H: [B, T, Dh]
    G: [B, K, Dg]
    valid_mask: [B, T] bool, True=valid token
    returns:
      Z: [B, K, Dh]
      alpha_info: {"alpha_full": [B, K, T]}
    """
    def __init__(self, d_h: int, d_g: int, d_att: int = 256, dropout: float = 0.0):
        super().__init__()
        self.h_proj = nn.Linear(d_h, d_att, bias=False)
        self.g_proj = nn.Linear(d_g, d_att, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(
        self,
        H: torch.Tensor,
        G: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        return_alpha: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, Dh = H.shape
        _, K, Dg = G.shape

        Hk = self.h_proj(H)          # [B,T,A]
        Gq = self.g_proj(G)          # [B,K,A]
        Hk = self.dropout(Hk)
        Gq = self.dropout(Gq)

        # logits[b,k,t] = dot(Hk[b,t], Gq[b,k])
        logits = torch.einsum("bta,bka->bkt", Hk, Gq)  # [B,K,T]

        if valid_mask is not None:
            if valid_mask.dtype != torch.bool:
                valid_mask = valid_mask != 0
            logits = logits.masked_fill(~valid_mask.unsqueeze(1), -1e9)

        alpha = F.softmax(logits, dim=-1)              # [B,K,T]
        Z = torch.einsum("bkt,btd->bkd", alpha, H)     # [B,K,Dh]

        alpha_info = {}
        if return_alpha:
            alpha_info["alpha_full"] = alpha
        return Z, alpha_info
