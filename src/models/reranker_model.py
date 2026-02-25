from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union
from transformers import AutoModel


class MaskedMeanPool(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, H: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        H: [B,T,D]
        valid_mask: [B,T] bool (True=valid)
        returns: [B,D]
        """
        if valid_mask is None:
            return H.mean(dim=1)

        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask != 0

        w = valid_mask.to(H.dtype).unsqueeze(-1)     # [B,T,1]
        denom = w.sum(dim=1).clamp_min(1.0)          # [B,1]
        return (H * w).sum(dim=1) / denom            # [B,D]


class RerankerMeanPoolConcatMLP(nn.Module):
    """
    Minimal working reranker.

    Input:
      H: [B,T,Dh] protein token reps (frozen encoder output or stored embeddings)
      valid_mask: [B,T] bool
      go_input_ids: [B*K,L]
      go_attention_mask: [B*K,L]
      K: int

    Output:
      logits: [B,K]

    Notes:
      - Text encoder can be frozen or LoRA-wrapped outside.
    """
    def __init__(
        self,
        text_model_name: str,
        d_h: int, #protein residue(token) embedding dim
        hidden_dim: int = 512,
        freeze_text_encoder: bool = True,
        dropout: float = 0.1,
        use_protein_ln: bool = True,
        use_go_ln: bool = True,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        d_g = int(self.text_encoder.config.hidden_size)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.pool = MaskedMeanPool()
        self.protein_ln = nn.LayerNorm(d_h) if use_protein_ln else nn.Identity()
        self.go_ln = nn.LayerNorm(d_g) if use_go_ln else nn.Identity()

        self.scorer = nn.Sequential(
            nn.Linear(d_h + d_g, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        H: torch.Tensor,                       # [B,T,Dh]
        valid_mask: Optional[torch.Tensor],    # [B,T]
        go_input_ids: torch.Tensor,            # [B*K,L]
        go_attention_mask: torch.Tensor,       # [B*K,L]
        K: int,
        return_alpha: bool = False,            # kept for trainer compat, always ignored
        **kwargs,
    ) -> torch.Tensor:
        B, T, Dh = H.shape
        BK, L = go_input_ids.shape
        if BK != B * K:
            raise RuntimeError(f"go batch must be B*K. got {BK} expected {B*K}")

        # 1) protein -> [B,Dh]
        prot_vec = self.pool(H, valid_mask)         # [B,Dh]
        prot_vec = self.protein_ln(prot_vec)

        # 2) go text -> [B*K,Dg]
        out = self.text_encoder(input_ids=go_input_ids, attention_mask=go_attention_mask)
        go_cls = out.last_hidden_state[:, 0]        # [B*K,Dg]
        go_cls = self.go_ln(go_cls)

        # 3) repeat prot for each candidate
        prot_rep = prot_vec.unsqueeze(1).expand(B, K, Dh).contiguous().view(B * K, Dh)

        # 4) score
        x = torch.cat([prot_rep, go_cls], dim=-1)   # [B*K, Dh+Dg]
        s = self.scorer(x).squeeze(-1)              # [B*K]
        logits = s.view(B, K)

        return logits