from __future__ import annotations

import math
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


class GoTokenAlignPooler(nn.Module):
    """
    Protein-conditioned GO token pooling.

    prot_vec: [BK, Dh]
    go_tokens: [BK, L, Dg]
    go_mask: [BK, L] (1/0 or bool)

    returns:
      pooled_go: [BK, Dg]
      alpha: [BK, L]
    """
    def __init__(
        self,
        d_prot: int,
        d_go: int,
        d_attn: Optional[int] = None,
        dropout: float = 0.1,
        use_go_residual: bool = True,
        use_out_ln: bool = True,
    ):
        super().__init__()
        d_attn = int(d_attn or d_go)

        self.q_proj = nn.Linear(d_prot, d_attn)
        self.k_proj = nn.Linear(d_go, d_attn)
        self.v_proj = nn.Linear(d_go, d_go)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.out_ln = nn.LayerNorm(d_go) if use_out_ln else nn.Identity()
        self.use_go_residual = bool(use_go_residual)

        self.scale = math.sqrt(float(d_attn))

    def forward(
        self,
        prot_vec: torch.Tensor,            # [BK, Dh]
        go_tokens: torch.Tensor,           # [BK, L, Dg]
        go_mask: Optional[torch.Tensor],   # [BK, L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        BK, L, Dg = go_tokens.shape

        q = self.q_proj(prot_vec).unsqueeze(1)       # [BK,1,Da]
        k = self.k_proj(go_tokens)                   # [BK,L,Da]
        v = self.v_proj(go_tokens)                   # [BK,L,Dg]

        scores = torch.sum(q * k, dim=-1) / self.scale   # [BK,L]

        if go_mask is not None:
            if go_mask.dtype != torch.bool:
                go_mask = go_mask != 0
            scores = scores.masked_fill(~go_mask, -1e9)

        alpha = torch.softmax(scores, dim=-1)            # [BK,L]
        alpha = self.dropout(alpha)

        pooled = torch.sum(alpha.unsqueeze(-1) * v, dim=1)   # [BK,Dg]

        if self.use_go_residual:
            cls_vec = go_tokens[:, 0, :]                    # [BK,Dg]
            pooled = pooled + cls_vec

        pooled = self.out_ln(pooled)
        return pooled, alpha


class RerankerMeanPoolConcatMLP(nn.Module):
    """
    Reranker with optional protein-conditioned GO token align pooler.

    Input:
      H: [B,T,Dh] protein token reps
      valid_mask: [B,T] bool
      go_input_ids: [B*K,L]
      go_attention_mask: [B*K,L]
      K: int

    Output:
      logits: [B,K]

    If return_alpha=True:
      returns dict:
        {
          "logits": [B,K],
          "alpha": [B,K,L] or None
        }
    """
    def __init__(
        self,
        text_model_name: str,
        d_h: int,
        hidden_dim: int = 512,
        freeze_text_encoder: bool = True,
        dropout: float = 0.1,
        use_protein_ln: bool = True,
        use_go_ln: bool = True,
        use_go_token_align_pooler: bool = True,
        go_align_attn_dim: Optional[int] = None,
        go_align_dropout: float = 0.1,
        use_go_residual: bool = True,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        d_g = int(self.text_encoder.config.hidden_size)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.use_go_token_align_pooler = bool(use_go_token_align_pooler)

        self.pool = MaskedMeanPool()
        self.protein_ln = nn.LayerNorm(d_h) if use_protein_ln else nn.Identity()
        self.go_ln = nn.LayerNorm(d_g) if use_go_ln else nn.Identity()

        if self.use_go_token_align_pooler:
            self.go_token_align_pooler = GoTokenAlignPooler(
                d_prot=d_h,
                d_go=d_g,
                d_attn=go_align_attn_dim,
                dropout=go_align_dropout,
                use_go_residual=use_go_residual,
                use_out_ln=False,   # external LN below
            )
        else:
            self.go_token_align_pooler = None

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
        return_alpha: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, Dh = H.shape
        BK, L = go_input_ids.shape
        if BK != B * K:
            raise RuntimeError(f"go batch must be B*K. got {BK} expected {B*K}")

        # 1) protein -> [B,Dh]
        prot_vec = self.pool(H, valid_mask)         # [B,Dh]
        prot_vec = self.protein_ln(prot_vec)

        # 2) repeat protein for each candidate
        prot_rep = prot_vec.unsqueeze(1).expand(B, K, Dh).contiguous().view(B * K, Dh)  # [BK,Dh]

        # 3) GO text encoder output
        out = self.text_encoder(input_ids=go_input_ids, attention_mask=go_attention_mask)
        go_tokens = out.last_hidden_state           # [BK,L,Dg]

        alpha = None
        if self.use_go_token_align_pooler:
            go_vec, alpha = self.go_token_align_pooler(
                prot_vec=prot_rep,
                go_tokens=go_tokens,
                go_mask=go_attention_mask,
            )                                       # [BK,Dg], [BK,L]
        else:
            go_vec = go_tokens[:, 0, :]             # CLS fallback [BK,Dg]

        go_vec = self.go_ln(go_vec)

        # 4) score
        x = torch.cat([prot_rep, go_vec], dim=-1)   # [BK, Dh+Dg]
        s = self.scorer(x).squeeze(-1)              # [BK]
        logits = s.view(B, K)                       # [B,K]

        if return_alpha:
            if alpha is not None:
                alpha = alpha.view(B, K, L)
            return {
                "logits": logits,
                "alpha": alpha,
            }

        return logits