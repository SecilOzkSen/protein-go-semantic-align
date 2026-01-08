# src/training/reranker_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import AutoModel
from src.models.go_token_align_pooler import GoTokenAlignPooler

# -----------------------------
# (B) Fast baseline: protein_vec + GO_CLS -> MLP
# -----------------------------
class RerankerConcatMLP(nn.Module):
    """
    protein_vec: [B, Dp]
    go_text: tokenized -> BiomedBERT CLS [B, Dt]
    score = MLP([protein_vec, go_cls])
    """
    def __init__(
        self,
        text_model_name: str,
        protein_dim: int,
        hidden_dim: int = 512,
        freeze_text_encoder: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        dt = self.text_encoder.config.hidden_size

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(protein_dim + dt, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, protein_vec, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        go_cls = out.last_hidden_state[:, 0]                  # [B,Dt]
        x = torch.cat([protein_vec, go_cls], dim=-1)
        return self.mlp(x).squeeze(-1)                        # [B]


# -----------------------------
# (A+) AlignPooler reranker: H tokens + GO_CLS -> GO-conditioned pooling -> MLP
# -----------------------------
class RerankerGoAlignPooler(nn.Module):
    """
    Input:
      H: [B,T,Dh]              protein token/residue reps (from your protein encoder pipeline)
      valid_mask: [B,T] bool   True=valid
      go_text: tokenized for K candidates per protein, shaped as [B*K, L]
    Steps:
      1) Encode GO text with BiomedBERT -> go_cls [B*K, Dt]
      2) Reshape go_cls -> G [B,K,Dt]
      3) Pool protein tokens conditioned on each GO: Z = pooler(H, G) -> [B,K,Dh]
      4) Score each pair with MLP on [Z || G]
    Output:
      logits: [B,K]
    """
    def __init__(
        self,
        text_model_name: str,
        d_h: int,
        d_att: int = 256,
        hidden_dim: int = 512,
        freeze_text_encoder: bool = True,
        dropout: float = 0.0,
        return_alpha: bool = False,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        d_g = self.text_encoder.config.hidden_size

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.pooler = GoTokenAlignPooler(d_h=d_h, d_g=d_g, d_att=d_att, dropout=dropout)

        # LN after pooling is safer
        self.protein_ln = nn.LayerNorm(d_h)
        self.go_ln = nn.LayerNorm(d_g)

        self.scorer = nn.Sequential(
            nn.Linear(d_h + d_g, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

        self._return_alpha_default = return_alpha

    def forward(
        self,
        H: torch.Tensor,                       # [B,T,Dh]
        valid_mask: Optional[torch.Tensor],    # [B,T]
        go_input_ids: torch.Tensor,            # [B*K,L]
        go_attention_mask: torch.Tensor,       # [B*K,L]
        K: int,
        return_alpha: Optional[bool] = None,
    ):
        if return_alpha is None:
            return_alpha = self._return_alpha_default

        B, T, Dh = H.shape
        BK, L = go_input_ids.shape
        assert BK == B * K, f"go batch must be B*K. got {BK} expected {B*K}"

        # 1) GO text encoder
        out = self.text_encoder(input_ids=go_input_ids, attention_mask=go_attention_mask)
        go_cls = out.last_hidden_state[:, 0]                 # [B*K, Dg]
        Dg = go_cls.size(-1)

        # 2) reshape to [B,K,Dg]
        G = go_cls.view(B, K, Dg)

        # 3) GO-conditioned pooling: Z [B,K,Dh]
        Z, alpha_info = self.pooler(H, G, valid_mask, return_alpha=return_alpha)

        Z = self.protein_ln(Z)
        G = self.go_ln(G)

        # 4) pairwise scoring
        x = torch.cat([Z, G], dim=-1)                        # [B,K,Dh+Dg]
        logits = self.scorer(x).squeeze(-1)                  # [B,K]

        if return_alpha:
            return logits, alpha_info
        return logits