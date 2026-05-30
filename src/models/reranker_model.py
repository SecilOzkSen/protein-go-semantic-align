from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.hidden_dim = int(hidden_dim)

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

        # Project both modalities to a shared scorer space
        self.prot_proj = nn.Linear(d_h, hidden_dim)
        self.go_proj = nn.Linear(d_g, hidden_dim)

        # Features: prot, go, |prot-go|, prot*go, cosine
        scorer_in_dim = hidden_dim * 4 + 1

        self.scorer = nn.Sequential(
            nn.Linear(scorer_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        H: torch.Tensor,                       # [B,T,Dh]
        valid_mask: Optional[torch.Tensor],   # [B,T]
        go_input_ids: torch.Tensor,           # [B*K,L]
        go_attention_mask: torch.Tensor,      # [B*K,L]
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

        # 4) shared scorer space
        prot_h = self.prot_proj(prot_rep)           # [BK,H]
        go_h = self.go_proj(go_vec)                 # [BK,H]

        # 5) richer interaction features
        diff = torch.abs(prot_h - go_h)             # [BK,H]
        prod = prot_h * go_h                        # [BK,H]
        cos = F.cosine_similarity(prot_h, go_h, dim=-1).unsqueeze(-1)  # [BK,1]

        x = torch.cat([prot_h, go_h, diff, prod, cos], dim=-1)         # [BK, 4H+1]
        s = self.scorer(x).squeeze(-1)                                    # [BK]
        logits = s.view(B, K)                                             # [B,K]

        if return_alpha:
            if alpha is not None:
                alpha = alpha.view(B, K, L)
            return {
                "logits": logits,
                "alpha": alpha,
            }

        return logits

class ResidueGoCrossAttentionReranker(nn.Module):
    """
    Direct P3a-HierCross reranker.

    Candidate GO text tokens cross-attend to residue-level protein embeddings.
    This is retrieval-conditioned and candidate-local, not full-ontology decoding.

    Inputs:
      H: [B,T,Dh]
      valid_mask: [B,T]
      go_input_ids: [B*K,L]
      go_attention_mask: [B*K,L]
      K: int
      retriever_score: optional [B,K]
      rank_feature: optional [B,K]

    Output:
      logits: [B,K]
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
        cross_dim: int = 256,
        cross_heads: int = 4,
        cross_dropout: float = 0.1,
        candidate_chunk_size: int = 16,
        use_retriever_features: bool = True,
        use_cls_residual: bool = True,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        d_g = int(self.text_encoder.config.hidden_size)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.hidden_dim = int(hidden_dim)
        self.cross_dim = int(cross_dim)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.use_retriever_features = bool(use_retriever_features)
        self.use_cls_residual = bool(use_cls_residual)

        self.pool = MaskedMeanPool()
        self.protein_ln = nn.LayerNorm(d_h) if use_protein_ln else nn.Identity()
        self.go_ln = nn.LayerNorm(d_g) if use_go_ln else nn.Identity()

        self.prot_cross_proj = nn.Linear(d_h, self.cross_dim)
        self.go_cross_proj = nn.Linear(d_g, self.cross_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.cross_dim,
            num_heads=int(cross_heads),
            dropout=float(cross_dropout),
            batch_first=True,
        )
        self.cross_ln = nn.LayerNorm(self.cross_dim)

        self.prot_proj = nn.Linear(d_h, hidden_dim)
        self.go_proj = nn.Linear(self.cross_dim, hidden_dim)

        # Features: prot, go, |diff|, product, cosine, optional retriever_score/rank
        scorer_in_dim = hidden_dim * 4 + 1 + (2 if self.use_retriever_features else 0)

        self.scorer = nn.Sequential(
            nn.Linear(scorer_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _masked_mean(H: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return H.mean(dim=1)
        if mask.dtype != torch.bool:
            mask = mask != 0
        w = mask.to(H.dtype).unsqueeze(-1)
        return (H * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def _encode_go_tokens(
        self,
        go_input_ids: torch.Tensor,
        go_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.text_encoder(input_ids=go_input_ids, attention_mask=go_attention_mask)
        go_tokens = out.last_hidden_state
        go_tokens = self.go_ln(go_tokens)
        return go_tokens

    def forward(
        self,
        H: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        go_input_ids: torch.Tensor,
        go_attention_mask: torch.Tensor,
        K: int,
        retriever_score: Optional[torch.Tensor] = None,
        rank_feature: Optional[torch.Tensor] = None,
        return_alpha: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, Dh = H.shape
        BK, L = go_input_ids.shape
        K = int(K)
        if BK != B * K:
            raise RuntimeError(f"go batch must be B*K. got {BK} expected {B*K}")

        if valid_mask is not None and valid_mask.dtype != torch.bool:
            valid_mask = valid_mask != 0

        H_norm = self.protein_ln(H)
        prot_vec = self.pool(H_norm, valid_mask)  # [B,Dh]
        prot_h_all = self.prot_proj(prot_vec)     # [B,H]

        # Protein residues projected once, then repeated per candidate chunk.
        H_cross = self.prot_cross_proj(H_norm)    # [B,T,C]
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask        # True means masked for MultiheadAttention

        logits_chunks = []
        alpha_chunks = [] if return_alpha else None
        csz = max(1, int(self.candidate_chunk_size))

        for s in range(0, K, csz):
            e = min(K, s + csz)
            Ck = e - s

            flat_s = s * B
            # Actual flat layout is [B*K,L], where row b*K + k. Need index rows by candidate slice.
            rows = []
            for b in range(B):
                base = b * K
                rows.extend(range(base + s, base + e))
            row_idx = torch.as_tensor(rows, dtype=torch.long, device=go_input_ids.device)

            ids_c = go_input_ids.index_select(0, row_idx)           # [B*Ck,L]
            mask_c = go_attention_mask.index_select(0, row_idx)     # [B*Ck,L]

            go_tok = self._encode_go_tokens(ids_c, mask_c)          # [B*Ck,L,Dg]
            go_q = self.go_cross_proj(go_tok)                       # [B*Ck,L,C]

            # Repeat protein residues for B*Ck rows in matching order.
            H_rep = H_cross.unsqueeze(1).expand(B, Ck, T, self.cross_dim).reshape(B * Ck, T, self.cross_dim)
            if key_padding_mask is not None:
                kpm = key_padding_mask.unsqueeze(1).expand(B, Ck, T).reshape(B * Ck, T)
            else:
                kpm = None

            # GO tokens query protein residues.
            ctx, attn_w = self.cross_attn(
                query=go_q,
                key=H_rep,
                value=H_rep,
                key_padding_mask=kpm,
                need_weights=return_alpha,
                average_attn_weights=False,
            )                                                       # [B*Ck,L,C]
            ctx = self.cross_ln(ctx + go_q)

            if mask_c is not None:
                go_valid = mask_c != 0
            else:
                go_valid = None
            go_vec = self._masked_mean(ctx, go_valid)               # [B*Ck,C]

            if self.use_cls_residual:
                # Add projected CLS/text summary as a stabilizer.
                cls = self.go_cross_proj(go_tok[:, 0, :])
                go_vec = self.cross_ln(go_vec + cls)

            prot_h = prot_h_all.unsqueeze(1).expand(B, Ck, self.hidden_dim).reshape(B * Ck, self.hidden_dim)
            go_h = self.go_proj(go_vec)

            diff = torch.abs(prot_h - go_h)
            prod = prot_h * go_h
            cos = F.cosine_similarity(prot_h, go_h, dim=-1).unsqueeze(-1)
            feats = [prot_h, go_h, diff, prod, cos]

            if self.use_retriever_features:
                if retriever_score is None:
                    rs = torch.zeros((B, Ck), dtype=H.dtype, device=H.device)
                else:
                    rs = retriever_score[:, s:e].to(dtype=H.dtype, device=H.device)
                if rank_feature is None:
                    rf = torch.zeros((B, Ck), dtype=H.dtype, device=H.device)
                else:
                    rf = rank_feature[:, s:e].to(dtype=H.dtype, device=H.device)
                feats.extend([rs.reshape(B * Ck, 1), rf.reshape(B * Ck, 1)])

            x = torch.cat(feats, dim=-1)
            logit_c = self.scorer(x).squeeze(-1).view(B, Ck)
            logits_chunks.append(logit_c)

            if return_alpha and attn_w is not None:
                # attn_w [B*Ck, heads, L, T]
                alpha_chunks.append(attn_w.detach().view(B, Ck, attn_w.size(1), L, T))

        logits = torch.cat(logits_chunks, dim=1)                    # [B,K]

        if return_alpha:
            alpha = torch.cat(alpha_chunks, dim=1) if alpha_chunks else None
            return {"logits": logits, "alpha": alpha}

        return logits
