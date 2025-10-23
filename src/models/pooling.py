from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

class AttnPool1D(nn.Module):
    """Z: [B, T, Dh] üzerinde T-boyunca öğrenilebilir havuzlama."""
    def __init__(self, d_in: int, d_hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        self.use_mlp = d_hidden > 0
        if self.use_mlp:
            self.proj1 = nn.Linear(d_in, d_hidden, bias=True)
            self.proj2 = nn.Linear(d_hidden, 1, bias=False)
        else:
            self.proj = nn.Linear(d_in, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z: torch.Tensor, t_mask: Optional[torch.Tensor] = None):
        # Z: [B,T,D], t_mask: [B,T] (1=keep, 0=pad)
        if self.use_mlp:
            x = torch.tanh(self.proj1(Z))
            x = self.dropout(x)
            logits = self.proj2(x).squeeze(-1)  # [B,T]
        else:
            logits = self.proj(Z).squeeze(-1)   # [B,T]
        if t_mask is not None:
            logits = logits.masked_fill(t_mask == 0, float("-inf"))
        w = torch.softmax(logits, dim=-1)       # [B,T]
        pooled = torch.bmm(w.unsqueeze(1), Z).squeeze(1)  # [B,D]
        return pooled, w


class GoSpecificWattiPooling(nn.Module):
    """
    GO(token) x Protein(residue) cross-attention:
      - Projeksiyonlar: Q=Wq*G, K=Wk*H
      - Logits = Q K^T / sqrt(P), alpha=softmax(logits, L)
      - Z = alpha H   (B,T,Dh)
    Bellek için:
      - T ekseninde q_chunk,
      - L ekseninde k_chunk ile çalışır; dev [B,T,L] oluşturmaz.
    Opsiyonel:
      - reduce: "none" | "mean" | "attn"  (T boyunca)
    """
    def __init__(
        self,
        d_h: int,
        d_g: int,
        d_proj: int = 256,
        dropout: float = 0.0,
        k_chunk: int = 2048,             # L boyunca parça
        reduce: str = "none",            # "none" | "mean" | "attn"
        attn_t_hidden: int = 0,
        attn_t_dropout: float = 0.0,
    ):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)
        self.k_chunk = k_chunk
        self.reduce = reduce
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.t_pool = None
        if self.reduce == "attn":
            self.t_pool = AttnPool1D(d_in=d_h, d_hidden=attn_t_hidden, dropout=attn_t_dropout)

    @torch.cuda.amp.autocast(enabled=False)  # dtype’ı dışarıdan (bf16/fp16) kontrol etmek daha güvenli
    def forward(
        self,
        H: torch.Tensor,                           # [B,L,Dh]
        G: torch.Tensor,                           # [B,T,Dg]
        mask: Optional[torch.Tensor] = None,       # [B,L] (True=PAD)  <-- BucketedGoWatti ile uyumlu
        return_alpha: bool = False,
        q_chunk: Optional[int] = None,             # T boyunca dilim; None -> otomatik makul
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, Dh = H.shape
        T = G.shape[1]
        device = H.device

        # Projeksiyonlar her zaman fp32'de; dışarıda autocast ediyorsan karışmasın
        K = self.Wk(H)              # [B,L,P]
        Q = self.Wq(G)              # [B,T,P]

        # mask: True=PAD --> -inf atacağız
        key_mask = None
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            key_mask = mask.unsqueeze(1)          # [B,1,L]

        # T boyunca dilim büyüklüğü
        t_step = q_chunk or max(1, min(T, 1024))

        Z_parts = []                 # her t-diliminin Z'si (B,t,Dh)
        alpha_parts = [] if return_alpha else None

        for ts in range(0, T, t_step):
            te = min(T, ts + t_step)
            Qc = Q[:, ts:te, :]                  # [B,t,P]

            # --- L boyunca parça parça logits inşa et ---
            logits_chunks = []
            for ls in range(0, L, self.k_chunk):
                le = min(L, ls + self.k_chunk)
                Kc = K[:, ls:le, :]              # [B,ℓ,P]
                # [B,t,ℓ] = [B,t,P] @ [B,P,ℓ]
                lc = torch.bmm(Qc, Kc.transpose(1, 2)) * self.scale
                if key_mask is not None:
                    km = key_mask[:, :, ls:le]   # [B,1,ℓ]
                    lc = lc.masked_fill(km, float("-inf"))
                logits_chunks.append(lc)

                # free
                del Kc, lc
                torch.cuda.empty_cache()

            # [B,t,L]
            logits = torch.cat(logits_chunks, dim=-1)
            # Dropout logits’e değil, istersen alpha sonrası uygulanır
            alpha = torch.softmax(logits, dim=-1)           # [B,t,L]
            # Zc = alpha @ H  -> [B,t,Dh]
            Zc = torch.bmm(alpha, H)

            Z_parts.append(Zc)
            if return_alpha:
                alpha_parts.append(alpha)

            # free
            del logits, alpha, Qc, logits_chunks, Zc
            torch.cuda.empty_cache()

        # [B,T,Dh]
        Z = torch.cat(Z_parts, dim=1)
        A = torch.cat(alpha_parts, dim=1) if return_alpha else None

        # --- T boyunca reduce (opsiyonel) ---
        if self.reduce == "none":
            return (Z, A) if return_alpha else Z
        elif self.reduce == "mean":
            Zp = Z.mean(dim=1)                     # [B,Dh]
            return (Zp, A) if return_alpha else Zp
        elif self.reduce == "attn":
            # Buraya istersen GO paddings için t_mask [B,T] geçebilirsin (şu an None)
            Zp, _ = self.t_pool(Z, t_mask=None)    # [B,Dh]
            return (Zp, A) if return_alpha else Zp
        else:
            raise ValueError(f"Unknown reduce='{self.reduce}'")