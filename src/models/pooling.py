import torch
import torch.nn as nn
from typing import Optional

class GoSpecificWattiPooling(nn.Module):
    """
    GO-özel dikkat havuzu (memory-friendly).
    Q = Wq * G  (B, T, P)
    K = Wk * H  (B, L, P)
    alpha = softmax( Q @ K^T )  (B, T, L)
    Z = alpha @ H               (B, T, Dh)
    """
    def __init__(self, d_h: int, d_g: int, d_proj: int = 256, dropout: float = 0.0, q_chunk: int = 128):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.q_chunk = q_chunk  # T ekseni için parça boyu

    def forward(
        self,
        H: torch.Tensor,                 # (B, L, Dh)
        G: torch.Tensor,                 # (B, T, Dg)
        mask: Optional[torch.Tensor]=None,  # (B, L) -> True=PAD/ignore
        return_alpha: bool=False
    ):
        B, L, Dh = H.shape
        T = G.shape[1]

        # Projeksiyonlar (AMP ile dışarıdan autocast kullanabilirsin)
        K = self.Wk(H)                   # (B, L, P)
        Q = self.Wq(G)                   # (B, T, P)
        Kt = K.transpose(1, 2).contiguous()  # (B, P, L)

        # Maskeyi (B, 1, L) şekline getir ki (B, t, L) logitlere yayınlanabilsin
        if mask is not None:
            # mask: True olan yerler -inf olmalı
            # gelen mask boolean/byte/fp olabilir—hepsini bool’a çekelim:
            m = mask.bool().unsqueeze(1)  # (B, 1, L)
        else:
            m = None

        Z_chunks = []
        alpha_chunks = [] if return_alpha else None

        # T’yi parça parça işle (softmax L üstünde; dolayısıyla T’yi bölmek güvenli)
        q_chunk = max(1, int(self.q_chunk))
        for t0 in range(0, T, q_chunk):
            t1 = min(T, t0 + q_chunk)
            Qc = Q[:, t0:t1, :]                     # (B, t, P)

            # logits = Qc @ K^T  -> (B, t, L)
            # torch.matmul, batched çarpımda daha bellek-dostu
            logits = torch.matmul(Qc, Kt) * self.scale  # (B, t, L)

            if m is not None:
                logits = logits.masked_fill(m, float('-inf'))

            # softmax L ekseninde
            alpha = torch.softmax(logits, dim=-1)       # (B, t, L)
            alpha = self.drop(alpha)

            # Zc = alpha @ H  -> (B, t, Dh)
            # (B, t, L) x (B, L, Dh) = (B, t, Dh)
            Zc = torch.matmul(alpha, H)

            Z_chunks.append(Zc)
            if return_alpha:
                # Dönüşte büyük bellek kullanmasın diye istersen CPU’ya taşıyabilirsin:
                alpha_chunks.append(alpha)  # ya da alpha.detach().cpu()

            # Ara tensörleri serbest bırak
            del logits, alpha, Qc, Zc
            torch.cuda.empty_cache()

        Z = torch.cat(Z_chunks, dim=1)  # (B, T, Dh)

        if return_alpha:
            A = torch.cat(alpha_chunks, dim=1)  # (B, T, L)
            return Z, A
        return Z