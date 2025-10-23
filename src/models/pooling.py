from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp


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
      - Q = Wq*G, K = Wk*H
      - logits = Q K^T / sqrt(P), alpha = softmax(logits, L)
      - Z = alpha H  (B,T,Dh)

    Bellek dostu:
      - T ekseninde q_chunk
      - L ekseninde k_chunk
      - streaming softmax: (B,t,L) logits/alpha'yı TAM boy asla tutmaz.

    Opsiyonel:
      - reduce: "none" | "mean" | "attn"  (T boyunca)
      - return_alpha=True ise yalnızca küçük örneklerde full alpha döner, aksi halde None.
    """
    def __init__(
        self,
        d_h: int,
        d_g: int,
        d_proj: int = 256,
        dropout: float = 0.0,
        k_chunk: int = 2048,               # L boyunca parça
        reduce: str = "none",              # "none" | "mean" | "attn"
        attn_t_hidden: int = 0,
        attn_t_dropout: float = 0.0,
        allow_full_alpha_tokens: int = 2_000_000,  # B*T*L bundan küçükse alpha döndür (güvenli)
    ):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)

        self.k_chunk = int(k_chunk)
        self.reduce = reduce
        self.allow_full_alpha_tokens = int(allow_full_alpha_tokens)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.t_pool = None
        if self.reduce == "attn":
            self.t_pool = AttnPool1D(d_in=d_h, d_hidden=attn_t_hidden, dropout=attn_t_dropout)

    @amp.autocast("cuda", enabled=False)  # AMP uyarısı giderildi; dtype kontrolü eğitim döngüsünde
    def forward(
        self,
        H: torch.Tensor,                           # [B,L,Dh]
        G: torch.Tensor,                           # [B,T,Dg]
        mask: Optional[torch.Tensor] = None,       # [B,L] (True=PAD)  <-- BucketedGoWatti ile uyumlu
        return_alpha: bool = False,
        q_chunk: Optional[int] = None,             # T boyunca dilim; None -> otomatik
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, Dh = H.shape
        T = G.shape[1]
        device = H.device
        dtype = H.dtype

        # Projeksiyonlar (fp32/compute dtype dışarıdan yönetilsin diye burada autocast kapalı)
        K = self.Wk(H).to(dtype)              # [B,L,P]
        Q = self.Wq(G).to(dtype)              # [B,T,P]

        # mask: True=PAD --> -inf
        key_mask = None
        if mask is not None:
            if mask.dtype is not torch.bool:
                mask = mask.bool()
            key_mask = mask                   # [B,L]

        # T boyunca dilim büyüklüğü
        t_step = int(q_chunk) if q_chunk is not None else max(1, min(T, 512))

        # Küçük örnekler için tam alpha yolu (güvenliyse)
        total_tokens = B * T * L
        can_full_alpha = return_alpha and (total_tokens <= self.allow_full_alpha_tokens) and (L <= self.k_chunk) and (T <= t_step)

        Z_parts = []                           # [B,t,Dh] parçaları
        alpha_parts = [] if can_full_alpha else None

        # ---- Sorgu ekseninde (T) ilerle ----
        for ts in range(0, T, t_step):
            te = min(T, ts + t_step)
            t = te - ts
            Qc = Q[:, ts:te, :]                        # [B,t,P]

            if can_full_alpha:
                # logits'i L boyunca birleştir, alpha üret (sadece küçük örneklerde)
                logits_chunks = []
                for ls in range(0, L, self.k_chunk):
                    le = min(L, ls + self.k_chunk)
                    Kc = K[:, ls:le, :]                # [B,ℓ,P]
                    lc = torch.bmm(Qc, Kc.transpose(1, 2)) * self.scale   # [B,t,ℓ]
                    if key_mask is not None:
                        mk = key_mask[:, ls:le]        # [B,ℓ]
                        lc = lc.masked_fill(mk.unsqueeze(1), float("-inf"))
                    logits_chunks.append(lc)
                    del Kc, lc
                logits = torch.cat(logits_chunks, dim=-1)                 # [B,t,L]
                alpha = torch.softmax(logits, dim=-1)                     # [B,t,L]
                Zc = torch.bmm(alpha, H)                                  # [B,t,Dh]
                Zc = self.dropout(Zc)
                Z_parts.append(Zc)
                alpha_parts.append(alpha)

                del logits, alpha, Qc, logits_chunks, Zc
                torch.cuda.empty_cache()
                continue

            # ---- Büyük örnekler: streaming softmax (logits/alpha full tutulmaz) ----
            # Online softmax istatistikleri
            # m: running max, s: running sumexp, n: running numerator (exp * H)
            m = torch.full((B, t, 1), -float("inf"), device=device, dtype=dtype)  # [B,t,1]
            s = torch.zeros((B, t, 1), device=device, dtype=dtype)                # [B,t,1]
            n = torch.zeros((B, t, Dh), device=device, dtype=dtype)               # [B,t,Dh]

            for ls in range(0, L, self.k_chunk):
                le = min(L, ls + self.k_chunk)
                Kc = K[:, ls:le, :]                          # [B,ℓ,P]
                Hc = H[:, ls:le, :]                          # [B,ℓ,Dh]

                lc = torch.bmm(Qc, Kc.transpose(1, 2)) * self.scale   # [B,t,ℓ]
                if key_mask is not None:
                    mk = key_mask[:, ls:le]                  # [B,ℓ]
                    lc = lc.masked_fill(mk.unsqueeze(1), float("-inf"))

                # Online max+sumexp güncellemesi
                m_chunk = lc.max(dim=-1, keepdim=True).values            # [B,t,1]
                m_new = torch.maximum(m, m_chunk)                        # [B,t,1]
                exp_prev = torch.exp(m - m_new)                          # [B,t,1]
                exp_chunk = torch.exp(lc - m_new)                        # [B,t,ℓ]

                s = s * exp_prev + exp_chunk.sum(dim=-1, keepdim=True)   # [B,t,1]
                # numerator güncelle: sum(exp * H)
                n = n * exp_prev + torch.bmm(exp_chunk, Hc)              # [B,t,Dh]
                m = m_new

                del Kc, Hc, lc, m_chunk, m_new, exp_prev, exp_chunk
                torch.cuda.empty_cache()

            Zc = n / s.clamp_min(1e-12)     # [B,t,Dh]
            Zc = self.dropout(Zc)
            Z_parts.append(Zc)

            del Qc, m, s, n, Zc
            torch.cuda.empty_cache()

        # Çıktı birleştir
        Z = torch.cat(Z_parts, dim=1)               # [B,T,Dh]
        A = torch.cat(alpha_parts, dim=1) if alpha_parts is not None else None  # [B,T,L] | None

        # ---- T boyunca reduce (opsiyonel) ----
        if self.reduce == "none":
            return (Z, A) if return_alpha else Z
        elif self.reduce == "mean":
            Zp = Z.mean(dim=1)                      # [B,Dh]
            return (Zp, A) if return_alpha else Zp
        elif self.reduce == "attn":
            # İstersen GO paddings için t_mask [B,T] geçebilirsin (şu an None)
            Zp, _ = self.t_pool(Z, t_mask=None)     # [B,Dh]
            return (Zp, A) if return_alpha else Zp
        else:
            raise ValueError(f"Unknown reduce='{self.reduce}'")