from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    GO(token) x Protein(residue) cross-attention (H:[B,L,Dh], G:[B,T,Dg]).

    Bellek güvenli "streaming" softmax:
      - T ekseninde q_chunk (t_step) ile,
      - L ekseninde k_chunk ile çalışır.
    Büyük [B,T,L] tensörleri oluşturmaz.
    Logit ve exp hesapları bf16'da; akümülatörler fp32'de tutulur.

    reduce: "none" | "mean" | "attn"  -> T boyunca indirgeme seçeneği.
    """
    def __init__(
        self,
        d_h: int,
        d_g: int,
        d_proj: int = 256,
        dropout: float = 0.0,
        k_chunk: int = 128,            # L boyunca parça
        reduce: str = "none",           # "none" | "mean" | "attn"
        attn_t_hidden: int = 0,
        attn_t_dropout: float = 0.0,
        q_chunk_default: int = 32,     # T boyunca varsayılan parça
    ):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)
        self.k_chunk = int(k_chunk)
        self.reduce = reduce
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.q_chunk_default = int(q_chunk_default)

        self.t_pool = None
        if self.reduce == "attn":
            self.t_pool = AttnPool1D(d_in=d_h, d_hidden=attn_t_hidden, dropout=attn_t_dropout)

    def forward(
        self,
        H: torch.Tensor,                           # [B,L,Dh]
        G: torch.Tensor,                           # [B,T,Dg]
        mask: Optional[torch.Tensor] = None,       # [B,L] (True=PAD)
        return_alpha: bool = False,
        q_chunk: Optional[int] = None,             # T boyunca dilim
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, L, Dh = H.shape
        T = G.shape[1]
        device = H.device

        # Projeksiyonlar fp32 (stabilite)
        K_full = self.Wk(H)              # [B,L,P] fp32
        Q_full = self.Wq(G)              # [B,T,P] fp32

        key_mask = None
        if mask is not None:
            key_mask = mask.bool().unsqueeze(1)  # [B,1,L]

        t_step = int(q_chunk) if q_chunk is not None else self.q_chunk_default
        t_step = max(1, min(T, t_step))

        # Çıktı ve akümülatör dtype’ları
        Z_out = torch.zeros(B, T, Dh, device=device, dtype=torch.float32)   # [B,T,Dh]
        comp_dtype = torch.bfloat16 if torch.cuda.is_available() else H.dtype

        # Numerator için mikro-parça (H üzerinden) boyutu
        # Büyükse OOM’a sebep olmasın diye makul küçük tutuyoruz.
        h_sub = min(self.k_chunk, 256)

        for ts in range(0, T, t_step):
            te = min(T, ts + t_step)
            Qc = Q_full[:, ts:te, :].to(comp_dtype)           # [B,t,P] bf16

            # running log-sum-exp parçalı softmax değişkenleri (fp32)
            m = torch.full((B, te - ts), -float('inf'), device=device, dtype=torch.float32)  # [B,t]
            s = torch.zeros(B, te - ts, device=device, dtype=torch.float32)                  # [B,t]
            z = torch.zeros(B, te - ts, Dh, device=device, dtype=torch.float32)             # [B,t,Dh]

            for ls in range(0, L, self.k_chunk):
                le = min(L, ls + self.k_chunk)

                # K dilimi bf16 (logits için)
                Kc = K_full[:, ls:le, :].to(comp_dtype)        # [B,ℓ,P] bf16

                # logits: [B,t,ℓ] (bf16)
                lc = torch.bmm(Qc, Kc.transpose(1, 2)) * self.scale
                if key_mask is not None:
                    km = key_mask[:, :, ls:le]                 # [B,1,ℓ]
                    lc = lc.masked_fill(km, float("-inf"))

                # chunk maksimumu (fp32'ye)
                lc_max = lc.max(dim=-1).values.to(torch.float32)             # [B,t]
                m_new = torch.maximum(m, lc_max)                              # [B,t]

                # önceki katkıları yeniden ölçekle
                exp_scale_prev = torch.exp(m - m_new)                         # [B,t]
                s = s * exp_scale_prev
                z = z * exp_scale_prev.unsqueeze(-1)

                # --- payda katkısı: sum exp(lc - m_new) ---
                exp_chunk_den = torch.exp(lc.to(comp_dtype) - m_new.unsqueeze(-1).to(comp_dtype))  # [B,t,ℓ] bf16
                s = s + exp_chunk_den.sum(dim=-1).to(torch.float32)          # [B,t]

                # --- pay (numerator) katkısı: mikro-parça ile ---
                # H için KOPYA/KAST yok -> H_sub fp32 kalır; exp_sub fp32'ye cast edilip bmm yapılır.
                for mls in range(ls, le, h_sub):
                    mle = min(le, mls + h_sub)
                    # logits alt parçasının exp'i: [B,t,h]
                    exp_sub = exp_chunk_den[:, :, (mls - ls):(mle - ls)]
                    # H alt parçası: [B,h,Dh] (fp32)
                    H_sub = H[:, mls:mle, :]  # fp32, kopya yok (view)
                    # bmm için aynı dtype olmalı -> exp_sub'ı fp32’ye çeviriyoruz (küçük h için güvenli)
                    z = z + torch.bmm(exp_sub.to(torch.float32), H_sub)       # [B,t,Dh] fp32 akümüle

                # güncelle
                m = m_new

                # serbest bırak
                del Kc, lc, lc_max, m_new, exp_scale_prev, exp_chunk_den
                torch.cuda.empty_cache()

            # t-diliminde normalize edilmiş çıktı
            eps = 1e-12
            Zc = z / (s.clamp_min(eps).unsqueeze(-1))                         # [B,t,Dh] fp32
            Zc = self.dropout(Zc).to(H.dtype)

            Z_out[:, ts:te, :] = Zc

            del Qc, m, s, z, Zc
            torch.cuda.empty_cache()

        # --- T boyunca reduce (opsiyonel) ---
        if self.reduce == "none":
            if return_alpha:
                return Z_out, None
            return Z_out
        elif self.reduce == "mean":
            Zp = Z_out.mean(dim=1)                     # [B,Dh]
            if return_alpha:
                return Zp, None
            return Zp
        elif self.reduce == "attn":
            Zp, _ = self.t_pool(Z_out, t_mask=None)    # [B,Dh]
            if return_alpha:
                return Zp, None
            return Zp
        else:
            raise ValueError(f"Unknown reduce='{self.reduce}'")