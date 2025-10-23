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
        k_chunk: int = 256,            # L boyunca parça
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
        q_chunk: Optional[int] = None,             # T boyunca dilim; None -> q_chunk_default
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, L, Dh = H.shape
        T = G.shape[1]
        device = H.device

        # Projeksiyonları fp32'de tut (stabilite)
        K_full = self.Wk(H)              # [B,L,P] fp32
        Q_full = self.Wq(G)              # [B,T,P] fp32

        # mask: True=PAD -> -inf uygulanacak
        key_mask = None
        if mask is not None:
            key_mask = mask.bool().unsqueeze(1)  # [B,1,L]

        # T boyunca dilim büyüklüğü
        t_step = int(q_chunk) if q_chunk is not None else self.q_chunk_default
        t_step = max(1, min(T, t_step))

        # Akümülatörler fp32
        Z_out = torch.zeros(B, T, Dh, device=device, dtype=torch.float32)   # [B,T,Dh]

        # Streaming softmax akışı için "running" değerler
        # Not: bunlar her t-dilimi için sıfırlanacak (global T boyunca birleştirip yazıyoruz).
        comp_dtype = torch.bfloat16 if torch.cuda.is_available() else H.dtype

        for ts in range(0, T, t_step):
            te = min(T, ts + t_step)
            Qc = Q_full[:, ts:te, :].to(comp_dtype)           # [B,t,P] bf16
            # t-dilimi için akümülatörleri sıfırla (fp32)
            m = torch.full((B, te - ts), -float('inf'), device=device, dtype=torch.float32)  # [B,t]
            s = torch.zeros(B, te - ts, device=device, dtype=torch.float32)                  # [B,t]
            z = torch.zeros(B, te - ts, Dh, device=device, dtype=torch.float32)             # [B,t,Dh]

            for ls in range(0, L, self.k_chunk):
                le = min(L, ls + self.k_chunk)
                Kc = K_full[:, ls:le, :].to(comp_dtype)        # [B,ℓ,P] bf16
                Hc = H[:, ls:le, :].to(comp_dtype)             # [B,ℓ,Dh] bf16

                # logits: [B,t,ℓ] (bf16)
                lc = torch.bmm(Qc, Kc.transpose(1, 2)) * self.scale

                if key_mask is not None:
                    km = key_mask[:, :, ls:le]                 # [B,1,ℓ]
                    lc = lc.masked_fill(km, float("-inf"))

                # chunk maksimumu (fp32'ye taşı)
                lc_max = lc.max(dim=-1).values.to(torch.float32)             # [B,t]
                m_new = torch.maximum(m, lc_max)                              # [B,t]

                # önceki katkıları yeniden ölçekle
                # s,z: fp32 (broadcast)
                exp_scale_prev = torch.exp(m - m_new)                         # [B,t]
                s = s * exp_scale_prev
                z = z * exp_scale_prev.unsqueeze(-1)

                # yeni chunk katkısı: exp(lc - m_new)
                # not: exp ve bmm bf16'da; sonra fp32'ye eklenir
                exp_chunk = torch.exp(lc.to(comp_dtype) - m_new.unsqueeze(-1).to(comp_dtype))   # [B,t,ℓ] bf16

                # payda katkısı
                s = s + exp_chunk.sum(dim=-1).to(torch.float32)              # [B,t]

                # pay katkısı: [B,t,ℓ] @ [B,ℓ,Dh] -> [B,t,Dh]
                num = torch.bmm(exp_chunk, Hc)                                # bf16
                z = z + num.to(torch.float32)                                 # fp32 akümülatör

                # güncelle
                m = m_new

                # serbest bırak
                del Kc, Hc, lc, lc_max, m_new, exp_scale_prev, exp_chunk, num
                torch.cuda.empty_cache()

            # t-diliminde softmax-normalize edilmiş çıktı: z / s (güvenlik için eps)
            eps = 1e-12
            Zc = z / (s.clamp_min(eps).unsqueeze(-1))                         # [B,t,Dh] fp32

            # dropout (eğer istenirse) ve type cast (orijinal dtype’ına)
            Zc = self.dropout(Zc).to(H.dtype)

            # çıktıya yaz
            Z_out[:, ts:te, :] = Zc

            # clean
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