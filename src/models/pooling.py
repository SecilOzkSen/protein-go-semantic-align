from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Not: torch.cuda.amp.autocast() yerine yeni API
autocast_cuda = lambda enabled: torch.amp.autocast(device_type="cuda", enabled=enabled)

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
      - T ekseninde q_chunk (t_step),
      - L ekseninde k_chunk ile çalışır (ℓ-bloklar).
    Opsiyonel:
      - reduce: "none" | "mean" | "attn"  (T boyunca)
    """
    def __init__(
        self,
        d_h: int,
        d_g: int,
        d_proj: int = 256,
        dropout: float = 0.0,
        k_chunk: int = 256,             # L boyunca parça (küçük tut; tepe VRAM’i belirler)
        reduce: str = "none",            # "none" | "mean" | "attn"
        attn_t_hidden: int = 0,
        attn_t_dropout: float = 0.0,
    ):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)
        self.k_chunk = int(k_chunk)
        self.reduce = reduce
        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0) else nn.Identity()

        self.t_pool = None
        if self.reduce == "attn":
            self.t_pool = AttnPool1D(d_in=d_h, d_hidden=attn_t_hidden, dropout=attn_t_dropout)

    @autocast_cuda(enabled=False)  # dtype’ı dışarıdan (bf16/fp16) kontrol etmek daha güvenli
    def forward(
        self,
        H: torch.Tensor,                           # [B,L,Dh] (genelde fp32 geliyor)
        G: torch.Tensor,                           # [B,T,Dg]
        mask: Optional[torch.Tensor] = None,       # [B,L] (True=PAD)
        return_alpha: bool = False,
        q_chunk: Optional[int] = None,             # T boyunca dilim; None -> otomatik
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, Dh = H.shape
        T = int(G.shape[1])
        device = H.device

        # Hesaplama dty: mümkünse bf16; değilse fp16; yoksa fp32
        if torch.cuda.is_available():
            comp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            comp_dtype = torch.float32

        # Projeksiyonlar: Q ve K için fp32 -> sonra comp_dtype
        Q32 = self.Wq(G.to(torch.float32))                     # [B,T,P] fp32
        Q = Q32.to(comp_dtype)                                 # [B,T,P] bf16/fp16
        del Q32

        # mask: True=PAD --> -inf atacağız
        key_mask = None
        if mask is not None:
            key_mask = mask.bool().unsqueeze(1)                # [B,1,L]

        # T boyunca dilim
        t_step = q_chunk or max(1, min(T, 256))

        Z_parts = []                             # her t-diliminin Z'si (B,t,Dh)
        alpha_parts = [] if return_alpha else None

        # --- T dilimleri ---
        for ts in range(0, T, t_step):
            te = min(T, ts + t_step)
            Qc = Q[:, ts:te, :]                                  # [B,t,P] comp
            t_len = te - ts

            # Log-sum-exp için akümülatörler (fp32, sayısal stabil)
            m = torch.full((B, t_len, 1), float("-inf"), device=device, dtype=torch.float32)  # [B,t,1]
            s = torch.zeros((B, t_len, 1), device=device, dtype=torch.float32)                # [B,t,1]
            z = torch.zeros((B, t_len, Dh), device=device, dtype=torch.float32)               # [B,t,Dh]

            if return_alpha:
                # alpha için de stream toplanacak: alpha = exp(logits - m_final) / s_final
                # Bunun için blok bazlı maksimumu ve toplamı zaten tutuyoruz, ancak
                # tam alpha’yı döndürmek ağır olur; bu yüzden None bırakıyoruz (isteğe göre eklenebilir).
                pass

            # --- L dilimleri ---
            for ls in range(0, L, self.k_chunk):
                le = min(L, ls + self.k_chunk)
                # 1) K = Wk(H_block) (fp32), sonra comp_dtype’a düş
                H_lin = H[:, ls:le, :].to(torch.float32)                     # [B,ℓ,Dh] fp32
                K32 = self.Wk(H_lin)                                         # [B,ℓ,P] fp32
                K = K32.to(comp_dtype)                                       # [B,ℓ,P] comp
                del K32

                # 2) logits_block = Qc @ K^T * scale  (comp_dtype)
                logits_block = torch.bmm(Qc, K.transpose(1, 2)) * self.scale # [B,t,ℓ] comp

                # 3) Maskeyi uygula (PAD -> -inf)
                if key_mask is not None:
                    km = key_mask[:, :, ls:le]                                # [B,1,ℓ] bool
                    # logits_block comp_dtype; -inf’i fp32 veriyoruz, cast edilir
                    logits_block = logits_block.masked_fill(km, float("-inf"))

                # 4) Blok maksimumu (fp32)
                m_block = torch.amax(logits_block.to(torch.float32), dim=2, keepdim=True)  # [B,t,1]
                m_new = torch.maximum(m, m_block)                                          # [B,t,1]

                # 5) exp(logits - m_new), comp_dtype’ta
                logits_shifted = (logits_block.to(comp_dtype) -
                                  m_new.to(comp_dtype))                                     # [B,t,ℓ] comp
                exp_block = torch.exp(logits_shifted)                                       # [B,t,ℓ] comp
                del logits_shifted, logits_block

                # 6) s ve z’yi güncelle
                # s_new = s*exp(m-m_new) + sum(exp_block)
                exp_m = torch.exp((m - m_new).to(torch.float32))                            # [B,t,1] fp32
                sum_exp = exp_block.sum(dim=2, keepdim=True).to(torch.float32)             # [B,t,1] fp32
                s_new = s * exp_m + sum_exp                                                # [B,t,1] fp32

                # z_new = z*exp(m-m_new) + exp_block @ H_block
                # H_block’ı comp_dtype’a düşür, bmm comp’da, sonra fp32 akümüle
                H_block_comp = H_lin.to(comp_dtype)                                        # [B,ℓ,Dh] comp
                contrib = torch.bmm(exp_block, H_block_comp).to(torch.float32)             # [B,t,Dh] fp32
                z = z * exp_m + contrib                                                    # [B,t,Dh] fp32

                # ileri
                m = m_new
                s = s_new

                # bellek serbest bırak
                del exp_block, H_block_comp, H_lin, K
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # --- final Z = z / s  ---
            Zc = z / torch.clamp_min(s, 1e-12)                           # [B,t,Dh] fp32
            Z_parts.append(Zc)

            # alpha istenirse burada tam alpha üretmek VRAM pahalı;
            # eğitimde gerekmediği için döndürmüyoruz.
            if return_alpha:
                alpha_parts.append(None)

            del Qc, z, s, m, Zc
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # [B,T,Dh]
        Z = torch.cat(Z_parts, dim=1) if len(Z_parts) > 1 else Z_parts[0]
        A = None  # alpha döndürmüyoruz; istenirse windowed/summary üretilebilir.

        # --- T boyunca reduce (opsiyonel) ---
        if self.reduce == "none":
            return (Z, A) if return_alpha else Z
        elif self.reduce == "mean":
            Zp = Z.mean(dim=1)                     # [B,Dh]
            return (Zp, A) if return_alpha else Zp
        elif self.reduce == "attn":
            Zp, _ = self.t_pool(Z, t_mask=None)    # [B,Dh]
            return (Zp, A) if return_alpha else Zp
        else:
            raise ValueError(f"Unknown reduce='{self.reduce}'")
