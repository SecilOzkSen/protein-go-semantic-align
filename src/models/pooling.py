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
    GO(token) x Protein(residue) cross-attention (memory-safe):
      - Q = Wq(G), K = Wk(H)
      - logits parça parça (t_chunk x k_chunk), softmax L boyunca LSE ile
      - Z = softmax(logits, L) @ H  (mikro-parçalarla, kopyasız)
    Opsiyonel T-reduce: "none" | "mean" | "attn"
    """
    def __init__(
        self,
        d_h: int,
        d_g: int,
        d_proj: int = 256,
        dropout: float = 0.0,
        k_chunk: int = 1024,             # L boyunca ana parça
        reduce: str = "none",            # "none" | "mean" | "attn"
        attn_t_hidden: int = 0,
        attn_t_dropout: float = 0.0,
        q_chunk_default: int = 32,       # T boyunca varsayılan dilim
        h_sub_init: int = 128            # L mikro-parça (OOM’da yarıya iner)
    ):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)

        self.k_chunk = int(k_chunk)
        self.reduce = reduce
        self.q_chunk_default = int(q_chunk_default)
        self.h_sub_init = int(h_sub_init)

        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0) else nn.Identity()

        self.t_pool = None
        if self.reduce == "attn":
            self.t_pool = AttnPool1D(d_in=d_h, d_hidden=attn_t_hidden, dropout=attn_t_dropout)

    @torch.cuda.amp.autocast(enabled=False)  # dışarıdan dtype kontrolü için
    def forward(
        self,
        H: torch.Tensor,                           # [B,L,Dh] fp32
        G: torch.Tensor,                           # [B,T,Dg] fp32
        mask: Optional[torch.Tensor] = None,       # [B,L] (True=PAD)
        return_alpha: bool = False,
        q_chunk: Optional[int] = None,             # T boyunca dilim
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, L, Dh = H.shape
        T = G.shape[1]
        device = H.device

        # --- Projeksiyonlar: Q/K’yi bf16’a düşür, büyük fp32’yi tutma
        comp_dtype = torch.bfloat16 if torch.cuda.is_available() else H.dtype
        K_full = self.Wk(H).to(comp_dtype)     # [B,L,P] bf16
        Q_full = self.Wq(G).to(comp_dtype)     # [B,T,P] bf16

        key_mask = None
        if mask is not None:
            key_mask = mask.bool().unsqueeze(1)       # [B,1,L]

        t_step = int(q_chunk) if q_chunk is not None else self.q_chunk_default
        t_step = max(1, min(T, t_step))

        # Çıktı (fp32, stabilite)
        Z_out = torch.zeros(B, T, Dh, device=device, dtype=torch.float32)

        # adaptif mikro-parça
        base_h_sub = max(16, min(self.k_chunk, self.h_sub_init))

        for ts in range(0, T, t_step):
            te = min(T, ts + t_step)
            Qc = Q_full[:, ts:te, :]                         # [B,t,P] bf16

            # LSE akümülatörleri (fp32)
            m = torch.full((B, te - ts), -float('inf'), device=device, dtype=torch.float32)  # [B,t]
            s = torch.zeros(B, te - ts, device=device, dtype=torch.float32)                  # [B,t]
            z = torch.zeros(B, te - ts, Dh, device=device, dtype=torch.float32)             # [B,t,Dh]

            for ls in range(0, L, self.k_chunk):
                le = min(L, ls + self.k_chunk)
                Kc = K_full[:, ls:le, :]                         # [B,ℓ,P] bf16

                # logits bf16
                lc = torch.bmm(Qc, Kc.transpose(1, 2)) * self.scale   # [B,t,ℓ] bf16
                if key_mask is not None:
                    lc = lc.masked_fill(key_mask[:, :, ls:le], float("-inf"))

                # LSE: yeni maksimum (fp32)
                lc_max = lc.max(dim=-1).values.to(torch.float32)      # [B,t]
                m_new = torch.maximum(m, lc_max)

                # önceki katkıları yeniden ölçekle
                scale_prev = torch.exp(m - m_new)                      # [B,t]
                s = s * scale_prev
                z = z * scale_prev.unsqueeze(-1)

                # payda katkısı (bf16) -> fp32’ye topla
                exp_den = torch.exp(lc - m_new.unsqueeze(-1).to(comp_dtype))  # [B,t,ℓ] bf16
                s = s + exp_den.sum(dim=-1).to(torch.float32)                  # [B,t]

                # --- pay (numerator) katkısı: BF16 BMM, SONUÇ fp32 ---
                # exp_sub’u fp32’ye döndürmek yerine H_sub’u bf16’a çeviriyoruz (küçük slice).
                h_sub = base_h_sub
                offset = 0
                while offset < (le - ls):
                    try:
                        mls = ls + offset
                        mle = min(le, mls + h_sub)
                        # [B,t,h]
                        exp_sub = exp_den[:, :, (mls - ls):(mle - ls)]
                        # [B,h,Dh]  -> küçük dilim, bf16 kopyası makul
                        H_sub_bf16 = H[:, mls:mle, :].to(comp_dtype, copy=True)
                        # bmm bf16 -> bf16, sonra fp32 akümüle
                        z = z + torch.bmm(exp_sub, H_sub_bf16).to(torch.float32)
                        offset = mle - ls
                    except torch.cuda.OutOfMemoryError:
                        # adaptif küçültme
                        if h_sub <= 16:
                            raise
                        h_sub = max(16, h_sub // 2)
                        torch.cuda.empty_cache()
                        continue

                m = m_new

                # serbest bırak
                del Kc, lc, lc_max, m_new, scale_prev, exp_den
                torch.cuda.empty_cache()

            # normalize et ve yaz
            eps = 1e-12
            Zc = z / (s.clamp_min(eps).unsqueeze(-1))                 # [B,t,Dh] fp32
            Zc = self.dropout(Zc).to(H.dtype)
            Z_out[:, ts:te, :] = Zc

            del Qc, m, s, z, Zc
            torch.cuda.empty_cache()

        # --- T boyunca reduce (opsiyonel) ---
        if self.reduce == "none":
            return (Z_out, None) if return_alpha else Z_out
        elif self.reduce == "mean":
            Zp = Z_out.mean(dim=1)
            return (Zp, None) if return_alpha else Zp
        elif self.reduce == "attn":
            Zp, _ = self.t_pool(Z_out, t_mask=None)
            return (Zp, None) if return_alpha else Zp
        else:
            raise ValueError(f"Unknown reduce='{self.reduce}'")