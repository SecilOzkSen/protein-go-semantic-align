# src/models/pooling.py
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Simple token-attention pooler over T (optional "reduce=attn")
# ============================================================
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


# ============================================================
# GO x Residue streaming cross-attention (memory safe)
# ============================================================
class GoSpecificWattiPooling(nn.Module):
    """
    GO(token) x Protein(residue) cross-attention (streaming):
      - Q = Wq(G[:, ts:te])  (t-chunk içinde proj)
      - K = Wk(H[:, ls:le])  (k-chunk içinde proj)
      - Softmax(L) için log-sum-exp streaming: full [B,T,L] yok
      - Z = sum(alpha_chunk @ H_chunk) (numerator da streaming)

    Bellek:
      - t-chunk (q_chunk) ve k-chunk (k_chunk) ile çalışır
      - Numerator fp32 akümülatör, ara çarpımlar bf16 (H100)

    Opsiyonel reduce: "none" | "mean" | "attn" (T boyunca)
    """
    def __init__(
        self,
        d_h: int,
        d_g: int,
        d_proj: int = 256,
        dropout: float = 0.0,
        k_chunk: int = 512,             # L boyunca başlangıç parçası
        reduce: str = "none",           # "none" | "mean" | "attn"
        attn_t_hidden: int = 0,
        attn_t_dropout: float = 0.0,
        q_chunk_default: int = 8,       # T boyunca başlangıç parçası
        h_sub_init: int = 64,           # numerator için H alt-parçası
    ):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)
        self.k_chunk = k_chunk
        self.q_chunk_default = q_chunk_default
        self.h_sub_init = h_sub_init
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.reduce = reduce

        self.t_pool = None
        if self.reduce == "attn":
            self.t_pool = AttnPool1D(d_in=d_h, d_hidden=attn_t_hidden, dropout=attn_t_dropout)

    @torch.amp.autocast('cuda', enabled=False)  # dışarıdan dtype kontrolü için
    def forward(
        self,
        H: torch.Tensor,                           # [B,L,Dh]
        G: torch.Tensor,                           # [B,T,Dg]
        mask: Optional[torch.Tensor] = None,       # [B,L] (True=PAD)
        return_alpha: bool = False,
        q_chunk: Optional[int] = None,             # T boyunca dilim; None -> default
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        device = H.device
        B, L, Dh = H.shape
        T = G.shape[1]

        # --- Hesaplama dtype'ı (CUDA'da bf16 tercih) ---
        bf16_ok = (device.type == "cuda")
        comp_dtype = torch.bfloat16 if bf16_ok else torch.float32

        # *** KRİTİK DÜZELTME: Projeksiyon katmanlarını comp_dtype'a al ***
        # Linear ağırlıkları ile input aynı dtype olmalı.
        if self.Wk.weight.dtype != comp_dtype:
            self.Wk.to(dtype=comp_dtype, device=device)
        if self.Wq.weight.dtype != comp_dtype:
            self.Wq.to(dtype=comp_dtype, device=device)

        # key mask (True=PAD)
        key_mask = None
        if mask is not None:
            key_mask = mask.bool().unsqueeze(1)      # [B,1,L]

        # parça boyutları (adaptif küçülecek)
        t_step = q_chunk or self.q_chunk_default
        k_step = self.k_chunk
        h_sub  = self.h_sub_init

        Z_parts: List[torch.Tensor] = []
        alpha_parts: List[torch.Tensor] = [] if return_alpha else None

        # ---- T boyunca dilimle ----
        for ts in range(0, T, t_step):
            te = min(T, ts + t_step)

            # Q'yu sadece bu t-chunk için projekte et
            Qc = self.Wq(G[:, ts:te, :].to(comp_dtype))   # [B,t,P] comp_dtype

            # LSE akümülatörleri: m,s (denominator), z (numerator)
            m = None
            s = None
            z = torch.zeros((B, te - ts, Dh), device=device, dtype=torch.float32)

            # ---- L boyunca dilimle ----
            ls = 0
            while ls < L:
                le = min(L, ls + k_step)
                # H/K alt-dilim comp_dtype
                try:
                    Hc = H[:, ls:le, :].to(comp_dtype)         # [B,ℓ,Dh]
                    Kc = self.Wk(Hc)                           # [B,ℓ,P] (comp_dtype)
                    # logits: [B,t,ℓ] = Qc [B,t,P] @ Kc^T [B,P,ℓ]
                    lc = torch.bmm(Qc, Kc.transpose(1, 2)) * self.scale  # comp_dtype
                    lc = lc.to(torch.float32)                               # LSE için fp32
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if k_step > 128:
                        k_step = max(128, k_step // 2)
                        continue
                    else:
                        raise

                # mask uygula
                if key_mask is not None:
                    km = key_mask[:, :, ls:le]      # [B,1,ℓ]
                    lc = lc.masked_fill(km, float("-inf"))

                # ---- LSE streaming + numerator streaming ----
                lc_max = lc.max(dim=-1, keepdim=True).values  # [B,t,1]
                if m is None:
                    m = lc_max
                    s = torch.exp(lc - m).sum(dim=-1, keepdim=True)  # [B,t,1]

                    start = 0
                    sub_size = h_sub
                    while start < (le - ls):
                        stop = min(le - ls, start + sub_size)
                        H_sub = Hc[:, start:stop, :]                         # [B,h,Dh] comp_dtype
                        w_sub = torch.exp(lc[:, :, start:stop] - m)          # [B,t,h] fp32
                        try:
                            z = z + torch.bmm(w_sub.to(comp_dtype), H_sub).to(torch.float32)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if sub_size > 16:
                                sub_size = max(16, sub_size // 2)
                                continue
                            else:
                                raise
                        start = stop
                else:
                    m_new = torch.maximum(m, lc_max)                         # [B,t,1]
                    scale_old = torch.exp(m - m_new)                         # [B,t,1]
                    s = scale_old * s + torch.exp(lc - m_new).sum(-1, keepdim=True)

                    start = 0
                    sub_size = h_sub
                    while start < (le - ls):
                        stop = min(le - ls, start + sub_size)
                        H_sub = Hc[:, start:stop, :]                         # [B,h,Dh] comp_dtype
                        w_sub = torch.exp(lc[:, :, start:stop] - m_new)      # [B,t,h] fp32
                        try:
                            z = z.mul_(scale_old).add_(
                                torch.bmm(w_sub.to(comp_dtype), H_sub).to(torch.float32)
                            )
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if sub_size > 16:
                                sub_size = max(16, sub_size // 2)
                                continue
                            else:
                                raise
                        start = stop
                    m = m_new

                # free chunk
                del Hc, Kc, lc
                torch.cuda.empty_cache()
                ls = le

            # t-chunk sonucu
            Zc = (z / s.clamp_min(1e-20)).to(torch.float32)  # [B,t,Dh]
            Zc = self.dropout(Zc)
            Z_parts.append(Zc)

            if return_alpha:
                # alpha üretimi (uzun L için pahalı; genelde False gönderin)
                alpha_chunks: List[torch.Tensor] = []
                ls = 0
                while ls < L:
                    le = min(L, ls + k_step)
                    with torch.no_grad():
                        Hc = H[:, ls:le, :].to(comp_dtype)
                        Kc = self.Wk(Hc)
                        lc = torch.bmm(Qc, Kc.transpose(1, 2)) * self.scale
                        lc = lc.to(torch.float32)
                        if key_mask is not None:
                            km = key_mask[:, :, ls:le]
                            lc = lc.masked_fill(km, float("-inf"))
                        alpha_sub = torch.exp(lc - m) / s
                        alpha_chunks.append(alpha_sub)   # [B,t,ℓ]
                        del Hc, Kc, lc, alpha_sub
                        torch.cuda.empty_cache()
                    ls = le
                A_t = torch.cat(alpha_chunks, dim=-1)     # [B,t,L]
                alpha_parts.append(A_t)

            # free Q chunk & accumulators
            del Qc, m, s, z, Zc
            torch.cuda.empty_cache()

        # [B,T,Dh]
        Z = torch.cat(Z_parts, dim=1)

        # --- T boyunca reduce (opsiyonel) ---
        if self.reduce == "none":
            if return_alpha:
                A = torch.cat(alpha_parts, dim=1)  # [B,T,L]
                return Z, A
            return Z

        elif self.reduce == "mean":
            Zp = Z.mean(dim=1)                     # [B,Dh]
            if return_alpha:
                A = torch.cat(alpha_parts, dim=1)
                return Zp, A
            return Zp

        elif self.reduce == "attn":
            Zp, _ = self.t_pool(Z, t_mask=None)    # [B,Dh]
            if return_alpha:
                A = torch.cat(alpha_parts, dim=1)
                return Zp, A
            return Zp

        else:
            raise ValueError(f"Unknown reduce='{self.reduce}'")