from typing import Optional, Tuple, List
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
    GO(token) x Protein(residue) streaming cross-attention:
      - Q = Wq(G_tchunk), K = Wk(H_kchunk)
      - Softmax(L) ve numerator log-sum-exp ile streaming (full [B,T,L] yok)
    reduce: "none" | "mean" | "attn"
    """
    def __init__(
        self,
        d_h: int,
        d_g: int,
        d_proj: int = 256,
        dropout: float = 0.0,
        k_chunk: int = 256,             # L boyunca başlangıç parçası
        reduce: str = "none",           # "none" | "mean" | "attn"
        attn_t_hidden: int = 0,
        attn_t_dropout: float = 0.0,
        q_chunk_default: int = 4,       # T boyunca başlangıç parçası
        h_sub_init: int = 32,           # numerator için H alt-parçası
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

    @torch.amp.autocast('cuda', enabled=False)  # dtype kontrolünü içeride yapıyoruz
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

        # key mask (True=PAD)
        key_mask = None
        if mask is not None:
            key_mask = mask.bool().unsqueeze(1)      # [B,1,L]

        # parça boyutları (OOM olursa küçülecek)
        t_step = q_chunk or self.q_chunk_default
        k_step = self.k_chunk
        h_sub  = self.h_sub_init

        Z_parts: List[torch.Tensor] = []
        alpha_parts: List[torch.Tensor] = [] if return_alpha else None

        # ---- T boyunca dilimle ----
        for ts in range(0, T, t_step):
            te = min(T, ts + t_step)

            # Q: önce girişleri Wq.weight.dtype'a çevir, sonra linear, sonra fp32
            Gc = G[:, ts:te, :]
            Qc_lin = Gc.to(self.Wq.weight.dtype)
            Qc32 = self.Wq(Qc_lin).to(torch.float32)             # [B,t,P] fp32

            # LSE akümülatörleri: m,s (denominator), z (numerator)
            m = None
            s = None
            z = torch.zeros((B, te - ts, Dh), device=device, dtype=torch.float32)

            # ---- L boyunca dilimle ----
            ls = 0
            while ls < L:
                le = min(L, ls + k_step)
                try:
                    # H ve K: yine dtype hizalaması -> linear -> fp32
                    Hc = H[:, ls:le, :]
                    Hc_lin = Hc.to(self.Wk.weight.dtype)         # [B,ℓ,Dh] weight dtype
                    Kc32 = self.Wk(Hc_lin).to(torch.float32)     # [B,ℓ,P] fp32
                    # logits: [B,t,ℓ] = Qc32 [B,t,P] @ Kc32^T [B,P,ℓ]
                    lc = torch.bmm(Qc32, Kc32.transpose(1, 2)) * self.scale  # fp32
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

                # ---- LSE streaming + numerator streaming (tamamı fp32) ----
                lc_max = lc.max(dim=-1, keepdim=True).values  # [B,t,1]
                if m is None:
                    m = lc_max
                    s = torch.exp(lc - m).sum(dim=-1, keepdim=True)  # [B,t,1]

                    start = 0
                    sub_size = h_sub
                    while start < (le - ls):
                        stop = min(le - ls, start + sub_size)
                        H_sub32 = H[:, ls+start:ls+stop, :].to(torch.float32)  # [B,h,Dh]
                        w_sub = torch.exp(lc[:, :, start:stop] - m)            # [B,t,h]
                        try:
                            z = z + torch.bmm(w_sub, H_sub32)                  # [B,t,Dh]
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if sub_size > 16:
                                sub_size = max(16, sub_size // 2)
                                continue
                            else:
                                raise
                        start = stop
                else:
                    m_new = torch.maximum(m, lc_max)                           # [B,t,1]
                    scale_old = torch.exp(m - m_new)                           # [B,t,1]
                    s = scale_old * s + torch.exp(lc - m_new).sum(-1, keepdim=True)

                    start = 0
                    sub_size = h_sub
                    while start < (le - ls):
                        stop = min(le - ls, start + sub_size)
                        H_sub32 = H[:, ls+start:ls+stop, :].to(torch.float32)  # [B,h,Dh]
                        w_sub = torch.exp(lc[:, :, start:stop] - m_new)        # [B,1,t,h] değil -> [B,t,h]
                        try:
                            z = z.mul_(scale_old).add_(torch.bmm(w_sub, H_sub32))
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
                del Hc, Hc_lin, Kc32, lc
                torch.cuda.empty_cache()
                ls = le

            # t-chunk sonucu
            Zc = (z / s.clamp_min(1e-20))  # [B,t,Dh] fp32
            Zc = self.dropout(Zc)
            Z_parts.append(Zc)

            if return_alpha:
                # alpha üretimi (m ve s ile tekrar tarama)
                alpha_chunks: List[torch.Tensor] = []
                ls = 0
                while ls < L:
                    le = min(L, ls + k_step)
                    with torch.no_grad():
                        Hc = H[:, ls:le, :]
                        Hc_lin = Hc.to(self.Wk.weight.dtype)
                        Kc32 = self.Wk(Hc_lin).to(torch.float32)
                        lc = torch.bmm(Qc32, Kc32.transpose(1, 2)) * self.scale
                        if key_mask is not None:
                            km = key_mask[:, :, ls:le]
                            lc = lc.masked_fill(km, float("-inf"))
                        alpha_sub = torch.exp(lc - m) / s
                        alpha_chunks.append(alpha_sub)   # [B,t,ℓ]
                        del Hc, Hc_lin, Kc32, lc, alpha_sub
                        torch.cuda.empty_cache()
                    ls = le
                A_t = torch.cat(alpha_chunks, dim=-1)     # [B,t,L]
                alpha_parts.append(A_t)

            # free Q chunk & accumulators
            del Gc, Qc_lin, Qc32, m, s, z, Zc
            torch.cuda.empty_cache()

        # [B,T,Dh]
        Z = torch.cat(Z_parts, dim=1)

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