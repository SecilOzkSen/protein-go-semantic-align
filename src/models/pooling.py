import math
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

class GoSpecificWattiPooling(nn.Module):
    """
    GO-özel attention pooling (Watti tarzı), bellek-dostu "streaming softmax" ile.
    H: (B, L, Dh)  - protein/rezidü dizisi
    G: (B, T, Dg)  - GO dizi/temsil
    Çıktı: Z: (B, T, Dh)

    Özellikler:
      - Q ve K/H için chunk'lama (q_chunk, k_chunk)
      - Online/streaming softmax (alpha materialize etmeden)
      - bf16'da matmul, FP32 akümülatör ile sayısal sağlamlık
      - mask alias: forward(mask=...) da çalışır (attn_mask ile eşlenir)
      - return_alpha=True -> bilinçli olarak dense yola düşer (OOM riski)
    """

    def __init__(
        self,
        d_h: int,
        d_g: int,
        d_proj: int = 256,
        dropout: float = 0.0,        # ileride gerekirse eklenir
        *,
        # Q-chunk "e" (expert sayısı) ile hizalansın isteniyorsa e'yi ver; verilirse q_chunk default = e
        e: int | None = None,
        q_chunk: int | None = None,  # None -> e varsa e, yoksa otomatik
        k_chunk: int = 512,
        use_streaming: bool = True,
        use_bf16: bool = True,
    ):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = d_proj ** -0.5

        self.e = e
        self.user_q_chunk = q_chunk
        self.k_chunk = int(k_chunk)
        self.use_streaming = bool(use_streaming)
        self.use_bf16 = bool(use_bf16)

        # basit dropout kancası (şimdilik etkin değil)
        self._dropout_p = float(dropout)

    # ---- public ----
    def forward(self, H, G, attn_mask=None, return_alpha: bool = False, **kwargs):
        """
        attn_mask: (B, L)  True/1 = PAD/ignora edilecek
        Eski çağrılar için alias:
           forward(..., mask=...)  -> attn_mask eşleştirilir.
        """
        if attn_mask is None and "mask" in kwargs:
            attn_mask = kwargs["mask"]

        # mask’ı bool’a normalle
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.bool()

        # Projeksiyonları olası bf16 AMP ile yap
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (self.use_bf16 and H.is_cuda)
            else nullcontext()
        )
        with amp_ctx:
            K = self.Wk(H)   # (B, L, P)
            Q = self.Wq(G)   # (B, T, P)

        # Q-chunk otomatik seçimi: kullanıcı verdisi > e > T tabanlı heuristik
        B, L, Dh = H.shape
        _, T, _ = Q.shape
        q_chunk = (
            int(self.user_q_chunk)
            if self.user_q_chunk is not None
            else (int(self.e) if self.e is not None else max(1, min(64, T)))
        )

        if return_alpha:
            # ---- DENSE FALLBACK (alpha materialize eder) ----
            # OOM riski vardır; gerçekten gerekliyse kullan.
            logits = torch.einsum("btd,bld->btl", Q, K) * self.scale  # (B,T,L)
            if attn_mask is not None:
                logits = logits.masked_fill(attn_mask.unsqueeze(1), float("-inf"))
            alpha = torch.softmax(logits, dim=-1)                     # (B,T,L)
            Z = torch.bmm(alpha, H)                                   # (B,T,Dh)
            return Z, alpha

        if self.use_streaming:
            return self._streaming_attend(Q, K, H, attn_mask, q_chunk, self.k_chunk), None
        else:
            # Küçük girişler için hızlı yol (alpha’yı yine materialize ETMEYİZ)
            logits = torch.einsum("btd,bld->btl", Q, K) * self.scale  # (B,T,L)
            if attn_mask is not None:
                logits = logits.masked_fill(attn_mask.unsqueeze(1), float("-inf"))
            # softmax(btl) * H(blD) -> btd; alpha’yı tutmadan direkt matmul:
            # exp(logits - m) / s  * H  =  (exp(logits - m) @ H) / s
            m = torch.amax(logits, dim=-1, keepdim=True)              # (B,T,1)
            ex = torch.exp(logits - m)                                # (B,T,L)
            if torch.isinf(m).any():  # Tamamı mask ise NaN yayılmasın
                ex = torch.where(torch.isfinite(m), ex, torch.zeros_like(ex))
            num = torch.bmm(ex, H)                                    # (B,T,Dh)
            den = ex.sum(dim=-1, keepdim=True).clamp_min(1e-20)      # (B,T,1)
            Z = num / den
            return Z, None

    # ---- private: streaming softmax ----
    @torch.no_grad()
    def _streaming_attend(self, Q, K, H, attn_mask, q_chunk: int, k_chunk: int):
        """
        Online/streaming softmax (vec) – alpha’yı asla materialize etmez.
        Her Q parçası için K/H parçalarını sırayla tarar, (m, s, z) akümülatörleriyle
        sayısal stabil softmax uygular.
        """
        device = H.device
        B, L, Dh = H.shape
        _, T, P = Q.shape

        # Çıktı ve akümülatörler (FP32)
        Z = torch.empty(B, T, Dh, device=device, dtype=H.dtype)
        # Döngüde Q’yu chunk’layacağız
        for t0 in range(0, T, q_chunk):
            t1 = min(t0 + q_chunk, T)
            Qc = Q[:, t0:t1, :]                     # (B, tq, P)
            tq = Qc.size(1)

            # akümülatörler: (B, tq, 1/Dh)
            m = torch.full((B, tq, 1), -float("inf"), device=device, dtype=torch.float32)  # max logits
            s = torch.zeros(B, tq, 1, device=device, dtype=torch.float32)                  # sum exp
            z = torch.zeros(B, tq, Dh, device=device, dtype=torch.float32)                 # weighted sum

            # K/H üzerinde k-chunk taraması
            for l0 in range(0, L, k_chunk):
                l1 = min(l0 + k_chunk, L)
                Kc = K[:, l0:l1, :]                 # (B, lk, P)
                Hc = H[:, l0:l1, :]                 # (B, lk, Dh)

                # logits_chunk: (B, tq, lk)
                logits = torch.einsum("btd,bld->btl", Qc, Kc) * self.scale

                if attn_mask is not None:
                    mc = attn_mask[:, l0:l1]        # (B, lk) [True=pad]
                    logits = logits.masked_fill(mc.unsqueeze(1), float("-inf"))

                # online softmax güncellemesi
                # m_new = max(m_prev, max(logits_chunk, dim=-1))
                m_chunk = torch.amax(logits, dim=-1, keepdim=True).to(torch.float32)   # (B,tq,1)
                m_new = torch.maximum(m, m_chunk)                                      # (B,tq,1)

                # exp ölçekleri
                exp_prev = torch.exp(m - m_new)                                        # (B,tq,1)
                # logits - m_new; buradan exp
                ex = torch.exp((logits.to(torch.float32)) - m_new)                     # (B,tq,lk)
                # sum exp
                s = s * exp_prev + ex.sum(dim=-1, keepdim=True)                        # (B,tq,1)

                # weighted sum (ex @ Hc)
                num_chunk = torch.bmm(ex, Hc.to(torch.float32))                         # (B,tq,Dh)
                z = z * exp_prev + num_chunk

                m = m_new

            # normalize: z / s
            Zc = z / s.clamp_min(1e-20)                                                # (B,tq,Dh)
            # çıktı dtype’ına döndür (genelde bf16/fp16 değil, dikkatli olmak istersek fp32 bırakabiliriz)
            Z[:, t0:t1, :] = Zc.to(Z.dtype)

        return Z