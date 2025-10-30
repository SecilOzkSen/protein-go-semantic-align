from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pooling import GoSpecificWattiPooling


class BucketedGoWatti(nn.Module):
    """
    Bucket-aware + streaming window aggregation:
      - L <= short_thr        : full-length pooling (tek shot)
      - short_thr < L <= mid_thr: küçük pencereler, akümülatif (streaming) pencere-attn
      - L > mid_thr           : büyük pencereler, akümülatif (streaming) pencere-attn

    Bellek dostu: (B,T,W,D) gibi dev tensörler OLUŞTURMAZ.
    """
    def __init__(self,
                 d_h: int,
                 d_g: int,
                 short_thr: int = 2048,
                 mid_thr: int = 8192,
                 win_small: int = 1536,
                 stride_small: int = 384,
                 win_large: int = 3072,
                 stride_large: int = 768):
        super().__init__()
        # Çekirdek cross-attn (zaten bellek-dostu, chunked)
        self.core = GoSpecificWattiPooling(d_h=d_h, d_g=d_g, d_proj=256, dropout=0.0)

        self.short_thr = int(short_thr)
        self.mid_thr = int(mid_thr)
        self.win_small, self.stride_small = int(win_small), int(stride_small)
        self.win_large, self.stride_large = int(win_large), int(stride_large)

        # Pencere-üstü attention için (W ekseni) lineerler
        self.win_query = nn.Linear(d_g, d_h, bias=False)
        self.win_key   = nn.Linear(d_h, d_h, bias=False)
        self.scale = d_h ** -0.5

    def _window_spans(self, L: int, win: int, stride: int):
        s = 0
        while s < L:
            e = min(s + win, L)
            yield (s, e)
            if e == L:
                break
            s += stride

    @torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    def forward(self,
                H: torch.Tensor,        # (B,L,D)
                G: torch.Tensor,        # (B,T,Gd)
                attn_mask: Optional[torch.Tensor] = None,  # (B,L) True=PAD
                return_alpha: bool = False):

        dev = self.win_query.weight.device  # modülün cihazı
        if H.device != dev:   H = H.to(dev, non_blocking=True)
        if G.device != dev:   G = G.to(dev, non_blocking=True)
        if attn_mask is not None and attn_mask.device != dev:
            attn_mask = attn_mask.to(dev, non_blocking=True)

        B, L, D = H.shape
        T = G.shape[1]
        device = H.device

        # KISA DİZİ: tek seferde
        if L <= self.short_thr:
            out = self.core(H, G, mask=attn_mask, return_alpha=return_alpha)
            # core, return_alpha=False ise Tensor; True ise (Z, None) döner
            return out

        # Hangi pencere ayarı?
        if L <= self.mid_thr:
            win, stride = self.win_small, self.stride_small
        else:
            win, stride = self.win_large, self.stride_large

        # Pencere-üstü attn için sabit sorgu (q_t)
        # Dtype bf16/ fp16; akümülatörler fp32
        q_t = self.win_query(G)  # (B,T,D)

        # Streaming softmax akümülatörleri (W boyunca)
        z_w = torch.zeros(B, T, D, device=device, dtype=torch.float32)  # pay
        s_w = torch.zeros(B, T, 1, device=device, dtype=torch.float32)  # payda
        m_w = torch.full((B, T, 1), -float("inf"), device=device, dtype=torch.float32)

        # Pencereleri tek tek işle
        for s, e in self._window_spans(L, win, stride):
            Hk = H[:, s:e, :]
            mk = attn_mask[:, s:e] if attn_mask is not None else None

            # Çekirdek: Zk = cross-attn(Hk, G) -> (B,T,D)
            Zk = self.core(Hk, G, mask=mk, return_alpha=False)  # (B,T,D) bf16/fp16

            # Pencere anahtarı ve logit (tek pencere = tek "token")
            # k_w: (B,T,D) -> skaler logit_w: (B,T,1)
            k_w = self.win_key(Zk)                     # (B,T,D)
            logits_w = (q_t * k_w).sum(dim=-1, keepdim=True) * self.scale  # (B,T,1)
            lw32 = logits_w.to(torch.float32)

            # Streaming softmax (W boyunca) — log-sum-exp
            m_new = torch.maximum(m_w, lw32)           # (B,T,1)
            exp_m = torch.exp(m_w - m_new)             # (B,T,1)
            exp_w = torch.exp(lw32 - m_new)            # (B,T,1)

            s_w = s_w * exp_m + exp_w                  # payda
            z_w = z_w * exp_m + (exp_w * Zk.to(torch.float32))  # pay
            m_w = m_new

            # ara tensörleri bırak
            del Hk, Zk, k_w, logits_w, lw32, m_new, exp_m, exp_w
            torch.cuda.empty_cache()

        # Son pencere-üstü havuzlama
        Z = (z_w / (s_w + 1e-8)).to(H.dtype)  # (B,T,D)

        # Bellek tasarrufu için full alpha döndürmüyoruz
        if not return_alpha:
            return Z
        return Z, None