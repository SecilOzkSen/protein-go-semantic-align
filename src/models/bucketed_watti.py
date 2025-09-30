from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pooling import GoSpecificWattiPooling

class BucketedGoWatti(nn.Module):
    """
    Bucket-aware wrapper:
      - L <= short_thr: full-length pooling
      - short_thr < L <= mid_thr: small windows
      - L > mid_thr: large windows
    Returns either full-length alpha or windowed alpha metadata depending on path.
    """
    def __init__(self,
                 d_h: int,
                 d_g: int,
                 short_thr: int = 2048,
                 mid_thr: int = 8192,
                 win_small: int = 1024,
                 stride_small: int = 256,
                 win_large: int = 2048,
                 stride_large: int = 512):
        super().__init__()
        self.core = GoSpecificWattiPooling(d_h=d_h, d_g=d_g, d_proj=256, dropout=0.1)
        self.short_thr = short_thr
        self.mid_thr = mid_thr
        self.win_small, self.stride_small = win_small, stride_small
        self.win_large, self.stride_large = win_large, stride_large

        # GO-conditional window aggregator
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

    def forward(self,
                H: torch.Tensor,        # (B,L,D)
                G: torch.Tensor,        # (B,T,Gd)
                attn_mask: Optional[torch.Tensor] = None,  # (B,L) bool where True=PAD
                return_alpha: bool = False):
        B, L, D = H.shape
        if L <= self.short_thr:
            Z, A = self.core(H, G, mask=attn_mask, return_alpha=True)  # (B,T,D), (B,T,L)
            if return_alpha:
                return Z, {"alpha_full": A}
            return Z

        # choose window config
        if L <= self.mid_thr:
            win, stride = self.win_small, self.stride_small
        else:
            win, stride = self.win_large, self.stride_large

        spans = list(self._window_spans(L, win, stride))
        win_Z = []
        win_A = [] if return_alpha else None

        for s, e in spans:
            Hk = H[:, s:e, :]
            mk = attn_mask[:, s:e] if attn_mask is not None else None
            Zk, Ak = self.core(Hk, G, mask=mk, return_alpha=True)  # (B,T,D), (B,T,e-s)
            win_Z.append(Zk)
            if return_alpha:
                if (e - s) < win:
                    pad = win - (e - s)
                    Ak = F.pad(Ak, (0, pad), value=0.0)
                win_A.append(Ak)

        # stack windows
        ZW = torch.stack(win_Z, dim=2)  # (B,T,W,D)
        # window-level attention
        q_t = self.win_query(G)                   # (B,T,D)
        k_w = self.win_key(ZW)                    # (B,T,W,D)
        logits = torch.einsum("btd,btwd->btw", q_t, k_w) * self.scale
        w = torch.softmax(logits, dim=-1)         # (B,T,W)
        Z = torch.einsum("btw,btwd->btd", w, ZW)  # (B,T,D)

        if not return_alpha:
            return Z

        AW = torch.stack(win_A, dim=2) if win_A is not None else None  # (B,T,W,win)
        return Z, {"alpha_windows": AW, "win_weights": w, "spans": spans}
