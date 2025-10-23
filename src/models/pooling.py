import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

class GoSpecificWattiPooling(nn.Module):
    """
    GO-specific attention pooling with streaming softmax:
    - Q tarafı chunk'lanır (T -> q_chunk)
    - K/H tarafı da tile'lanır (L -> k_chunk)
    - alpha (B,T,L) materialize edilmez; iki geçişli/stabil softmax ile Z = softmax(QK^T)H akümüle edilir.
    """
    def __init__(self, d_h, d_g, d_proj=256, dropout=0.0,
                 q_chunk: int = 16,         # T için chunk
                 k_chunk: int = 512,        # L için tile
                 use_streaming: bool = True,
                 use_bf16: bool = True):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)
        self.q_chunk = q_chunk
        self.k_chunk = k_chunk
        self.use_streaming = use_streaming
        self.use_bf16 = use_bf16

    def forward(self, H, G, attn_mask=None, return_alpha=False):
        """
        H: (B, L, Dh)  protein/ESM embeddings
        G: (B, T, Dg)  GO embeddings
        attn_mask: (B, L)  True=pad/ignore  (eski koddaki gibi)
        """
        # Projeksiyonlar hafif, bf16 altında güvenli; skor/exp FP32'ye alınacak.
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (self.use_bf16 and H.is_cuda) else nullcontext()
        with amp_ctx:
            K = self.Wk(H)                 # (B, L, P)  bf16/fp16
            Q = self.Wq(G)                 # (B, T, P)
            K = K.contiguous()
            Q = Q.contiguous()

        if self.use_streaming and not return_alpha:
            return self._streaming_attend(Q, K, H, attn_mask)

        # ---- Fallback (sadece return_alpha=True için) ----
        # (Bu yol alpha’yı materialize eder; OOM riski vardır.)
        logits = torch.einsum('btd,bld->btl', Q, K) * self.scale  # (B,T,L)
        if attn_mask is not None:
            logits = logits.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        alpha = torch.softmax(logits, dim=-1)                     # (B,T,L)
        Z = torch.bmm(alpha, H)                                   # (B,T,Dh)
        return (Z, alpha) if return_alpha else Z

    @torch.no_grad()
    def _streaming_attend(self, Q, K, H, attn_mask):
        """
        alpha = softmax(QK^T) üretmeden Z = alpha @ H hesaplar.
        İki geçişli/stabil softmax (tile-by-tile) uygular.
        Geri dönüş: (B, T, Dh)
        """
        B, L, P = K.shape
        _, T, _ = Q.shape
        Dh = H.size(-1)

        # Çıkış dtype'ını girişle uyumlu tut; akümülasyon FP32'de.
        out_dtype = H.dtype

        Z_all = []
        for t0 in range(0, T, self.q_chunk):
            t1 = min(t0 + self.q_chunk, T)
            q = Q[:, t0:t1, :].to(torch.float32)    # (B, Tc, P) fp32’de skor/exp daha stabil

            # Online softmax istatistikleri:
            # m: running max, lse: running sum(exp), z: running numerator (exp * H)
            m   = torch.full((B, q.size(1), 1), -float('inf'), device=Q.device, dtype=torch.float32)
            lse = torch.zeros((B, q.size(1)),    device=Q.device, dtype=torch.float32)
            z   = torch.zeros((B, q.size(1), Dh), device=Q.device, dtype=torch.float32)

            for l0 in range(0, L, self.k_chunk):
                l1 = min(l0 + self.k_chunk, L)
                k_tile = K[:, l0:l1, :].to(torch.float32)          # (B, Lc, P)
                h_tile = H[:, l0:l1, :].to(torch.float32)          # (B, Lc, Dh)

                scores = torch.einsum('btd,bld->btl', q, k_tile) * self.scale  # (B,Tc,Lc) fp32

                if attn_mask is not None:
                    # attn_mask: (B,L) True/1 = pad/ignore
                    msk = attn_mask[:, l0:l1].unsqueeze(1)         # (B,1,Lc)
                    scores = scores.masked_fill(msk, float('-inf'))

                # Tile içi max
                tile_max = scores.max(dim=-1, keepdim=True).values # (B,Tc,1)
                new_m = torch.maximum(m, tile_max)                 # (B,Tc,1)

                # Eski katkıyı yeni max'a göre ölçekle
                exp_m_scale = torch.exp(m - new_m)                 # (B,Tc,1)
                z = z * exp_m_scale
                lse = lse * exp_m_scale.squeeze(-1)

                # Bu tile’ın katkısı
                exp_scores = torch.exp(scores - new_m)             # (B,Tc,Lc)
                z = z + torch.bmm(exp_scores, h_tile)              # (B,Tc,Dh)
                lse = lse + exp_scores.sum(dim=-1)                 # (B,Tc)

                m = new_m
                # ara tensörleri asap bırak
                del k_tile, h_tile, scores, tile_max, new_m, exp_m_scale, exp_scores

            # Normalize et
            Zt = z / (lse.clamp_min(1e-6).unsqueeze(-1))           # (B,Tc,Dh)
            Z_all.append(Zt.to(out_dtype))
            del q, m, lse, z, Zt

        Z = torch.cat(Z_all, dim=1)  # (B,T,Dh)
        return Z