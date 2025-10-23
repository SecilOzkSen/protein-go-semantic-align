import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

class GoSpecificWattiPooling(nn.Module):
    """
    Memory-friendly GO attention pooling:
      - Precompute K = Wk(H) once
      - Chunk Q across T dimension (q_chunk)
      - Compute logits/softmax per-chunk, immediately multiply with H
      - No full alpha kept in memory unless return_alpha=True
    """
    def __init__(self, d_h, d_g, d_proj=256, dropout=0.0,
                 q_chunk: int = 24,   # <= T; 32 iyi bir başlangıç
                 use_autocast: bool = True,
                 use_bf16: bool = True):  # H100 için bf16 çok stabil
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)
        self.q_chunk = q_chunk
        self.use_autocast = use_autocast
        self.use_bf16 = use_bf16

    def forward(self, H, G, mask=None, return_alpha=False):
        """
        H: (B, L, D_h)
        G: (B, T, D_g)
        mask: (B, L)  True=maskla/inf, False=geçerli  (senin kodundaki ile uyumlu)
        """
        B, L, Dh = H.shape
        T = G.shape[1]
        device = H.device

        # Otomatik q_chunk (isteğe bağlı): çok büyük T gelirse küçült
        q_chunk = self.q_chunk if self.q_chunk and self.q_chunk > 0 else T
        q_chunk = min(q_chunk, T)

        # Hesaplama dtypes
        acc_dtype = torch.bfloat16 if (self.use_bf16 and torch.cuda.is_available()) else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=acc_dtype) if (self.use_autocast and H.is_cuda) else nullcontext()

        # K'yi bir defa projekte et
        with amp_ctx:
            K = self.Wk(H)                   # (B, L, P)
            KT = K.transpose(1, 2).contiguous()  # (B, P, L)

        out_chunks = []
        alpha_chunks = [] if return_alpha else None

        for s in range(0, T, q_chunk):
            e = min(s + q_chunk, T)
            Gc = G[:, s:e, :]                # (B, Tc, Dg)
            with amp_ctx:
                Qc = self.Wq(Gc)             # (B, Tc, P)
                # logits: (B, Tc, L)
                logits = torch.bmm(Qc, KT) * self.scale
                if mask is not None:
                    # mask True ise -inf (senin mask semantiğine göre)
                    logits = logits.masked_fill(mask.unsqueeze(1), float("-inf"))

                # Softmax'ı fp32'de yap, sonra geri döndür
                alpha = F.softmax(logits.to(torch.float32), dim=-1).to(Qc.dtype)

                # Zc = alpha @ H  -> (B, Tc, Dh)
                Zc = torch.bmm(alpha, H)

            out_chunks.append(Zc)
            if return_alpha:
                # Dikkat: bu büyük olabilir; sadece debug için aç
                alpha_chunks.append(alpha.detach().to("cpu"))

            # Ara tensörleri serbest bırak
            del Gc, Qc, logits, alpha, Zc
            torch.cuda.empty_cache()

        Z = torch.cat(out_chunks, dim=1)     # (B, T, Dh)
        if return_alpha:
            return Z, alpha_chunks
        return Z