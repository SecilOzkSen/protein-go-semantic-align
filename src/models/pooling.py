from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

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
    GO(token) x Protein(residue) cross-attention (bellek-dostu, parçalı):
      - Q = Wq * G  (GO tarafı)  [B,t,P]
      - K = Wk * H  (Residue)    [B,ℓ,P]
      - Logits = Q K^T / sqrt(P)
      - α softmax'ı 'streaming' ve mikro-k bloklarıyla hesaplanır:
          z  = Σ exp(lc - m_new) @ H_block      (fp32 akümülatör)
          s  = Σ exp(lc - m_new)
          out= z / s
      Notlar:
        * T'yi q_chunk ile, L'yi k_chunk ve inner_k ile dilimliyoruz.
        * Projeksiyonlar autocast(bf16) içinde; akümülatörler fp32.
    """
    def __init__(
        self,
        d_h: int,
        d_g: int,
        d_proj: int = 256,
        dropout: float = 0.0,
        # Bellek kontrol parametreleri:
        k_chunk: int = 512,         # L boyunca ana parça boyu
        inner_k: int = 64,          # L parçası içinde mikro parça
        q_chunk_default: int = 64,  # T boyunca parça boyu varsayılan
        reduce: str = "none",        # "none" | "mean" | "attn"
        attn_t_hidden: int = 0,
        attn_t_dropout: float = 0.0,
    ):
        super().__init__()
        self.Wk = nn.Linear(d_h, d_proj, bias=False)
        self.Wq = nn.Linear(d_g, d_proj, bias=False)
        self.scale = (d_proj ** -0.5)
        self.k_chunk = int(k_chunk)
        self.inner_k = int(inner_k)
        self.q_chunk_default = int(q_chunk_default)
        self.reduce = reduce
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.t_pool = None
        if self.reduce == "attn":
            self.t_pool = AttnPool1D(d_in=d_h, d_hidden=attn_t_hidden, dropout=attn_t_dropout)

    # autocast'ı açık tut: Lineerler ve bmm'ler bf16; akümülatörler fp32.
    @torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    def forward(
        self,
        H: torch.Tensor,                           # [B,L,Dh]
        G: torch.Tensor,                           # [B,T,Dg]
        mask: Optional[torch.Tensor] = None,       # [B,L] (True=PAD)
        return_alpha: bool = False,
        q_chunk: Optional[int] = None,             # T dilim boyu; None -> q_chunk_default
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        dev = self.Wq.weight.device  # modülün cihazı
        if H.device != dev:   H = H.to(dev, non_blocking=True)
        if G.device != dev:   G = G.to(dev, non_blocking=True)
        if mask is not None and mask.device != dev:
            mask = mask.to(dev, non_blocking=True)

        B, L, Dh = H.shape
        T = G.shape[1]
        device = H.device

        # mask: True=PAD --> -inf atacağız
        key_mask = None
        if mask is not None:
            key_mask = mask.bool().unsqueeze(1)  # [B,1,L]

        t_step = int(q_chunk or self.q_chunk_default)
        t_step = max(1, min(T, t_step))

        Z_parts = []
        # α'yı tam saklamak bellek-düşmanı; istenirse None döneceğiz.
        alpha_parts = None

        # --- T boyunca dilimle ---
        for ts in range(0, T, t_step):
            te = min(T, ts + t_step)
            Gc = G[:, ts:te, :]                 # [B,t,Dg]
            Qc = self.Wq(Gc)                    # [B,t,P] (bf16)

            # fp32 akümülatörler (numerik stabilite)
            t = Qc.shape[1]
            z = torch.zeros(B, t, Dh, device=device, dtype=torch.float32)   # pay
            s = torch.zeros(B, t, 1,  device=device, dtype=torch.float32)   # payda
            m = torch.full((B, t, 1), -float("inf"), device=device, dtype=torch.float32)

            # --- L boyunca ana parça ---
            for ls in range(0, L, self.k_chunk):
                le = min(L, ls + self.k_chunk)

                # --- bu ana parça içinde mikro-k döngüsü ---
                for ks in range(ls, le, self.inner_k):
                    ke = min(le, ks + self.inner_k)

                    H_block = H[:, ks:ke, :]              # [B,ki,Dh] (bf16)
                    K_block = self.Wk(H_block)            # [B,ki,P]  (bf16)

                    # logits bloğu: [B,t,ki] -> fp32
                    lc = torch.bmm(Qc, K_block.transpose(1, 2)) * self.scale  # bf16
                    lc = lc.to(torch.float32)

                    if key_mask is not None:
                        km = key_mask[:, :, ks:ke]        # [B,1,ki]
                        lc = lc.masked_fill(km, float("-inf"))

                    # running log-sum-exp güncellemesi (blok bazında)
                    block_max = lc.max(dim=-1, keepdim=True).values    # [B,t,1]
                    m_new = torch.maximum(m, block_max)                # [B,t,1]

                    # eski katkıyı yeni baza dönüştür
                    exp_m = torch.exp(m - m_new)                       # [B,t,1]
                    # yeni bloğun katkısı
                    exp_block = torch.exp(lc - m_new)                  # [B,t,ki]

                    # payda
                    s = s * exp_m + exp_block.sum(dim=-1, keepdim=True)    # [B,t,1]
                    # pay: (exp_block @ H_block)
                    # exp_block fp32, H_block fp16/bf16 -> fp32 çarpıp ekle
                    H_block = H_block.to(exp_block.dtype)
                    z = z * exp_m + torch.bmm(exp_block, H_block.to(torch.float32))  # [B,t,Dh]

                    m = m_new  # güncelle

                    # ara tensörleri serbest bırak
                    del H_block, K_block, lc, block_max, m_new, exp_m, exp_block
                    torch.cuda.empty_cache()

            # bu T-dilimi için çıktı
            Zc = (z / (s + 1e-8)).to(H.dtype)    # [B,t,Dh] tekrar bf16/fp16’e
            Z_parts.append(Zc)

            # serbest bırak
            del Gc, Qc, z, s, m, Zc
            torch.cuda.empty_cache()

        # [B,T,Dh]
        Z = torch.cat(Z_parts, dim=1)
        A = None  # alpha'yı full döndürmüyoruz (bellek dostu değil)

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