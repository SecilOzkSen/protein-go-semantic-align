import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from src.models.projection import ProjectionHead
from src.models.bucketed_watti import BucketedGoWatti
from src.encoders import BioMedBERTEncoder


class ProteinGoAligner(nn.Module):
    """
    Two-branch aligner for Protein (residue-level) and GO (text) representations.
    - Core path: forward(H, G, mask, return_alpha)
      * If G shape == [B, T, d_g]  -> per-token (T-match) mode, returns (B, T) scores
      * If G shape == [B, K, d_g]  -> candidate-pool mode, returns (B, K) scores
    """
    def __init__(
        self,
        d_h: int,
        d_g: Optional[int] = None,          # if None and go_encoder provided -> infer
        d_z: int = 768,
        go_encoder: Optional[BioMedBERTEncoder] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = bool(normalize)
        self.go_encoder: Optional[BioMedBERTEncoder] = go_encoder
        if self.go_encoder is not None and d_g is None:
            d_g = int(self.go_encoder.model.config.hidden_size)
        if d_g is None:
            raise ValueError("d_g must be provided if go_encoder is None.")

        # Pooler + projection heads
        self.pooler = BucketedGoWatti(d_h=d_h, d_g=d_g)
        self.proj_p = ProjectionHead(d_in=d_h, d_out=d_z)  # protein side
        self.proj_g = ProjectionHead(d_in=d_g, d_out=d_z)  # GO text side

    # ------------------ Core tensor path ------------------
    def forward(
        self,
        H: torch.Tensor,          # [B, T, d_h]
        G: torch.Tensor,          # [B, T, d_g]  (T-match)  OR  [B, K, d_g] (candidate pool)
        mask: torch.Tensor,       # [B, T] bool
        return_alpha: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[dict]]]:
        """
        Returns:
          - if G is [B,T,d_g] : scores -> [B, T]
          - if G is [B,K,d_g] : scores -> [B, K]
          If return_alpha=True, also returns alpha_info (dict or None).
        """
        device = H.device
        B, T, d_h = H.shape

        # ---- MODE A) T-match (per-token) ----
        if G.dim() == 3 and G.size(1) == T:
            # pooler çıktısını esnek şekilde aç
            pool_out = self.pooler(H, G, attn_mask=mask, return_alpha=return_alpha)
            if isinstance(pool_out, tuple):
                Z = pool_out[0]
                alpha_info = None
                for x in pool_out[1:]:
                    if isinstance(x, dict):
                        alpha_info = x
                        break
            else:
                Z = pool_out
                alpha_info = None

            # Projeksiyon
            Zp = self.proj_p(Z)      # [B, T, d_z]
            Gz = self.proj_g(G)      # [B, T, d_z]

            if self.normalize:
                Zp = F.normalize(Zp, dim=-1)
                Gz = F.normalize(Gz, dim=-1)

            # Per-token skor
            scores_bt = torch.einsum("btd,btd->bt", Zp, Gz)  # [B, T]

            if return_alpha:
                return scores_bt, (alpha_info or {})
            return scores_bt

        # ---- MODE B) Candidate-pool (G: [B, K, d_g]) ----
        if G.dim() == 3 and G.size(1) != T:
            K = G.size(1)
            d_g = G.size(2)

            # Adayı token boyutuna yayınla ve (B*K, T, ·) seviyesinde pooler çalıştır
            G_rep = G.unsqueeze(2).expand(B, K, T, d_g).contiguous().view(B * K, T, d_g)   # [B*K, T, d_g]
            H_rep = H.unsqueeze(1).expand(B, K, T, d_h).contiguous().view(B * K, T, d_h)   # [B*K, T, d_h]
            m_rep = mask.unsqueeze(1).expand(B, K, T).contiguous().view(B * K, T)          # [B*K, T]

            pool_out = self.pooler(H_rep, G_rep, attn_mask=m_rep, return_alpha=False)
            Z = pool_out[0] if isinstance(pool_out, tuple) else pool_out                  # [B*K, T, d_h]

            # Projeksiyon (aynı yayınlama ile)
            Zp = self.proj_p(Z)                              # [B*K, T, d_z]
            Gz = self.proj_g(G_rep)                          # [B*K, T, d_z]

            if self.normalize:
                Zp = F.normalize(Zp, dim=-1)
                Gz = F.normalize(Gz, dim=-1)

            # Per-token skor, sonra geçerli tokenlar boyunca ortalama
            tok_scores = torch.einsum("btd,btd->bt", Zp, Gz)     # [B*K, T]
            valid = (~m_rep).float()                              # [B*K, T], True=valid varsayımı
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            tok_scores = (tok_scores * valid).sum(dim=1, keepdim=True) / denom   # [B*K, 1]
            scores_bk = tok_scores.view(B, K)                                     # [B, K]

            if return_alpha:
                # Aday modunda alpha büyük olur; eğitim akışın bunu kullanmıyor → None döndür.
                return scores_bk, {}
            return scores_bk

        raise ValueError(f"Unexpected G shape: {tuple(G.shape)} (expected [B,T,d_g] or [B,K,d_g])")

    # ------------------ Helpers (senin bıraktığın gibi) ------------------
    @staticmethod
    def _weighted_avg(
        vecs: torch.Tensor,               # [N, d]
        idx: torch.Tensor,                # [M]
        w: Optional[torch.Tensor] = None  # [M] or None
    ) -> torch.Tensor:
        if idx.numel() == 0:
            return torch.zeros(vecs.size(-1), device=vecs.device)
        sel = vecs.index_select(0, idx)  # [M, d]
        if w is None or w.numel() == 0:
            return sel.mean(dim=0)
        w = w.to(sel.device).float()
        w = w / (w.sum().clamp_min(1e-6))
        return (sel * w.unsqueeze(-1)).sum(dim=0)

    def _broadcast_batch_go(
        self,
        uniq_G: torch.Tensor,                  # [G, d_g]
        pos_local: List[torch.Tensor],         # len=B; each [Mi] (local indices into uniq)
        pos_w: Optional[List[torch.Tensor]],   # len=B; each [Mi] or None
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        For each protein i, compute weighted average of its positive GO vectors (from uniq_G),
        producing g_i in R^{d_g}. Then repeat across T tokens -> [B, T, d_g].
        """
        B = len(pos_local)
        d_g = int(uniq_G.size(-1))
        G_list: List[torch.Tensor] = []
        for i in range(B):
            idx = pos_local[i].to(uniq_G.device)
            w = None
            if pos_w is not None and i < len(pos_w) and pos_w[i] is not None:
                w = pos_w[i].to(uniq_G.device)
            g_i = self._weighted_avg(uniq_G, idx, w=w)  # [d_g]
            G_list.append(g_i)

        G_b = torch.stack(G_list, dim=0).to(device)      # [B, d_g]
        G_bt = G_b.unsqueeze(1).expand(B, T, d_g).contiguous()
        return G_bt