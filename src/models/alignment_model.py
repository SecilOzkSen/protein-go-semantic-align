
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from src.models.projection import ProjectionHead
from src.models.bucketed_watti import BucketedGoWatti
from src.encoders import BioMedBERTEncoder


class ProteinGoAligner(nn.Module):
    """
    Two-branch aligner for Protein (residue-level) and GO (text) representations.
    """
    def __init__(
        self,
        d_h: int,
        d_g: Optional[int] = None,
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

        self.pooler = BucketedGoWatti(d_h=d_h, d_g=d_g)
        self.proj_p = ProjectionHead(d_in=d_h, d_out=d_z)  # protein
        self.proj_g = ProjectionHead(d_in=d_g, d_out=d_z)  # GO

    def _norm(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # daima fp32 normla → sayısal kararlılık
        return F.normalize(x.float(), p=2, dim=dim)

    def forward(
        self,
        H: torch.Tensor,         # [B, T, d_h]
        G: torch.Tensor,         # [B, T, d_g]  veya  [B, K, d_g]
        mask: torch.Tensor,      # [B, T] bool
        return_alpha: bool = False,
        cand_chunk_k: int = 256, # aday kipinde K'yi parça parça işle
    ):
        device = H.device
        B, T, d_h = H.shape

        # --------------------------
        # 1) POZİTİF KİP (G.shape[1] == T)
        # --------------------------
        if G.dim() == 3 and G.size(1) == T:
            # standart yol: tek G dizisi için token skorları
            Z, alpha_info = self.pooler(H, G, attn_mask=mask, return_alpha=return_alpha)  # [B, T, d_h]
            Zp = self.proj_p(Z)  # [B, T, d_z]
            Gz = self.proj_g(G)  # [B, T, d_z]

            if self.normalize:
                Zp = self._norm(Zp, dim=-1)
                Gz = self._norm(Gz, dim=-1)

            # token-başı kosinüs benzerliği → [B, T]
            scores = (Zp * Gz).sum(dim=-1)
            return (scores, alpha_info) if return_alpha else (scores, {})

        # --------------------------
        # 2) ADAY KİP (G.shape[1] == K)
        # --------------------------
        if G.dim() != 3:
            raise ValueError("G must be 3D: [B, T, d_g] or [B, K, d_g].")
        K = G.size(1)
        d_g = G.size(2)

        # Hafızayı patlatmamak için K'yi chunk'la.
        # Her chunk için:
        #   - H ve mask 'k' kez tekrarlanır (B*k, T, *)
        #   - G_chunk her aday için T boyunca yayınlanır (B*k, T, d_g)
        #   - pooler + proj + skor hesaplanır
        #   - token skorları T üzerinde ortalanıp [B, k] elde edilir
        scores_list = []
        alpha_info_global = {}  # aday kipinde alpha sadece debug için gerekli değil -> boş döneceğiz

        # TF32 / autocast üst düzeyde zaten aktif; burada ek bir şey gerekmiyor.

        k_step = int(max(1, cand_chunk_k))
        for s in range(0, K, k_step):
            e = min(K, s + k_step)
            k = e - s  # bu chunk boyu

            G_chunk = G[:, s:e, :]  # [B, k, d_g]

            # (B, T, d_h) -> (B, k, T, d_h) -> (B*k, T, d_h)
            H_rep = H.unsqueeze(1).expand(B, k, T, d_h).reshape(B * k, T, d_h)
            mask_rep = mask.unsqueeze(1).expand(B, k, T).reshape(B * k, T)

            # (B, k, d_g) -> (B, k, T, d_g) -> (B*k, T, d_g)
            G_rep = G_chunk.unsqueeze(2).expand(B, k, T, d_g).reshape(B * k, T, d_g)

            # pooler + proj
            Z_rep, _ = self.pooler(H_rep, G_rep, attn_mask=mask_rep, return_alpha=False)   # [B*k, T, d_h]
            Zp = self.proj_p(Z_rep)                                                        # [B*k, T, d_z]
            Gz = self.proj_g(G_rep)                                                        # [B*k, T, d_z]

            if self.normalize:
                Zp = self._norm(Zp, dim=-1)
                Gz = self._norm(Gz, dim=-1)

            # aday başına token skorlarını ortala → [B*k]
            s_tok = (Zp * Gz).sum(dim=-1)                  # [B*k, T]
            s_bk = s_tok.mean(dim=-1).reshape(B, k)        # [B, k]
            scores_list.append(s_bk)

            # ara tensörleri serbest bırak
            del H_rep, mask_rep, G_rep, Z_rep, Zp, Gz, s_tok, s_bk
            torch.cuda.empty_cache()

        scores = torch.cat(scores_list, dim=1)  # [B, K]
        return (scores, alpha_info_global) if return_alpha else (scores, {})