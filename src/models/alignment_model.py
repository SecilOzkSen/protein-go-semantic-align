import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.projection import ProjectionHead
from src.models.bucketed_watti import BucketedGoWatti
from src.encoders import BioMedBERTEncoder


class ProteinGoAligner(nn.Module):
    """
    Two-branch aligner for Protein (residue-level) and GO (text) representations.

    İki kip:
      1) Pozitif kip:  G: [B, T, d_g]  -> scores: [B, T]  (+ opsiyonel alpha_info)
      2) Aday kip:     G: [B, K, d_g]  -> scores: [B, K]  (chunk'lı, hafıza dostu)
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
        self.proj_p = ProjectionHead(d_in=d_h, d_out=d_z)  # protein side
        self.proj_g = ProjectionHead(d_in=d_g, d_out=d_z)  # GO text side

    # ---- helpers ----
    def _norm(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # sayısal kararlılık: önce fp32
        return F.normalize(x.float(), p=2, dim=dim)

    def _pool(self, H: torch.Tensor, G: torch.Tensor, mask: torch.Tensor, return_alpha: bool):
        """
        BucketedGoWatti çıktılarını sağlamlaştırır:
        - Z'yi her durumda döndürür
        - alpha_info varsa dict, yoksa {}
        """
        out = self.pooler(H, G, attn_mask=mask, return_alpha=return_alpha)
        if isinstance(out, tuple):
            # En azından Z hep ilk eleman
            Z = out[0]
            # İkinci eleman dict değilse de, alpha_info'yu boş döndür
            alpha_info = out[1] if (len(out) >= 2 and isinstance(out[1], dict)) else {}
        else:
            Z = out
            alpha_info = {}
        return Z, alpha_info

    def forward(
            self,
            H: torch.Tensor,  # [B, T, d_h]
            G: torch.Tensor,  # [B, T, d_g]  veya  [B, K, d_g]
            mask: torch.Tensor,  # [B, T] bool
            return_alpha: bool = False,
            cand_chunk_k: int = 16,  # <<< daha küçük, OOM güvenli default
    ):
        device = H.device
        B, T, d_h = H.shape

        # ---- Pozitif kip: G ikinci boyutu T ise ----
        if G.dim() == 3 and G.size(1) == T:
            Z, alpha_info = self._pool(H, G, mask, return_alpha=return_alpha)  # [B, T, d_h], dict
            Zp = self.proj_p(Z)  # [B, T, d_z]
            Gz = self.proj_g(G)  # [B, T, d_z]
            if self.normalize:
                Zp = self._norm(Zp, dim=-1)
                Gz = self._norm(Gz, dim=-1)
            scores = (Zp * Gz).sum(dim=-1)  # [B, T]
            return (scores, alpha_info) if return_alpha else scores

        # ---- Aday kip: G ikinci boyutu K ise ----
        if G.dim() != 3:
            raise ValueError("G must be 3D: [B, T, d_g] or [B, K, d_g].")

        K = int(G.size(1))
        d_g = int(G.size(2))
        k_step = max(1, int(cand_chunk_k))  # sağlamlaştır

        scores_list = []
        # Aday kipinde alpha gerekmiyor; boş döneceğiz.
        for s in range(0, K, k_step):
            e = min(K, s + k_step)
            k = e - s
            G_chunk = G[:, s:e, :]  # [B, k, d_g]

            # (B, T, d_h) -> (B, k, T, d_h) -> (B*k, T, d_h)
            H_rep = H.unsqueeze(1).expand(B, k, T, d_h).reshape(B * k, T, d_h)
            mask_rep = mask.unsqueeze(1).expand(B, k, T).reshape(B * k, T)

            # (B, k, d_g) -> (B, k, T, d_g) -> (B*k, T, d_g)
            G_rep = G_chunk.unsqueeze(2).expand(B, k, T, d_g).reshape(B * k, T, d_g)

            # pool + proj (alpha yok)
            Z_rep, _ = self._pool(H_rep, G_rep, mask_rep, return_alpha=False)  # [B*k, T, d_h]
            Zp = self.proj_p(Z_rep)  # [B*k, T, d_z]
            Gz = self.proj_g(G_rep)  # [B*k, T, d_z]
            if self.normalize:
                Zp = self._norm(Zp, dim=-1)
                Gz = self._norm(Gz, dim=-1)

            s_tok = (Zp * Gz).sum(dim=-1)  # [B*k, T]
            s_bk = s_tok.mean(dim=-1).reshape(B, k)  # [B, k]
            scores_list.append(s_bk)

            # agresif temizlik
            del H_rep, mask_rep, G_rep, Z_rep, Zp, Gz, s_tok, s_bk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scores = torch.cat(scores_list, dim=1)  # [B, K]
        return scores