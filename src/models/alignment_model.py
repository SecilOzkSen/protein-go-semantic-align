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
    Backward compatible:
      - Old mode: forward(H, G, mask)  -> uses given tensors directly
      - New mode: forward(batch_dict)  -> encodes GO texts with GoEncoder (+LoRA)
    """
    def __init__(
        self,
        d_h: int,
        d_g: Optional[int] = None,          # if None, will be set from go_encoder hidden_size
        d_z: int = 768,
        go_encoder: Optional[BioMedBERTEncoder] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = bool(normalize)
        self.go_encoder: Optional[BioMedBERTEncoder] = go_encoder
        if self.go_encoder is not None and d_g is None:
            # take hidden size from encoder
            d_g = int(self.go_encoder.model.config.hidden_size)

        if d_g is None:
            raise ValueError("d_g must be provided if go_encoder is None.")

        # --- Pooler + projection heads ---
        self.pooler = BucketedGoWatti(d_h=d_h, d_g=d_g)
        self.proj_p = ProjectionHead(d_in=d_h, d_out=d_z)  # protein side
        self.proj_g = ProjectionHead(d_in=d_g, d_out=d_z)  # GO text side

    # ------------------ Core tensor path ------------------
    def forward(
        self,
        H: torch.Tensor,          # [B, T, d_h]
        G: torch.Tensor,          # [B, T, d_g]
        mask: torch.Tensor,       # [B, T] bool
        return_alpha: bool = False
    ):
        Z, alpha_info = self.pooler(H, G, attn_mask=mask, return_alpha=return_alpha)  # (B, T, d_h), dict
        Zp = self.proj_p(Z)  # (B, T, d_z)
        Gz = self.proj_g(G)  # (B, T, d_z)

        if self.normalize:
            Zp = F.normalize(Zp, dim=-1)
            Gz = F.normalize(Gz, dim=-1)

        scores = torch.einsum("btd,btd->bt", Zp, Gz)  # cosine per token, then summed along last dim
        return (scores, alpha_info) if return_alpha else (scores, None)

    # ------------------ Helpers ------------------
    @staticmethod
    def _weighted_avg(
        vecs: torch.Tensor,               # [N, d]
        idx: torch.Tensor,                # [M] indices into vecs
        w: Optional[torch.Tensor] = None  # [M] weights or None
    ) -> torch.Tensor:
        if idx.numel() == 0:
            # edge case: no positives → return zero vec (will be normalized later in broadcast)
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
        pos_w: Optional[List[torch.Tensor]],   # len=B; each [Mi] weights (or None)
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        For each protein i, compute weighted average of its positive GO vectors (from uniq_G),
        producing g_i in R^{d_g}. Then repeat across T tokens: G[i] -> [T, d_g].
        Returns: [B, T, d_g]
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

        # Stack to [B, d_g] then broadcast to [B, T, d_g]
        G_b = torch.stack(G_list, dim=0).to(device)      # [B, d_g]
        G_bt = G_b.unsqueeze(1).expand(B, T, d_g).contiguous()
        return G_bt
