import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

from src.models.projection import ProjectionHead
from src.models.bucketed_watti import BucketedGoWatti
from src.encoders import BioMedBERTEncoder


class ProteinGoAligner(nn.Module):
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
    @staticmethod
    def _norm(x, dim=-1, eps=1e-6, norm_chunk: int = 0):
        """
        L2 normalize without materializing a full fp32 copy.
        - Compute norms in fp32 for stability, but per-chunk.
        - Return in original dtype (bf16/fp16 friendly).
        """
        dtype = x.dtype
        if norm_chunk is None or norm_chunk <= 0:
            n = torch.linalg.vector_norm(x.to(torch.float32), dim=dim, keepdim=True).clamp_min(eps)
            return x / n.to(dtype)

        outs = []
        # chunk along the first dimension that is "batch-like".
        # Gz genelde (B, T, Dh). Burada ilk eksen B*T olabilir; cat edilen ekseni koruyoruz.
        N = x.shape[0]
        for s in range(0, N, norm_chunk):
            xe = x[s:s + norm_chunk]
            ne = torch.linalg.vector_norm(xe.to(torch.float32), dim=dim, keepdim=True).clamp_min(eps)
            outs.append(xe / ne.to(dtype))
            del xe, ne
        return torch.cat(outs, dim=0)

    def _pool(self, H: torch.Tensor, G: torch.Tensor, mask: torch.Tensor, return_alpha: bool):
        out = self.pooler(H, G, attn_mask=mask, return_alpha=return_alpha)
        if isinstance(out, tuple):
            Z = out[0]
            alpha_info = out[1] if (len(out) >= 2 and isinstance(out[1], dict)) else {}
        else:
            Z, alpha_info = out, {}
        return Z, alpha_info

    def forward(
        self,
        H: torch.Tensor,         # [B, T, d_h]
        G: torch.Tensor,         # [B, T, d_g]  veya  [B, K, d_g]
        mask: torch.Tensor,      # [B, T] bool
        return_alpha: bool = False,
        cand_chunk_k: int = 16,  # K-yönlü chunk
        pos_chunk_t: int = 256,  # T-yönlü chunk
    ):
        device = H.device
        B, T, d_h = H.shape

        # =========================
        #   POZİTİF YOL (G ~ [B,T,*])
        # =========================
        if G.dim() == 3 and G.size(1) == T:
            # T ekseninde parça parça işle, alpha_info’yu da birleştir
            scores_parts: List[torch.Tensor] = []
            alpha_full_parts: List[torch.Tensor] = []
            aw_parts: List[torch.Tensor] = []
            spans: List = []
            win_weights: Optional[torch.Tensor] = None

            for s in range(0, T, max(1, int(pos_chunk_t))):
                e = min(T, s + max(1, int(pos_chunk_t)))
                Hs = H[:, s:e, :]                  # [B, t, d_h]
                Gs = G[:, s:e, :]                  # [B, t, d_g]
                ms = mask[:, s:e]                  # [B, t]

                Zs, ainfo = self._pool(Hs, Gs, ms, return_alpha=return_alpha)
                Zp = self.proj_p(Zs)               # [B, t, d_z]
                Gz = self.proj_g(Gs)               # [B, t, d_z]
                if self.normalize:
                    Zp = self._norm(Zp, dim=-1, norm_chunk=256)
                    Gz = self._norm(Gz, dim=-1, norm_chunk=256)
                scores_parts.append((Zp * Gz).sum(dim=-1))   # [B, t]

                # alpha biriktir
                if return_alpha and isinstance(ainfo, dict):
                    if "alpha_full" in ainfo:
                        alpha_full_parts.append(ainfo["alpha_full"])  # [B, t, L]
                    if all(k in ainfo for k in ["alpha_windows", "win_weights", "spans"]):
                        aw_parts.append(ainfo["alpha_windows"])       # [B, t, W, win]
                        if win_weights is None:
                            win_weights = ainfo["win_weights"]
                        if not spans:
                            spans = ainfo["spans"]

                # temizlik
                del Hs, Gs, ms, Zs, Zp, Gz, ainfo
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            scores = torch.cat(scores_parts, dim=1)       # [B, T]

            if not return_alpha:
                return scores

            alpha_info_out: Dict[str, torch.Tensor] = {}
            if alpha_full_parts:
                alpha_info_out["alpha_full"] = torch.cat(alpha_full_parts, dim=1)   # [B, T, L]
            if aw_parts:
                alpha_info_out["alpha_windows"] = torch.cat(aw_parts, dim=1)        # [B, T, W, win]
                if win_weights is not None:
                    alpha_info_out["win_weights"] = win_weights
                if spans:
                    alpha_info_out["spans"] = spans
            return scores, alpha_info_out

        # =========================
        #   ADAY YOL (G ~ [B,K,*])
        # =========================
        if G.dim() != 3:
            raise ValueError("G must be 3D: [B, T, d_g] or [B, K, d_g].")

        K = int(G.size(1))
        d_g = int(G.size(2))
        k_step = max(1, int(cand_chunk_k))

        scores_list = []
        for s in range(0, K, k_step):
            e = min(K, s + k_step)
            k = e - s
            G_chunk = G[:, s:e, :]                                   # [B, k, d_g]

            # (B, T, d_h) -> (B, k, T, d_h) -> (B*k, T, d_h)
            H_rep = H.unsqueeze(1).expand(B, k, T, d_h).reshape(B * k, T, d_h)
            mask_rep = mask.unsqueeze(1).expand(B, k, T).reshape(B * k, T)

            # (B, k, d_g) -> (B, k, T, d_g) -> (B*k, T, d_g)
            G_rep = G_chunk.unsqueeze(2).expand(B, k, T, d_g).reshape(B * k, T, d_g)

            Z_rep, _ = self._pool(H_rep, G_rep, mask_rep, return_alpha=False)  # [B*k, T, d_h]
            Zp = self.proj_p(Z_rep)                                            # [B*k, T, d_z]
            Gz = self.proj_g(G_rep)                                            # [B*k, T, d_z]
            if self.normalize:
                Zp = self._norm(Zp, dim=-1)
                Gz = self._norm(Gz, dim=-1)

            s_tok = (Zp * Gz).sum(dim=-1)                                      # [B*k, T]
            s_bk = s_tok.mean(dim=-1).reshape(B, k)                            # [B, k]
            scores_list.append(s_bk)

            del H_rep, mask_rep, G_rep, Z_rep, Zp, Gz, s_tok, s_bk, G_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scores = torch.cat(scores_list, dim=1)  # [B, K]
        return scores
