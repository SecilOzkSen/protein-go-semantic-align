'''
1-Slots + meanGO
model = ProteinGoAligner(
    d_h=1280,
    d_g=768,
    d_z=768,
    normalize=True,
    protein_pool_type="slots",
    protein_n_slots=4,
    go_pool_type="mean")
2- Slots + token alignment
model = ProteinGoAligner(
    d_h=1280,
    d_g=768,
    d_z=768,
    normalize=True,
    protein_pool_type="slots",
    protein_n_slots=4,
    go_pool_type="token_align")

Note:
    1.	protein_pool_type="slots", go_pool_type="mean"
	2.	protein_pool_type="slots", go_pool_type="token_align"

'''

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.projection import ProjectionHead
from src.encoders import BioMedBERTEncoder
from src.models.go_token_align_pooler import GoTokenAlignPooler


class AttentionPool1D(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Linear(d_in, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                     # [B, T, D]
        mask: Optional[torch.Tensor] = None # [B, T] bool, True=valid
    ):
        s = self.score(self.dropout(x)).squeeze(-1)  # [B, T]

        if mask is not None:
            neg_inf = torch.finfo(s.dtype).min
            s = s.masked_fill(~mask, neg_inf)

        a = torch.softmax(s, dim=1)                  # [B, T]
        pooled = torch.sum(x * a.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, a


class ProteinSlotExtractor(nn.Module):
    """
    Residue embeddings -> multiple protein slots

    Input:
        H:    [B, T, D]
        mask: [B, T] bool, True=valid

    Output:
        slots: [B, S, D]
        attn:  [B, S, T]
    """
    def __init__(
        self,
        d_in: int,
        n_slots: int = 4,
        d_attn: Optional[int] = None,
        dropout: float = 0.1,
        use_residual: bool = False,
        use_output_ln: bool = True,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.n_slots = int(n_slots)
        self.d_attn = int(d_attn or d_in)

        self.slot_queries = nn.Parameter(torch.randn(self.n_slots, self.d_attn) * 0.02)

        self.q_proj = nn.Linear(self.d_attn, self.d_attn)
        self.k_proj = nn.Linear(self.d_in, self.d_attn)
        self.v_proj = nn.Linear(self.d_in, self.d_in)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.out_ln = nn.LayerNorm(self.d_in) if use_output_ln else nn.Identity()
        self.use_residual = bool(use_residual)

        if self.use_residual:
            self.residual_proj = nn.Linear(self.d_attn, self.d_in)
        else:
            self.residual_proj = None

        self.scale = math.sqrt(float(self.d_attn))

    def forward(
        self,
        H: torch.Tensor,                     # [B, T, D]
        mask: Optional[torch.Tensor] = None # [B, T] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = H.shape

        if mask is not None and mask.dtype != torch.bool:
            mask = mask != 0

        q = self.q_proj(self.slot_queries).unsqueeze(0).expand(B, self.n_slots, self.d_attn)  # [B,S,Da]
        k = self.k_proj(H)  # [B,T,Da]
        v = self.v_proj(H)  # [B,T,D]

        scores = torch.einsum("bsd,btd->bst", q, k) / self.scale  # [B,S,T]

        if mask is not None:
            mask_expanded = ~mask.unsqueeze(1)  # [B,1,T]
            scores = scores.masked_fill(mask_expanded, -1e4)

        attn = torch.softmax(scores, dim=-1)  # [B,S,T]
        attn = self.dropout(attn)

        slots = torch.einsum("bst,btd->bsd", attn, v)  # [B,S,D]

        if self.use_residual:
            slots = slots + self.residual_proj(q)

        slots = self.out_ln(slots)
        return slots, attn


class ProteinGoAligner(nn.Module):
    def __init__(
        self,
        d_h: int,
        d_g: Optional[int] = None,
        d_z: int = 768,
        go_encoder: Optional[BioMedBERTEncoder] = None,
        normalize: bool = True,
        protein_pool_type: str = "mean",   # "mean" | "attn" | "go_align" | "slots"
        protein_n_slots: int = 4,
        go_pool_type: str = "mean",        # used when protein_pool_type == "slots": "mean" | "token_align"
    ):
        super().__init__()

        if protein_pool_type not in {"mean", "attn", "go_align", "slots"}:
            raise ValueError(f"Unsupported protein_pool_type: {protein_pool_type}")

        if go_pool_type not in {"mean", "token_align"}:
            raise ValueError(f"Unsupported go_pool_type: {go_pool_type}")

        self.normalize = bool(normalize)
        self.go_encoder = go_encoder
        self.protein_pool_type = protein_pool_type
        self.protein_n_slots = int(protein_n_slots)
        self.go_pool_type = go_pool_type

        if self.go_encoder is not None and d_g is None:
            d_g = int(self.go_encoder.model.config.hidden_size)
        if d_g is None:
            raise ValueError("d_g must be provided if go_encoder is None.")

        # Existing go-align pooler, only for protein_pool_type == "go_align"
        self.go_token_align_pooler = None
        if self.protein_pool_type == "go_align":
            self.go_token_align_pooler = GoTokenAlignPooler(
                d_h=d_h,
                d_g=d_g,
                d_att=256,
                dropout=0.05,
            )

        self.protein_attn_pool = AttentionPool1D(d_h, dropout=0.05)

        self.slot_extractor = None
        if self.protein_pool_type == "slots":
            self.slot_extractor = ProteinSlotExtractor(
                d_in=d_h,
                n_slots=self.protein_n_slots,
                d_attn=d_h,
                dropout=0.05,
                use_residual=False,
                use_output_ln=True,
            )

        self.proj_p = ProjectionHead(d_in=d_h, d_out=d_z)
        self.proj_g = ProjectionHead(d_in=d_g, d_out=d_z)

        self.protein_ln = nn.LayerNorm(d_h)
        self.go_ln = nn.LayerNorm(d_g)
        #TODO: Erase
        self.step = 0

    @staticmethod
    def _norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        n = torch.linalg.vector_norm(x.to(torch.float32), dim=dim, keepdim=True).clamp_min(eps)
        return x / n.to(x.dtype)

    @staticmethod
    def _masked_mean(
        x: torch.Tensor,                     # [N, L, D]
        mask: Optional[torch.Tensor] = None # [N, L] bool
    ) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)

        if mask.dtype != torch.bool:
            mask = mask != 0

        w = mask.to(x.dtype).unsqueeze(-1)   # [N,L,1]
        denom = w.sum(dim=1).clamp_min(1.0)  # [N,1]
        return (x * w).sum(dim=1) / denom    # [N,D]

    def forward(
        self,
        H: torch.Tensor,                       # [B, T, Dh]
        G: torch.Tensor,                       # [B, K, Dg] OR [B, K, L, Dg]
        mask: Optional[torch.Tensor],          # [B, T], bool, True=valid
        go_mask: Optional[torch.Tensor] = None,# [B, K, L] for token GO
        return_alpha: bool = False,
        **kwargs
    ):
        if mask is not None and mask.dtype != torch.bool:
            mask = mask != 0

        alpha_info = {}

        if G.dim() not in {3, 4}:
            raise ValueError("G must be [B, K, Dg] or [B, K, L, Dg].")

        B, T, Dh = H.shape
        if G.dim() == 3:
            _, K, Dg = G.shape
            L = None
        else:
            _, K, L, Dg = G.shape

        # --------------------------------------------------
        # mean pooling baseline
        # --------------------------------------------------
        if self.protein_pool_type == "mean":
            if mask is not None:
                w = mask.to(H.dtype).unsqueeze(-1)      # [B,T,1]
                denom = w.sum(dim=1).clamp_min(1.0)     # [B,1]
                h_pool = (H * w).sum(dim=1) / denom     # [B,Dh]
            else:
                h_pool = H.mean(dim=1)                  # [B,Dh]

            h_pool = self.protein_ln(h_pool)            # [B,Dh]
            Z = h_pool.unsqueeze(1).expand(B, K, Dh)    # [B,K,Dh]

            if G.dim() == 4:
                if go_mask is None:
                    raise ValueError("go_mask must be provided when G is [B,K,L,Dg].")
                G_flat = G.view(B * K, L, Dg)
                go_mask_flat = go_mask.view(B * K, L)
                G = self._masked_mean(G_flat, go_mask_flat).view(B, K, Dg)

            G = self.go_ln(G)
            Zp = self.proj_p(Z)
            Gz = self.proj_g(G)

            if self.normalize:
                Zp = self._norm(Zp, dim=-1)
                Gz = self._norm(Gz, dim=-1)

            scores = (Zp * Gz).sum(dim=-1)              # [B,K]

            if return_alpha:
                return scores, alpha_info
            return scores

        # --------------------------------------------------
        # attention pooling baseline
        # --------------------------------------------------
        elif self.protein_pool_type == "attn":
            h_pool, attn = self.protein_attn_pool(H, mask)  # [B,Dh], [B,T]
            h_pool = self.protein_ln(h_pool)
            Z = h_pool.unsqueeze(1).expand(B, K, Dh)

            if return_alpha:
                alpha_info["protein_attn"] = attn

            if G.dim() == 4:
                if go_mask is None:
                    raise ValueError("go_mask must be provided when G is [B,K,L,Dg].")
                G_flat = G.view(B * K, L, Dg)
                go_mask_flat = go_mask.view(B * K, L)
                G = self._masked_mean(G_flat, go_mask_flat).view(B, K, Dg)

            G = self.go_ln(G)
            Zp = self.proj_p(Z)
            Gz = self.proj_g(G)

            if self.normalize:
                Zp = self._norm(Zp, dim=-1)
                Gz = self._norm(Gz, dim=-1)

            scores = (Zp * Gz).sum(dim=-1)              # [B,K]

            if return_alpha:
                return scores, alpha_info
            return scores

        # --------------------------------------------------
        # go_align branch
        # --------------------------------------------------
        elif self.protein_pool_type == "go_align":
            if G.dim() != 3:
                raise ValueError("For protein_pool_type='go_align', G must be [B, K, Dg].")

            Z, gtap_alpha = self.go_token_align_pooler(
                H, G, mask, return_alpha=return_alpha
            )  # [B,K,Dh] depending on your external implementation

            Z = self.protein_ln(Z)

            if return_alpha:
                alpha_info.update(gtap_alpha)

            G = self.go_ln(G)

            Zp = self.proj_p(Z)
            Gz = self.proj_g(G)

            if self.normalize:
                Zp = self._norm(Zp, dim=-1)
                Gz = self._norm(Gz, dim=-1)

            scores = (Zp * Gz).sum(dim=-1)

            if return_alpha:
                return scores, alpha_info
            return scores

        # --------------------------------------------------
        # slots branch
        # --------------------------------------------------
        else:  # "slots"
            slots, slot_attn = self.slot_extractor(H, mask)  # [B,S,Dh], [B,S,T]
            #TODO: Erase later
            with torch.no_grad():
                import torch.nn.functional as F
                a = F.normalize(slot_attn.detach().float(), dim=-1)  # [B,S,T]
                attn_sim = torch.matmul(a, a.transpose(1, 2))  # [B,S,S]

                eye = torch.eye(attn_sim.size(1), dtype=torch.bool, device=attn_sim.device)
                offdiag = attn_sim[:, ~eye]

                print("\n[DBG-SLOT-ATTN]")
                print("slot_attn:", tuple(slot_attn.shape), slot_attn.dtype)
                print("attn cosine b0:")
                print(attn_sim[0].detach().cpu())
                print(
                    "attn offdiag mean/max/min:",
                    offdiag.mean().item(),
                    offdiag.max().item(),
                    offdiag.min().item(),
                )

                entropy = -(slot_attn.detach().float() * (slot_attn.detach().float() + 1e-8).log()).sum(dim=-1)
                print("attn entropy mean:", entropy.mean().item())
                print("attn entropy b0:", entropy[0].detach().cpu())
            slots = self.protein_ln(slots)
            import torch.nn.functional as F
            with torch.no_grad():
                s = F.normalize(slots.detach().float(), dim=-1)
                slot_sim = torch.matmul(s, s.transpose(1, 2))  # [B,S,S]

                eye = torch.eye(slot_sim.size(1), dtype=torch.bool, device=slot_sim.device)
                offdiag = slot_sim[:, ~eye]
                #TODO: erase
                if self.step % 100 == 0:
                    print("\n[DBG-SLOT]")
                    print("slots:", tuple(slots.shape), slots.dtype)
                    print("slot cosine b0:")
                    print(slot_sim[0].detach().cpu())
                    print(
                        "offdiag mean/max/min:",
                        offdiag.mean().item(),
                        offdiag.max().item(),
                        offdiag.min().item(),
                    )
                self.step += 1

            if return_alpha:
                alpha_info["protein_slot_attn"] = slot_attn

            # ----------------------------------------------
            # slots + mean GO
            # ----------------------------------------------
            if self.go_pool_type == "mean":
                if G.dim() == 4:
                    if go_mask is None:
                        raise ValueError("go_mask must be provided when G is [B,K,L,Dg].")
                    G_flat = G.view(B * K, L, Dg)
                    go_mask_flat = go_mask.view(B * K, L)
                    G = self._masked_mean(G_flat, go_mask_flat).view(B, K, Dg)

                G = self.go_ln(G)                        # [B,K,Dg]

                Zp = self.proj_p(slots)                 # [B,S,Dz]
                Gz = self.proj_g(G)                     # [B,K,Dz]

                if self.normalize:
                    Zp = self._norm(Zp, dim=-1)
                    Gz = self._norm(Gz, dim=-1)

                # [B,S,K]
                sim = torch.einsum("bsd,bkd->bsk", Zp, Gz)

                # [B,K]
                scores = sim.max(dim=1).values

                if return_alpha:
                    alpha_info["slot_go_sim"] = sim
                    return scores, alpha_info
                return scores

            # ----------------------------------------------
            # slots + token-level GO alignment
            # ----------------------------------------------
            else:  # go_pool_type == "token_align"
                if G.dim() != 4:
                    raise ValueError(
                        "For go_pool_type='token_align', G must be [B, K, L, Dg]."
                    )
                if go_mask is None:
                    raise ValueError("go_mask must be provided for token_align mode.")

                G = self.go_ln(G)                       # [B,K,L,Dg]

                Zp = self.proj_p(slots)                # [B,S,Dz]
                Gz = self.proj_g(G)                    # [B,K,L,Dz]

                if self.normalize:
                    Zp = self._norm(Zp, dim=-1)
                    Gz = self._norm(Gz, dim=-1)

                # [B,S,K,L]
                sim = torch.einsum("bsd,bkld->bskl", Zp, Gz)

                # TODO: Erase later
                with torch.no_grad():
                    sim_f = sim.detach().float()

                    print("\n[DBG-SIM]")
                    print("sim:", tuple(sim.shape))
                    print(
                        "sim min/max/mean/std:",
                        sim_f.min().item(),
                        sim_f.max().item(),
                        sim_f.mean().item(),
                        sim_f.std().item(),
                    )

                    max_over_slots = sim_f.max(dim=1).values  # [B,K]
                    mean_over_slots = sim_f.mean(dim=1)

                    print(
                        "max_over_slots mean/std:",
                        max_over_slots.mean().item(),
                        max_over_slots.std().item(),
                    )

                    print(
                        "mean_over_slots mean/std:",
                        mean_over_slots.mean().item(),
                        mean_over_slots.std().item(),
                    )

                if go_mask.dtype != torch.bool:
                    go_mask = go_mask != 0

                fill_value = -1e4 if sim.dtype == torch.float16 else -1e9
                sim = sim.masked_fill(~go_mask.unsqueeze(1), fill_value)

                # best token per slot -> [B,S,K]
                best_token = sim.max(dim=-1).values

                # best slot per GO -> [B,K]
                scores = best_token.max(dim=1).values

                if return_alpha:
                    alpha_info["slot_token_sim"] = sim
                    alpha_info["best_token_per_slot"] = best_token
                    return scores, alpha_info
                return scores