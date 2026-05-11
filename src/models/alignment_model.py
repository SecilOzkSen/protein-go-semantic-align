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

ProteinSlotExtractor
        ↓
encode_protein_for_scoring
        ↓
score_from_encoded_protein
        ↓
scores

'''

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.projection import ProjectionHead
from src.encoders import BioMedBERTEncoder
from src.models.go_token_align_pooler import GoTokenAlignPooler
import torch.nn.functional as F


class AttentionPool1D(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.score = nn.Linear(d_in, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # [B,T,D]
        mask: Optional[torch.Tensor] = None    # [B,T] bool, True=valid
    ):
        if mask is not None and mask.dtype != torch.bool:
            mask = mask != 0
        if mask is not None:
            valid_counts = mask.sum(dim=1)
            if (valid_counts == 0).any():
                raise RuntimeError("AttentionPool1D received a row with zero valid residues.")

        x_score = self.ln(x)
        s = self.score(self.dropout(x_score)).squeeze(-1)  # [B,T]

        if mask is not None:
            s = s.masked_fill(~mask, -1e4)

        a = torch.softmax(s.float(), dim=1).to(dtype=x.dtype)  # [B,T]
        pooled = torch.sum(x * a.unsqueeze(-1), dim=1)         # [B,D]

        return pooled, a

class GatedMeanAttnPool1D(nn.Module):
    def __init__(self, d_in: int, dropout: float = 0.05):
        super().__init__()

        self.attn_pool = AttentionPool1D(d_in=d_in, dropout=dropout)

        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_in),
            nn.Linear(2 * d_in, d_in // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 4, 1),
        )

        # Start close to mean pooling.
        # sigmoid(-2) ≈ 0.12, so attention initially contributes weakly.
        nn.init.constant_(self.gate[-1].bias, -2.0)

    @staticmethod
    def masked_mean(
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)

        if mask.dtype != torch.bool:
            mask = mask != 0

        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(eps)

    def forward(
        self,
        x: torch.Tensor,                       # [B,T,D]
        mask: Optional[torch.Tensor] = None,
    ):
        h_mean = self.masked_mean(x, mask)     # [B,D]
        h_attn, alpha = self.attn_pool(x, mask)

        gate_in = torch.cat([h_mean, h_attn], dim=-1)  # [B,2D]
        gate = torch.sigmoid(self.gate(gate_in))       # [B,1]

        h = (1.0 - gate) * h_mean + gate * h_attn      # [B,D]

        info = {
            "protein_attn_alpha": alpha,
            "protein_attn_gate": gate,
        }

        return h, info


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
        self.last_slot_div_loss = None

        if self.use_residual:
            self.residual_proj = nn.Linear(self.d_attn, self.d_in)
        else:
            self.residual_proj = None

        self.scale = math.sqrt(float(self.d_attn))

    def forward(
            self,
            H: torch.Tensor,  # [B, T, D]
            mask: Optional[torch.Tensor] = None  # [B, T] bool
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

        # dropout sonrası attention toplamı bozulmasın
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        a = F.normalize(attn.float(), dim=-1)
        attn_sim = torch.matmul(a, a.transpose(1, 2))  # [B,S,S]

        S = attn_sim.size(1)
        eye = torch.eye(S, dtype=torch.bool, device=attn_sim.device)
        offdiag = attn_sim[:, ~eye]

        self.last_slot_div_loss = offdiag.pow(2).mean()

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
            protein_pool_type: str = "mean",  # "mean" | "attn" | "go_align" | "slots"
            protein_n_slots: int = 4,
            go_pool_type: str = "mean",  # used when protein_pool_type == "slots": "mean" | "token_align"
    ):
        super().__init__()

        if protein_pool_type not in {"mean", "attn", "mean_attn_gate", "slots"}:
            raise ValueError(f"Unsupported protein_pool_type: {protein_pool_type}")
        if protein_pool_type == "go_align":
            raise ValueError(
                "protein_pool_type='go_align' is deprecated in the current retriever. "
                "Use 'mean_attn_gate', 'mean', 'attn', or 'slots'."
            )
        if go_pool_type == "token_align" and protein_pool_type != "slots":
            raise ValueError(
                "go_pool_type='token_align' requires protein_pool_type='slots'. "
                f"Got protein_pool_type={protein_pool_type}."
            )

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

        self.protein_attn_pool = AttentionPool1D(d_h, dropout=0.05)
        self.protein_mean_attn_gate_pool = GatedMeanAttnPool1D(
            d_in=d_h,
            dropout=0.05,
        )

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

        self.go_segment_names = ["name", "namespace", "definition", "is_a", "part_of"]
        self.n_go_segments = len(self.go_segment_names)

        self.go_segment_gate = nn.Sequential(
            nn.LayerNorm(d_g),
            nn.Linear(d_g, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 1),
        )

        # Start uniform over present segments.
        # This avoids the gate randomly preferring one segment at step 0.
        nn.init.zeros_(self.go_segment_gate[-1].weight)
        nn.init.zeros_(self.go_segment_gate[-1].bias)

        self.last_go_segment_info = {}
        self.last_pool_info = {}

    @staticmethod
    def _norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        n = torch.linalg.vector_norm(x.to(torch.float32), dim=dim, keepdim=True).clamp_min(eps)
        return x / n.to(x.dtype)

    @staticmethod
    def _masked_mean(
            x: torch.Tensor,  # [N, L, D]
            mask: Optional[torch.Tensor] = None  # [N, L] bool
    ) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)

        if mask.dtype != torch.bool:
            mask = mask != 0

        w = mask.to(x.dtype).unsqueeze(-1)  # [N,L,1]
        denom = w.sum(dim=1).clamp_min(1.0)  # [N,1]
        return (x * w).sum(dim=1) / denom  # [N,D]

    def encode_protein_for_scoring(self, H, mask=None):
        """
        Returns projected protein representation used for scoring.
        For slots: [B, S, Dz]
        For pooled: [B, Dz]
        """
        if mask is not None and mask.dtype != torch.bool:
            mask = mask != 0

        if self.protein_pool_type == "slots":
            slots, slot_attn = self.slot_extractor(H, mask)  # [B,S,Dh], [B,S,T]

            # slot diversity loss already stored inside slot_extractor if you added it there
            Zp = self.protein_ln(slots)
            Zp = self.proj_p(Zp)

            if self.normalize:
                Zp = self._norm(Zp, dim=-1)

            return Zp

        elif self.protein_pool_type == "attn":
            h_pool, _ = self.protein_attn_pool(H, mask)
            Zp = self.protein_ln(h_pool)
            Zp = self.proj_p(Zp)

            if self.normalize:
                Zp = self._norm(Zp, dim=-1)

            return Zp

        elif self.protein_pool_type == "mean_attn_gate":
            h_pool, pool_info = self.protein_mean_attn_gate_pool(H, mask)
            self.last_pool_info = pool_info
            Zp = self.protein_ln(h_pool)
            Zp = self.proj_p(Zp)

            if self.normalize:
                Zp = self._norm(Zp, dim=-1)

            return Zp

        elif self.protein_pool_type == "mean":
            if mask is not None:
                w = mask.to(H.dtype).unsqueeze(-1)
                h_pool = (H * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
            else:
                h_pool = H.mean(dim=1)

            Zp = self.protein_ln(h_pool)
            Zp = self.proj_p(Zp)

            if self.normalize:
                Zp = self._norm(Zp, dim=-1)

            return Zp

        else:
            raise ValueError(f"Unsupported protein_pool_type: {self.protein_pool_type}")

    def score_from_encoded_protein(self, Zp, G, go_mask=None):
        """
        Zp:
          slots mode: [B,S,Dz]
          pooled mode: [B,Dz]

        G:
          pooled candidates: [B,K,Dg]
          token candidates:  [B,K,L,Dg]
        """

        # token-align case: G = [B,K,L,Dg]
        if G.dim() == 4:
            if Zp.dim() != 3:
                raise RuntimeError("Token-align scoring requires slot protein representation [B,S,Dz].")

            Gz = self.go_ln(G)
            Gz = self.proj_g(Gz)

            if self.normalize:
                Gz = self._norm(Gz, dim=-1)

            sim = torch.einsum("bsd,bkld->bskl", Zp, Gz)

            # sim: [B,S,K,L]
            if go_mask is not None:
                if go_mask.dtype != torch.bool:
                    go_mask = go_mask != 0
                fill_value = -1e4 if sim.dtype == torch.float16 else -1e9
                sim = sim.masked_fill(~go_mask.unsqueeze(1), fill_value)

            # token-level evidence per slot
            slot_scores = sim.max(dim=-1).values  # [B,S,K]

            # candidate score = best slot evidence
            scores = slot_scores.max(dim=1).values  # [B,K]

            return scores

        # pooled GO case: G = [B,K,Dg]
        elif G.dim() == 3:
            Gz = self.go_ln(G)
            Gz = self.proj_g(Gz)

            if self.normalize:
                Gz = self._norm(Gz, dim=-1)

            if Zp.dim() == 3:
                # slots: [B,S,D] x [B,K,D] -> [B,S,K] -> [B,K]
                sim = torch.einsum("bsd,bkd->bsk", Zp, Gz)
                scores = sim.max(dim=1).values
            else:
                # pooled: [B,D] x [B,K,D] -> [B,K]
                scores = torch.einsum("bd,bkd->bk", Zp, Gz)

            return scores

        else:
            raise RuntimeError(f"Unsupported G shape: {tuple(G.shape)}")

    def forward(
            self,
            H: torch.Tensor,  # [B, T, Dh]
            G: torch.Tensor,  # [B, K, Dg] OR [B, K, L, Dg]
            mask: Optional[torch.Tensor],
            go_mask: Optional[torch.Tensor] = None,
            return_alpha: bool = False,
            **kwargs
    ):
        if return_alpha:
            raise RuntimeError(
                "return_alpha=True is disabled in the unified scoring path. "
                "Use explicit debug hooks instead."
            )

        if mask is not None and mask.dtype != torch.bool:
            mask = mask != 0

        if G.dim() not in {3, 4}:
            raise ValueError("G must be [B,K,Dg] or [B,K,L,Dg].")

        if G.dim() == 4 and go_mask is None:
            raise ValueError("go_mask must be provided when G is [B,K,L,Dg].")

        Zp = self.encode_protein_for_scoring(H, mask)
        scores = self.score_from_encoded_protein(Zp, G, go_mask=go_mask)

        return scores

    def encode_go_segment_aware(
            self,
            seg_input_ids: torch.Tensor,  # [G, S, L]
            seg_attention_mask: torch.Tensor,  # [G, S, L]
            seg_present: torch.Tensor,  # [G, S] bool
    ):
        """
        Segment-aware GO encoder for B1.

        Input:
          seg_input_ids:      [G, S, L]
          seg_attention_mask: [G, S, L]
          seg_present:        [G, S]

        Output dict:
          pooled:          [G, Dg] raw GO encoder space
          segment_embs:    [G, S, Dg]
          segment_weights: [G, S]
          segment_present: [G, S]

        Important:
          This returns RAW GO embeddings.
          Do NOT apply go_ln/proj_g/normalize here.
          score_from_encoded_protein() will handle projection.
        """
        if self.go_encoder is None:
            raise RuntimeError("encode_go_segment_aware requires self.go_encoder.")

        if seg_input_ids.dim() != 3:
            raise RuntimeError(f"seg_input_ids must be [G,S,L], got {tuple(seg_input_ids.shape)}")
        if seg_attention_mask.dim() != 3:
            raise RuntimeError(f"seg_attention_mask must be [G,S,L], got {tuple(seg_attention_mask.shape)}")
        if seg_present.dim() != 2:
            raise RuntimeError(f"seg_present must be [G,S], got {tuple(seg_present.shape)}")

        G, S, L = seg_input_ids.shape

        if S != self.n_go_segments:
            raise RuntimeError(
                f"Expected {self.n_go_segments} GO segments {self.go_segment_names}, got S={S}"
            )

        if seg_attention_mask.shape != seg_input_ids.shape:
            raise RuntimeError(
                f"seg_attention_mask shape {tuple(seg_attention_mask.shape)} "
                f"does not match seg_input_ids {tuple(seg_input_ids.shape)}"
            )

        if seg_present.shape != (G, S):
            raise RuntimeError(
                f"seg_present shape {tuple(seg_present.shape)} expected {(G, S)}"
            )

        if seg_present.dtype != torch.bool:
            seg_present = seg_present != 0

        seg_present = seg_present.to(device=seg_input_ids.device)

        if (seg_present.sum(dim=1) == 0).any():
            bad = torch.nonzero(seg_present.sum(dim=1) == 0, as_tuple=False).flatten()[:5]
            raise RuntimeError(f"Some GO rows have zero present segments. Example rows: {bad.tolist()}")

        flat_ids = seg_input_ids.reshape(G * S, L)
        flat_mask = seg_attention_mask.reshape(G * S, L)

        out = self.go_encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            output_mode="pooled",
        )

        if isinstance(out, tuple):
            out = out[0]
        elif isinstance(out, dict):
            if "pooled" in out:
                out = out["pooled"]
            elif "pooler_output" in out:
                out = out["pooler_output"]
            else:
                raise RuntimeError("go_encoder dict output missing 'pooled' or 'pooler_output'.")

        if not torch.is_tensor(out):
            raise RuntimeError(f"Unsupported go_encoder output type: {type(out)}")

        if out.dim() != 2:
            raise RuntimeError(f"Expected flat segment embeddings [G*S,D], got {tuple(out.shape)}")

        Dg = out.size(-1)

        segment_embs = torch.nan_to_num(out).view(G, S, Dg).contiguous()  # [G,S,Dg]

        # Segment gate in raw GO encoder space.
        seg_logits = self.go_segment_gate(segment_embs).squeeze(-1)  # [G,S]
        seg_logits = seg_logits.float().masked_fill(~seg_present, -1e9)

        segment_weights = torch.softmax(seg_logits, dim=-1).to(dtype=segment_embs.dtype)  # [G,S]

        pooled = torch.einsum("gs,gsd->gd", segment_weights, segment_embs)  # [G,Dg]
        pooled = torch.nan_to_num(pooled).contiguous()

        self.last_go_segment_info = {
            "segment_weights": segment_weights.detach(),
            "segment_present": seg_present.detach(),
            "segment_names": self.go_segment_names,
        }

        return {
            "pooled": pooled,
            "segment_embs": segment_embs,
            "segment_weights": segment_weights,
            "segment_present": seg_present,
        }
