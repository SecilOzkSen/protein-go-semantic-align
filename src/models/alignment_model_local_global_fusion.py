"""Local SmoothMIL + global pooled GO scoring, with trainable per-(protein, GO) gate.

This is a score-level fusion, not a gate between mean and attention pooling.
"""
import torch
import torch.nn.functional as F
from src.models.alignment_model_v2_original import ProteinGoAligner as LocalAligner


class ProteinGoAligner(LocalAligner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Local parent initializes these as compatibility-only frozen modules.
        for module in (self.protein_mean_attn_gate_pool, self.proj_p, self.fusion_gate):
            for param in module.parameters():
                param.requires_grad_(True)
        # GO segment gate remains frozen/unused, as in the local-only control.
        for param in self.go_segment_gate.parameters():
            param.requires_grad_(False)

    def score_fine_grained(self, H, mask, name_raw, definition_raw,
                           definition_present, parent_raw, parent_present,
                           global_go_raw):
        # Preserve original local SmoothMIL, GO name/definition and parent penalty.
        local = super().score_fine_grained(
            H, mask, name_raw, definition_raw, definition_present,
            parent_raw, parent_present, global_go_raw,
        )
        local_diag = dict(self.last_h_diagnostics)

        # Independent global evidence using one pooled protein vector.
        zp = self.encode_protein_for_scoring(H, mask).float()  # [B,Dz]
        gn = self.project_go(name_raw).float()
        gd = self.project_go(definition_raw).float()
        B, G = zp.shape[0], gn.shape[0]
        chunks = []
        for st in range(0, G, self.go_chunk_size):
            ed = min(G, st + self.go_chunk_size)
            sn = zp @ gn[st:ed].T
            sd = zp @ gd[st:ed].T
            dp = definition_present[st:ed].bool()[None, :]
            child = torch.where(dp, .5 * (sn + sd), sn)
            pr = parent_raw[st:ed]
            pm = parent_present[st:ed].bool()
            if pm.any():
                pg = self.project_go(pr.reshape(-1, pr.shape[-1])).float()
                pg = pg.reshape(ed - st, pr.shape[1], -1)
                parent_scores = torch.einsum('bd,cpd->bcp', zp, pg)
                parent_scores = parent_scores.masked_fill(~pm[None, :, :], 0.0)
                parent_avg = parent_scores.sum(-1) / pm.sum(-1).clamp_min(1)[None, :]
                adjustment = (-self.parent_strength * pm.any(-1)[None, :]
                              * torch.relu(child - parent_avg))
            else:
                adjustment = torch.zeros_like(child)
            chunks.append(child + adjustment)
        global_scores = torch.cat(chunks, dim=1)
        if local.shape != global_scores.shape:
            raise RuntimeError(f'Fusion score shape mismatch: {local.shape} vs {global_scores.shape}')

        # GO-specific scalar evidence, gate = proportion assigned to global branch.
        features = torch.stack((local, global_scores, local - global_scores), dim=-1)
        alpha_global = torch.sigmoid(self.fusion_gate(features)).squeeze(-1)
        fused = (1.0 - alpha_global) * local + alpha_global * global_scores
        if not torch.isfinite(fused).all():
            raise RuntimeError('Nonfinite fusion scores')
        self.last_h_diagnostics = {
            **local_diag,
            'h/local_only': 0.0,
            'h/fusion_alpha_global_mean': float(alpha_global.detach().mean().item()),
            'h/fusion_local_global_abs_gap': float((local.detach() - global_scores.detach()).abs().mean().item()),
        }
        return fused
