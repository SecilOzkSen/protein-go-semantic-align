"""Global protein pooling ablation, separate from local-only V2 model.

Select with GLOBAL_POOL_ABLATION=mean|attention|mean_attn_gate.
Keep GO name/definition/parent scoring identical to local-only scorer;
replace local SmoothMIL aggregation with a single projected global protein vector.
"""
import os
import torch
import torch.nn.functional as F
from src.models.alignment_model_v2_original import ProteinGoAligner as LocalAligner
from src.models.alignment_model_v2_original import GatedMeanAttnPool1D


class ProteinGoAligner(LocalAligner):
    def __init__(self, *args, **kwargs):
        variant = os.environ.get('GLOBAL_POOL_ABLATION', '').strip().lower()
        if variant not in {'mean', 'attention', 'mean_attn_gate'}:
            raise ValueError('Set GLOBAL_POOL_ABLATION=mean|attention|mean_attn_gate')
        kwargs['protein_pool_type'] = 'mean_attn_gate'  # compatibility with parent constructor
        super().__init__(*args, **kwargs)
        self.global_pool_variant = variant
        for p in self.proj_local.parameters():
            p.requires_grad_(False)
        for p in self.proj_p.parameters():
            p.requires_grad_(True)
        for p in self.protein_mean_attn_gate_pool.parameters():
            p.requires_grad_(variant != 'mean')
        if variant == 'attention':
            for p in self.protein_mean_attn_gate_pool.gate.parameters():
                p.requires_grad_(False)
        # GO segment gate is unused in score_fine_grained, as in local baseline.
        for p in self.go_segment_gate.parameters():
            p.requires_grad_(False)
        self.protein_pool_type = variant

    def encode_protein_for_scoring(self, H, mask=None):
        if H.ndim != 3:
            raise ValueError(f'Expected [B,T,D], got {tuple(H.shape)}')
        if mask is not None:
            mask = mask.bool()
        pool = self.protein_mean_attn_gate_pool
        if self.global_pool_variant == 'mean':
            pooled = GatedMeanAttnPool1D.masked_mean(H, mask)
        elif self.global_pool_variant == 'attention':
            pooled, _ = pool.attn_pool(H, mask)
        else:
            pooled, _ = pool(H, mask)
        z = self.proj_p(pooled)
        return self._norm(z, dim=-1) if self.normalize else z

    def score_fine_grained(self, H, mask, name_raw, definition_raw,
                           definition_present, parent_raw, parent_present,
                           global_go_raw):
        zp = self.encode_protein_for_scoring(H, mask).float()  # [B,D]
        gn = self.project_go(name_raw).float()
        gd = self.project_go(definition_raw).float()
        B, G = zp.shape[0], gn.shape[0]
        chunks, penalties = [], []
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
                pg = pg.reshape(ed-st, pr.shape[1], -1)
                parent_scores = torch.einsum('bd,cpd->bcp', zp, pg)
                parent_scores = parent_scores.masked_fill(~pm[None, :, :], 0)
                parent_avg = parent_scores.sum(-1) / pm.sum(-1).clamp_min(1)[None, :]
                adjustment = -self.parent_strength * pm.any(-1)[None, :] * torch.relu(child - parent_avg)
            else:
                adjustment = torch.zeros_like(child)
            chunks.append(child + adjustment)
            penalties.append(adjustment)
        self.last_h_diagnostics = {
            'h/local_only': 0.0,
            'h/parent_adjustment_mean_abs': float(torch.cat(penalties, 1).detach().abs().mean().item()),
            'h/window_count_mean': 1.0,
        }
        return torch.cat(chunks, dim=1)
