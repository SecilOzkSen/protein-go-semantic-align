from __future__ import annotations
import torch
from typing import List
import torch.nn.functional as F

class VectorResources:
    def __init__(self, faiss_index, go_embs: torch.Tensor, align_dim:int = 768, query_projector = None):
        self.faiss_index = faiss_index    # FAISS flat/IP or IVF/HNSW; must expose .search
        self.go_embs = go_embs.float()    # [G, d] (assumed L2-normalized upstream)
        # ensure normalized
        self.go_embs = torch.nn.functional.normalize(self.go_embs, p=2, dim=1)
        self.align_dim = align_dim
        self.query_projector = query_projector

    def set_backends(self, faiss_index, go_embs: torch.Tensor):
        self.faiss_index = faiss_index
        self.go_embs = torch.nn.functional.normalize(go_embs.float(), p=2, dim=1)


    @torch.no_grad()
    def project_queries_to_index(self, Q: torch.Tensor) -> torch.Tensor:
        # Q: [B, 1280] or [B, 768]
        if Q.dim() == 1:
            Q = Q.unsqueeze(0)
        if Q.size(1) != self.align_dim:
            if self.query_projector is None:
                raise ValueError(f"No query_projector set for dim {Q.size(1)} -> {self.align_dim}")
            Q = self.query_projector(Q)  # W_h uygula
        return torch.nn.functional.normalize(Q, p=2, dim=1)

    @torch.no_grad()
    def coarse_prot_vecs(self, prot_emb_pad: torch.Tensor, prot_attn_mask: torch.Tensor) -> torch.Tensor:
        # mean pooling with mask -> [B, d]
        mask = prot_attn_mask.float()  # [B,L]
        x = prot_emb_pad * mask.unsqueeze(-1)     # [B,L,D]
        s = x.sum(dim=1)                           # [B,D]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        v = s / denom
        v = torch.nn.functional.normalize(v, p=2, dim=1)
        return v

    @torch.no_grad()
    def true_prot_vecs(self, prot_emb_pad: torch.Tensor, prot_attn_mask: torch.Tensor, watti_or_model=None) -> torch.Tensor:
        # If a weighting provider is given, try to obtain weights; else fallback to mean
        if watti_or_model is None:
            return self.coarse_prot_vecs(prot_emb_pad, prot_attn_mask)
        # Expect a callable: weights = f(emb, mask) -> [B, L] non-negative
        try:
            weights = watti_or_model(prot_emb_pad, prot_attn_mask)  # [B, L]
            if not isinstance(weights, torch.Tensor):
                weights = torch.as_tensor(weights, dtype=prot_emb_pad.dtype, device=prot_emb_pad.device)
            weights = weights.clamp_min(0)
            # apply masked weighted pooling
            weights = weights * prot_attn_mask  # zero out pads
            wsum = (prot_emb_pad * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            v = wsum / denom
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            return v
        except Exception:
            # fallback to mean if anything goes wrong
            return self.coarse_prot_vecs(prot_emb_pad, prot_attn_mask)

    @torch.no_grad()
    def query(self, Q: torch.Tensor, topM: int, return_scores: bool = False):
        """
        Project (if needed), L2-normalize and retrieve topM GO indices.

        Args:
            Q: [B, d_q] protein queries. If d_q != align_dim, uses self.query_projector.
            topM: shortlist size to retrieve.
            return_scores: if True, also return FAISS distances/similarities (torch.float32).

        Returns:
            ids: LongTensor [B, topM] (indices into self.go_embs)
            (optional) scores: FloatTensor [B, topM]
        """
        # 1) project to index dim & normalize
        Qp = self.project_queries_to_index(Q)  # [B, align_dim], L2 normed
        # 2) FAISS search
        D, I = self.coarse_search(Qp, topM)  # tensors (genelde CPU)
        I = I.to(Q.device, dtype=torch.long)
        if return_scores:
            return I, D.to(Q.device).contiguous()
        return I

    @torch.no_grad()
    def coarse_search(self, queries: torch.Tensor, topM: int):
        # queries: [B, d] torch tensor (L2-normalized)
        q_np = queries.detach().cpu().numpy().astype('float32')
        D, I = self.faiss_index.search(q_np, topM)  # returns numpy arrays
        return torch.from_numpy(D), torch.from_numpy(I)

    @torch.no_grad()
    def gather_go_embs(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B,M] or [L]; return matching rows from go_embs
        if ids.dim() == 1:
            return self.go_embs.index_select(0, ids)
        elif ids.dim() == 2:
            out = []
            for b in range(ids.size(0)):
                out.append(self.go_embs.index_select(0, ids[b]))
            return torch.stack(out, dim=0)
        else:
            raise ValueError(f"ids shape {ids.shape} not supported")

class GoMemoryBank:
    def __init__(self, init_embs: torch.Tensor, row2id: List[int]):

        self.id2row = {int(i): int(r) for r,i in enumerate(row2id)}
        self.embs = torch.as_tensor(init_embs).cpu()
        self.embs = F.normalize(self.embs, p=2, dim=1)

    def lookup(self, ids):
        rows = [self.id2row[int(i)] for i in ids]
        return self.embs[rows]

    def update(self, ids, new_embs):  # new_embs: (len(ids), d) on CPU?
        for j,i in enumerate(ids):
            self.embs[self.id2row[int(i)]] = new_embs[j]

