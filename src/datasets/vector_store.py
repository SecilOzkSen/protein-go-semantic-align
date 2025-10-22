from __future__ import annotations
from typing import Optional, List
import torch
import torch.nn.functional as F
from src.miners.queue_miner import MoCoQueue

Tensor = torch.Tensor

class VectorResources:
    """
    Unified vector backend:
      - Optional projector for queries (protein->GO space).
      - Two negative/search sources:
          (a) MoCoQueue (if attached and non-empty)  -> fast, freshest negatives
          (b) go_embs dense bank on CPU              -> fallback / coarse retrieval
    Backward compatible with your original API.
    """

    def __init__(self,
                 faiss_index: Optional[object],
                 go_embs: torch.Tensor,
                 align_dim: int = 768,
                 query_projector: Optional[torch.nn.Module] = None,
                 device = "cuda"):
        # FAISS is deprecated
        self.faiss_index = None
        self.align_dim = int(align_dim)
        self.query_projector = query_projector
        # keep GO bank on CPU, L2-normalized
        self.go_embs = F.normalize(go_embs.float().cpu(), p=2, dim=1) if go_embs.numel() > 0 else go_embs
        self.device = device
        self.queue = MoCoQueue(dim=self.align_dim, device=self.device, K=65536)
        # optional MoCo queue


    # ---------- attachments ----------
    def set_query_projector(self, projector: torch.nn.Module) -> None:
        self.query_projector = projector

    def set_align_dim(self, dim: int) -> None:
        self.align_dim = int(dim)
        self.queue.on_change_dim(self.align_dim)

    def attach_queue(self, queue: torch.nn.Module) -> None:
        self.queue = queue

    def detach_queue(self) -> None:
        self.queue = None

    def set_backends(self, faiss_index, go_embs: torch.Tensor):
        # keep signature; ignore faiss
        self.faiss_index = None
        if go_embs is not None and go_embs.numel() > 0:
            self.go_embs = F.normalize(go_embs.float().cpu(), p=2, dim=1)

    # ---------- query projection ----------
    @torch.no_grad()
    def project_queries_to_index(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Q: [B, d_q] or [d_q]; projector varsa uygular, yoksa align_dim'e kırpar.
        Dönen tensör L2-normalize edilir.
        """
        if Q.dim() == 1:
            Q = Q.unsqueeze(0)

        d_q = Q.size(-1)
        if self.query_projector is not None:
            proj_dev = next(self.query_projector.parameters()).device
            if Q.device != proj_dev:
                Q = Q.to(proj_dev, non_blocking=True, dtype=Q.dtype)
            was_train = self.query_projector.training
            self.query_projector.eval()
            Q = self.query_projector(Q)
            if was_train:
                self.query_projector.train()
        elif d_q != self.align_dim:
            Q = Q[..., :self.align_dim]

        return F.normalize(Q, dim=1)

    # ---------- unified coarse search ----------
    @torch.no_grad()
    def _search_queue(self, queries: torch.Tensor, topM: int):
        # Use MoCoQueue if attached & ready
        if self.queue is None or not hasattr(self.queue, "topk") or self.queue.size() == 0:
            # empty result
            B = queries.shape[0]
            z = torch.empty(B, 0, device=queries.device, dtype=torch.float32)
            i = torch.empty(B, 0, device=queries.device, dtype=torch.long)
            return z, i
        return self.queue.topk(queries, topM)

    @torch.no_grad()
    def _search_go_bank(self, queries: torch.Tensor, topM: int):
        # cosine (both normalized)
        if self.go_embs is None or self.go_embs.numel() == 0:
            B = queries.shape[0]
            z = torch.empty(B, 0, device=queries.device, dtype=torch.float32)
            i = torch.empty(B, 0, device=queries.device, dtype=torch.long)
            return z, i
        G = self.go_embs.to(queries.device, non_blocking=True)  # [G, d]
        S = queries @ G.t()                                     # [B, G]
        k_eff = min(int(topM), S.size(1))
        vals, idx = torch.topk(S, k=k_eff, dim=1, largest=True, sorted=True)
        return vals.contiguous(), idx.contiguous()

    @torch.no_grad()
    def coarse_search(self, queries: torch.Tensor, topM: int):
        """
        Eski API korunur. Önce queue varsa oradan, yetmezse GO bank’tan tamamlar.
        Dönen indeksler:
          - önce queue (0..Kf-1) için negatif relative id'ler (kaynak ayrımı yapmıyoruz),
          - sonra GO bank indeksleri.
        Bu metodu doğrudan kullanıyorsan kaynak ayrımı gerekirse kendin tut.
        """
        # normalize edilmiş queries bekliyoruz; yine de güvenlik:
        queries = F.normalize(queries.float(), dim=1)

        # 1) queue
        q_vals, q_idx = self._search_queue(queries, topM)
        remain = max(0, int(topM) - q_idx.size(1))

        if remain > 0:
            g_vals, g_idx = self._search_go_bank(queries, remain)
            vals = torch.cat([q_vals, g_vals], dim=1) if q_vals.numel() else g_vals
            idx  = torch.cat([q_idx,  g_idx],  dim=1) if q_idx.numel() else g_idx
        else:
            vals, idx = q_vals, q_idx

        return vals, idx

    @torch.no_grad()
    def query(self, Q: torch.Tensor, topM: int, return_scores: bool = False):
        """
        Protein query → projekte et → L2-normalize → (queue +/or GO bank) topM.
        """
        Qp = self.project_queries_to_index(Q)       # [B, align_dim], normed
        D, I = self.coarse_search(Qp, topM)         # [B, topM]
        I = I.to(Q.device, dtype=torch.long)
        if return_scores:
            return I, D.to(Q.device)
        return I

    # ---------- protein vecs ----------
    @torch.no_grad()
    def coarse_prot_vecs(self, prot_emb_pad: torch.Tensor, prot_attn_mask: torch.Tensor) -> torch.Tensor:
        m = prot_attn_mask.float()
        v = (prot_emb_pad * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)
        return F.normalize(v, p=2, dim=1)

    @torch.no_grad()
    def true_prot_vecs(self, prot_emb_pad: torch.Tensor, prot_attn_mask: torch.Tensor, watti_or_model=None) -> torch.Tensor:
        if watti_or_model is None:
            return self.coarse_prot_vecs(prot_emb_pad, prot_attn_mask)
        try:
            weights = watti_or_model(prot_emb_pad, prot_attn_mask)  # [B, L]
            if not isinstance(weights, torch.Tensor):
                weights = torch.as_tensor(weights, dtype=prot_emb_pad.dtype, device=prot_emb_pad.device)
            weights = weights.clamp_min(0) * prot_attn_mask
            v = (prot_emb_pad * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-6)
            return F.normalize(v, p=2, dim=1)
        except Exception:
            return self.coarse_prot_vecs(prot_emb_pad, prot_attn_mask)