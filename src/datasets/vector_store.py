from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn.functional as F

class VectorResources:
    def __init__(self,
                 faiss_index: Optional[object],
                 go_embs: torch.Tensor,
                 align_dim: int = 768,
                 query_projector: Optional[torch.nn.Module] = None):
        # FAISS artık opsiyonel ve kullanılmıyor
        self.faiss_index = None
        self.align_dim = int(align_dim)
        self.query_projector = query_projector
        self.go_embs = F.normalize(go_embs.float().cpu(), p=2, dim=1)  # bank CPU'da tutulur

    def set_backends(self, faiss_index, go_embs: torch.Tensor):
        # FAISS'i yok sayıyoruz; yalnızca embs güncellenir
        self.faiss_index = None
        self.go_embs = F.normalize(go_embs.float().cpu(), p=2, dim=1)

    @torch.no_grad()
    def project_queries_to_index(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Q: [B, d_q] veya [d_q]. Eğer d_q != align_dim ve projector varsa projekte edilir.
        Sonuç L2-normalize edilir.
        """
        if Q.dim() == 1:
            Q = Q.unsqueeze(0)

        d_q = Q.size(-1)
        if self.query_projector is not None:
            # Projector'ın cihazına taşıyıp ileri geçir
            proj_dev = next(self.query_projector.parameters()).device
            if Q.device != proj_dev:
                Q = Q.to(proj_dev, non_blocking=True, dtype=getattr(getattr(self.query_projector, 'weight', None), 'dtype', Q.dtype))
            was_training = getattr(self.query_projector, "training", False)
            try: self.query_projector.eval()
            except Exception: pass
            Q = self.query_projector(Q)
            try:
                if was_training: self.query_projector.train()
            except Exception:
                pass
        else:
            # Projector yok: boyut zaten uyumluysa bırak; değilse ilk align_dim'i al (uyarı yok, hızlı fallback)
            if d_q != self.align_dim:
                Q = Q[..., :self.align_dim]

        return F.normalize(Q, dim=1)

    @torch.no_grad()
    def coarse_search(self, queries: torch.Tensor, topM: int):
        """
        FAISS'siz arama: full-matrix cosine/IP benzerlik + topk.
        queries normalize edilmiş olmalı; go_embs zaten normalize.
        D: [B, topM] skorlar, I: [B, topM] indeksler (go_embs satır indeksleri).
        """
        # Hesaplamayı query'nin cihazında yap
        device = queries.device
        G = self.go_embs.to(device, non_blocking=True)  # büyük olabilir; gerekirse chunk'lanabilir
        # IP = cosine (normalize edilmiş vektörlerde)
        S = queries @ G.t()                    # [B, G]
        vals, idx = torch.topk(S, k=min(topM, S.size(1)), dim=1)
        return vals.contiguous(), idx.contiguous()

    @torch.no_grad()
    def query(self, Q: torch.Tensor, topM: int, return_scores: bool = False):
        """
        Protein sorgularını indeks boyutuna projekte eder, normalize eder ve
        PyTorch topk ile en benzer GO vektörlerini döndürür.
        """
        Qp = self.project_queries_to_index(Q)       # [B, align_dim], normed
        D, I = self.coarse_search(Qp, topM)         # [B, topM]
        I = I.to(Q.device, dtype=torch.long)
        if return_scores:
            return I, D.to(Q.device)
        return I

    @torch.no_grad()
    def coarse_prot_vecs(self, prot_emb_pad: torch.Tensor, prot_attn_mask: torch.Tensor) -> torch.Tensor:
        # mean pooling + mask
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

    @torch.no_grad()
    def gather_go_embs(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() == 1:
            return self.go_embs.index_select(0, ids.cpu()).to(ids.device)
        elif ids.dim() == 2:
            out = []
            for b in range(ids.size(0)):
                out.append(self.go_embs.index_select(0, ids[b].cpu()).to(ids.device))
            return torch.stack(out, dim=0)
        else:
            raise ValueError(f"ids shape {ids.shape} not supported")

class GoMemoryBank:
    def __init__(self, init_embs: torch.Tensor, row2id: List[int]):
        self.id2row = {int(i): int(r) for r, i in enumerate(row2id)}
        self.embs = F.normalize(torch.as_tensor(init_embs).cpu(), p=2, dim=1)

    def lookup(self, ids):
        rows = [self.id2row[int(i)] for i in ids]
        return self.embs[rows]

    def update(self, ids, new_embs):  # new_embs: (len(ids), d) CPU bekleriz
        new_embs = F.normalize(new_embs.cpu(), p=2, dim=1)
        for j, i in enumerate(ids):
            self.embs[self.id2row[int(i)]] = new_embs[j]
