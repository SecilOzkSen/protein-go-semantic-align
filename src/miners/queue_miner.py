from __future__ import annotations
import torch
import torch.nn.functional as F

class QueueMiner:
    """
    Lightweight miner in the style of MoCo/CLIP.
    - Stores the latest G/O (GO text) embeddings in a fixed-size queue (memory bank).
    - At each step: queries @ queue^T → returns the hardest k negatives.
    Note: You choose which modality to keep in the queue (usually the GO text side).
    """
    def __init__(self, dim: int, K: int = 8192, device: str = "cuda", normalize: bool = True):
        self.K = int(K)
        self.device = device
        self.normalize = normalize
        self.queue = torch.empty(self.K, dim, device=device)
        torch.nn.init.normal_(self.queue, std=0.02)
        self.queue = F.normalize(self.queue, dim=1)
        self.ptr = 0
        self.total = 0

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        """
        keys: [N, D] (usually GO text embeddings). Does not require grad (detach).
        """
        if self.normalize:
            keys = F.normalize(keys, dim=1)
        n = keys.shape[0]
        if n >= self.K:
            self.queue.copy_(keys[-self.K:])
            self.ptr = 0
            self.total = self.K
            return
        end = (self.ptr + n) % self.K
        if end > self.ptr:
            self.queue[self.ptr:end].copy_(keys)
        else:
            remain = self.K - self.ptr
            self.queue[self.ptr:].copy_(keys[:remain])
            self.queue[:end].copy_(keys[remain:])
        self.ptr = end
        self.total = min(self.total + n, self.K)

    @torch.no_grad()
    def get_negatives(self, queries: torch.Tensor, k_hard: int = 32) -> torch.Tensor:
        """
        queries: [B, D] (usually protein embeddings)
        return: neg_embs [B, k_hard, D] (hard negatives selected from the queue)
        """
        if self.total == 0:
            # If the queue is empty, return None so the trainer falls back to in-batch negatives
            return None
        Q = self.queue[:self.total]  # [T, D]
        if self.normalize:
            queries = F.normalize(queries, dim=1)
        sims = queries @ Q.T   # [B, T]
        k = min(int(k_hard), Q.shape[0])
        vals, idx = sims.topk(k, dim=1)  # [B, k]
        negs = Q[idx]                     # [B, k, D]
        return negs
