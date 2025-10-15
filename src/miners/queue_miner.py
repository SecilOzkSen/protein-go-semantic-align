from __future__ import annotations
import torch
import torch.nn.functional as F

class MoCoQueue(torch.nn.Module):
    """
    Lightweight miner in the style of MoCo/CLIP.
    - Stores the latest G/O (GO text) embeddings in a fixed-size queue (memory bank).
    - At each step: queries @ queue^T → returns the hardest k negatives.
    Note: You choose which modality to keep in the queue (usually the GO text side).
    """

    def __init__(self, dim: int, K: int = 131072, device="cuda"):
        super().__init__()  # <— super init
        self.K = int(K)
        self.register_buffer("queue", torch.zeros(dim, K, dtype=torch.float32, device=device))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long, device=device))

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor):
        keys = F.normalize(keys.float(), dim=1)
        b, d = keys.shape
        ptr = int(self.ptr.item())
        if ptr + b <= self.K:
            self.queue[:, ptr:ptr + b] = keys.T
        else:
            first = self.K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :b - first] = keys[first:].T
        self.ptr[0] = (ptr + b) % self.K
        self.filled[0] = torch.clamp(self.filled + b, max=self.K)

    @torch.no_grad()
    def get_all_neg(self) -> torch.Tensor:
        Kf = int(self.filled.item())
        return self.queue[:, :Kf].T  # [Kf, dim]
