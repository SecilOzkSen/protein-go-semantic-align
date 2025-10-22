import torch

class MoCoQueue(torch.nn.Module):
    """
    FIFO queue storing GO vectors **and their global GO ids** for false-negative masking.
    """
    def __init__(self, dim: int, K: int, device: str = "cuda"):
        super().__init__()
        self.K = int(K)
        self.register_buffer("queue", torch.zeros(self.K, dim, dtype=torch.float32, device=device))
        self.register_buffer("ids",   torch.full((self.K,), -1, dtype=torch.long, device=device))
        self.register_buffer("valid", torch.zeros(self.K, dtype=torch.bool, device=device))
        self._ptr = 0

    @torch.no_grad()
    def enqueue(self, vecs: torch.Tensor, ids: torch.Tensor):
        assert vecs.ndim == 2 and ids.ndim == 1 and vecs.size(0) == ids.size(0)
        n = vecs.size(0)
        idx = (torch.arange(n, device=vecs.device) + self._ptr) % self.K
        self.queue.index_copy_(0, idx, vecs)
        self.ids.index_copy_(0, idx, ids)
        self.valid.index_fill_(0, idx, True)
        self._ptr = int((self._ptr + n) % self.K)

    @torch.no_grad()
    def get_all_neg(self):
        if not self.valid.any():
            return None, None
        m = self.valid
        return self.queue[m], self.ids[m]