# src/training/moco_queue.py  (veya sizde neredeyse)
import torch

class MoCoQueue(torch.nn.Module):
    def __init__(self, dim: int, K: int, device: str = "cuda"):
        super().__init__()
        self.K = int(K)
        self.register_buffer("queue", torch.zeros(self.K, dim, dtype=torch.float32, device=device))  # [K, D]
        self.register_buffer("ids",   torch.full((self.K,), -1, dtype=torch.long, device=device))    # [K]
        self.register_buffer("valid", torch.zeros(self.K, dtype=torch.bool, device=device))          # [K]
        self._ptr = 0

    @property
    def device(self) -> torch.device:
        return self.queue.device

    @torch.no_grad()
    def on_change_dim(self, new_dim: int):
        new_dim = int(new_dim)
        cur_dim = int(self.queue.size(1))
        if new_dim == cur_dim:
            return
        dev, K = self.queue.device, self.K
        self.queue = torch.zeros(K, new_dim, dtype=torch.float32, device=dev)
        self.ids.zero_().sub_(1)    # tekrar -1 yap
        self.valid.zero_()
        self._ptr = 0

    @torch.no_grad()
    def enqueue(self, vecs: torch.Tensor, ids: torch.Tensor):
        # vecs: [N, D], ids: [N]
        dev = self.device
        # === cihaz + dtype hizalama ===
        if vecs.device != dev:
            vecs = vecs.to(dev, non_blocking=True)
        if ids.device != dev:
            ids = ids.to(dev, non_blocking=True)
        if ids.dtype != torch.long:
            ids = ids.long()
        n = vecs.size(0)
        idx = (torch.arange(n, device=vecs.device) + self._ptr) % self.K
        self.queue.index_copy_(0, idx, vecs)
        self.ids.index_copy_(0, idx, ids)
        self.valid.index_fill_(0, idx, True)
        self._ptr = int((self._ptr + n) % self.K)

    @torch.no_grad()
    def get_all_neg(self):
        # Her durumda tuple döndür!
        if not self.valid.any():
            return None, None
        m = self.valid
        return self.queue[m], self.ids[m]

    @torch.no_grad()
    def to_(self, dev: torch.device | str):
        """ İsteğe bağlı: explicit taşıma kolaylığı """
        self.queue = self.queue.to(dev, non_blocking=True)
        self.ptr = self.ptr.to(dev, non_blocking=True)
        return self