import torch

class MoCoQueue(torch.nn.Module):
    """
    FIFO queue storing GO vectors and their global GO ids (kept internally).
    API (trainer uyumlu):
      - enqueue(vecs [N,dim], ids [N])
      - get_all_neg() -> Tensor [M, dim] (sadece vektörler)
      - on_change_dim(new_dim) -> kuyruğu güvenli sıfırlar
    """
    def __init__(self, dim: int, K: int, device: str = "cuda"):
        super().__init__()
        self.K = int(K)
        # ŞEKİL: [K, dim]  (trainer get_all_neg -> [M, dim] dönecek)
        self.register_buffer("queue", torch.zeros(self.K, dim, dtype=torch.float32, device=device))
        self.register_buffer("ids",   torch.full((self.K,), -1, dtype=torch.long, device=device))
        self.register_buffer("valid", torch.zeros(self.K, dtype=torch.bool, device=device))
        self._ptr = 0

    @torch.no_grad()
    def on_change_dim(self, new_dim: int):
        new_dim = int(new_dim)
        cur_dim = int(self.queue.size(1))          # <<< DOĞRU EKSEN: dim
        if new_dim == cur_dim:
            return

        device = self.queue.device
        dtype  = self.queue.dtype
        K      = self.K

        # Tüm buffer'ları yeni boyuta göre SIFIRLA
        self.queue = torch.zeros(K, new_dim, dtype=dtype, device=device)
        self.ids = torch.full((K,), -1, dtype=torch.long, device=device)
        self.valid = torch.zeros(K, dtype=torch.bool, device=device)
        self._ptr = 0

    @torch.no_grad()
    def enqueue(self, vecs: torch.Tensor, ids: torch.Tensor):
        # vecs: [N, dim], ids: [N]
        assert vecs.ndim == 2 and ids.ndim == 1 and vecs.size(0) == ids.size(0)
        n = int(vecs.size(0))
        # hedef indeksler
        idx = (torch.arange(n, device=self.queue.device) + self._ptr) % self.K
        # kopya
        self.queue.index_copy_(0, idx, vecs.to(self.queue.dtype))
        self.ids.index_copy_(0, idx, ids.to(self.ids.dtype))
        self.valid[idx] = True
        self._ptr = int((self._ptr + n) % self.K)

    @torch.no_grad()
    def get_all_neg(self):
        # Trainer sadece EMBEDDING bekliyor
        if not torch.any(self.valid):
            return None
        return self.queue[self.valid]   # [M, dim]