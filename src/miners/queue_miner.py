import torch

class MoCoQueue(torch.nn.Module):
    def __init__(self, dim: int, K: int, device: str = "cuda"):
        super().__init__()
        self.K = int(K)
        self.proj_dim = int(dim)
        # projected queue lives on GPU, used for mining
        self.register_buffer(
            "queue_proj",
            torch.zeros(self.K, self.proj_dim, dtype=torch.float32, device=device)
        )  # [K, Dz]

        # ids / valid live on CPU
        self.register_buffer(
            "ids",
            torch.full((self.K,), -1, dtype=torch.long, device="cpu")
        )  # [K]
        self.register_buffer(
            "valid",
            torch.zeros(self.K, dtype=torch.bool, device="cpu")
        )  # [K]

        # raw queue stays on CPU, lazy init because raw dim may not be known at __init__
        self.queue_raw: torch.Tensor | None = None  # [K, Dg] on CPU

        self._ptr = 0

    @property
    def device(self) -> torch.device:
        return self.queue_proj.device

    @torch.no_grad()
    def reset(self):
        self.queue_proj.zero_()
        self.ids.fill_(-1)
        self.valid.zero_()
        if self.queue_raw is not None:
            self.queue_raw.zero_()
        self._ptr = 0

    @torch.no_grad()
    def on_change_dim(self, new_dim: int):
        """
        Change projected queue dimension.
        Raw queue is not touched here because its dim is independent.
        """
        new_dim = int(new_dim)
        cur_dim = int(self.queue_proj.size(1))
        if new_dim == cur_dim:
            return

        dev, K = self.queue_proj.device, self.K
        self._buffers["queue_proj"] = torch.zeros(K, new_dim, dtype=torch.float32, device=dev)
        self._buffers["ids"] = torch.full((K,), -1, dtype=torch.long, device="cpu")
        self._buffers["valid"] = torch.zeros(K, dtype=torch.bool, device="cpu")
        self.queue_raw = None
        self.proj_dim = new_dim
        self._ptr = 0

    @torch.no_grad()
    def _ensure_raw_queue(self, raw_dim: int):
        raw_dim = int(raw_dim)
        if self.queue_raw is None:
            self.queue_raw = torch.zeros(self.K, raw_dim, dtype=torch.float32, device="cpu")
            return

        cur_raw_dim = int(self.queue_raw.size(1))
        if cur_raw_dim != raw_dim:
            self.queue_raw = torch.zeros(self.K, raw_dim, dtype=torch.float32, device="cpu")
            self.ids.fill_(-1)
            self.valid.zero_()
            self._ptr = 0

    @torch.no_grad()
    def enqueue(self, proj_vecs: torch.Tensor, raw_vecs: torch.Tensor, ids: torch.Tensor):
        """
        proj_vecs: [N, Dz] projected+normalized GO vectors, for mining, stored on GPU
        raw_vecs:  [N, Dg] raw GO encoder outputs, for scorer candidates, stored on CPU
        ids:       [N] global GO ids, stored on CPU
        """
        if proj_vecs.dim() != 2:
            raise ValueError(f"proj_vecs must be [N,D], got {tuple(proj_vecs.shape)}")
        if raw_vecs.dim() != 2:
            raise ValueError(f"raw_vecs must be [N,D], got {tuple(raw_vecs.shape)}")
        if ids.dim() != 1:
            raise ValueError(f"ids must be [N], got {tuple(ids.shape)}")
        if proj_vecs.size(0) != raw_vecs.size(0) or proj_vecs.size(0) != ids.size(0):
            raise ValueError(
                f"enqueue size mismatch: proj={tuple(proj_vecs.shape)} "
                f"raw={tuple(raw_vecs.shape)} ids={tuple(ids.shape)}"
            )

        if int(proj_vecs.size(1)) != int(self.queue_proj.size(1)):
            raise ValueError(
                f"Projected queue dim mismatch: got {int(proj_vecs.size(1))}, "
                f"expected {int(self.queue_proj.size(1))}. Call on_change_dim first."
            )

        self._ensure_raw_queue(int(raw_vecs.size(1)))

        dev = self.device

        # projected vectors -> GPU queue
        proj_vecs = proj_vecs.detach().to(dev, dtype=torch.float32, non_blocking=True)

        # raw vectors + ids -> CPU queue
        raw_vecs = raw_vecs.detach().to("cpu", dtype=torch.float32)
        ids = ids.detach().to("cpu", dtype=torch.long)

        n = int(proj_vecs.size(0))
        if n == 0:
            return

        # If batch is larger than queue, keep only last K items
        if n > self.K:
            proj_vecs = proj_vecs[-self.K:]
            raw_vecs = raw_vecs[-self.K:]
            ids = ids[-self.K:]
            n = self.K

        idx_cpu = (torch.arange(n, device="cpu") + self._ptr) % self.K
        idx_gpu = idx_cpu.to(dev, non_blocking=True)

        self.queue_proj.index_copy_(0, idx_gpu, proj_vecs)
        self.queue_raw.index_copy_(0, idx_cpu, raw_vecs)
        self.ids.index_copy_(0, idx_cpu, ids)
        self.valid.index_fill_(0, idx_cpu, True)

        self._ptr = int((self._ptr + n) % self.K)

    @torch.no_grad()
    def get_all_neg(self, convert_proj_device: torch.device | str | None = None):
        """
        Returns:
          proj: GPU by default, or moved if convert_proj_device is provided
          raw:  CPU
          ids:  CPU
        """
        if not self.valid.any():
            return None

        m_cpu = self.valid  # CPU bool mask

        proj = self.queue_proj[m_cpu.to(self.queue_proj.device, non_blocking=True)]
        raw = None if self.queue_raw is None else self.queue_raw[m_cpu]
        ids = self.ids[m_cpu]

        if convert_proj_device is not None and str(convert_proj_device) != str(self.queue_proj.device):
            proj = proj.to(convert_proj_device, non_blocking=True)

        return proj, raw, ids

    @torch.no_grad()
    def to_(self, dev: torch.device | str):
        """
        Move only projected queue.
        Raw queue, ids, valid remain on CPU by design.
        """
        self._buffers["queue_proj"] = self.queue_proj.to(dev, non_blocking=True)
        return self
