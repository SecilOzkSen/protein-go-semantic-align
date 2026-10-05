from typing import Sequence, Dict, Mapping, Optional
import torch

def build_go_index(row2id: Sequence[int]) -> Dict[int, int]:
    # global GO id -> column index (0..n_go-1)
    return {int(gid): int(i) for i, gid in enumerate(row2id)}

def mask_from_globals(
    terms: Sequence[int],
    go_index: Mapping[int, int],
    n_go: int,
    device: Optional[torch.device] = None
) -> torch.BoolTensor:
    dev = device if device is not None else torch.device("cpu")
    m = torch.zeros(n_go, dtype=torch.bool, device=dev)
    for g in terms:
        j = go_index.get(int(g), -1)
        if j >= 0:
            m[j] = True
    return m