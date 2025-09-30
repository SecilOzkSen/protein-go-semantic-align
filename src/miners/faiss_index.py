import faiss, numpy as np, torch, random
from src.go.go_cache import GoLookupCache

class FaissIndexManager:
    @staticmethod
    def build_from_cache(go_cache: GoLookupCache, use_gpu: bool=True, use_idmap: bool=True):
        embs = go_cache.embs.detach().cpu().to(torch.float32).numpy()
        N, D = embs.shape
        base = faiss.IndexFlatIP(D)
        if use_idmap:
            index = faiss.IndexIDMap2(base)
            ids = np.asarray(go_cache.row2id, dtype=np.int64)
            index.add_with_ids(embs, ids)
        else:
            index = base
            index.add(embs)
        if use_gpu:
            try:
                index = faiss.index_cpu_to_all_gpus(index)
            except Exception:
                pass
        return index

    @staticmethod
    def save_cpu(index, path: str):
        # GPU index to CPU
        try:
            idx_cpu = faiss.index_gpu_to_cpu(index)
        except Exception:
            idx_cpu = index
        faiss.write_index(idx_cpu, path)

    @staticmethod
    def load_cpu(path: str, use_gpu: bool=True):
        idx = faiss.read_index(path)  # CPU
        if use_gpu:
            try:
                idx = faiss.index_cpu_to_all_gpus(idx)
            except Exception:
                pass
        return idx

    @staticmethod
    def self_check_top1(index, embs: np.ndarray, ids: np.ndarray, nsamples: int=256, seed: int=42) -> float:
        rng = random.Random(seed)
        idxs = [rng.randrange(0, embs.shape[0]) for _ in range(min(nsamples, embs.shape[0]))]
        Q = embs[idxs]
        _, I = index.search(Q, 1)
        return float((I.reshape(-1) == ids[idxs]).mean())
