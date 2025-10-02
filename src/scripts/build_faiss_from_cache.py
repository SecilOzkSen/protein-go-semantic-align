"""
Build a FAISS index from a phase-locked GO embedding cache.

This script:
  - Loads processed/go_text_embeddings.pt (embs + id2row/row2id),
  - Ensures L2-normalization (cosine ≡ inner product),
  - Builds a FAISS IndexIDMap2(FlatIP) on CPU with GO ids,
  - Optionally creates a GPU copy for runtime search (not saved),
  - Saves the CPU index to disk (and a small .meta.json sidecar),
  - Runs a self-check (top-1 self-retrieval).

Why IndexIDMap2(FlatIP)?
- FlatIP == cosine if vectors are L2-normalized.
- IDMap2 stores YOUR GO ids as FAISS ids → search returns GO ids directly.
"""

from __future__ import annotations
import os, json, datetime, random
from typing import Dict, Any, List
from src.miners.faiss_index import FaissIndexManager
import numpy as np
import torch
from src.configs.data_classes import TrainingReadyDataPaths
import torch.nn.functional as F
from src.go.go_cache import GoLookupCache
import faiss
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")

MAKE_GPU_COPY = True
SELF_CHECK_SAMPLES = 256
SEED = 42

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

def inspect_tensor(t, name="x"):
    print(f"[{name}] type: {type(t)}")
    if isinstance(t, torch.Tensor):
        try:
            print(f"[{name}] device={t.device}, dtype={t.dtype}, shape={tuple(t.shape)}")
            print(f"[{name}] stride={t.stride()}, storage_offset={t.storage_offset()}")
            print(f"[{name}] is_contiguous={t.is_contiguous()}")
            # minik bir okuma, storage hatasına hızlı yakalanır:
            _ = t.reshape(-1)[:8]
            print(f"[{name}] head values:", _.tolist())
        except Exception as e:
            print(f"[{name}] meta/peek sırasında hata:", repr(e))
    else:
        print(f"[{name}] Tensor değil! (örn. numpy/list/dict)")


def load_go_cache(path: str) -> Dict[str, Any]:
    """
    Load GO cache produced by build_go_embeddings.py.
    Ensures row2id/ id2row consistency.
    """
    blob = torch.load(path, map_location="cpu")
    # TODO: remove here! - normalization should be done during embedding creation. But for safety, we keep it.
    blob['embs'] = F.normalize(blob['embs'], p=2, dim=1) # FAISS expects L2 normalized vectors!

    assert "embs" in blob and "id2row" in blob, "Cache missing 'embs' or 'id2row'."
    if "row2id" not in blob:
        id2row = blob["id2row"]
        row2id = [None] * len(id2row)
        for gid, r in id2row.items():
            row2id[r] = int(gid)
        blob["row2id"] = row2id
    return blob

def _maybe_make_gpu_copy(index):
    """Try to create a GPU copy; fall back to the given index on failure."""
    try:
        import faiss
        gpu = faiss.index_cpu_to_all_gpus(index)
        # round-trip to ensure compatibility on some builds
        index_cpu = faiss.index_gpu_to_cpu(gpu)
        gpu = faiss.index_cpu_to_all_gpus(index_cpu)
        print("[INFO] Using FAISS GPU index (runtime only).")
        return gpu
    except Exception as e:
        print(f"[WARN] GPU index creation failed: {e} → using CPU index.")
        return index


def self_check_top1(index, embs_np: np.ndarray, ids_np: np.ndarray, nsamples: int = 256, seed: int = 42) -> float:
    """
    Randomly pick nsamples rows, query their vectors, and check top-1 id.
    Returns recall@1 in [0,1] (can be <1.0 if many duplicates).
    """
    # to ensure .search exists
    nsamples = min(nsamples, embs_np.shape[0])
    rng = random.Random(seed)
    idxs = [rng.randrange(0, embs_np.shape[0]) for _ in range(nsamples)]
    Q = embs_np[idxs, :]
    _, I = index.search(Q, 1)  # top-1
    got = I.reshape(-1)
    truth = ids_np[idxs]
    return float((got == truth).mean())


def faiss_index_builder(go_cache_path:str, save_index_path:str, save_meta_path:str):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # 1) Load cache
    print(f"[INFO] Loading cache: {go_cache_path}")
    cache = load_go_cache(go_cache_path)
    embs: torch.Tensor = cache["embs"].contiguous()   # [N, D]
    meta: Dict[str, Any] = cache.get("meta", {})
    N, D = int(embs.shape[0]), int(embs.shape[1])
    print(f"[INFO] Cache stats: N={N:,}, D={D}, normalized={bool(meta.get('normalized', False))}")

    # 2) Ensure float32 + L2 normalization
    if embs.dtype != torch.float32:
        embs = embs.to(torch.float32)

    # --- Sanity check: are embeddings really L2-normalized? ---
    row_norms = torch.sqrt(torch.clamp((embs * embs).sum(dim=1), min=1e-30))
    mean_norm = float(row_norms.mean())
    print(f"[CHECK] Mean L2 norm across embeddings = {mean_norm:.4f}")

    if abs(mean_norm - 1.0) > 0.02:
        print("[WARN] Embeddings may not be properly L2-normalized. "
              "FAISS cosine similarity will give wrong results unless fixed.")
    else:
        print("[INFO] Embeddings look properly normalized (=1).")
    # -----------------------------------------------------------

    row2id_np = np.asarray(cache["row2id"], dtype=np.int64)

    # 3) Build CPU index — prefer FaissIndexManager if available

    cpu_index = FaissIndexManager.build_from_cache(
            GoLookupCache(embs=embs, row2id=row2id_np, id2row=cache["id2row"]),
            use_gpu=False, use_idmap=True
        )

    print("[INFO] Built CPU index via FaissIndexManager.")

    assert cpu_index.ntotal == N, f"FAISS ntotal {cpu_index.ntotal} != N {N}"

    # 4) Optional GPU copy (runtime only)
    runtime_index = _maybe_make_gpu_copy(cpu_index) if MAKE_GPU_COPY else cpu_index

    # 5) Self-check (recall@1 with self-queries)
    r1 = self_check_top1(runtime_index, embs.cpu().numpy(), row2id_np, nsamples=SELF_CHECK_SAMPLES, seed=SEED)
    print(f"[CHECK] top-1 self-retrieval recall ≈ {r1:.4f}  "
          f"(expected ~1.0; duplicates can lower this)")

    # 6) Save CPU index
    os.makedirs(os.path.dirname(save_index_path) or ".", exist_ok=True)
    try:
        faiss.write_index(cpu_index, str(save_index_path))
        print(f"[OK] Wrote CPU FAISS index -> {save_index_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to write FAISS index: {e}")

    # 7) Save meta sidecar
    sidecar = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "n": N, "d": D,
        "index_type": "IndexIDMap2(FlatIP)",
        "normalized": True,
        "go_cache_path": os.path.abspath(save_meta_path),
        "model_name": meta.get("model_name"),
        "phase": meta.get("phase"),
        "drift_vs": meta.get("compare_to"),
        "drift": meta.get("drift"),
    }
    with open(save_meta_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote meta → {save_meta_path}")

    print("[DONE] FAISS build complete.")

def main():
    for i in range(len(TrainingReadyDataPaths.phases)):
        print(f"==============   Starting phase - {i+1}: ========================")
        print("\n\n")
        path_dict = TrainingReadyDataPaths.phases[i]
        faiss_index_builder(go_cache_path=path_dict["embeddings"],
                            save_meta_path=path_dict["meta"],
                            save_index_path=path_dict["ip"])
        print("\n\n")
        print("================================================================")


if __name__ == "__main__":
    main()
