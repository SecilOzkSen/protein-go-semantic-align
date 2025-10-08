import os, json
import torch
import faiss
from src.configs.data_classes import TrainingReadyDataPaths

def load_faiss_index_for_phase(phase: int, to_gpu: bool = False):
    """
    go_faiss_ip.faiss and meta.json loading.
    Checks if 'phase' matches the real phase in meta.json
    """
    if phase < 0:
        raise RuntimeError("Phase value must be bigger than 0.")

    phases_dict = TrainingReadyDataPaths().phases[phase]
    # 1) meta check
    if not os.path.isfile(phases_dict["meta"]):
        raise FileNotFoundError(phases_dict["meta"])
    with open(phases_dict["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    if str(meta.get("phase")) != str(f"phase_{phase+1}"):
        raise RuntimeError(f"Meta phase mismatch: meta={meta.get('phase')} vs arg={phase+1}")

    # 2) index
    if not os.path.isfile(phases_dict["ip"]):
        raise FileNotFoundError(phases_dict["ip"])

    cpu_idx = faiss.read_index(str(phases_dict["ip"]))
    index = cpu_idx
    if to_gpu:
        try:
            index = faiss.index_cpu_to_all_gpus(cpu_idx)
        except Exception:
            pass

    # Cache for look-up
    if not os.path.isfile(phases_dict["embeddings"]):
        raise FileNotFoundError(phases_dict["embeddings"])
    cache = torch.load(phases_dict["embeddings"], map_location="cpu")
    embs  = cache["embs"]
    row2id = cache.get("row2id")
    if row2id is None:
        # reconstruct
        id2row = cache["id2row"]
        row2id = [None] * len(id2row)
        for gid, r in id2row.items():
            row2id[r] = int(gid)

    return index #cache, meta, embs
