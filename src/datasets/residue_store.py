"""
ESM residue embedding loader (post-hoc stitching / overlap trim).

- Expects shards saved as: embeddings/esm_embed_*.pt
  Each shard is a dict: { protein_id: np.ndarray[L_concat, D] }
  where L_concat includes duplicated residues from sliding-window overlap.

- If you provide the original sequence lengths (e.g., from sequences_full.pkl),
  we can trim the concatenation without re-embedding:
      L_concat = L_seq + (n_segments - 1) * overlap
  ⇒ Simply drop (L_concat - L_seq) rows from the **tail**.

Usage
-----
    store = ESMResidueStore(embed_dir, seq_len_lookup=seq_len_dict, overlap=256)
    H = store.get(protein_id)  # torch.FloatTensor [L, D]
"""
from __future__ import annotations
import pickle
from typing import Dict, Optional
from src.configs.paths import GOOGLE_DRIVE_MANIFEST_CACHE
import torch

class ESMResidueStore:
    def __init__(self,
                 embed_dir: str,
                 seq_len_lookup: Optional[Dict[str, int]] = None, # proteinid -> sequence length.
                 overlap: Optional[int] = None,
                 cache_shards: bool = True,
                 pro_manifest_file :str = GOOGLE_DRIVE_MANIFEST_CACHE # manifest file
                 ):
        self.embed_dir = embed_dir
        self.seq_len_lookup = seq_len_lookup or {}
        self.overlap = overlap  # kept for reference/logging; trimming needs only seq_len
        self.cache_shards = bool(cache_shards)

        # load manifest
        with open(pro_manifest_file, 'rb') as f:
            self.pid2shard: Dict[str, str] = pickle.load(f)
        # Optional shard cache
        self._cache: Dict[str, dict] = {}

    def _load_shard(self, path: str) -> dict:
        if self.cache_shards and path in self._cache:
            return self._cache[path]
        try:
            shard = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Error loading shard {path}: {e}")
        if self.cache_shards:
            self._cache[path] = shard
        return shard

    def get(self, protein_id: str) -> torch.Tensor:
        path = self.pid2shard.get(str(protein_id), None)
        if path is None:
            raise KeyError(f"Protein '{protein_id}' not found in {self.embed_dir}")
        shard = self._load_shard(path)
        arr = shard.get(str(protein_id), None)
        if arr is None:
            # some shards might have non-str keys; attempt fallback
            arr = shard.get(protein_id, None)
        if arr is None:
            raise KeyError(f"Protein '{protein_id}' missing in shard {path}")

        # Convert to torch tensor
        H = torch.as_tensor(arr).float()  # [L_concat, D]

        # Post-hoc stitching: trim tail duplicates if we know L_seq
        L_concat = int(H.shape[0])
        L_seq = int(self.seq_len_lookup.get(str(protein_id), 0))
        if L_seq > 0 and L_concat > L_seq:
            drop = L_concat - L_seq
            if drop > 0:
                H = H[:-drop, :]
        return H

