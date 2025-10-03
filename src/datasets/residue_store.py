"""
ESM residue embedding loader (post-hoc stitching / overlap merge).

- Expects shards saved as: embeddings/esm_embed_*.pt
  Each shard is a dict: { protein_id: np.ndarray[L_concat, D] }
  where L_concat includes duplicated residues from sliding-window overlap.

- If you provide the original sequence lengths (e.g., from sequences_full.pkl),
  we can properly stitch overlapping segments back to [L_seq, D].
  Instead of dropping the tail, we merge overlapping residues
  (averaging their embeddings) so that positions align exactly.

Usage
-----
    store = ESMResidueStore(embed_dir,
                            seq_len_lookup=seq_len_dict,
                            max_len=1024,
                            overlap=256)
    H = store.get(protein_id)  # torch.FloatTensor [L_seq, D]
"""
from __future__ import annotations
import pickle
from typing import Dict, Optional
from src.configs.paths import GOOGLE_DRIVE_MANIFEST_CACHE
import torch

class ESMResidueStore:
    def __init__(self,
                 embed_dir: str,
                 seq_len_lookup: Optional[Dict[str, int]] = None,  # protein_id -> sequence length
                 max_len: Optional[int] = None,                   # window size used during extraction
                 overlap: Optional[int] = None,                   # overlap used during extraction
                 cache_shards: bool = True,
                 pro_manifest_file: str = GOOGLE_DRIVE_MANIFEST_CACHE):
        self.embed_dir = embed_dir
        self.seq_len_lookup = seq_len_lookup or {}
        self.max_len = max_len
        self.overlap = overlap
        self.cache_shards = bool(cache_shards)

        # load manifest mapping protein_id -> shard path
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

    def _stitch_with_overlap(self, H: torch.Tensor, L_seq: int) -> torch.Tensor:
        """
        Reconstructs residue embeddings from overlapped segments.
        Overlapping positions are averaged to preserve alignment.

        H: [L_concat, D]  -> concatenated segment embeddings
        L_seq: true sequence length
        Returns: [L_seq, D]
        """
        if self.max_len is None or self.overlap is None:
            raise ValueError("max_len and overlap must be provided for stitching.")

        D = H.size(1)
        step = self.max_len - self.overlap
        if step <= 0:
            raise ValueError("Invalid parameters: max_len must be greater than overlap.")

        # Compute segment start indices (based on extraction logic)
        starts = []
        s = 0
        while s < L_seq:
            starts.append(s)
            if s + self.max_len >= L_seq:
                break
            s += step

        # Expected segment lengths (last one may be shorter)
        seg_lens = [min(self.max_len, L_seq - st) for st in starts]

        if sum(seg_lens) != H.size(0):
            raise ValueError(
                f"Concat length mismatch for stitching: "
                f"got {H.size(0)}, expected {sum(seg_lens)}. "
                "Check max_len/overlap/sequence length consistency."
            )

        # Merge overlaps by averaging
        out = H.new_zeros((L_seq, D))
        cnt = H.new_zeros((L_seq, 1))
        cursor = 0
        for st, sl in zip(starts, seg_lens):
            out[st:st+sl] += H[cursor:cursor+sl]
            cnt[st:st+sl] += 1
            cursor += sl
        out = out / cnt.clamp_min(1e-8)
        return out

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

        # Post-hoc stitching
        L_concat = int(H.shape[0])
        L_seq = int(self.seq_len_lookup.get(str(protein_id), 0))

        if L_seq > 0 and L_concat != L_seq:
            H = self._stitch_with_overlap(H, L_seq)

        return H