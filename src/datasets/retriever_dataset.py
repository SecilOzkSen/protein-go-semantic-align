from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class RetrieverV2DumpDataset(Dataset):
    """
    Memory-mapped dataset for Retriever-v2 full-GO dumps.

    Per-protein tensors:
        protein_z         [D]
        retriever_scores  [G]
        labels            [G]

    Shared tensors / metadata:
        go_z              [G, D]
        eval_go_ids       [G]
        protein_ids       [N]

    Notes
    -----
    - .npy arrays are opened with mmap_mode="r".
    - go_z is shared and is NOT returned for every sample.
    - The dataset validates dump shapes at construction time.
    - Float16 dump arrays are converted to torch tensors without changing
      dtype. The trainer can decide when to cast/move them.
    """

    def __init__(self, dump_dir: str | Path):
        super().__init__()

        self.dump_dir = Path(dump_dir).expanduser().resolve()

        if not self.dump_dir.exists():
            raise FileNotFoundError(
                f"Dump directory does not exist: {self.dump_dir}"
            )

        done = self.dump_dir / "DONE"
        if not done.exists():
            raise RuntimeError(
                f"Dump is incomplete, missing DONE marker: {done}"
            )

        metadata_path = self.dump_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing metadata.json: {metadata_path}"
            )

        with metadata_path.open("r", encoding="utf-8") as f:
            self.metadata: Dict[str, Any] = json.load(f)

        files = self.metadata.get("files", {})

        self.protein_z_path = self._resolve_file(
            files.get("protein_z", "protein_z.float16.npy")
        )
        self.go_z_path = self._resolve_file(
            files.get("go_z", "go_z.float16.npy")
        )
        self.scores_path = self._resolve_file(
            files.get("scores", "retriever_scores.float16.npy")
        )
        self.labels_path = self._resolve_file(
            files.get("labels", "labels.int8.npy")
        )
        self.eval_go_ids_path = self._resolve_file(
            files.get("eval_go_ids", "eval_go_ids.int64.npy")
        )
        self.protein_ids_path = self._resolve_file(
            files.get("protein_ids", "protein_ids.json")
        )

        # Memory-mapped arrays.
        self.protein_z = np.load(
            self.protein_z_path,
            mmap_mode="r",
        )
        self.go_z = np.load(
            self.go_z_path,
            mmap_mode="r",
        )
        self.retriever_scores = np.load(
            self.scores_path,
            mmap_mode="r",
        )
        self.labels = np.load(
            self.labels_path,
            mmap_mode="r",
        )
        self.eval_go_ids = np.load(
            self.eval_go_ids_path,
            mmap_mode="r",
        )

        with self.protein_ids_path.open("r", encoding="utf-8") as f:
            self.protein_ids = json.load(f)

        self._validate()

    def _resolve_file(self, name: str) -> Path:
        path = self.dump_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dump file: {path}"
            )
        return path

    def _validate(self) -> None:
        if self.protein_z.ndim != 2:
            raise RuntimeError(
                f"protein_z must be [N, D], got {self.protein_z.shape}"
            )

        if self.go_z.ndim != 2:
            raise RuntimeError(
                f"go_z must be [G, D], got {self.go_z.shape}"
            )

        if self.retriever_scores.ndim != 2:
            raise RuntimeError(
                "retriever_scores must be [N, G], "
                f"got {self.retriever_scores.shape}"
            )

        if self.labels.ndim != 2:
            raise RuntimeError(
                f"labels must be [N, G], got {self.labels.shape}"
            )

        if self.eval_go_ids.ndim != 1:
            raise RuntimeError(
                f"eval_go_ids must be [G], got {self.eval_go_ids.shape}"
            )

        n, d = self.protein_z.shape
        g, go_d = self.go_z.shape

        if d != go_d:
            raise RuntimeError(
                f"Embedding dimension mismatch: "
                f"protein D={d}, GO D={go_d}"
            )

        if self.retriever_scores.shape != (n, g):
            raise RuntimeError(
                "retriever_scores shape mismatch: "
                f"got {self.retriever_scores.shape}, expected {(n, g)}"
            )

        if self.labels.shape != (n, g):
            raise RuntimeError(
                "labels shape mismatch: "
                f"got {self.labels.shape}, expected {(n, g)}"
            )

        if self.eval_go_ids.shape[0] != g:
            raise RuntimeError(
                "eval_go_ids length mismatch: "
                f"got {self.eval_go_ids.shape[0]}, expected {g}"
            )

        if len(self.protein_ids) != n:
            raise RuntimeError(
                "protein_ids length mismatch: "
                f"got {len(self.protein_ids)}, expected {n}"
            )

        meta_n = self.metadata.get("n_samples")
        meta_g = self.metadata.get("n_go")
        meta_d = self.metadata.get("d_z")

        if meta_n is not None and int(meta_n) != n:
            raise RuntimeError(
                f"metadata n_samples={meta_n}, actual={n}"
            )

        if meta_g is not None and int(meta_g) != g:
            raise RuntimeError(
                f"metadata n_go={meta_g}, actual={g}"
            )

        if meta_d is not None and int(meta_d) != d:
            raise RuntimeError(
                f"metadata d_z={meta_d}, actual={d}"
            )

        # Experiment-B invariant: this must be a full-GO dump.
        if self.metadata.get("candidate_truncation") is True:
            raise RuntimeError(
                "Experiment B requires full-GO dumps, "
                "but metadata says candidate_truncation=True."
            )

    @property
    def num_go(self) -> int:
        return int(self.go_z.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.go_z.shape[1])

    def get_go_z_tensor(
            self,
            *,
            dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Materialize the shared GO bank once.

        Use this outside the per-sample DataLoader path, then move it to the
        desired device once in the trainer.
        """
        arr = np.asarray(self.go_z).copy()
        return torch.from_numpy(arr).to(dtype=dtype)

    def get_eval_go_ids_tensor(self) -> torch.Tensor:
        arr = np.asarray(self.eval_go_ids).copy()
        return torch.from_numpy(arr).long()

    def __len__(self) -> int:
        return int(self.protein_z.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Copy each row because mmap arrays are read-only. This avoids
        # PyTorch warnings about non-writable NumPy buffers.
        protein_z = torch.from_numpy(
            np.array(self.protein_z[idx], copy=True)
        )

        retriever_scores = torch.from_numpy(
            np.array(self.retriever_scores[idx], copy=True)
        )

        labels = torch.from_numpy(
            np.array(self.labels[idx], copy=True)
        ).float()

        return {
            "protein_z": protein_z,
            "retriever_scores": retriever_scores,
            "labels": labels,
            "protein_id": str(self.protein_ids[idx]),
            "index": int(idx),
        }


def validate_matching_go_banks(
        train_dataset: RetrieverV2DumpDataset,
        other_dataset: RetrieverV2DumpDataset,
) -> None:
    """
    Ensure train/val/test dumps use the same GO column ordering.

    This must pass before Experiment B training/evaluation.
    """
    train_ids = np.asarray(train_dataset.eval_go_ids)
    other_ids = np.asarray(other_dataset.eval_go_ids)

    if train_ids.shape != other_ids.shape:
        raise RuntimeError(
            "GO universe size mismatch: "
            f"train={train_ids.shape}, other={other_ids.shape}"
        )

    if not np.array_equal(train_ids, other_ids):
        mismatch = np.flatnonzero(train_ids != other_ids)
        example = mismatch[:10].tolist()

        raise RuntimeError(
            "GO column ordering mismatch between dumps. "
            f"First mismatch positions: {example}"
        )

    if train_dataset.embedding_dim != other_dataset.embedding_dim:
        raise RuntimeError(
            "Embedding dimension mismatch between dumps: "
            f"train={train_dataset.embedding_dim}, "
            f"other={other_dataset.embedding_dim}"
        )
