"""Experiment D: leakage-aware ESM-1b 5-nearest-neighbour builder.

Run: python -m src.moe.neighbour_builder --train_ids ... --val_ids ...
     --embed_dir ... --out_dir ...

This builds protein neighbours only; no GO annotations are read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.datasets.residue_store import ESMResidueStore

LOG = logging.getLogger("moe.neighbour_builder")


def read_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError(f"Expected JSON list of protein IDs: {path}")
        ids = [str(v).strip() for v in obj]
    else:
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [pid for pid in ids if pid]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate protein IDs in {path}; resolve before neighbour search")
    if not ids:
        raise ValueError(f"Empty protein ID list: {path}")
    return ids


def load_exclusion_groups(path: Path | None) -> dict[str, str]:
    """Optional JSON mapping protein_id -> sequence-hash/group ID.

    Equal group IDs are excluded, including exact sequence duplicates with different IDs.
    """
    if path is None:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("--exclusion_groups must be JSON object: protein_id -> group_id")
    return {str(k): str(v) for k, v in obj.items()}


def embed_one(store: ESMResidueStore, pid: str) -> tuple[np.ndarray, int]:
    h = store.get(pid)
    if not isinstance(h, torch.Tensor):
        h = torch.as_tensor(h)
    if h.ndim != 2 or h.shape[0] == 0:
        raise ValueError(f"Invalid residue tensor for {pid}: {tuple(h.shape)}")
    if not torch.isfinite(h).all():
        raise ValueError(f"Non-finite ESM residue embeddings for {pid}")
    # Unpadded residue store: all returned rows are valid residues.
    vec = h.float().mean(dim=0)
    norm = torch.linalg.vector_norm(vec)
    if not torch.isfinite(norm) or norm.item() < 1e-12:
        raise ValueError(f"Zero/invalid mean embedding for {pid}")
    return (vec / norm).cpu().numpy().astype(np.float32), int(h.shape[0])


def embed_ids(store: ESMResidueStore, ids: list[str], split: str, log_every: int):
    rows, lengths = [], []
    dim = None
    for i, pid in enumerate(ids, 1):
        v, length = embed_one(store, pid)
        if dim is None:
            dim = len(v)
        if len(v) != dim:
            raise ValueError(f"Embedding dimension mismatch for {pid}: {len(v)} vs {dim}")
        rows.append(v)
        lengths.append(length)
        if log_every > 0 and i % log_every == 0:
            LOG.info("[%s] embedded %d/%d", split, i, len(ids))
    return np.stack(rows), np.asarray(lengths, dtype=np.int32)


def topk_neighbours(query: np.ndarray, bank: np.ndarray, query_ids: list[str],
                    bank_ids: list[str], k: int, block_size: int,
                    exclusion_groups: dict[str, str]):
    """Exact cosine search, bounded-memory query blocks, self/group exclusions."""
    if k <= 0 or block_size <= 0:
        raise ValueError("k and block_size must be positive")
    if query.shape[1] != bank.shape[1]:
        raise ValueError("Query/bank embedding dimension mismatch")
    bank_index = {pid: j for j, pid in enumerate(bank_ids)}
    group_index: dict[str, list[int]] = {}
    for j, pid in enumerate(bank_ids):
        if pid in exclusion_groups:
            group_index.setdefault(exclusion_groups[pid], []).append(j)
    out_idx = np.empty((len(query_ids), k), dtype=np.int32)
    out_scores = np.empty((len(query_ids), k), dtype=np.float32)
    for start in range(0, len(query_ids), block_size):
        end = min(start + block_size, len(query_ids))
        sim = query[start:end] @ bank.T
        for local, pid in enumerate(query_ids[start:end]):
            if pid in bank_index:
                sim[local, bank_index[pid]] = -np.inf
            group = exclusion_groups.get(pid)
            if group is not None:
                sim[local, group_index.get(group, [])] = -np.inf
            valid = np.flatnonzero(np.isfinite(sim[local]))
            if len(valid) < k:
                raise RuntimeError(f"{pid}: only {len(valid)} eligible neighbours (need {k})")
            chosen = valid[np.argpartition(-sim[local, valid], k - 1)[:k]]
            chosen = chosen[np.lexsort((chosen, -sim[local, chosen]))]
            out_idx[start + local] = chosen
            out_scores[start + local] = sim[local, chosen]
        LOG.info("[search] %d/%d queries", end, len(query_ids))
    return out_idx, out_scores


def save_split(out: Path, split: str, ids: list[str], bank_ids: list[str],
               indices: np.ndarray, scores: np.ndarray):
    np.save(out / f"{split}_neighbour_indices.npy", indices)
    np.save(out / f"{split}_neighbour_scores.npy", scores)
    # Keep protein IDs as JSON, avoiding unsafe object-array loading.
    (out / f"{split}_protein_ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    # Each row is aligned to split_protein_ids.json.
    (out / f"{split}_neighbour_ids.json").write_text(
        json.dumps([[bank_ids[int(j)] for j in row] for row in indices], indent=2),
        encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Experiment D ESM-1b neighbour builder")
    p.add_argument("--train_ids", type=Path, required=True,
                   help="Use EXACT filtered Experiment C training IDs")
    p.add_argument("--val_ids", type=Path, required=True,
                   help="Use EXACT filtered Experiment C validation IDs")
    p.add_argument("--embed_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--exclusion_groups", type=Path, default=None,
                   help="Optional JSON protein_id -> sequence/group hash")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    marker = a.out_dir / "metadata.json"
    if marker.exists() and not a.overwrite:
        raise FileExistsError(f"{marker} exists. Pass --overwrite to replace")
    train_ids, val_ids = read_ids(a.train_ids), read_ids(a.val_ids)
    groups = load_exclusion_groups(a.exclusion_groups)
    LOG.info("Train=%d Val=%d k=%d", len(train_ids), len(val_ids), a.k)
    store = ESMResidueStore(str(a.embed_dir), max_len=None, overlap=None)
    bank, train_lengths = embed_ids(store, train_ids, "train", a.log_every)
    val, val_lengths = embed_ids(store, val_ids, "val", a.log_every)
    np.save(a.out_dir / "train_esm_mean_normalized.npy", bank)
    np.save(a.out_dir / "val_esm_mean_normalized.npy", val)
    np.save(a.out_dir / "train_residue_lengths.npy", train_lengths)
    np.save(a.out_dir / "val_residue_lengths.npy", val_lengths)
    (a.out_dir / "bank_protein_ids.json").write_text(json.dumps(train_ids, indent=2), encoding="utf-8")
    for split, ids, matrix in (("train", train_ids, bank), ("val", val_ids, val)):
        idx, score = topk_neighbours(matrix, bank, ids, train_ids, a.k,
                                     a.block_size, groups)
        save_split(a.out_dir, split, ids, train_ids, idx, score)
        LOG.info("[%s] score mean=%.4f min=%.4f max=%.4f", split,
                 float(score.mean()), float(score.min()), float(score.max()))
    meta = {
        "method": "raw_ESM1b_residue_masked_mean_L2_cosine_exact_knn",
        "train_count": len(train_ids), "val_count": len(val_ids),
        "dimension": int(bank.shape[1]), "k": a.k,
        "embedding_dir": str(a.embed_dir),
        "train_ids_file": str(a.train_ids), "val_ids_file": str(a.val_ids),
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "val_ids_sha256": hashlib.sha256("\n".join(val_ids).encode()).hexdigest(),
        "sequence_duplicate_exclusion": a.exclusion_groups is not None,
        "exclusion_groups_file": str(a.exclusion_groups) if a.exclusion_groups else None,
        "scope": "training-bank-only; train self excluded; no GO annotations accessed",
        "length_min_max_train": [int(train_lengths.min()), int(train_lengths.max())],
    }
    marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("DONE -> %s", a.out_dir)


if __name__ == "__main__":
    main()
