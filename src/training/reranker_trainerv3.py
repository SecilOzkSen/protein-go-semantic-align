from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from src.metrics.cafa import compute_term_aupr
from src.models.reranker_model_v3 import SemExpInteractionMLP, SemExpScoreOnlyReranker


# -----------------------------
# GO metadata helpers
# -----------------------------

def _go_to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    if not s:
        return None
    if s.startswith("GO:"):
        s = s.split(":", 1)[1]
    if s.startswith("GO_"):
        s = s.split("_", 1)[1]
    try:
        return int(s)
    except Exception:
        return None


def _norm_ns(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"mf", "mfo", "molecular_function", "molecular function"}:
        return "MF"
    if s in {"bp", "bpo", "biological_process", "biological process"}:
        return "BP"
    if s in {"cc", "cco", "cellular_component", "cellular component"}:
        return "CC"
    return None


def load_go_namespace_map(go_basic_json: str | Path) -> Dict[int, str]:
    path = Path(go_basic_json)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[int, str] = {}

    def add(term: Any, fallback_gid: Any = None):
        if not isinstance(term, dict):
            gid = _go_to_int(fallback_gid)
            ns = _norm_ns(term)
            if gid is not None and ns is not None:
                out[gid] = ns
            return
        gid = _go_to_int(term.get("id") or term.get("go_id") or term.get("GO") or term.get("go") or fallback_gid)
        ns = _norm_ns(term.get("namespace") or term.get("aspect") or term.get("branch") or term.get("ontology"))
        if gid is not None and ns is not None:
            out[gid] = ns

    if isinstance(data, list):
        for term in data:
            add(term)
    elif isinstance(data, dict):
        if isinstance(data.get("terms"), list):
            for term in data["terms"]:
                add(term)
        else:
            for k, v in data.items():
                add(v, fallback_gid=k)
    return out


def load_true_ids_rows(dump_dir: str | Path) -> List[List[int]]:
    dump_dir = Path(dump_dir)
    npy = dump_dir / "true_go_ids.npy"
    js = dump_dir / "true_go_ids.json"
    if npy.exists():
        arr = np.load(npy, mmap_mode="r")
        rows: List[List[int]] = []
        for row in arr:
            rows.append([int(x) for x in row if int(x) >= 0])
        return rows
    if js.exists():
        with js.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [[int(x) for x in row if int(x) >= 0] for row in data]
    raise FileNotFoundError(f"Missing true_go_ids.npy/json in {dump_dir}")


def build_go_meta_bank(
    *,
    eval_go_ids: np.ndarray,
    train_dump: str | Path,
    go_basic_json: str | Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      meta_bank [G,5] = MF/BP/CC one-hot + normalized log-count + normalized IC
      ns_id [G]       = 0 MF, 1 BP, 2 CC, -1 unknown
    """
    eval_go_ids = np.asarray(eval_go_ids, dtype=np.int64)
    id2idx = {int(g): i for i, g in enumerate(eval_go_ids.tolist())}

    counts = np.zeros(len(eval_go_ids), dtype=np.float32)
    for row in load_true_ids_rows(train_dump):
        for gid in set(int(x) for x in row):
            j = id2idx.get(gid)
            if j is not None:
                counts[j] += 1.0

    ns_map = load_go_namespace_map(go_basic_json)
    ns_id = np.full(len(eval_go_ids), -1, dtype=np.int8)
    onehot = np.zeros((len(eval_go_ids), 3), dtype=np.float32)
    for i, gid in enumerate(eval_go_ids.tolist()):
        ns = ns_map.get(int(gid))
        if ns == "MF":
            ns_id[i] = 0
            onehot[i, 0] = 1.0
        elif ns == "BP":
            ns_id[i] = 1
            onehot[i, 1] = 1.0
        elif ns == "CC":
            ns_id[i] = 2
            onehot[i, 2] = 1.0

    log_count = np.log1p(counts)
    if log_count.std() > 0:
        log_count_z = (log_count - log_count.mean()) / (log_count.std() + 1e-6)
    else:
        log_count_z = log_count * 0.0

    # Empirical IC. This is not GO-conditional CAFA IC, but useful as calibration feature.
    n_train = max(1, len(load_true_ids_rows(train_dump)))
    prob = (counts + 1.0) / (float(n_train) + 2.0)
    ic = -np.log2(prob)
    if ic.std() > 0:
        ic_z = (ic - ic.mean()) / (ic.std() + 1e-6)
    else:
        ic_z = ic * 0.0

    meta = np.concatenate([onehot, log_count_z[:, None], ic_z[:, None]], axis=1).astype(np.float32)
    stats = {
        "n_eval_go": int(len(eval_go_ids)),
        "namespace_counts": {
            "MF": int((ns_id == 0).sum()),
            "BP": int((ns_id == 1).sum()),
            "CC": int((ns_id == 2).sum()),
            "UNK": int((ns_id < 0).sum()),
        },
        "train_count_mean": float(counts.mean()),
        "train_count_max": float(counts.max()),
        "log_count_mean": float(log_count.mean()),
        "log_count_std": float(log_count.std()),
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
    }
    return meta, ns_id, stats


# -----------------------------
# Dataset
# -----------------------------

class P3aSemExpCandidateDataset(Dataset):
    def __init__(
        self,
        dump_dir: str | Path,
        topk: int,
        score_mean: float,
        score_std: float,
        go_meta_bank: np.ndarray,
        ns_id_bank: np.ndarray,
        semexp_stats: Dict[str, float],
    ):
        self.dump_dir = Path(dump_dir)
        self.topk = int(topk)
        self.score_mean = float(score_mean)
        self.score_std = float(score_std) if score_std > 0 else 1.0

        self.eval_go_ids = np.load(self.dump_dir / "eval_go_ids.npy", mmap_mode="r")
        self.go_z = np.load(self.dump_dir / "go_z.float16.npy", mmap_mode="r")
        self.protein_z = np.load(self.dump_dir / "protein_z.float16.npy", mmap_mode="r")
        self.top_cols = np.load(self.dump_dir / "top_go_cols.int32.npy", mmap_mode="r")
        self.top_scores = np.load(self.dump_dir / "top_scores.float32.npy", mmap_mode="r")
        self.top_labels = np.load(self.dump_dir / "top_labels.int8.npy", mmap_mode="r")
        self.true_go_ids = np.load(self.dump_dir / "true_go_ids.npy", mmap_mode="r")

        valid_path = self.dump_dir / "top_valid.int8.npy"
        self.top_valid = np.load(valid_path, mmap_mode="r") if valid_path.exists() else None

        self.semexp_stats = semexp_stats or {"seed_score_mean": 0.0, "seed_score_std": 1.0}
        arr_shape = self.top_scores.shape
        self.direct_p3a = _load_optional_matrix(self.dump_dir, "direct_p3a.int8.npy", arr_shape, np.int8, 1)
        self.parent_expansion = _load_optional_matrix(self.dump_dir, "parent_expansion.int8.npy", arr_shape, np.int8, 0)
        self.child_expansion = _load_optional_matrix(self.dump_dir, "child_expansion.int8.npy", arr_shape, np.int8, 0)
        self.sibling_expansion = _load_optional_matrix(self.dump_dir, "sibling_expansion.int8.npy", arr_shape, np.int8, 0)
        self.text_neighbor_expansion = _load_optional_matrix(self.dump_dir, "text_neighbor_expansion.int8.npy", arr_shape, np.int8, 0)
        self.seed_score = _load_optional_matrix(self.dump_dir, "seed_score.float32.npy", arr_shape, np.float32, 0.0)
        self.seed_rank = _load_optional_matrix(self.dump_dir, "seed_rank.int32.npy", arr_shape, np.int32, -1)
        self.relation_distance = _load_optional_matrix(self.dump_dir, "relation_distance.int16.npy", arr_shape, np.int16, -1)
        self.text_neighbor_sim = _load_optional_matrix(self.dump_dir, "text_neighbor_sim.float32.npy", arr_shape, np.float32, 0.0)

        pids_path = self.dump_dir / "protein_ids.json"
        if pids_path.exists():
            with pids_path.open("r", encoding="utf-8") as f:
                self.protein_ids = json.load(f)
        else:
            self.protein_ids = [f"row_{i}" for i in range(self.top_scores.shape[0])]

        if self.topk > self.top_scores.shape[1]:
            raise ValueError(f"Requested topk={self.topk}, but dump has only {self.top_scores.shape[1]}")

        self.go_meta_bank = np.asarray(go_meta_bank, dtype=np.float32)
        self.ns_id_bank = np.asarray(ns_id_bank, dtype=np.int8)

        self.rank_feature = (
            np.log1p(np.arange(1, self.topk + 1, dtype=np.float32)) / np.log1p(float(self.topk))
        ).astype(np.float32)

    def __len__(self) -> int:
        return int(self.top_scores.shape[0])

    @property
    def dim(self) -> int:
        return int(self.protein_z.shape[1])

    @property
    def n_go(self) -> int:
        return int(self.eval_go_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cols = np.asarray(self.top_cols[idx, : self.topk], dtype=np.int64)
        scores = np.asarray(self.top_scores[idx, : self.topk], dtype=np.float32)
        labels = np.asarray(self.top_labels[idx, : self.topk], dtype=np.float32)

        if self.top_valid is not None:
            valid = np.asarray(self.top_valid[idx, : self.topk], dtype=np.float32)
        else:
            valid = np.ones(self.topk, dtype=np.float32)

        scores = (scores - self.score_mean) / max(self.score_std, 1e-6)
        scores = np.where(valid > 0, scores, 0.0).astype(np.float32)

        direct = np.asarray(self.direct_p3a[idx, : self.topk], dtype=np.float32) * valid
        parent = np.asarray(self.parent_expansion[idx, : self.topk], dtype=np.float32) * valid
        child = np.asarray(self.child_expansion[idx, : self.topk], dtype=np.float32) * valid
        sibling = np.asarray(self.sibling_expansion[idx, : self.topk], dtype=np.float32) * valid
        text_flag = np.asarray(self.text_neighbor_expansion[idx, : self.topk], dtype=np.float32) * valid
        expanded = np.where((valid > 0) & (direct <= 0), 1.0, 0.0).astype(np.float32)

        seed_score = np.asarray(self.seed_score[idx, : self.topk], dtype=np.float32)
        seed_score = (seed_score - self.semexp_stats["seed_score_mean"]) / max(self.semexp_stats["seed_score_std"], 1e-6)
        seed_score = np.where(valid > 0, seed_score, 0.0).astype(np.float32)

        seed_rank = np.asarray(self.seed_rank[idx, : self.topk], dtype=np.float32)
        seed_rank = np.where(seed_rank >= 0, seed_rank, float(self.topk))
        seed_rank_feat = (np.log1p(seed_rank + 1.0) / np.log1p(float(self.topk + 1))).astype(np.float32)
        seed_rank_feat = np.where(valid > 0, seed_rank_feat, 0.0).astype(np.float32)

        rel_dist = np.asarray(self.relation_distance[idx, : self.topk], dtype=np.float32)
        rel_dist = np.where(rel_dist >= 0, np.minimum(rel_dist, 5.0) / 5.0, 0.0).astype(np.float32)
        rel_dist = np.where(valid > 0, rel_dist, 0.0).astype(np.float32)

        text_sim = np.asarray(self.text_neighbor_sim[idx, : self.topk], dtype=np.float32)
        text_sim = np.where(valid > 0, text_sim, 0.0).astype(np.float32)

        semexp_feat = np.stack(
            [direct, parent, child, sibling, text_flag, expanded, seed_score, seed_rank_feat, rel_dist, text_sim],
            axis=-1,
        ).astype(np.float32)

        item = {
            "protein_id": str(self.protein_ids[idx]),
            "protein_z": torch.from_numpy(np.asarray(self.protein_z[idx], dtype=np.float32).copy()),
            "go_z": torch.from_numpy(np.asarray(self.go_z[cols], dtype=np.float32).copy()),
            "retriever_score": torch.from_numpy(scores.copy()),
            "rank_feature": torch.from_numpy(self.rank_feature.copy()),
            "go_meta": torch.from_numpy(self.go_meta_bank[cols].copy()),
            "semexp_feat": torch.from_numpy(semexp_feat.copy()),
            "label": torch.from_numpy(labels.copy()),
            "valid_mask": torch.from_numpy(valid.copy()),
            "top_cols": torch.from_numpy(cols.copy()),
            "top_ids": torch.from_numpy(np.asarray(self.eval_go_ids[cols], dtype=np.int64).copy()),
            "true_go_ids": torch.from_numpy(np.asarray(self.true_go_ids[idx], dtype=np.int64).copy()),
            "ns_id": torch.from_numpy(self.ns_id_bank[cols].astype(np.int64).copy()),
        }
        return item


def estimate_score_stats(dump_dir: str | Path, topk: int, max_rows: Optional[int] = None) -> Tuple[float, float]:
    dump_dir = Path(dump_dir)
    scores = np.load(dump_dir / "top_scores.float32.npy", mmap_mode="r")
    n = scores.shape[0] if max_rows is None else min(int(max_rows), scores.shape[0])
    arr = np.asarray(scores[:n, :topk], dtype=np.float32)

    valid_path = dump_dir / "top_valid.int8.npy"
    if valid_path.exists():
        valid = np.asarray(np.load(valid_path, mmap_mode="r")[:n, :topk], dtype=bool)
    else:
        valid = np.ones_like(arr, dtype=bool)

    mask = valid & np.isfinite(arr) & (arr > -1e5)
    if not mask.any():
        return float(arr.mean()), float(arr.std() + 1e-6)
    vals = arr[mask]
    return float(vals.mean()), float(vals.std() + 1e-6)



def _load_optional_matrix(dump_dir: Path, filename: str, shape: Tuple[int, int], dtype: Any, fill: float = 0.0) -> np.ndarray:
    path = dump_dir / filename
    if path.exists():
        return np.load(path, mmap_mode="r")
    # Returned arrays are only used through row slices. A dense fallback is okay for non-SemExp dumps.
    return np.full(shape, fill, dtype=dtype)


def estimate_semexp_stats(dump_dir: str | Path, topk: int, max_rows: Optional[int] = None) -> Dict[str, float]:
    """Stats for semantic-expansion seed scores, computed over valid candidates."""
    dump_dir = Path(dump_dir)
    scores = np.load(dump_dir / "top_scores.float32.npy", mmap_mode="r")
    n = scores.shape[0] if max_rows is None else min(int(max_rows), scores.shape[0])
    shape = (n, int(topk))

    valid_path = dump_dir / "top_valid.int8.npy"
    if valid_path.exists():
        valid = np.asarray(np.load(valid_path, mmap_mode="r")[:n, :topk], dtype=bool)
    else:
        valid = np.ones(shape, dtype=bool)

    seed_score_path = dump_dir / "seed_score.float32.npy"
    if seed_score_path.exists():
        seed = np.asarray(np.load(seed_score_path, mmap_mode="r")[:n, :topk], dtype=np.float32)
        mask = valid & np.isfinite(seed) & (seed > -1e5)
        if mask.any():
            vals = seed[mask]
            seed_mean = float(vals.mean())
            seed_std = float(vals.std() + 1e-6)
        else:
            seed_mean, seed_std = 0.0, 1.0
    else:
        seed_mean, seed_std = 0.0, 1.0

    return {"seed_score_mean": seed_mean, "seed_score_std": seed_std}

def estimate_pos_weight(dump_dir: str | Path, topk: int, max_value: float = 50.0) -> float:
    dump_dir = Path(dump_dir)
    labels = np.load(dump_dir / "top_labels.int8.npy", mmap_mode="r")[:, :topk]
    valid_path = dump_dir / "top_valid.int8.npy"
    if valid_path.exists():
        valid = np.load(valid_path, mmap_mode="r")[:, :topk].astype(bool)
    else:
        valid = np.ones_like(labels, dtype=bool)
    y = labels[valid].astype(np.int64)
    pos = int(y.sum())
    total = int(y.size)
    if pos <= 0:
        return 1.0
    neg = max(1, total - pos)
    return float(min(max_value, neg / max(1, pos)))


def collate_candidate_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"protein_id": [x["protein_id"] for x in items]}
    tensor_keys = [
        "protein_z", "go_z", "retriever_score", "rank_feature", "go_meta", "semexp_feat", "label",
        "valid_mask", "top_cols", "top_ids", "true_go_ids", "ns_id",
    ]
    for k in tensor_keys:
        out[k] = torch.stack([x[k] for x in items], dim=0)
    return out


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def build_model(kind: str, dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    if kind in {"semexp_interaction_mlp", "branch_pairwise_interaction_mlp"}:
        return SemExpInteractionMLP(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
    if kind in {"semexp_score_only", "branch_score_only"}:
        return SemExpScoreOnlyReranker(hidden_dim=max(32, hidden_dim // 4), dropout=dropout)
    raise ValueError(f"Unknown model_kind: {kind}")


# -----------------------------
# Metrics
# -----------------------------

def compute_fmax_and_aupr(y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 501) -> Dict[str, float]:
    y_true = y_true.astype(np.int32, copy=False)
    y_score = y_score.astype(np.float32, copy=False)
    finite = np.isfinite(y_score)
    if not finite.any():
        return {"fmax": 0.0, "aupr": 0.0, "best_threshold": 0.0, "precision": 0.0, "recall": 0.0}

    s_min = float(y_score[finite].min())
    s_max = float(y_score[finite].max())
    if s_min == s_max:
        return {"fmax": 0.0, "aupr": 0.0, "best_threshold": s_min, "precision": 0.0, "recall": 0.0}

    thresholds = np.linspace(s_min, s_max, int(n_thresholds), dtype=np.float32)
    one_minus = 1 - y_true
    best = {"fmax": 0.0, "best_threshold": float(thresholds[0]), "precision": 0.0, "recall": 0.0}

    for t in thresholds:
        y_hat = (y_score >= float(t)).astype(np.int32)
        tp = int((y_hat & y_true).sum())
        fp = int((y_hat & one_minus).sum())
        fn = int(((1 - y_hat) & y_true).sum())
        prec = tp / max(1e-12, tp + fp)
        rec = tp / max(1e-12, tp + fn)
        f = (2.0 * prec * rec) / max(1e-12, prec + rec)
        if f > best["fmax"]:
            best = {"fmax": float(f), "best_threshold": float(t), "precision": float(prec), "recall": float(rec)}

    try:
        aupr = float(compute_term_aupr(y_true, y_score))
    except Exception:
        aupr = 0.0
    best["aupr"] = aupr
    return best


def branch_calibrated_fmax(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ns_id: np.ndarray,
    branch_thresholds: Dict[int, float],
) -> Dict[str, float]:
    y_hat = np.zeros_like(y_true, dtype=np.int32)
    for b, t in branch_thresholds.items():
        cols = ns_id == int(b)
        if cols.any():
            y_hat[:, cols] = (y_score[:, cols] >= float(t)).astype(np.int32)

    tp = int((y_hat & y_true).sum())
    fp = int((y_hat & (1 - y_true)).sum())
    fn = int(((1 - y_hat) & y_true).sum())
    prec = tp / max(1e-12, tp + fp)
    rec = tp / max(1e-12, tp + fn)
    f = (2.0 * prec * rec) / max(1e-12, prec + rec)
    return {"branch_calibrated_fmax": float(f), "branch_calibrated_precision": float(prec), "branch_calibrated_recall": float(rec)}


@torch.no_grad()
def evaluate_reranker(
    model: nn.Module,
    loader: DataLoader,
    dataset: P3aSemExpCandidateDataset,
    device: torch.device,
    ns_id_bank: np.ndarray,
    n_thresholds: int = 501,
    non_candidate_margin: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    all_valid: List[np.ndarray] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        logits = model(
            protein_z=batch["protein_z"],
            go_z=batch["go_z"],
            retriever_score=batch["retriever_score"],
            rank_feature=batch["rank_feature"],
            go_meta=batch["go_meta"],
            semexp_feat=batch["semexp_feat"],
        )
        all_logits.append(logits.detach().cpu().float().numpy())
        all_cols.append(batch["top_cols"].detach().cpu().numpy())
        all_true.append(batch["true_go_ids"].detach().cpu().numpy())
        all_valid.append(batch["valid_mask"].detach().cpu().numpy().astype(bool))

    cand_logits = np.concatenate(all_logits, axis=0).astype(np.float32)
    top_cols = np.concatenate(all_cols, axis=0).astype(np.int64)
    true_ids = np.concatenate(all_true, axis=0).astype(np.int64)
    valid_mask = np.concatenate(all_valid, axis=0).astype(bool)

    valid_scores = cand_logits[np.isfinite(cand_logits) & valid_mask]
    if valid_scores.size == 0:
        raise RuntimeError("No valid candidate scores during eval.")
    fill_value = float(valid_scores.min() - non_candidate_margin)
    cand_logits = np.where(valid_mask & np.isfinite(cand_logits), cand_logits, fill_value)

    N = cand_logits.shape[0]
    G = dataset.n_go
    y_score = np.full((N, G), fill_value, dtype=np.float32)
    y_true = np.zeros((N, G), dtype=np.int8)

    rows = np.arange(N)[:, None]
    y_score[rows, top_cols] = cand_logits

    eval_go_ids = np.asarray(dataset.eval_go_ids, dtype=np.int64)
    id2col = {int(g): i for i, g in enumerate(eval_go_ids.tolist())}
    for i in range(N):
        for gid in true_ids[i]:
            gid = int(gid)
            if gid < 0:
                continue
            j = id2col.get(gid)
            if j is not None:
                y_true[i, j] = 1

    metrics = compute_fmax_and_aupr(y_true, y_score, n_thresholds=n_thresholds)

    # Branch-specific Fmax/AUPR and branch-calibrated global Fmax.
    branch_thresholds: Dict[int, float] = {}
    for bid, name in [(0, "mf"), (1, "bp"), (2, "cc")]:
        cols = ns_id_bank == bid
        if cols.any() and y_true[:, cols].sum() > 0:
            m = compute_fmax_and_aupr(y_true[:, cols], y_score[:, cols], n_thresholds=n_thresholds)
            metrics[f"{name}_fmax"] = m["fmax"]
            metrics[f"{name}_aupr"] = m["aupr"]
            metrics[f"{name}_best_threshold"] = m["best_threshold"]
            branch_thresholds[bid] = m["best_threshold"]
        else:
            metrics[f"{name}_fmax"] = 0.0
            metrics[f"{name}_aupr"] = 0.0
            metrics[f"{name}_best_threshold"] = 0.0

    if branch_thresholds:
        metrics.update(branch_calibrated_fmax(y_true, y_score, ns_id_bank, branch_thresholds))

    # Candidate diagnostics.
    labels = np.asarray(dataset.top_labels[:, : dataset.topk], dtype=np.int8)
    valid = np.asarray(dataset.top_valid[:, : dataset.topk], dtype=np.int8) if dataset.top_valid is not None else np.ones_like(labels)
    true_counts = (np.asarray(dataset.true_go_ids) >= 0).sum(axis=1).clip(min=1)
    hits = (labels * valid).sum(axis=1)
    coverage = float(np.mean(hits / true_counts))
    oracle = float((2.0 * hits.sum()) / max(1e-12, 2.0 * hits.sum() + (true_counts - hits).sum()))
    metrics.update({"candidate_coverage": coverage, "oracle_microF": oracle, "num_samples": int(N), "num_go": int(G), "topk": int(dataset.topk)})
    return metrics


# -----------------------------
# Training
# -----------------------------

def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor, hard_neg_k: int = 64) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    B = logits.size(0)
    for i in range(B):
        v = valid[i].bool()
        pos = logits[i][v & (labels[i] > 0.5)]
        neg = logits[i][v & (labels[i] <= 0.5)]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        if neg.numel() > hard_neg_k:
            neg = torch.topk(neg, k=int(hard_neg_k), largest=True).values
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        losses.append(F.softplus(-diff).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    lambda_pair: float,
    hard_neg_k: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_pair = 0.0
    total_valid = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        logits = model(
            protein_z=batch["protein_z"],
            go_z=batch["go_z"],
            retriever_score=batch["retriever_score"],
            rank_feature=batch["rank_feature"],
            go_meta=batch["go_meta"],
            semexp_feat=batch["semexp_feat"],
        )
        labels = batch["label"].float()
        valid = batch["valid_mask"].float()

        bce_mat = criterion(logits, labels)
        bce = (bce_mat * valid).sum() / valid.sum().clamp_min(1.0)
        pair = pairwise_ranking_loss(logits, labels, valid, hard_neg_k=hard_neg_k)
        loss = bce + float(lambda_pair) * pair

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        n = float(valid.sum().detach().item())
        total_valid += n
        total_loss += float(loss.detach().item()) * n
        total_bce += float(bce.detach().item()) * n
        total_pair += float(pair.detach().item()) * n

    denom = max(1.0, total_valid)
    return {"train_loss": total_loss / denom, "train_bce": total_bce / denom, "train_pair": total_pair / denom}


@dataclass
class P3aSemExpRerankerConfig:
    train_dump: str
    val_dump: str
    test_dump: Optional[str]
    out_dir: str
    go_basic_json: str = "/workspace/data/go_vocab.json"
    topk: int = 1000
    model_kind: str = "semexp_interaction_mlp"
    hidden_dim: int = 512
    dropout: float = 0.10
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    patience: int = 2
    grad_clip: float = 1.0
    pos_weight_max: float = 50.0
    lambda_pair: float = 0.25
    hard_neg_k: int = 64
    n_thresholds: int = 501
    monitor: str = "fmax"
    device: str = "cuda:0"


class P3aSemExpRerankerTrainer:
    def __init__(self, cfg: P3aSemExpRerankerConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.score_mean, self.score_std = estimate_score_stats(cfg.train_dump, cfg.topk)
        self.semexp_stats = estimate_semexp_stats(cfg.train_dump, cfg.topk)
        eval_go_ids = np.load(Path(cfg.train_dump) / "eval_go_ids.npy", mmap_mode="r").astype(np.int64)
        self.go_meta_bank, self.ns_id_bank, self.meta_stats = build_go_meta_bank(
            eval_go_ids=eval_go_ids,
            train_dump=cfg.train_dump,
            go_basic_json=cfg.go_basic_json,
        )

        self.train_ds = P3aSemExpCandidateDataset(cfg.train_dump, cfg.topk, self.score_mean, self.score_std, self.go_meta_bank, self.ns_id_bank, self.semexp_stats)
        self.val_ds = P3aSemExpCandidateDataset(cfg.val_dump, cfg.topk, self.score_mean, self.score_std, self.go_meta_bank, self.ns_id_bank, self.semexp_stats)
        self.test_ds = P3aSemExpCandidateDataset(cfg.test_dump, cfg.topk, self.score_mean, self.score_std, self.go_meta_bank, self.ns_id_bank, self.semexp_stats) if cfg.test_dump else None

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_candidate_batch,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_candidate_batch,
        )
        self.test_loader = None
        if self.test_ds is not None:
            self.test_loader = DataLoader(
                self.test_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                collate_fn=collate_candidate_batch,
            )

        self.model = build_model(cfg.model_kind, self.train_ds.dim, cfg.hidden_dim, cfg.dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.pos_weight = estimate_pos_weight(cfg.train_dump, cfg.topk, cfg.pos_weight_max)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight, device=self.device), reduction="none")

        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.history: List[Dict[str, float]] = []

        cfg_dump = asdict(cfg)
        cfg_dump.update({"score_mean": self.score_mean, "score_std": self.score_std, "pos_weight": self.pos_weight, "meta_stats": self.meta_stats, "semexp_stats": self.semexp_stats})
        with (self.out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg_dump, f, indent=2)

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": int(epoch),
                "metrics": metrics,
                "score_mean": self.score_mean,
                "score_std": self.score_std,
                "pos_weight": self.pos_weight,
                "meta_stats": self.meta_stats,
                "semexp_stats": self.semexp_stats,
                "config": asdict(self.cfg),
            },
            path,
        )

    def fit(self):
        print(f"[p3a-semexp-reranker] device={self.device}")
        print(f"[p3a-semexp-reranker] model={self.cfg.model_kind} topk={self.cfg.topk}")
        print(f"[p3a-semexp-reranker] score_mean={self.score_mean:.4f} score_std={self.score_std:.4f} pos_weight={self.pos_weight:.2f}")
        print(f"[p3a-semexp-reranker] lambda_pair={self.cfg.lambda_pair} hard_neg_k={self.cfg.hard_neg_k}")
        print(f"[p3a-semexp-reranker] meta_stats={self.meta_stats}")
        print(f"[p3a-semexp-reranker] semexp_stats={self.semexp_stats}")

        for epoch in range(self.cfg.epochs):
            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                grad_clip=self.cfg.grad_clip,
                lambda_pair=self.cfg.lambda_pair,
                hard_neg_k=self.cfg.hard_neg_k,
            )
            val_metrics = evaluate_reranker(
                self.model,
                self.val_loader,
                self.val_ds,
                self.device,
                ns_id_bank=self.ns_id_bank,
                n_thresholds=self.cfg.n_thresholds,
            )
            val_metrics.update(train_metrics)
            val_metrics["epoch"] = float(epoch)
            self.history.append(val_metrics)

            print(
                "[val] epoch", epoch,
                "| loss", f"{train_metrics['train_loss']:.5f}",
                "| bce", f"{train_metrics['train_bce']:.5f}",
                "| pair", f"{train_metrics['train_pair']:.5f}",
                "| fmax", f"{val_metrics['fmax']:.4f}",
                "| bc_fmax", f"{val_metrics.get('branch_calibrated_fmax', 0.0):.4f}",
                "| aupr", f"{val_metrics['aupr']:.4f}",
                "| MF", f"{val_metrics.get('mf_fmax', 0.0):.4f}",
                "| BP", f"{val_metrics.get('bp_fmax', 0.0):.4f}",
                "| CC", f"{val_metrics.get('cc_fmax', 0.0):.4f}",
                "| oracle", f"{val_metrics['oracle_microF']:.4f}",
                "| coverage", f"{val_metrics['candidate_coverage']:.4f}",
            )

            monitor_value = float(val_metrics.get(self.cfg.monitor, val_metrics["fmax"]))
            if monitor_value > self.best_metric:
                self.best_metric = monitor_value
                self.best_epoch = epoch
                self.bad_epochs = 0
                self.save_checkpoint(self.out_dir / "best.pt", epoch, val_metrics)
                print(f"[p3a-semexp-reranker] new best {self.cfg.monitor}={monitor_value:.4f} at epoch={epoch}")
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.cfg.patience:
                    print(f"[p3a-semexp-reranker] early stop at epoch={epoch}, best_epoch={self.best_epoch}")
                    break

            with (self.out_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

        if self.test_loader is not None:
            best = torch.load(self.out_dir / "best.pt", map_location=self.device, weights_only=False)
            self.model.load_state_dict(best["model"])
            test_metrics = evaluate_reranker(
                self.model,
                self.test_loader,
                self.test_ds,
                self.device,
                ns_id_bank=self.ns_id_bank,
                n_thresholds=self.cfg.n_thresholds,
            )
            print("[test]", json.dumps(test_metrics, indent=2))
            with (self.out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)
