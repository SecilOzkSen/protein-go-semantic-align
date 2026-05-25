from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from src.metrics.cafa import compute_term_aupr
from src.models.reranker_modelv2 import CandidateInteractionMLP, ScoreOnlyReranker


@dataclass
class DumpMetadata:
    dump_dir: str
    n_samples: int
    n_go: int
    topk_available: int
    dim: int


class CandidateDumpDataset(Dataset):
    """
    Protein-level dataset backed by clean candidate dump files.

    Required files:
      eval_go_ids.npy
      go_z.float16.npy
      protein_z.float16.npy
      top_go_cols.int32.npy
      top_scores.float32.npy
      top_labels.int8.npy

    Optional but recommended:
      protein_ids.json
      true_go_ids.npy

    top_go_ids.int64.npy is NOT required.
    GO ids are reconstructed as eval_go_ids[top_go_cols].
    """

    def __init__(
        self,
        dump_dir: str | Path,
        topk: int,
        score_mean: Optional[float] = None,
        score_std: Optional[float] = None,
    ):
        self.dump_dir = Path(dump_dir)
        self.topk = int(topk)

        self.eval_go_ids = np.load(self.dump_dir / "eval_go_ids.npy", mmap_mode="r")
        self.go_z = np.load(self.dump_dir / "go_z.float16.npy", mmap_mode="r")
        self.protein_z = np.load(self.dump_dir / "protein_z.float16.npy", mmap_mode="r")
        self.top_cols = np.load(self.dump_dir / "top_go_cols.int32.npy", mmap_mode="r")
        self.top_scores = np.load(self.dump_dir / "top_scores.float32.npy", mmap_mode="r")
        self.top_labels = np.load(self.dump_dir / "top_labels.int8.npy", mmap_mode="r")

        protein_ids_path = self.dump_dir / "protein_ids.json"
        if protein_ids_path.exists():
            with protein_ids_path.open("r", encoding="utf-8") as f:
                self.protein_ids = json.load(f)
        else:
            self.protein_ids = [f"row_{i}" for i in range(int(self.top_scores.shape[0]))]

        true_path = self.dump_dir / "true_go_ids.npy"
        if true_path.exists():
            self.true_go_ids = np.load(true_path, mmap_mode="r")
        else:
            self.true_go_ids = None

        valid_path = self.dump_dir / "top_valid.int8.npy"
        if valid_path.exists():
            self.top_valid = np.load(valid_path, mmap_mode="r")
        else:
            self.top_valid = None

        if self.topk > self.top_scores.shape[1]:
            raise ValueError(
                f"Requested topk={self.topk}, but dump has only {self.top_scores.shape[1]} candidates."
            )

        if self.top_cols.shape != self.top_scores.shape:
            raise RuntimeError(
                f"top_go_cols shape {self.top_cols.shape} != top_scores shape {self.top_scores.shape}"
            )

        if self.top_labels.shape != self.top_scores.shape:
            raise RuntimeError(
                f"top_labels shape {self.top_labels.shape} != top_scores shape {self.top_scores.shape}"
            )

        if self.protein_z.shape[0] != self.top_scores.shape[0]:
            raise RuntimeError(
                f"protein_z rows {self.protein_z.shape[0]} != top_scores rows {self.top_scores.shape[0]}"
            )

        if len(self.protein_ids) != self.top_scores.shape[0]:
            raise RuntimeError(
                f"protein_ids length {len(self.protein_ids)} != top_scores rows {self.top_scores.shape[0]}"
            )

        self.score_mean = float(score_mean) if score_mean is not None else 0.0
        self.score_std = float(score_std) if score_std is not None and score_std > 0 else 1.0

        self.rank_feature = (
            np.log1p(np.arange(1, self.topk + 1, dtype=np.float32))
            / np.log1p(float(self.topk))
        ).astype(np.float32)

    def __len__(self) -> int:
        return int(self.top_scores.shape[0])

    @property
    def dim(self) -> int:
        return int(self.protein_z.shape[1])

    @property
    def n_go(self) -> int:
        return int(self.eval_go_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        cols = np.asarray(self.top_cols[idx, : self.topk], dtype=np.int64)

        # Reconstruct global GO ids from eval_go_ids and top columns.
        top_ids = np.asarray(self.eval_go_ids[cols], dtype=np.int64)

        pz = np.asarray(self.protein_z[idx], dtype=np.float32)
        gz = np.asarray(self.go_z[cols], dtype=np.float32)

        scores = np.asarray(self.top_scores[idx, : self.topk], dtype=np.float32)
        labels = np.asarray(self.top_labels[idx, : self.topk], dtype=np.float32)

        scores = (scores - self.score_mean) / max(self.score_std, 1e-6)

        item: Dict[str, torch.Tensor | str] = {
            "protein_id": str(self.protein_ids[idx]),
            "protein_z": torch.from_numpy(pz.copy()),
            "go_z": torch.from_numpy(gz.copy()),
            "retriever_score": torch.from_numpy(scores.copy()),
            "rank_feature": torch.from_numpy(self.rank_feature.copy()),
            "label": torch.from_numpy(labels.copy()),
            "top_cols": torch.from_numpy(cols.astype(np.int64)),
            "top_ids": torch.from_numpy(top_ids.astype(np.int64)),
        }
        if self.top_valid is not None:
            valid = np.asarray(self.top_valid[idx, : self.topk], dtype=np.float32)
        else:
            valid = np.ones(self.topk, dtype=np.float32)

        item["valid_mask"] = torch.from_numpy(valid.copy())

        if self.true_go_ids is not None:
            item["true_go_ids"] = torch.from_numpy(
                np.asarray(self.true_go_ids[idx], dtype=np.int64).copy()
            )

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

    finite = np.isfinite(arr)
    not_filler = arr > -1e5
    mask = valid & finite & not_filler

    if not mask.any():
        return float(arr.mean()), float(arr.std() + 1e-6)

    vals = arr[mask]
    return float(vals.mean()), float(vals.std() + 1e-6)


def estimate_pos_weight(dump_dir: str | Path, topk: int, max_value: float = 50.0) -> float:
    labels = np.load(Path(dump_dir) / "top_labels.int8.npy", mmap_mode="r")
    y = np.asarray(labels[:, :topk], dtype=np.int64)
    pos = int(y.sum())
    total = int(y.size)
    neg = max(1, total - pos)
    if pos <= 0:
        return 1.0
    return float(min(max_value, neg / max(1, pos)))


def collate_candidate_batch(items: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
    out: Dict[str, torch.Tensor | List[str]] = {}
    out["protein_id"] = [x["protein_id"] for x in items]

    tensor_keys = [
        "protein_z",
        "go_z",
        "retriever_score",
        "rank_feature",
        "label",
        "top_cols",
        "top_ids",
        "true_go_ids",
        "valid_mask",
    ]

    for k in tensor_keys:
        if k in items[0]:
            out[k] = torch.stack([x[k] for x in items], dim=0)

    return out


def build_model(kind: str, dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    if kind == "score_only":
        return ScoreOnlyReranker(hidden_dim=max(16, hidden_dim // 16), dropout=dropout)
    if kind == "interaction_mlp":
        return CandidateInteractionMLP(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_embeddings=True,
            use_score=True,
            use_rank=True,
        )
    if kind == "embedding_only":
        return CandidateInteractionMLP(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_embeddings=True,
            use_score=False,
            use_rank=False,
        )
    raise ValueError(f"Unknown reranker kind: {kind}")


def move_batch(batch: Dict[str, torch.Tensor | List[str]], device: torch.device) -> Dict[str, torch.Tensor | List[str]]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def compute_fmax_and_aupr(y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 101) -> Dict[str, float]:
    y_true = y_true.astype(np.int32, copy=False)
    y_score = y_score.astype(np.float32, copy=False)

    finite = np.isfinite(y_score)
    if not finite.any():
        return {"fmax": 0.0, "aupr": 0.0, "best_threshold": 0.0, "precision": 0.0, "recall": 0.0}

    s_min = float(y_score[finite].min())
    s_max = float(y_score[finite].max())
    if s_min == s_max:
        return {"fmax": 0.0, "aupr": 0.0, "best_threshold": s_min, "precision": 0.0, "recall": 0.0}

    thresholds = np.linspace(s_min, s_max, n_thresholds, dtype=np.float32)

    best = {"fmax": 0.0, "best_threshold": float(thresholds[0]), "precision": 0.0, "recall": 0.0}
    one_minus = 1 - y_true

    for t in thresholds:
        y_hat = (y_score >= t).astype(np.int32)
        tp = int((y_hat & y_true).sum())
        fp = int((y_hat & one_minus).sum())
        fn = int(((1 - y_hat) & y_true).sum())

        prec = tp / max(1e-12, tp + fp)
        rec = tp / max(1e-12, tp + fn)
        f = (2.0 * prec * rec) / max(1e-12, prec + rec)
        if f > best["fmax"]:
            best = {
                "fmax": float(f),
                "best_threshold": float(t),
                "precision": float(prec),
                "recall": float(rec),
            }

    if compute_term_aupr is not None:
        try:
            aupr = float(compute_term_aupr(y_true, y_score))
        except Exception:
            aupr = 0.0
    else:
        aupr = 0.0

    best["aupr"] = aupr
    return best


@torch.no_grad()
def evaluate_reranker(
    model: nn.Module,
    loader: DataLoader,
    dataset: CandidateDumpDataset,
    device: torch.device,
    non_candidate_margin: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    all_true: List[np.ndarray] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        logits = model(
            protein_z=batch["protein_z"],
            go_z=batch["go_z"],
            retriever_score=batch["retriever_score"],
            rank_feature=batch["rank_feature"],
        )
        if "valid_mask" in batch:
            valid = batch["valid_mask"].bool()
            logits = logits.masked_fill(~valid, -1e6)
        all_logits.append(logits.detach().cpu().float().numpy())
        all_cols.append(batch["top_cols"].detach().cpu().numpy())
        if "true_go_ids" not in batch:
            raise RuntimeError(
                "Batch is missing true_go_ids. "
                "Check that true_go_ids.npy exists in the dump directory and "
                "collate_candidate_batch includes 'true_go_ids'."
            )

        all_true.append(batch["true_go_ids"].detach().cpu().numpy())

    cand_logits = np.concatenate(all_logits, axis=0).astype(np.float32)
    top_cols = np.concatenate(all_cols, axis=0).astype(np.int64)
    true_ids = np.concatenate(all_true, axis=0).astype(np.int64)

    N = cand_logits.shape[0]
    G = dataset.n_go
    fill_value = float(np.nanmin(cand_logits) - non_candidate_margin)

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
            j = id2col.get(gid, None)
            if j is not None:
                y_true[i, j] = 1

    metrics = compute_fmax_and_aupr(y_true, y_score)

    # Candidate-label diagnostics on the selected K.
    labels = np.asarray(dataset.top_labels[:, : dataset.topk], dtype=np.int8)
    true_counts = (np.asarray(dataset.true_go_ids) >= 0).sum(axis=1).clip(min=1)
    hits = labels.sum(axis=1)
    coverage = float(np.mean(hits / true_counts))
    oracle_micro_f = float((2.0 * hits.sum()) / max(1e-12, 2.0 * hits.sum() + (true_counts - hits).sum()))

    metrics.update({
        "candidate_coverage": coverage,
        "oracle_microF": oracle_micro_f,
        "num_samples": int(N),
        "num_go": int(G),
        "topk": int(dataset.topk),
    })
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        logits = model(
            protein_z=batch["protein_z"],
            go_z=batch["go_z"],
            retriever_score=batch["retriever_score"],
            rank_feature=batch["rank_feature"],
        )
        labels = batch["label"].float()

        loss_mat = criterion(logits, labels)

        if "valid_mask" in batch:
            valid = batch["valid_mask"].float()
            loss = (loss_mat * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            loss = loss_mat.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n = int(labels.numel())
        total_loss += float(loss.detach().item()) * n
        total_items += n

    return total_loss / max(1, total_items)


@dataclass
class RerankerTrainConfig:
    train_dump: str
    val_dump: str
    test_dump: Optional[str]
    out_dir: str
    topk: int = 500
    model_kind: str = "interaction_mlp"
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
    monitor: str = "fmax"
    device: str = "cuda:0"


class RerankerTrainer:
    def __init__(self, cfg: RerankerTrainConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        score_mean, score_std = estimate_score_stats(cfg.train_dump, cfg.topk)
        self.score_mean = score_mean
        self.score_std = score_std

        self.train_ds = CandidateDumpDataset(cfg.train_dump, cfg.topk, score_mean, score_std)
        self.val_ds = CandidateDumpDataset(cfg.val_dump, cfg.topk, score_mean, score_std)
        self.test_ds = CandidateDumpDataset(cfg.test_dump, cfg.topk, score_mean, score_std) if cfg.test_dump else None

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

        pos_weight = estimate_pos_weight(cfg.train_dump, cfg.topk, cfg.pos_weight_max)
        self.pos_weight = pos_weight
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device),
            reduction="none",
        )

        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.history: List[Dict[str, float]] = []

        with open(self.out_dir / "config.json", "w", encoding="utf-8") as f:
            d = asdict(cfg)
            d.update({"score_mean": score_mean, "score_std": score_std, "pos_weight": pos_weight})
            json.dump(d, f, indent=2)

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "score_mean": self.score_mean,
                "score_std": self.score_std,
                "config": asdict(self.cfg),
            },
            path,
        )

    def fit(self):
        print(f"[reranker] device={self.device}")
        print(f"[reranker] model={self.cfg.model_kind} topk={self.cfg.topk}")
        print(f"[reranker] score_mean={self.score_mean:.4f} score_std={self.score_std:.4f} pos_weight={self.pos_weight:.2f}")

        for epoch in range(self.cfg.epochs):
            loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                grad_clip=self.cfg.grad_clip,
            )
            val_metrics = evaluate_reranker(self.model, self.val_loader, self.val_ds, self.device)
            val_metrics["train_loss"] = float(loss)
            val_metrics["epoch"] = float(epoch)

            self.history.append(val_metrics)
            print(
                "[val] epoch", epoch,
                "| loss", f"{loss:.5f}",
                "| fmax", f"{val_metrics['fmax']:.4f}",
                "| aupr", f"{val_metrics['aupr']:.4f}",
                "| oracle", f"{val_metrics['oracle_microF']:.4f}",
                "| coverage", f"{val_metrics['candidate_coverage']:.4f}",
            )

            monitor_value = float(val_metrics.get(self.cfg.monitor, val_metrics["fmax"]))
            if monitor_value > self.best_metric:
                self.best_metric = monitor_value
                self.best_epoch = epoch
                self.bad_epochs = 0
                self.save_checkpoint(self.out_dir / "best.pt", epoch, val_metrics)
                print(f"[reranker] new best {self.cfg.monitor}={monitor_value:.4f} at epoch={epoch}")
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.cfg.patience:
                    print(f"[reranker] early stop at epoch={epoch}, best_epoch={self.best_epoch}")
                    break

            with open(self.out_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

        if self.test_loader is not None:
            best = torch.load(self.out_dir / "best.pt", map_location=self.device, weights_only=False)
            self.model.load_state_dict(best["model"])
            test_metrics = evaluate_reranker(self.model, self.test_loader, self.test_ds, self.device)
            print("[test]", json.dumps(test_metrics, indent=2))
            with open(self.out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)
