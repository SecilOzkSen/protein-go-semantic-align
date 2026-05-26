
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
from src.models.reranker_union_model import (
    SourceAwareCandidateInteractionMLP,
    SourceAwareScoreOnlyReranker,
)

class SourceAwareUnionDataset(Dataset):
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

        required_source_files = [
            "source_in_a.int8.npy",
            "source_in_b.int8.npy",
            "source_score_a.float32.npy",
            "source_score_b.float32.npy",
            "source_rank_a.float32.npy",
            "source_rank_b.float32.npy",
        ]
        missing = [x for x in required_source_files if not (self.dump_dir / x).exists()]
        if missing:
            raise FileNotFoundError(
                f"Source-aware union dump is missing {missing}. "
                "Build it with build_source_aware_union_candidate_dump.py."
            )

        self.source_in_a = np.load(self.dump_dir / "source_in_a.int8.npy", mmap_mode="r")
        self.source_in_b = np.load(self.dump_dir / "source_in_b.int8.npy", mmap_mode="r")
        self.source_score_a = np.load(self.dump_dir / "source_score_a.float32.npy", mmap_mode="r")
        self.source_score_b = np.load(self.dump_dir / "source_score_b.float32.npy", mmap_mode="r")
        self.source_rank_a = np.load(self.dump_dir / "source_rank_a.float32.npy", mmap_mode="r")
        self.source_rank_b = np.load(self.dump_dir / "source_rank_b.float32.npy", mmap_mode="r")

        valid_path = self.dump_dir / "top_valid.int8.npy"
        if valid_path.exists():
            self.top_valid = np.load(valid_path, mmap_mode="r")
        else:
            self.top_valid = None

        protein_ids_path = self.dump_dir / "protein_ids.json"
        with protein_ids_path.open("r", encoding="utf-8") as f:
            self.protein_ids = json.load(f)

        true_path = self.dump_dir / "true_go_ids.npy"
        if not true_path.exists():
            raise FileNotFoundError(true_path)
        self.true_go_ids = np.load(true_path, mmap_mode="r")

        if self.topk > self.top_scores.shape[1]:
            raise ValueError(f"Requested topk={self.topk}, dump has {self.top_scores.shape[1]} candidates")

        n = self.top_scores.shape[0]
        for name, arr in [
            ("protein_z", self.protein_z),
            ("top_cols", self.top_cols),
            ("top_labels", self.top_labels),
            ("source_in_a", self.source_in_a),
            ("source_in_b", self.source_in_b),
            ("source_score_a", self.source_score_a),
            ("source_score_b", self.source_score_b),
            ("source_rank_a", self.source_rank_a),
            ("source_rank_b", self.source_rank_b),
        ]:
            if arr.shape[0] != n:
                raise RuntimeError(f"{name} rows {arr.shape[0]} != top_scores rows {n}")

        if len(self.protein_ids) != n:
            raise RuntimeError(f"protein_ids length {len(self.protein_ids)} != rows {n}")

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
        top_ids = np.asarray(self.eval_go_ids[cols], dtype=np.int64)

        scores = np.asarray(self.top_scores[idx, : self.topk], dtype=np.float32)
        scores = (scores - self.score_mean) / max(self.score_std, 1e-6)

        if self.top_valid is not None:
            valid = np.asarray(self.top_valid[idx, : self.topk], dtype=np.float32)
        else:
            valid = np.ones(self.topk, dtype=np.float32)

        item = {
            "protein_id": str(self.protein_ids[idx]),
            "protein_z": torch.from_numpy(np.asarray(self.protein_z[idx], dtype=np.float32).copy()),
            "go_z": torch.from_numpy(np.asarray(self.go_z[cols], dtype=np.float32).copy()),
            "retriever_score": torch.from_numpy(scores.copy()),
            "rank_feature": torch.from_numpy(self.rank_feature.copy()),
            "label": torch.from_numpy(np.asarray(self.top_labels[idx, : self.topk], dtype=np.float32).copy()),
            "valid_mask": torch.from_numpy(valid.copy()),
            "top_cols": torch.from_numpy(cols.astype(np.int64)),
            "top_ids": torch.from_numpy(top_ids.astype(np.int64)),
            "true_go_ids": torch.from_numpy(np.asarray(self.true_go_ids[idx], dtype=np.int64).copy()),
            "source_in_a": torch.from_numpy(np.asarray(self.source_in_a[idx, : self.topk], dtype=np.float32).copy()),
            "source_in_b": torch.from_numpy(np.asarray(self.source_in_b[idx, : self.topk], dtype=np.float32).copy()),
            "source_score_a": torch.from_numpy(np.asarray(self.source_score_a[idx, : self.topk], dtype=np.float32).copy()),
            "source_score_b": torch.from_numpy(np.asarray(self.source_score_b[idx, : self.topk], dtype=np.float32).copy()),
            "source_rank_a": torch.from_numpy(np.asarray(self.source_rank_a[idx, : self.topk], dtype=np.float32).copy()),
            "source_rank_b": torch.from_numpy(np.asarray(self.source_rank_b[idx, : self.topk], dtype=np.float32).copy()),
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
        return 0.0, 1.0
    vals = arr[mask]
    return float(vals.mean()), float(vals.std() + 1e-6)


def estimate_pos_weight(dump_dir: str | Path, topk: int, max_value: float = 50.0) -> float:
    dump_dir = Path(dump_dir)
    labels = np.load(dump_dir / "top_labels.int8.npy", mmap_mode="r")
    y = np.asarray(labels[:, :topk], dtype=np.int64)

    valid_path = dump_dir / "top_valid.int8.npy"
    if valid_path.exists():
        valid = np.asarray(np.load(valid_path, mmap_mode="r")[:, :topk], dtype=bool)
    else:
        valid = np.ones_like(y, dtype=bool)

    yv = y[valid]
    pos = int(yv.sum())
    total = int(yv.size)
    neg = max(1, total - pos)
    if pos <= 0:
        return 1.0
    return float(min(max_value, neg / max(1, pos)))


def collate_union_batch(items: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
    out: Dict[str, torch.Tensor | List[str]] = {}
    out["protein_id"] = [x["protein_id"] for x in items]
    tensor_keys = [
        "protein_z", "go_z", "retriever_score", "rank_feature", "label", "valid_mask",
        "top_cols", "top_ids", "true_go_ids",
        "source_in_a", "source_in_b", "source_score_a", "source_score_b", "source_rank_a", "source_rank_b",
    ]
    for k in tensor_keys:
        out[k] = torch.stack([x[k] for x in items], dim=0)
    return out


def move_batch(batch: Dict[str, torch.Tensor | List[str]], device: torch.device) -> Dict[str, torch.Tensor | List[str]]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def build_model(kind: str, dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    if kind == "source_aware_score_only":
        return SourceAwareScoreOnlyReranker(hidden_dim=max(64, hidden_dim // 4), dropout=dropout)
    if kind == "source_aware_interaction_mlp":
        return SourceAwareCandidateInteractionMLP(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_embeddings=True,
            use_union_score=True,
            use_union_rank=True,
            use_source_features=True,
        )
    raise ValueError(f"Unknown source-aware reranker kind: {kind}")


def compute_fmax_and_aupr(y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 501) -> Dict[str, float]:
    y_true = y_true.astype(np.int32, copy=False)
    y_score = y_score.astype(np.float32, copy=False)

    finite = np.isfinite(y_score) & (y_score > -1e5)
    if not finite.any():
        return {"fmax": 0.0, "aupr": 0.0, "best_threshold": 0.0, "precision": 0.0, "recall": 0.0}

    s_min = float(y_score[finite].min())
    s_max = float(y_score[finite].max())
    if s_min == s_max:
        return {"fmax": 0.0, "aupr": 0.0, "best_threshold": s_min, "precision": 0.0, "recall": 0.0}

    thresholds = np.linspace(s_min, s_max, int(n_thresholds), dtype=np.float32)
    best = {"fmax": 0.0, "best_threshold": float(thresholds[0]), "precision": 0.0, "recall": 0.0}
    one_minus = 1 - y_true

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

    if compute_term_aupr is not None:
        try:
            aupr = float(compute_term_aupr(y_true, y_score))
        except Exception:
            aupr = 0.0
    else:
        aupr = 0.0
    best["aupr"] = aupr
    return best


def make_full_y_true(true_ids: np.ndarray, eval_go_ids: np.ndarray) -> np.ndarray:
    N = int(true_ids.shape[0])
    G = int(eval_go_ids.shape[0])
    y_true = np.zeros((N, G), dtype=np.int8)
    id2col = {int(g): i for i, g in enumerate(np.asarray(eval_go_ids, dtype=np.int64).tolist())}
    for i in range(N):
        for gid in true_ids[i]:
            gid = int(gid)
            if gid < 0:
                continue
            j = id2col.get(gid)
            if j is not None:
                y_true[i, j] = 1
    return y_true


@torch.no_grad()
def evaluate_reranker(
    model: nn.Module,
    loader: DataLoader,
    dataset: SourceAwareUnionDataset,
    device: torch.device,
    n_thresholds: int = 501,
    non_candidate_margin: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    all_logits, all_cols, all_true, all_valid = [], [], [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        logits = model(
            protein_z=batch["protein_z"],
            go_z=batch["go_z"],
            retriever_score=batch["retriever_score"],
            rank_feature=batch["rank_feature"],
            source_score_a=batch["source_score_a"],
            source_score_b=batch["source_score_b"],
            source_rank_a=batch["source_rank_a"],
            source_rank_b=batch["source_rank_b"],
            source_in_a=batch["source_in_a"],
            source_in_b=batch["source_in_b"],
        )
        all_logits.append(logits.detach().cpu().float().numpy())
        all_cols.append(batch["top_cols"].detach().cpu().numpy())
        all_true.append(batch["true_go_ids"].detach().cpu().numpy())
        all_valid.append(batch["valid_mask"].detach().cpu().numpy().astype(bool))

    cand_logits = np.concatenate(all_logits, axis=0).astype(np.float32)
    top_cols = np.concatenate(all_cols, axis=0).astype(np.int64)
    true_ids = np.concatenate(all_true, axis=0).astype(np.int64)
    valid_mask = np.concatenate(all_valid, axis=0).astype(bool)

    N = cand_logits.shape[0]
    G = dataset.n_go

    valid_scores = cand_logits[np.isfinite(cand_logits) & valid_mask]
    if valid_scores.size == 0:
        raise RuntimeError("No valid candidate scores found during evaluation.")
    fill_value = float(valid_scores.min() - non_candidate_margin)
    cand_logits = np.where(valid_mask & np.isfinite(cand_logits), cand_logits, fill_value)

    y_score = np.full((N, G), fill_value, dtype=np.float32)
    rows = np.arange(N)[:, None]
    y_score[rows, top_cols] = cand_logits

    y_true = make_full_y_true(true_ids, np.asarray(dataset.eval_go_ids, dtype=np.int64))
    metrics = compute_fmax_and_aupr(y_true, y_score, n_thresholds=n_thresholds)

    labels = np.asarray(dataset.top_labels[:, : dataset.topk], dtype=np.int8)
    valid_diag = np.asarray(dataset.top_valid[:, : dataset.topk], dtype=np.int8) if dataset.top_valid is not None else np.ones_like(labels)
    true_counts = (np.asarray(dataset.true_go_ids) >= 0).sum(axis=1).clip(min=1)
    hits = (labels * valid_diag).sum(axis=1)
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
    total_valid = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        logits = model(
            protein_z=batch["protein_z"],
            go_z=batch["go_z"],
            retriever_score=batch["retriever_score"],
            rank_feature=batch["rank_feature"],
            source_score_a=batch["source_score_a"],
            source_score_b=batch["source_score_b"],
            source_rank_a=batch["source_rank_a"],
            source_rank_b=batch["source_rank_b"],
            source_in_a=batch["source_in_a"],
            source_in_b=batch["source_in_b"],
        )
        labels = batch["label"].float()
        valid = batch["valid_mask"].float()

        loss_mat = criterion(logits, labels)
        loss = (loss_mat * valid).sum() / valid.sum().clamp_min(1.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n = float(valid.sum().detach().item())
        total_loss += float(loss.detach().item()) * n
        total_valid += n

    return total_loss / max(1.0, total_valid)


@dataclass
class SourceAwareRerankerConfig:
    train_dump: str
    val_dump: str
    test_dump: Optional[str]
    out_dir: str
    topk: int = 1000
    model_kind: str = "source_aware_interaction_mlp"
    hidden_dim: int = 512
    dropout: float = 0.10
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    patience: int = 2
    grad_clip: float = 1.0
    pos_weight_max: float = 50.0
    monitor: str = "fmax"
    device: str = "cuda:0"
    n_thresholds: int = 501


class SourceAwareRerankerTrainer:
    def __init__(self, cfg: SourceAwareRerankerConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        score_mean, score_std = estimate_score_stats(cfg.train_dump, cfg.topk)
        self.score_mean = score_mean
        self.score_std = score_std

        self.train_ds = SourceAwareUnionDataset(cfg.train_dump, cfg.topk, score_mean, score_std)
        self.val_ds = SourceAwareUnionDataset(cfg.val_dump, cfg.topk, score_mean, score_std)
        self.test_ds = SourceAwareUnionDataset(cfg.test_dump, cfg.topk, score_mean, score_std) if cfg.test_dump else None

        self.train_loader = DataLoader(
            self.train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            pin_memory=True, collate_fn=collate_union_batch,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
            pin_memory=True, collate_fn=collate_union_batch,
        )
        self.test_loader = None
        if self.test_ds is not None:
            self.test_loader = DataLoader(
                self.test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                pin_memory=True, collate_fn=collate_union_batch,
            )

        self.model = build_model(cfg.model_kind, self.train_ds.dim, cfg.hidden_dim, cfg.dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        pos_weight = estimate_pos_weight(cfg.train_dump, cfg.topk, cfg.pos_weight_max)
        self.pos_weight = pos_weight
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device), reduction="none"
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
        print(f"[source-aware-reranker] device={self.device}")
        print(f"[source-aware-reranker] model={self.cfg.model_kind} topk={self.cfg.topk}")
        print(f"[source-aware-reranker] score_mean={self.score_mean:.4f} score_std={self.score_std:.4f} pos_weight={self.pos_weight:.2f}")

        for epoch in range(self.cfg.epochs):
            loss = train_one_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.device, self.cfg.grad_clip)
            val_metrics = evaluate_reranker(self.model, self.val_loader, self.val_ds, self.device, n_thresholds=self.cfg.n_thresholds)
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
                print(f"[source-aware-reranker] new best {self.cfg.monitor}={monitor_value:.4f} at epoch={epoch}")
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.cfg.patience:
                    print(f"[source-aware-reranker] early stop at epoch={epoch}, best_epoch={self.best_epoch}")
                    break

            with open(self.out_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

        if self.test_loader is not None:
            best = torch.load(self.out_dir / "best.pt", map_location=self.device, weights_only=False)
            self.model.load_state_dict(best["model"])
            test_metrics = evaluate_reranker(self.model, self.test_loader, self.test_ds, self.device, n_thresholds=self.cfg.n_thresholds)
            print("[test]", json.dumps(test_metrics, indent=2))
            with open(self.out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)
