from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models.retrievercal_model import RetrieverCal, ScoreSetCal, EmbCal, EmbSetCal


def _first_existing(base: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = base / name
        if p.exists():
            return p
    return None


class CandidateDumpDataset(Dataset):
    def __init__(
        self,
        dump_dir: str | Path,
        topk: int,
        score_mean: float,
        score_std: float,
        embedding_dump: str | Path | None = None,
        require_embeddings: bool = False,
    ):
        self.dump_dir = Path(dump_dir)
        self.topk = int(topk)
        self.score_mean = float(score_mean)
        self.score_std = float(score_std) if score_std > 0 else 1.0
        self.embedding_dir = Path(embedding_dump) if embedding_dump else self.dump_dir
        self.require_embeddings = bool(require_embeddings)

        self.eval_go_ids = np.load(self.dump_dir / "eval_go_ids.npy", mmap_mode="r").astype(np.int64)
        self.top_cols = np.load(self.dump_dir / "top_go_cols.int32.npy", mmap_mode="r")
        self.top_scores = np.load(self.dump_dir / "top_scores.float32.npy", mmap_mode="r")
        self.labels = np.load(self.dump_dir / "top_labels.int8.npy", mmap_mode="r")
        self.true_go_ids = np.load(self.dump_dir / "true_go_ids.npy", mmap_mode="r")
        valid_path = self.dump_dir / "top_valid.int8.npy"
        self.valid = np.load(valid_path, mmap_mode="r") if valid_path.exists() else None

        pids_path = self.dump_dir / "protein_ids.json"
        if pids_path.exists():
            with pids_path.open("r", encoding="utf-8") as f:
                self.protein_ids = [str(x) for x in json.load(f)]
        else:
            self.protein_ids = [f"row_{i}" for i in range(self.top_scores.shape[0])]

        if self.topk > self.top_scores.shape[1]:
            raise ValueError(f"Requested topk={self.topk}, dump has {self.top_scores.shape[1]}")

        self.rank_feature = (
            np.log1p(np.arange(1, self.topk + 1, dtype=np.float32)) / np.log1p(float(self.topk))
        ).astype(np.float32)

        self.has_embeddings = False
        self.protein_z = None
        self.go_z = None
        self.protein_row_map = None
        self.go_col_map = None
        self.emb_dim = None
        self._setup_embeddings()

    def _setup_embeddings(self) -> None:
        prot_path = _first_existing(self.embedding_dir, ["protein_z.float16.npy", "protein_z.float32.npy", "protein_z.npy"])
        go_path = _first_existing(self.embedding_dir, ["go_z.float16.npy", "go_z.float32.npy", "go_z.npy"])
        if prot_path is None or go_path is None:
            if self.require_embeddings:
                raise FileNotFoundError(
                    f"Embeddings required but missing protein_z/go_z in {self.embedding_dir}. "
                    "Copy protein_z.float16.npy and go_z.float16.npy from the source dump or pass --*_embedding_dump."
                )
            return

        self.protein_z = np.load(prot_path, mmap_mode="r")
        self.go_z = np.load(go_path, mmap_mode="r")
        self.emb_dim = int(self.go_z.shape[1])
        self.has_embeddings = True

        # GO id order may differ between candidate dump and embedding source dump.
        src_eval_path = self.embedding_dir / "eval_go_ids.npy"
        if src_eval_path.exists():
            src_eval = np.load(src_eval_path, mmap_mode="r").astype(np.int64)
            if len(src_eval) == len(self.eval_go_ids) and np.array_equal(src_eval, self.eval_go_ids):
                self.go_col_map = None
            else:
                id2src = {int(g): i for i, g in enumerate(src_eval.tolist())}
                missing = [int(g) for g in self.eval_go_ids.tolist() if int(g) not in id2src]
                if missing:
                    raise KeyError(f"{len(missing)} eval GO ids missing in embedding dump. Examples: {missing[:10]}")
                self.go_col_map = np.asarray([id2src[int(g)] for g in self.eval_go_ids.tolist()], dtype=np.int64)
        else:
            if self.go_z.shape[0] != len(self.eval_go_ids):
                raise RuntimeError(
                    f"go_z rows={self.go_z.shape[0]} but candidate eval_go_ids={len(self.eval_go_ids)} and no source eval_go_ids.npy."
                )
            self.go_col_map = None

        # Protein row order may differ too.
        src_pids_path = self.embedding_dir / "protein_ids.json"
        if src_pids_path.exists():
            with src_pids_path.open("r", encoding="utf-8") as f:
                src_pids = [str(x) for x in json.load(f)]
            if src_pids == self.protein_ids:
                self.protein_row_map = None
            else:
                pid2src = {p: i for i, p in enumerate(src_pids)}
                missing = [p for p in self.protein_ids if p not in pid2src]
                if missing:
                    raise KeyError(f"{len(missing)} protein IDs missing in embedding dump. Examples: {missing[:10]}")
                self.protein_row_map = np.asarray([pid2src[p] for p in self.protein_ids], dtype=np.int64)
        else:
            if self.protein_z.shape[0] != len(self.protein_ids):
                raise RuntimeError(
                    f"protein_z rows={self.protein_z.shape[0]} but proteins={len(self.protein_ids)} and no source protein_ids.json."
                )
            self.protein_row_map = None

    def __len__(self) -> int:
        return int(self.top_scores.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cols = np.asarray(self.top_cols[idx, : self.topk], dtype=np.int64)
        cand_ids = self.eval_go_ids[cols].astype(np.int64)
        scores = np.asarray(self.top_scores[idx, : self.topk], dtype=np.float32)
        labels = np.asarray(self.labels[idx, : self.topk], dtype=np.float32)
        if self.valid is not None:
            valid = np.asarray(self.valid[idx, : self.topk], dtype=np.bool_)
        else:
            valid = np.ones(self.topk, dtype=np.bool_)

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        scores = np.clip(scores, -20.0, 20.0).astype(np.float32)
        scores[~valid] = 0.0
        labels[~valid] = 0.0

        score_z = ((scores - self.score_mean) / max(self.score_std, 1e-6)).astype(np.float32)
        score_z[~valid] = 0.0

        item = {
            "protein_id": str(self.protein_ids[idx]),
            "cand_ids": torch.from_numpy(cand_ids.copy()).long(),
            "score_z": torch.from_numpy(score_z.copy()).float(),
            "rank_feature": torch.from_numpy(self.rank_feature.copy()).float(),
            "labels": torch.from_numpy(labels.copy()).float(),
            "valid": torch.from_numpy(valid.copy()).bool(),
            "true_go_ids": torch.from_numpy(np.asarray(self.true_go_ids[idx], dtype=np.int64).copy()).long(),
        }

        if self.has_embeddings:
            prot_row = int(idx if self.protein_row_map is None else self.protein_row_map[idx])
            src_cols = cols if self.go_col_map is None else self.go_col_map[cols]
            protein_z = np.asarray(self.protein_z[prot_row], dtype=np.float32)
            go_z = np.asarray(self.go_z[src_cols], dtype=np.float32)
            item["protein_z"] = torch.from_numpy(protein_z.copy()).float()
            item["go_z"] = torch.from_numpy(go_z.copy()).float()

        return item


def collate_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"protein_id": [x["protein_id"] for x in items]}
    tensor_keys = ["cand_ids", "score_z", "rank_feature", "labels", "valid", "true_go_ids"]
    if "protein_z" in items[0]:
        tensor_keys += ["protein_z", "go_z"]
    for k in tensor_keys:
        out[k] = torch.stack([x[k] for x in items], dim=0)
    return out


def estimate_score_stats(dump_dir: str | Path, topk: int, max_rows: Optional[int] = None) -> Tuple[float, float]:
    d = Path(dump_dir)
    scores = np.load(d / "top_scores.float32.npy", mmap_mode="r")
    n = scores.shape[0] if max_rows is None else min(scores.shape[0], int(max_rows))
    arr = np.asarray(scores[:n, :topk], dtype=np.float32)
    valid_path = d / "top_valid.int8.npy"
    if valid_path.exists():
        valid = np.asarray(np.load(valid_path, mmap_mode="r")[:n, :topk], dtype=bool)
    else:
        valid = np.ones_like(arr, dtype=bool)
    mask = valid & np.isfinite(arr) & (arr > -1e5)
    if not mask.any():
        return 0.0, 1.0
    vals = arr[mask]
    return float(vals.mean()), float(vals.std() + 1e-6)


def compute_global_fmax_aupr_from_items(items: List[Tuple[float, int]], n_true_total: int) -> Dict[str, float]:
    if n_true_total <= 0 or len(items) == 0:
        return {"fmax": 0.0, "aupr": 0.0, "best_threshold": 0.0, "precision": 0.0, "recall": 0.0}
    items = [(float(s), int(y)) for s, y in items if math.isfinite(float(s))]
    if not items:
        return {"fmax": 0.0, "aupr": 0.0, "best_threshold": 0.0, "precision": 0.0, "recall": 0.0}
    items.sort(key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    best_f = 0.0
    best_t = items[0][0]
    best_p = 0.0
    best_r = 0.0
    aupr = 0.0
    prev_rec = 0.0
    for score, is_true in items:
        if is_true:
            tp += 1
        else:
            fp += 1
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, n_true_total)
        if prec + rec > 0:
            f = 2.0 * prec * rec / (prec + rec)
            if f > best_f:
                best_f = f
                best_t = score
                best_p = prec
                best_r = rec
        dr = rec - prev_rec
        if dr > 0:
            aupr += prec * dr
            prev_rec = rec
    return {"fmax": float(best_f), "aupr": float(aupr), "best_threshold": float(best_t), "precision": float(best_p), "recall": float(best_r)}


def soft_f1_loss(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.sigmoid(logits)
    p = torch.where(valid, p, torch.zeros_like(p))
    y = torch.where(valid, labels.float(), torch.zeros_like(labels.float()))
    tp = (p * y).sum(dim=1)
    fp = (p * (1.0 - y)).sum(dim=1)
    fn = ((1.0 - p) * y).sum(dim=1)
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return 1.0 - f1.mean()


def cardinality_loss(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    p = torch.where(valid, p, torch.zeros_like(p))
    y = torch.where(valid, labels.float(), torch.zeros_like(labels.float()))
    pred_count = p.sum(dim=1)
    true_count = y.sum(dim=1)
    return F.mse_loss(torch.log1p(pred_count), torch.log1p(true_count))


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
    try:
        return int(s)
    except Exception:
        return None


def load_child_to_parents_json(path: str | Path) -> Dict[int, List[int]]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[int, List[int]] = {}
    for child, parents in raw.items():
        child_i = _go_to_int(child)
        if child_i is None:
            continue
        xs = []
        for p in parents or []:
            parent = p[0] if isinstance(p, (list, tuple)) else p
            pi = _go_to_int(parent)
            if pi is not None:
                xs.append(pi)
        out[child_i] = xs
    return out


def make_dag_edge_mask(cand_ids: torch.Tensor, valid: torch.Tensor, child_to_parents: Dict[int, List[int]], device: torch.device) -> torch.Tensor:
    B, K = cand_ids.shape
    out = torch.zeros((B, K, K), dtype=torch.bool)
    cand_np = cand_ids.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    for b in range(B):
        row = [int(x) for x in cand_np[b].tolist()]
        id_to_pos: Dict[int, List[int]] = {}
        for j, gid in enumerate(row):
            if valid_np[b, j]:
                id_to_pos.setdefault(gid, []).append(j)
        for i, child in enumerate(row):
            if not valid_np[b, i]:
                continue
            for parent in child_to_parents.get(child, []):
                for j in id_to_pos.get(int(parent), []):
                    out[b, i, j] = True
    return out.to(device=device, non_blocking=True)


def dag_loss_fn(logits: torch.Tensor, valid: torch.Tensor, edge_mask: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    if edge_mask is None or not edge_mask.any():
        return logits.sum() * 0.0
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)
    edge_mask = edge_mask.bool() & valid_pair
    if not edge_mask.any():
        return logits.sum() * 0.0
    logits_safe = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
    child = logits_safe.unsqueeze(2).expand_as(edge_mask)
    parent = logits_safe.unsqueeze(1).expand_as(edge_mask)
    diffs = child[edge_mask] - parent[edge_mask] - float(margin)
    return F.softplus(diffs.clamp(-30.0, 30.0)).mean()


@dataclass
class CalibConfig:
    train_dump: str
    val_dump: str
    out_dir: str
    model_kind: str = "retrievercal"  # retrievercal | scoresetcal | embcal | embsetcal
    topk: int = 500
    batch_size: int = 512
    num_workers: int = 0
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    n_layers: int = 1
    n_heads: int = 4
    dropout: float = 0.10
    pos_weight_max: float = 50.0
    lambda_f1: float = 0.25
    lambda_card: float = 0.05
    lambda_dag: float = 0.0
    dag_margin: float = 0.0
    dag_parents_json: str = ""
    eval_every_steps: int = 1000
    patience: int = 8
    device: str = "cuda:0"
    score_stat_rows: int = 0
    monitor: str = "fmax_full"
    train_embedding_dump: str = ""
    val_embedding_dump: str = ""
    proj_dim: int = 0


class CalibTrainer:
    def __init__(self, cfg: CalibConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        max_rows = None if cfg.score_stat_rows <= 0 else int(cfg.score_stat_rows)
        self.score_mean, self.score_std = estimate_score_stats(cfg.train_dump, cfg.topk, max_rows=max_rows)
        kind = cfg.model_kind.lower()
        requires_emb = kind in {"embcal", "embsetcal", "embeddingcal", "embeddingsetcal", "step3"}
        self.train_ds = CandidateDumpDataset(
            cfg.train_dump,
            cfg.topk,
            self.score_mean,
            self.score_std,
            embedding_dump=cfg.train_embedding_dump or None,
            require_embeddings=requires_emb,
        )
        self.val_ds = CandidateDumpDataset(
            cfg.val_dump,
            cfg.topk,
            self.score_mean,
            self.score_std,
            embedding_dump=cfg.val_embedding_dump or None,
            require_embeddings=requires_emb,
        )
        self.train_loader = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_batch, pin_memory=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_batch, pin_memory=True)
        if kind in {"retrievercal", "step1"}:
            self.model: nn.Module = RetrieverCal(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
        elif kind in {"scoresetcal", "setcal", "step2"}:
            self.model = ScoreSetCal(hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads, dropout=cfg.dropout, max_k=max(1024, cfg.topk))
        elif kind in {"embcal", "embeddingcal"}:
            if self.train_ds.emb_dim is None:
                raise RuntimeError("EmbCal requires embeddings but emb_dim is None")
            self.model = EmbCal(emb_dim=self.train_ds.emb_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout, proj_dim=(cfg.proj_dim or None))
        elif kind in {"embsetcal", "embeddingsetcal", "step3"}:
            if self.train_ds.emb_dim is None:
                raise RuntimeError("EmbSetCal requires embeddings but emb_dim is None")
            self.model = EmbSetCal(
                emb_dim=self.train_ds.emb_dim,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                max_k=max(1024, cfg.topk),
                proj_dim=(cfg.proj_dim or None),
            )
        else:
            raise ValueError(f"Unknown model_kind: {cfg.model_kind}")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.pos_weight = self._estimate_pos_weight()
        self.child_to_parents = load_child_to_parents_json(cfg.dag_parents_json) if cfg.lambda_dag > 0 else {}
        self.best_metric = -float("inf")
        self.bad_evals = 0
        with (self.out_dir / "config.json").open("w", encoding="utf-8") as f:
            d = asdict(cfg)
            d.update({"score_mean": self.score_mean, "score_std": self.score_std, "pos_weight": self.pos_weight})
            json.dump(d, f, indent=2)

    def _estimate_pos_weight(self) -> float:
        labels = np.asarray(self.train_ds.labels[:, : self.cfg.topk], dtype=np.int8)
        if self.train_ds.valid is not None:
            valid = np.asarray(self.train_ds.valid[:, : self.cfg.topk], dtype=bool)
        else:
            valid = np.ones_like(labels, dtype=bool)
        y = labels[valid].astype(np.int64)
        pos = int(y.sum())
        total = int(y.size)
        if pos <= 0:
            return 1.0
        neg = max(1, total - pos)
        return float(min(float(self.cfg.pos_weight_max), neg / max(1, pos)))

    def _move(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        kind = self.cfg.model_kind.lower()
        if kind in {"embcal", "embeddingcal", "embsetcal", "embeddingsetcal", "step3"}:
            return self.model(batch["score_z"], batch["rank_feature"], batch["valid"], batch["protein_z"], batch["go_z"])
        return self.model(batch["score_z"], batch["rank_feature"], batch["valid"])

    def _loss(self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        labels = batch["labels"].float()
        valid = batch["valid"].bool()
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
        valid_logits = logits[valid]
        valid_labels = labels[valid]
        if valid_logits.numel() == 0:
            bce = logits.sum() * 0.0
        else:
            pos_weight = torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(valid_logits, valid_labels, pos_weight=pos_weight, reduction="mean")
        lf1 = soft_f1_loss(logits, labels, valid) if self.cfg.lambda_f1 > 0 else logits.sum() * 0.0
        lcard = cardinality_loss(logits, labels, valid) if self.cfg.lambda_card > 0 else logits.sum() * 0.0
        if self.cfg.lambda_dag > 0 and self.child_to_parents:
            edge_mask = make_dag_edge_mask(batch["cand_ids"], valid, self.child_to_parents, self.device)
            ldag = dag_loss_fn(logits, valid, edge_mask, margin=self.cfg.dag_margin)
        else:
            ldag = logits.sum() * 0.0
        total = bce + self.cfg.lambda_f1 * lf1 + self.cfg.lambda_card * lcard + self.cfg.lambda_dag * ldag
        return {"loss": total, "bce": bce, "soft_f1": lf1, "card": lcard, "dag": ldag}

    def train(self) -> None:
        logging.info("[calib] device=%s model=%s topk=%d", self.device, self.cfg.model_kind, self.cfg.topk)
        logging.info("[calib] score_mean=%.4f score_std=%.4f pos_weight=%.2f", self.score_mean, self.score_std, self.pos_weight)
        if getattr(self.train_ds, "has_embeddings", False):
            logging.info("[calib] embeddings enabled emb_dim=%s train_emb=%s val_emb=%s", self.train_ds.emb_dim, self.train_ds.embedding_dir, self.val_ds.embedding_dir)
        global_step = 0
        for epoch in range(self.cfg.epochs):
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"train e{epoch}", leave=False):
                batch = self._move(batch)
                logits = self._forward(batch)
                losses = self._loss(logits, batch)
                loss = losses["loss"]
                if not torch.isfinite(loss):
                    logging.warning("[calib] non-finite loss at step=%d, skipping", global_step)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                global_step += 1
                if self.cfg.eval_every_steps > 0 and global_step % self.cfg.eval_every_steps == 0:
                    metrics = self.evaluate()
                    logging.info("[val@step%d] %s", global_step, metrics)
                    self._maybe_save(epoch, global_step, metrics)
                    if self.bad_evals >= self.cfg.patience:
                        logging.info("[calib] early stop at epoch=%d step=%d", epoch, global_step)
                        return
            metrics = self.evaluate()
            logging.info("[val] epoch=%d %s", epoch, metrics)
            self._maybe_save(epoch, global_step, metrics)
            if self.bad_evals >= self.cfg.patience:
                logging.info("[calib] early stop at epoch=%d step=%d", epoch, global_step)
                return

    def _maybe_save(self, epoch: int, step: int, metrics: Dict[str, float]) -> None:
        monitor = self.cfg.monitor
        val = float(metrics.get(monitor, metrics.get("fmax_full", -float("inf"))))
        if val > self.best_metric:
            self.best_metric = val
            self.bad_evals = 0
            ckpt = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": int(epoch),
                "step": int(step),
                "metrics": metrics,
                "config": asdict(self.cfg),
                "score_mean": self.score_mean,
                "score_std": self.score_std,
                "pos_weight": self.pos_weight,
            }
            torch.save(ckpt, self.out_dir / f"best_{monitor}.pt")
            torch.save(ckpt, self.out_dir / f"best_{monitor}_step{step}_epoch{epoch}.pt")
            with (self.out_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logging.info("[checkpoint] best %s=%.4f", monitor, val)
        else:
            self.bad_evals += 1

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        items: List[Tuple[float, int]] = []
        n_true_full = 0
        n_true_cand = 0
        hit1 = hit5 = hit10 = 0
        n_prot = 0
        losses: List[float] = []
        bces: List[float] = []
        f1s: List[float] = []
        cards: List[float] = []
        dags: List[float] = []
        for batch in tqdm(self.val_loader, desc="eval", leave=False):
            batch = self._move(batch)
            logits = self._forward(batch)
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            logits = torch.where(batch["valid"].bool(), logits, torch.full_like(logits, -1e9))
            loss_dict = self._loss(logits, batch)
            losses.append(float(loss_dict["loss"].detach().cpu()))
            bces.append(float(loss_dict["bce"].detach().cpu()))
            f1s.append(float(loss_dict["soft_f1"].detach().cpu()))
            cards.append(float(loss_dict["card"].detach().cpu()))
            dags.append(float(loss_dict["dag"].detach().cpu()))
            sc = logits.detach().cpu().numpy()
            y = batch["labels"].detach().cpu().numpy()
            valid = batch["valid"].detach().cpu().numpy().astype(bool)
            cand = batch["cand_ids"].detach().cpu().numpy()
            true_ids = batch["true_go_ids"].detach().cpu().numpy()
            B, K = sc.shape
            for i in range(B):
                pos_set = set(int(x) for x in true_ids[i].tolist() if int(x) >= 0)
                n_true_full += len(pos_set)
                n_true_cand += int(((y[i] > 0) & valid[i]).sum())
                for j in range(K):
                    if not valid[i, j]:
                        continue
                    items.append((float(sc[i, j]), int(y[i, j])))
                order = np.argsort(-sc[i])
                order = [j for j in order if valid[i, j]]
                top1 = [int(cand[i, order[0]])] if order else []
                top5 = [int(cand[i, j]) for j in order[: min(5, len(order))]]
                top10 = [int(cand[i, j]) for j in order[: min(10, len(order))]]
                hit1 += 1 if any(g in pos_set for g in top1) else 0
                hit5 += 1 if any(g in pos_set for g in top5) else 0
                hit10 += 1 if any(g in pos_set for g in top10) else 0
                n_prot += 1
        topk_m = compute_global_fmax_aupr_from_items(items, n_true_cand)
        full_m = compute_global_fmax_aupr_from_items(items, n_true_full)
        return {
            "fmax_topk": topk_m["fmax"],
            "aupr_topk": topk_m["aupr"],
            "fmax_full": full_m["fmax"],
            "aupr_full": full_m["aupr"],
            "hits@1": hit1 / max(1, n_prot),
            "hits@5": hit5 / max(1, n_prot),
            "hits@10": hit10 / max(1, n_prot),
            "retrieval_recall@K": n_true_cand / max(1, n_true_full),
            "oracle_microF@K": (2.0 * n_true_cand) / max(1e-12, 2.0 * n_true_cand + (n_true_full - n_true_cand)),
            "n_true_candidate": float(n_true_cand),
            "n_true_full": float(n_true_full),
            "n_pairs_scored": float(len(items)),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "bce": float(np.mean(bces)) if bces else 0.0,
            "soft_f1_loss": float(np.mean(f1s)) if f1s else 0.0,
            "card_loss": float(np.mean(cards)) if cards else 0.0,
            "dag_loss": float(np.mean(dags)) if dags else 0.0,
        }
