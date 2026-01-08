import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc
from typing import Tuple, Dict, Any, Optional, List


# -----------------------------
# Helpers: move batch to device
# -----------------------------
def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def _find_first_key(batch: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in batch and batch[k] is not None:
            return k
    return None

import numpy as np

import numpy as np

def cafa_metrics_sanity(y_true, y_pred, name="val", sample=2048, seed=0):
    """
    Fast sanity checks for CAFA-style multi-label eval.
    y_true, y_pred: array-like [N, M]
    Prints diagnostics and raises AssertionError on hard inconsistencies.
    """

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    # 1) Bir protein seç
    i = 0
    pos_cols = np.where(yt[i] > 0)[0]
    top_cols = np.argsort(-yp[i])[:50]

    print("[DBG][CAFA-check] pos_cols[:10] =", pos_cols[:10].tolist())
    print("[DBG][CAFA-check] top_cols[:10] =", top_cols[:10].tolist())
    print("[DBG][CAFA-check] overlap@50 =", int(np.intersect1d(pos_cols, top_cols).size))

    # 0) shape + finiteness
    assert yt.ndim == 2 and yp.ndim == 2, "y_true/y_pred must be 2D [N,M]"
    assert yt.shape == yp.shape, f"shape mismatch: y_true={yt.shape} y_pred={yp.shape}"
    assert np.isfinite(yp).all(), "NaN/Inf found in y_pred"

    N, M = yt.shape

    # 1) label density sanity
    true_pos = float(yt.sum())
    pos_per_prot = true_pos / max(1, N)
    pos_per_label = true_pos / max(1, M)
    assert true_pos > 0, "y_true is all-zero (label mapping broken)"

    # 2) prediction distribution
    mn = float(yp.min())
    mx = float(yp.max())
    mean = float(yp.mean())
    std = float(yp.std())
    assert std > 0.0, "y_pred has zero std (constant predictions)"

    # 3) quick separation proxy: positives should score higher than negatives
    pos_idx = np.argwhere(yt > 0)
    neg_idx = np.argwhere(yt == 0)
    assert len(pos_idx) > 0 and len(neg_idx) > 0, "cannot sample pos/neg indices"

    rng = np.random.default_rng(seed)
    ps = pos_idx[rng.integers(0, len(pos_idx), size=min(sample, len(pos_idx)))]
    ns = neg_idx[rng.integers(0, len(neg_idx), size=min(sample, len(neg_idx)))]
    pos_mean = float(np.mean(yp[ps[:, 0], ps[:, 1]]))
    neg_mean = float(np.mean(yp[ns[:, 0], ns[:, 1]]))
    assert pos_mean > neg_mean, f"alignment broken: pos_mean={pos_mean:.6f} <= neg_mean={neg_mean:.6f}"

    # 4) per-protein signal sanity: top1 should exceed median by some margin
    top1 = np.sort(yp, axis=1)[:, -1]
    med = np.median(yp, axis=1)
    gap = float(np.mean(top1 - med))
    assert gap > 0.0, f"no per-protein score contrast: avg(top1-median)={gap:.6f}"

    # 5) threshold sweep sanity: does predicted positive count change with threshold?
    # Use a few percentiles of y_pred as thresholds (fast, scale-agnostic)
    thr_list = np.quantile(yp, [0.99, 0.95, 0.9, 0.75, 0.5])
    pred_counts = [int((yp >= thr).sum()) for thr in thr_list]
    assert len(set(pred_counts)) > 1, f"thresholding ineffective, pred_counts={pred_counts}"

    print(
        f"[DBG][{name}] N={N} M={M} "
        f"true_pos={int(true_pos)} pos/prot={pos_per_prot:.3f} pos/label={pos_per_label:.3f} | "
        f"y_pred min/max/mean/std={mn:.4f}/{mx:.4f}/{mean:.4f}/{std:.4f} | "
        f"pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f} gap(top1-med)={gap:.4f} | "
        f"thr_counts={pred_counts}"
    )

    # Return a small dict in case you want to log it
    return {
        "N": N, "M": M,
        "true_pos": true_pos,
        "pos_per_prot": pos_per_prot,
        "pos_per_label": pos_per_label,
        "y_pred_min": mn, "y_pred_max": mx, "y_pred_mean": mean, "y_pred_std": std,
        "pos_mean": pos_mean, "neg_mean": neg_mean,
        "top1_minus_median": gap,
        "thr_counts": pred_counts,
    }


# ----------------------------------------
# Collect logits/probs + labels from dict batch
# ----------------------------------------
@torch.no_grad()
def collect_probs_and_labels_from_dict_batch(
    model,
    data_loader,
    device,
    label_key: Optional[str] = None,
    input_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_pred: [N, G] probabilities (sigmoid applied if needed)
      y_true: [N, G] multi-hot {0,1}

    Batch is expected to be a dict coming from your collate.
    We try to auto-detect label tensor key if label_key is not provided.
    We try model(batch) first, then fallback to model(batch[input_key]) if needed.
    """

    model.eval()

    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    # 1) auto-detect label key if not given
    label_candidates = [
        "y_true", "labels", "label", "y",
        "go_multi_hot", "go_targets", "targets", "target", "multi_hot"
    ]

    # 2) auto-detect input key if you want model(x) fallback
    input_candidates = [
        "prot_emb_pad", "prot_emb", "H", "x"
    ]

    for batch in data_loader:
        if not isinstance(batch, dict):
            raise TypeError(f"Expected dict batch, got: {type(batch)}")

        # detect keys once per batch (safe)
        lk = label_key or _find_first_key(batch, label_candidates)
        if lk is None:
            raise KeyError(f"Could not find labels in batch. Tried keys: {label_candidates}. Got keys: {list(batch.keys())}")

        ik = input_key or _find_first_key(batch, input_candidates)

        # move everything to device (safe, keeps non-tensors as-is)
        batch_dev = _to_device(batch, device)
        y = batch_dev[lk]

        # y should be [B, G]
        if not torch.is_tensor(y):
            raise TypeError(f"Label batch[{lk}] is not a tensor: {type(y)}")

        # forward: prefer model(batch_dev)
        try:
            out = model(batch_dev)
        except Exception:
            if ik is None:
                raise RuntimeError(
                    "model(batch) failed and no input_key could be inferred for model(x) fallback. "
                    f"Available keys: {list(batch.keys())}"
                )
            x = batch_dev[ik]
            out = model(x)

        # out can be logits or probs
        if not torch.is_tensor(out):
            raise TypeError(f"Model output is not a tensor: {type(out)}")

        # If output looks like logits, apply sigmoid.
        # Heuristic: if any values < 0 or > 1, treat as logits.
        if (out.min().item() < 0.0) or (out.max().item() > 1.0):
            probs = torch.sigmoid(out)
        else:
            probs = out

        preds.append(probs.detach().float().cpu().numpy())
        labels.append(y.detach().float().cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(labels, axis=0)
    y_true = (y_true > 0.0).astype(np.int32)
    return y_pred, y_true

import numpy as np
from typing import Tuple

def compute_fmax(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_thresholds: int = 101,
    eps: float = 1e-8
) -> Tuple[float, float]:
    """
    Proper CAFA-style protein-centric Fmax:
      - compute F1 per protein
      - average over proteins with at least 1 true label
      - maximize over thresholds
    """
    y_true = (y_true > 0).astype(np.bool_)
    y_pred = y_pred.astype(np.float32)

    thresholds = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)

    # only proteins with at least 1 true label
    true_cnt = y_true.sum(axis=1)
    has_true = true_cnt > 0
    if not np.any(has_true):
        return 0.0, 0.0

    yt = y_true[has_true]
    yp = y_pred[has_true]

    best_f, best_t = 0.0, 0.0

    for t in thresholds:
        pred = (yp >= t)

        tp = (pred & yt).sum(axis=1).astype(np.float32)
        fp = (pred & ~yt).sum(axis=1).astype(np.float32)
        fn = (~pred & yt).sum(axis=1).astype(np.float32)

        prec_i = tp / np.clip(tp + fp, 1.0, None)
        rec_i  = tp / np.clip(tp + fn, 1.0, None)

        f1_i = (2.0 * prec_i * rec_i) / np.clip(prec_i + rec_i, eps, None)

        f = float(np.mean(f1_i))
        if f > best_f:
            best_f, best_t = f, float(t)

    return best_f, best_t


# -----------------------------
# CAFA-style protein-centric Fmax
# -----------------------------
def compute_fmax_old(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_thresholds: int = 101,
    eps: float = 1e-8
) -> Tuple[float, float]:
    """
    CAFA-style protein-centric Fmax.

    precision is averaged only over proteins where we predicted at least 1 term
    recall is averaged only over proteins with at least 1 true label
    """

    _ = cafa_metrics_sanity(y_true, y_pred, name="val")

    y_true = (y_true > 0).astype(np.int32)

    fmax, best_t = 0.0, 0.0
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    for t in thresholds:
        pred = (y_pred >= t).astype(np.int32)

        tp = (pred & y_true).sum(axis=1).astype(np.float32)
        fp = (pred & (1 - y_true)).sum(axis=1).astype(np.float32)
        fn = ((1 - pred) & y_true).sum(axis=1).astype(np.float32)

        denom_p = tp + fp
        denom_r = tp + fn

        has_pred = denom_p > 0
        has_true = denom_r > 0

        prec = float(np.mean(tp[has_pred] / (denom_p[has_pred] + eps))) if np.any(has_pred) else 0.0
        rec  = float(np.mean(tp[has_true] / (denom_r[has_true] + eps))) if np.any(has_true) else 0.0

        f = (2.0 * prec * rec) / (prec + rec + eps)
        if f > fmax:
            fmax, best_t = float(f), float(t)

    return fmax, best_t


# -----------------------------
# Term-centric AUPR
# -----------------------------
def compute_term_aupr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = (y_true > 0).astype(np.int32)

    auprs = []
    G = y_true.shape[1]
    for g in range(G):
        if y_true[:, g].sum() < 1:
            continue
        p, r, _ = precision_recall_curve(y_true[:, g], y_pred[:, g])
        auprs.append(auc(r, p))
    return float(np.mean(auprs)) if auprs else 0.0


# -----------------------------
# One-shot eval helper
# -----------------------------
@torch.no_grad()
def eval_fmax_aupr_from_loader(
    model,
    data_loader,
    device,
    label_key: Optional[str] = None,
    input_key: Optional[str] = None,
    num_thresholds: int = 101
) -> Dict[str, float]:
    y_pred, y_true = collect_probs_and_labels_from_dict_batch(
        model=model,
        data_loader=data_loader,
        device=device,
        label_key=label_key,
        input_key=input_key,
    )
    fmax, best_t = compute_fmax(y_true, y_pred, num_thresholds=num_thresholds)
    aupr = compute_term_aupr(y_true, y_pred)
    return {"cafa_fmax": float(fmax), "best_t": float(best_t), "cafa_aupr": float(aupr)}
