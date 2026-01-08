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

def cafa_metrics(y_true, y_pred, name="val"):
    # y_true, y_pred: [N, M]
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    true_pos = yt.sum()
    pos_per_prot = true_pos / yt.shape[0]

    print(f"[DBG][CAFA] true_pos={true_pos:.0f}  pos_per_prot={pos_per_prot:.4f}")

    mn = float(yp.min())
    mx = float(yp.max())
    mean = float(yp.mean())
    std = float(yp.std())

    print(f"[DBG][CAFA] y_pred min={mn:.6f} max={mx:.6f} mean={mean:.6f} std={std:.6f}")

    # pozitif ve negatif indeksleri bul
    pos_idx = np.argwhere(yt > 0)
    neg_idx = np.argwhere(yt == 0)

    if len(pos_idx) > 0 and len(neg_idx) > 0:
        rng = np.random.default_rng(0)

        ps = pos_idx[rng.integers(0, len(pos_idx), size=min(1024, len(pos_idx)))]
        ns = neg_idx[rng.integers(0, len(neg_idx), size=min(1024, len(neg_idx)))]

        pos_mean = float(np.mean(yp[ps[:, 0], ps[:, 1]]))
        neg_mean = float(np.mean(yp[ns[:, 0], ns[:, 1]]))

        print(f"[DBG][CAFA] score_mean pos={pos_mean:.6f} neg={neg_mean:.6f} (want pos > neg)")

    print(f"[DBG][{name}] y_true shape={yt.shape} y_pred shape={yp.shape}")
    assert yt.shape == yp.shape, "shape mismatch"

    # 1) label density sanity
    true_pos = yt.sum()
    print(f"[DBG][{name}] true_pos={true_pos:.0f}  pos_per_prot={true_pos/yt.shape[0]:.3f}")
    assert true_pos > 0, "y_true all-zero (label mapping broken)"

    # 2) prediction range sanity
    mn, mx, mean, std = float(yp.min()), float(yp.max()), float(yp.mean()), float(yp.std())
    print(f"[DBG][{name}] y_pred min/max/mean/std = {mn:.4f} {mx:.4f} {mean:.4f} {std:.4f}")
    assert np.isfinite([mn, mx, mean, std]).all(), "NaN/Inf in y_pred"

    # 3) per-protein signal sanity: top score vs median
    top1 = np.sort(yp, axis=1)[:, -1]
    med = np.median(yp, axis=1)
    gap = float(np.mean(top1 - med))
    print(f"[DBG][{name}] avg(top1 - median) = {gap:.4f}")

    # 4) quick overlap proxy (NO threshold): do positives get higher scores?
    # sample 512 positives, 512 negatives
    pos_idx = np.argwhere(yt > 0)
    neg_idx = np.argwhere(yt == 0)
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        rng = np.random.default_rng(0)
        ps = pos_idx[rng.integers(0, len(pos_idx), size=min(512, len(pos_idx)))]
        ns = neg_idx[rng.integers(0, len(neg_idx), size=min(512, len(neg_idx)))]
        pos_mean = float(np.mean(yp[ps[:,0], ps[:,1]]))
        neg_mean = float(np.mean(yp[ns[:,0], ns[:,1]]))
        print(f"[DBG][{name}] score_mean pos={pos_mean:.4f} neg={neg_mean:.4f} (want pos>neg)")


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


# -----------------------------
# CAFA-style protein-centric Fmax
# -----------------------------
def compute_fmax(
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

    cafa_metrics(y_true, y_pred, name="val")

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
