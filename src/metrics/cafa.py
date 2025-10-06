import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import torch


@torch.no_grad()
def collect_logits_and_labels(model, data_loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run a full pass and collect sigmoid logits and labels as numpy."""
    model.eval()
    preds, labels = [], []
    for batch in data_loader:
        # adapt these 2 lines to your batch structure if needed
        x, y = batch  # x: model inputs, y: multi-hot labels [B, G]
        x = (x.to(device) if isinstance(x, torch.Tensor) else x)
        y = y.to(device)

        scores = model(x)  # [B, G] (raw logits)
        probs = torch.sigmoid(scores)  # turn into probabilities
        preds.append(probs.cpu().numpy())
        labels.append(y.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)  # [N, G]
    y_true = np.concatenate(labels, axis=0)  # [N, G]
    return y_pred, y_true


def compute_fmax(y_true: np.ndarray, y_pred: np.ndarray, num_thresholds: int = 101):
    """CAFA-style protein-centric Fmax (macro over proteins)."""
    fmax, best_t = 0.0, 0.0
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    for t in thresholds:
        pred = (y_pred >= t).astype(np.int32)
        tp = (pred * y_true).sum(axis=1)
        fp = (pred * (1 - y_true)).sum(axis=1)
        fn = ((1 - pred) * y_true).sum(axis=1)

        prec = np.mean(tp / (tp + fp + 1e-8))
        rec = np.mean(tp / (tp + fn + 1e-8))
        f = (2 * prec * rec) / (prec + rec + 1e-8)
        if f > fmax:
            fmax, best_t = f, t
    return float(fmax), float(best_t)


def compute_term_aupr(y_true: np.ndarray, y_pred: np.ndarray):
    """Term-centric AUPR averaged over GO terms with at least one positive."""
    auprs = []
    G = y_true.shape[1]
    for g in range(G):
        if y_true[:, g].sum() < 1:
            continue
        p, r, _ = precision_recall_curve(y_true[:, g], y_pred[:, g])
        auprs.append(auc(r, p))
    return float(np.mean(auprs)) if auprs else 0.0
