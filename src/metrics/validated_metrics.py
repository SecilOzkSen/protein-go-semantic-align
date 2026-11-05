"""Explicit validation metrics for protein-GO retrieval.

These are NOT the MZSGO temporal benchmark metrics. All unannotated pairs
are treated as negatives for validation diagnostics.
"""
from __future__ import annotations
import numpy as np


def _check(y_true, scores):
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 2 or s.shape != y.shape or y.shape[0] == 0:
        raise ValueError('Expected matching nonempty [proteins, GO] matrices')
    if not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Nonfinite scores or nonbinary labels')
    if (y.sum(axis=1) == 0).any():
        raise ValueError('Validation includes protein without annotated positives')
    return y.astype(bool), s


def compute_retrieval_metrics(scores, labels):
    """Return protein-wise AP, ROC-AUC and positive-block coverage.

    AP uses deterministic stable descending ranking; ties retain GO column order.
    AUC uses average ranks for ties and skips all-positive proteins.
    """
    y, s = _check(labels, scores)
    aps, aucs, pbcs = [], [], []
    for yy, ss in zip(y, s):
        npos = int(yy.sum())
        order = np.argsort(-ss, kind='stable')
        rel = yy[order]
        positions = np.flatnonzero(rel) + 1
        aps.append(float(np.mean(np.arange(1, npos + 1) / positions)))
        pbcs.append(float(rel[:npos].mean()))
        nneg = len(yy) - npos
        if nneg:
            # Pairwise AUC: ties count as 0.5; rank-based without O(P*N) allocation.
            asc = np.argsort(ss, kind='stable')
            sorted_s = ss[asc]
            ranks = np.empty(len(ss), dtype=np.float64)
            i = 0
            while i < len(ss):
                j = i + 1
                while j < len(ss) and sorted_s[j] == sorted_s[i]:
                    j += 1
                ranks[asc[i:j]] = (i + 1 + j) / 2.0
                i = j
            auc = (ranks[yy].sum() - npos * (npos + 1) / 2) / (npos * nneg)
            aucs.append(float(auc))
    return {
        'protein_wise_map': float(np.mean(aps)),
        'pairwise_accuracy': float(np.mean(aucs)) if aucs else float('nan'),
        'positive_block_coverage': float(np.mean(pbcs)),
    }


def protein_centric_f_at_threshold(y_true, scores, threshold):
    y, s = _check(y_true, scores)
    pred = s >= float(threshold)
    tp = (pred & y).sum(axis=1)
    npred = pred.sum(axis=1)
    precision = float(np.mean(tp[npred > 0] / npred[npred > 0])) if (npred > 0).any() else 0.0
    recall = float(np.mean(tp / y.sum(axis=1)))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compute_protein_centric_fmax(y_true, scores):
    """Validation Fmax at every distinct raw-score threshold (plus no-prediction).

    NOT MZSGO's 101-point [0,1] grid. This uses raw score values and
    includes tied scores together. The threshold is selected on validation.
    """
    y, s = _check(y_true, scores)
    # Vectorized across proteins; iterate unique score values in descending order.
    n, g = s.shape
    order = np.argsort(-s, axis=1, kind='stable')
    ordered_s = np.take_along_axis(s, order, axis=1)
    ordered_y = np.take_along_axis(y, order, axis=1)
    tp = np.zeros(n, dtype=np.int64)
    pred = np.zeros(n, dtype=np.int64)
    pos = y.sum(axis=1)
    # Event sweep with heap to avoid materializing all thresholds * matrix.
    import heapq
    heap = [(-float(ordered_s[i, 0]), i) for i in range(n)]
    heapq.heapify(heap)
    best_f, best_t = 0.0, float('inf')
    while heap:
        threshold = -heap[0][0]
        while heap and -heap[0][0] == threshold:
            _, i = heapq.heappop(heap)
            j = pred[i]
            while j < g and ordered_s[i, j] == threshold:
                tp[i] += int(ordered_y[i, j])
                j += 1
            pred[i] = j
            if j < g:
                heapq.heappush(heap, (-float(ordered_s[i, j]), i))
        nonempty = pred > 0
        p = float(np.mean(tp[nonempty] / pred[nonempty])) if nonempty.any() else 0.0
        r = float(np.mean(tp / pos))
        f = 2*p*r/(p+r) if p+r else 0.0
        if f > best_f:
            best_f, best_t = f, threshold
    return float(best_f), float(best_t)


def compute_term_aupr(y_true, scores):
    """Macro term average precision (AP), not trapezoidal PR-AUC.

    Terms without positive validation labels are excluded. Returns the macro
    average of sklearn-compatible non-interpolated average precision per term.
    """
    y, s = _check(y_true, scores)
    aps = []
    for j in range(y.shape[1]):
        yy, ss = y[:, j], s[:, j]
        npos = int(yy.sum())
        if npos == 0:
            continue
        order = np.argsort(-ss, kind='stable')
        ys, xs = yy[order], ss[order]
        ends = np.r_[np.flatnonzero(xs[1:] != xs[:-1]) + 1, len(xs)]
        cum = np.cumsum(ys)
        tp = cum[ends - 1]
        prec = tp / ends
        recall_inc = np.diff(np.r_[0, tp]) / npos
        aps.append(float(np.sum(prec * recall_inc)))
    return float(np.mean(aps)) if aps else float('nan')
