"""Threshold-independent protein-GO ranking metrics.

All metrics average over proteins, not protein-GO pairs.
"""
import numpy as np


def compute_retrieval_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    if scores.ndim != 2 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Expected finite scores and matching labels [proteins, GO]')
    aucs, aps, pbcs = [], [], []
    for s, y in zip(scores, labels):
        npos = int(y.sum())
        nneg = len(y) - npos
        if not npos or not nneg:
            continue
        order = np.argsort(-s, kind='stable')
        ordered = y[order]
        ranks = np.arange(1, len(s) + 1, dtype=np.float64)
        aps.append(float(((np.cumsum(ordered) / ranks) * ordered).sum() / npos))
        pbcs.append(float(ordered[:npos].sum() / npos))
        neg = np.sort(s[~y])
        pos = s[y]
        lo = np.searchsorted(neg, pos, side='left')
        hi = np.searchsorted(neg, pos, side='right')
        aucs.append(float(np.mean((lo + 0.5 * (hi - lo)) / nneg)))
    if not aucs:
        raise ValueError('No proteins with both positive and negative GO labels')
    return {
        'pairwise_accuracy': float(np.mean(aucs)),
        'positive_block_coverage': float(np.mean(pbcs)),
        'protein_wise_map': float(np.mean(aps)),
    }
