"""Ranking diagnostics on a shared full GO candidate universe.

This module depends only on NumPy and the Python standard library. It never
filters the protein cohort or candidate score matrix to the unseen group.
Unannotated pairs are treated as negatives for these validation diagnostics.

Protein AP uses stable descending scores, with ascending GO IDs breaking ties.
Term AP uses score-tie groups, matching non-interpolated average precision.
Those are deliberately different tie conventions, preserving the original
protein-ranking and term-AP metric definitions.
"""
from __future__ import annotations

import numpy as np


__all__ = ["evaluate_rankings"]


def _go_id(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must contain integer or canonical string GO IDs")
    if isinstance(value, (int, np.integer)):
        result = int(value)
    elif isinstance(value, (str, np.str_)):
        text = str(value).strip()
        if text.startswith("GO:"):
            text = text[3:]
        if not text or not text.isascii() or not text.isdigit():
            raise ValueError(f"Invalid GO ID in {name}: {value!r}")
        result = int(text)
    else:
        raise ValueError(f"Invalid GO ID in {name}: {value!r}")
    if result < 0:
        raise ValueError(f"GO IDs in {name} must be nonnegative")
    return result


def _go_ids(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of GO IDs, not one string")
    try:
        result = [_go_id(value, name) for value in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of GO IDs") from exc
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique GO IDs")
    return result


def _validate(scores, labels, go_ids, seen_go_ids, unseen_go_ids, protein_ids, ks):
    s, y = np.asarray(scores), np.asarray(labels)
    if s.ndim != 2 or y.shape != s.shape or not all(s.shape):
        raise ValueError("scores and labels must be matching nonempty [proteins, GO] matrices")
    if s.dtype.kind not in "fiu" or y.dtype.kind not in "bfiu":
        raise ValueError("scores must be real numeric values and labels must be binary")
    # Validate in blocks without allocating another full [N,G] matrix. Memmapped
    # inputs remain usable and no per-group dense label matrices are created.
    for start in range(0, s.shape[0], 256):
        ss, yy = s[start:start + 256], y[start:start + 256]
        if not np.isfinite(ss).all():
            raise ValueError("scores must be finite")
        if not ((yy == 0) | (yy == 1)).all():
            raise ValueError("labels must contain only 0 or 1")

    ids = _go_ids(go_ids, "go_ids")
    if len(ids) != s.shape[1]:
        raise ValueError("go_ids length differs from the score matrix columns")
    if any(left >= right for left, right in zip(ids, ids[1:])):
        raise ValueError(
            "go_ids must be strictly ascending; reorder scores and labels columns "
            "with the GO IDs before evaluation"
        )
    seen = set(_go_ids(seen_go_ids, "seen_go_ids"))
    unseen = set(_go_ids(unseen_go_ids, "unseen_go_ids"))
    universe = set(ids)
    if seen & unseen:
        raise ValueError("seen_go_ids and unseen_go_ids must be disjoint")
    if not (seen | unseen) <= universe:
        raise ValueError("seen/unseen GO IDs must belong to the scored candidate universe")

    if isinstance(protein_ids, (str, bytes)):
        raise ValueError("protein_ids must be an iterable, not one string")
    try:
        raw_pids = list(protein_ids)
    except TypeError as exc:
        raise ValueError("protein_ids must be an iterable") from exc
    if any(value is None or not str(value).strip() for value in raw_pids):
        raise ValueError("protein_ids must be nonempty identifiers")
    pids = [str(value) for value in raw_pids]
    if len(pids) != s.shape[0] or len(set(pids)) != len(pids):
        raise ValueError("protein_ids must be unique and match the score matrix rows")

    try:
        cutoffs = tuple(ks)
    except TypeError as exc:
        raise ValueError("ks must be an iterable of positive integers") from exc
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
           or int(k) < 1 for k in cutoffs):
        raise ValueError("ks must contain positive integers")
    cutoffs = tuple(int(k) for k in cutoffs)
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("ks must contain unique cutoffs")
    return s, y, ids, seen, unseen, pids, cutoffs


def _descending_order(values):
    values = np.asarray(values)
    # Bitwise complement reverses integer order exactly, including uints and the
    # signed minimum. Float conversion would lose distinctions above 2**53.
    descending_key = np.bitwise_not(values) if values.dtype.kind in "iu" else -values
    return np.argsort(descending_key, kind="stable")


def _term_average_precision(scores, labels, n_positive):
    """Non-interpolated AP across every protein, including all-negative rows.

    A constant-score column has AP equal to its positive prevalence. That is
    the constant-score baseline, not the expected AP of random permutations.
    """
    order = _descending_order(scores)
    xs, ys = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(xs[1:] != xs[:-1]) + 1, len(xs)]
    tp = np.cumsum(ys, dtype=np.int64)[ends - 1]
    precision = tp / ends
    recall_increments = np.diff(np.r_[0, tp]) / n_positive
    return float(np.sum(precision * recall_increments))


def evaluate_rankings(scores, labels, go_ids, seen_go_ids, unseen_go_ids,
                      protein_ids, ks=(50, 100, 200, 500, 1000)):
    """Evaluate all, seen and unseen targets against the same full GO ranking.

    Parameters
    ----------
    scores, labels : array-like, shape [N, G]
        Scores and binary full validation truth, for ALL validation proteins and
        ALL active candidate GO terms. Inputs are not mutated. A protein may
        have no positives, globally or within either target group.
    go_ids : iterable
        Unique, ascending integer GO IDs or equivalent ``GO:0000001`` strings.
        The same order must index the score and label columns. Unsorted IDs are
        rejected rather than silently changing score/label identity.
    seen_go_ids, unseen_go_ids : iterable
        Disjoint subsets of the scored GO IDs. Candidate terms outside both
        sets still participate in every ranking and the ``all`` group; their
        term-row status is ``unclassified``.
    protein_ids : iterable
        Unique identifiers in matrix-row order.
    ks : iterable of positive integers
        Recall cutoffs. A cutoff greater than G includes all G candidates.

    Returns
    -------
    metrics : dict
        Keys ``all``, ``seen`` and ``unseen``. Protein mAP and protein recall
        average over proteins with a target positive in that group. Micro recall
        pools the group's positive pairs; macro term recall averages over its
        validation-positive terms. Term AP scores every validation protein and
        averages only terms with at least one validation positive.
    term_rows : list[dict]
        One row per candidate, with integer ``go_id``, ``train_status``,
        ``n_positive_pairs``, ``ap`` and ``recall@K`` fields.
    protein_rows : list[dict]
        One row per validation protein. Flat fields are ``protein_id`` and
        ``{group}/n_positive_pairs``, ``{group}/ap``, ``{group}/recall@K``.

    Unsupported metrics are None, never NaN or a fabricated zero. Genuine
    zero recall/AP with positive support remains numeric zero.
    """
    s, y, ids, seen, unseen, pids, cutoffs = _validate(
        scores, labels, go_ids, seen_go_ids, unseen_go_ids, protein_ids, ks
    )
    n_proteins, n_terms = s.shape
    effective_ks = np.asarray([min(k, n_terms) for k in cutoffs], dtype=np.int64)
    masks = {
        "all": np.ones(n_terms, dtype=bool),
        "seen": np.asarray([gid in seen for gid in ids], dtype=bool),
        "unseen": np.asarray([gid in unseen for gid in ids], dtype=bool),
    }
    term_positive_counts = np.count_nonzero(y, axis=0)
    term_hits = np.zeros((len(cutoffs), n_terms), dtype=np.int64)
    accumulators = {
        group: {"n_proteins": 0, "ap_sum": 0.0,
                "protein_recall_sum": np.zeros(len(cutoffs), dtype=np.float64),
                "hits": np.zeros(len(cutoffs), dtype=np.int64)}
        for group in masks
    }
    protein_rows = []

    # Rank once per protein, then use only positive ranks for all three groups.
    # Additional working storage is O(G + K*G), not O(groups*N*G).
    for row_index in range(n_proteins):
        order = _descending_order(s[row_index])
        positive_ranks = np.flatnonzero(y[row_index, order]) + 1
        positive_columns = order[positive_ranks - 1]
        for k_index, hit_count in enumerate(
                np.searchsorted(positive_ranks, effective_ks, side="right")):
            # A GO appears at most once in a protein's positive list.
            term_hits[k_index, positive_columns[:hit_count]] += 1

        protein_row = {"protein_id": pids[row_index]}
        for group, mask in masks.items():
            ranks = positive_ranks[mask[positive_columns]]
            n_positive = len(ranks)
            protein_row[f"{group}/n_positive_pairs"] = n_positive
            if n_positive:
                ap = float(np.mean(np.arange(1, n_positive + 1, dtype=np.float64) / ranks))
                hits = np.searchsorted(ranks, effective_ks, side="right")
                recall = hits / n_positive
                acc = accumulators[group]
                acc["n_proteins"] += 1
                acc["ap_sum"] += ap
                acc["protein_recall_sum"] += recall
                acc["hits"] += hits
                protein_row[f"{group}/ap"] = ap
                for index, k in enumerate(cutoffs):
                    protein_row[f"{group}/recall@{k}"] = float(recall[index])
            else:
                protein_row[f"{group}/ap"] = None
                for k in cutoffs:
                    protein_row[f"{group}/recall@{k}"] = None
        protein_rows.append(protein_row)

    term_ap = np.full(n_terms, np.nan, dtype=np.float64)
    term_rows = []
    for col_index, gid in enumerate(ids):
        n_positive = int(term_positive_counts[col_index])
        ap = None
        if n_positive:
            ap = _term_average_precision(s[:, col_index], y[:, col_index], n_positive)
            term_ap[col_index] = ap
        term_row = {
            "go_id": gid,
            "train_status": "seen" if gid in seen else "unseen" if gid in unseen else "unclassified",
            "n_positive_pairs": n_positive,
            "ap": ap,
        }
        for index, k in enumerate(cutoffs):
            term_row[f"recall@{k}"] = (
                float(term_hits[index, col_index] / n_positive) if n_positive else None
            )
        term_rows.append(term_row)

    metrics = {}
    for group, mask in masks.items():
        supported = mask & (term_positive_counts > 0)
        n_supported = int(np.count_nonzero(supported))
        n_positive = int(term_positive_counts[mask].sum())
        acc = accumulators[group]
        n_target_proteins = acc["n_proteins"]
        result = {
            "protein_wise_map": float(acc["ap_sum"] / n_target_proteins) if n_target_proteins else None,
            "macro_term_ap": float(term_ap[supported].mean()) if n_supported else None,
            "n_proteins_with_positives": n_target_proteins,
            "n_terms_with_positives": n_supported,
            "n_positive_pairs": n_positive,
        }
        for index, k in enumerate(cutoffs):
            result[f"protein_recall@{k}"] = (
                float(acc["protein_recall_sum"][index] / n_target_proteins) if n_target_proteins else None
            )
            result[f"micro_recall@{k}"] = float(acc["hits"][index] / n_positive) if n_positive else None
            result[f"macro_term_recall@{k}"] = (
                float(np.mean(term_hits[index, supported] / term_positive_counts[supported]))
                if n_supported else None
            )
        metrics[group] = result
    return metrics, term_rows, protein_rows
