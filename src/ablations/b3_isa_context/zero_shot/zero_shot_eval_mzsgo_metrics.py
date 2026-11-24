"""MZSGO's protein-centric threshold-grid metrics on supplied probabilities.

The metric rules match ``utils/util.py:evaluate_annotations`` and
``utils/test_zero.py:evaluate_unseen_labels`` at the recorded official commit.
The caller supplies a fixed [0, 1] score matrix. This module does not choose a
cosine adapter, fit calibration, transform scores, or propagate GO predictions.

All proteins must already have been scored with the original model's batching.
Only this metric calculation removes rows without truth in its label subset.
"""
from __future__ import annotations

import numpy as np


__all__ = ["evaluate_mzsgo"]

REFERENCE_REPOSITORY = "https://github.com/toxic-byte/MZSGO"
REFERENCE_COMMIT = "c3184146614c917982e4f50a9b2226d680d2c8ec"
REFERENCE_SOURCE_SHA256 = {
    "utils/util.py": "0dafc1ca9b1c11fe9c2f7a0e800fd05f2a3bc747bad0848acf03d8c168be52d0",
    "utils/test_zero.py": "d15a208f3bf6d8ed4baeab4f894dd288cb41796fbea250a6f43e147712d4f393",
    "trainer.py": "9052707de5ac0e2198ce58df4bc897bc84d56efcc37ef939a88c1e8a3c81a0f7",
}
_MAX_BLOCK_ROWS = 256
_TARGET_BLOCK_CELLS = 1_000_000


def _go_ids(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of GO IDs")
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of GO IDs") from exc
    result = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"Invalid GO ID in {name}: {value!r}")
        if isinstance(value, (int, np.integer)):
            identifier = int(value)
        elif isinstance(value, (str, np.str_)):
            text = str(value).strip()
            if text.startswith("GO:"):
                text = text[3:]
            if not text or not text.isascii() or not text.isdigit():
                raise ValueError(f"Invalid GO ID in {name}: {value!r}")
            identifier = int(text)
        else:
            raise ValueError(f"Invalid GO ID in {name}: {value!r}")
        if identifier < 0:
            raise ValueError(f"GO IDs in {name} must be nonnegative")
        result.append(identifier)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique GO IDs")
    return result


def _validate(probabilities, labels, go_ids, seen_go_ids, unseen_go_ids):
    probabilities, labels = np.asarray(probabilities), np.asarray(labels)
    if probabilities.ndim != 2 or labels.shape != probabilities.shape:
        raise ValueError("probabilities and labels must be matching [proteins, GO] matrices")
    if probabilities.dtype.kind not in "fiu" or labels.dtype.kind not in "bfiu":
        raise ValueError("probabilities must be real numeric values and labels must be binary")
    for start in range(0, probabilities.shape[0], _MAX_BLOCK_ROWS):
        scores = probabilities[start:start + _MAX_BLOCK_ROWS]
        truth = labels[start:start + _MAX_BLOCK_ROWS]
        if not np.isfinite(scores).all():
            raise ValueError("probabilities must be finite")
        if not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError("probabilities must lie in [0, 1]; apply the declared adapter before evaluation")
        if not ((truth == 0) | (truth == 1)).all():
            raise ValueError("labels must contain only 0 or 1")
    ids = _go_ids(go_ids, "go_ids")
    if len(ids) != probabilities.shape[1]:
        raise ValueError("go_ids length differs from the probability matrix columns")
    seen, unseen = set(_go_ids(seen_go_ids, "seen_go_ids")), set(_go_ids(unseen_go_ids, "unseen_go_ids"))
    if seen & unseen:
        raise ValueError("seen_go_ids and unseen_go_ids must be disjoint")
    if not (seen | unseen) <= set(ids):
        raise ValueError("seen/unseen GO IDs must belong to the scored candidate universe")
    return probabilities, labels.astype(bool, copy=False), ids, seen, unseen


def _block_rows(n_terms):
    return max(1, min(_MAX_BLOCK_ROWS, _TARGET_BLOCK_CELLS // max(1, n_terms)))


def _evaluate_subset(probabilities, labels, columns):
    """Reproduce the official reduction order without a dense integer matrix.

    The integer buffers contain one count per threshold and protein, never one
    per threshold, protein AND GO term. Protein block boundaries cannot change
    floating point averaging: counts are assembled in the original row order
    before each official precision/recall reduction is evaluated.
    """
    columns = np.asarray(columns, dtype=np.intp)
    n_terms = int(columns.size)
    rows_per_block = _block_rows(n_terms)
    positive_counts = np.zeros(labels.shape[0], dtype=np.int64)
    for start in range(0, labels.shape[0], rows_per_block):
        end = min(start + rows_per_block, labels.shape[0])
        row_ids = np.arange(start, end, dtype=np.intp)
        positive_counts[start:end] = np.count_nonzero(labels[np.ix_(row_ids, columns)], axis=1)
    retained_rows = np.flatnonzero(positive_counts > 0)
    n_proteins = int(retained_rows.size)
    n_positive_pairs = int(positive_counts.sum())
    thresholds = np.linspace(0, 1, 101)
    precisions, recalls, f_scores = [], [], []
    if n_proteins:
        tp_grid = np.empty((len(thresholds), n_proteins), dtype=np.int64)
        pred_grid = np.empty_like(tp_grid)
        for start in range(0, n_proteins, rows_per_block):
            end = min(start + rows_per_block, n_proteins)
            indices = np.ix_(retained_rows[start:end], columns)
            block_scores, block_truth = probabilities[indices], labels[indices]
            for threshold_index, threshold in enumerate(thresholds):
                # Keep the original scalar comparison and input score dtype.
                # In particular, do not round scores or replace this with a
                # histogram whose floating-point boundary rules can differ.
                predicted = block_scores >= threshold
                pred_grid[threshold_index, start:end] = np.count_nonzero(predicted, axis=1)
                tp_grid[threshold_index, start:end] = np.count_nonzero(predicted & block_truth, axis=1)
        real_count = positive_counts[retained_rows]
        for true_positives, prediction_count in zip(tp_grid, pred_grid):
            has_prediction = prediction_count > 0
            n_with_prediction = np.sum(has_prediction)
            if n_with_prediction > 0:
                precision_values = true_positives[has_prediction] / prediction_count[has_prediction]
                precision = np.sum(precision_values) / n_with_prediction
            else:
                precision = 0.0
            recall = np.sum(true_positives / real_count) / n_proteins
            precisions.append(precision)
            recalls.append(recall)
            f_scores.append(2 * precision * recall / (precision + recall)
                            if precision + recall > 0 else 0.0)
    else:
        # The official function returns five zeros before building a curve in
        # this case. The returned zero curve makes this diagnostic explicit.
        precisions = [0.0] * len(thresholds)
        recalls = [0.0] * len(thresholds)
        f_scores = [0.0] * len(thresholds)

    best = int(np.argmax(f_scores))
    precision_array, recall_array = np.asarray(precisions), np.asarray(recalls)
    # Deliberately retain the original default (unstable) sort, including tied
    # recalls. No deduplication, precision envelope, or endpoints are added.
    order = np.argsort(recall_array)
    trapezoid = getattr(np, "trapz", None)
    if trapezoid is None:
        trapezoid = np.trapezoid
    area = trapezoid(precision_array[order], recall_array[order])
    return {
        "aupr": float(area), "fmax": float(f_scores[best]),
        "precision": float(precisions[best]), "recall": float(recalls[best]),
        "threshold": float(thresholds[best]), "n_proteins": n_proteins,
        "n_terms": n_terms, "n_positive_pairs": n_positive_pairs,
        "curve": {"thresholds": thresholds.tolist(),
                  "precision": precision_array.tolist(), "recall": recall_array.tolist()},
    }


def evaluate_mzsgo(probabilities, labels, go_ids, seen_go_ids, unseen_go_ids):
    """Return official-style overall, seen/unseen AUPR/Fmax and their AUPR H.

    ``go_ids`` describes columns in the supplied matrix. Seen/unseen inputs
    describe train-positive status; each group is additionally restricted to
    GO terms with positive support in these evaluation labels, exactly as in
    MZSGO's ``identify_unseen_labels``. Overall evaluation keeps every column.
    An empty seen/unseen group returns None. Inputs are never modified.

    Probability conversion is the caller's separate, recorded responsibility.
    This function performs no fitting, calibration, temperature scaling,
    sigmoid, cosine remapping, or ontology propagation.
    """
    probabilities, labels, ids, seen, unseen = _validate(
        probabilities, labels, go_ids, seen_go_ids, unseen_go_ids)
    positive_support = np.count_nonzero(labels, axis=0) > 0
    seen_columns = [j for j, gid in enumerate(ids) if gid in seen and positive_support[j]]
    unseen_columns = [j for j, gid in enumerate(ids) if gid in unseen and positive_support[j]]
    result = {
        "all": _evaluate_subset(probabilities, labels, np.arange(len(ids))),
        "seen": _evaluate_subset(probabilities, labels, seen_columns) if seen_columns else None,
        "unseen": _evaluate_subset(probabilities, labels, unseen_columns) if unseen_columns else None,
    }
    if result["seen"] is None or result["unseen"] is None:
        harmonic_mean = None
    else:
        seen_area, unseen_area = result["seen"]["aupr"], result["unseen"]["aupr"]
        harmonic_mean = (0.0 if seen_area == 0 or unseen_area == 0 else
                         float(2 * seen_area * unseen_area / (seen_area + unseen_area)))
    result["harmonic_mean"] = harmonic_mean
    result["protocol"] = {
        "name": "mzsgo_official_protein_centric_threshold_grid_v1",
        "reference_repository": REFERENCE_REPOSITORY, "reference_commit": REFERENCE_COMMIT,
        "reference_source_sha256": dict(REFERENCE_SOURCE_SHA256),
        "reference_functions": ["utils/util.py:evaluate_annotations",
                                "utils/test_zero.py:identify_unseen_labels",
                                "utils/test_zero.py:evaluate_unseen_labels",
                                "utils/test_zero.py:compute_harmonic_mean"],
        "numpy_version": str(np.__version__), "input_probability_dtype": str(probabilities.dtype),
        "input_shape": [int(x) for x in probabilities.shape],
        "score_transform_in_this_module": "none; caller supplies probabilities in [0,1]",
        "thresholds": {"start": 0.0, "stop": 1.0, "count": 101, "comparison": ">="},
        "all_columns": "all supplied active GO columns, including zero-support columns",
        "group_columns": "supplied train-status GO subset intersect positive evaluation support",
        "protein_mask": "positive truth within evaluated columns; applied only inside metric",
        "precision_average": "retained proteins with at least one prediction at this threshold",
        "recall_average": "all retained proteins with positive truth in evaluated columns",
        "area": "trapezoidal integration after np.argsort(recall), default sorting kind",
        "area_function": "numpy.trapz" if hasattr(np, "trapz") else "numpy.trapezoid",
        "added_endpoints": False, "precision_envelope": False, "recall_ties_collapsed": False,
        "fmax_ties": "first maximum in ascending threshold order",
        "empty_group": "seen/unseen=None; overall returns zero metrics and a zero diagnostic curve",
        "counts": "n_proteins is retained rows; n_positive_pairs is positive protein-GO pairs",
        "ontology_propagation": False, "macro_term_average_precision": False,
    }
    return result
