"""GOR2023 and CAFA-compatible IA-weighted evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def load_information_accretion(path: str | Path) -> dict[str, float]:
    """Load a headerless ``GO_ID IA`` file used by CAFA-evaluator."""
    values: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid IA line {line_number} in {path}: {raw!r}")
            go_id, value_raw = fields
            value = float(value_raw)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid IA value for {go_id}: {value}")
            values[go_id] = value
    if not values:
        raise ValueError(f"No IA values loaded from {path}")
    return values


def _go_string(value: object) -> str:
    text = str(value).strip()
    if text.upper().startswith("GO:"):
        return "GO:" + text.split(":", 1)[1].zfill(7)
    if text.isdigit():
        return f"GO:{int(text):07d}"
    raise ValueError(f"Invalid GO identifier: {value!r}")


def align_information_accretion(
        go_ids: Sequence[object],
        information_accretion: Mapping[str, float],
) -> np.ndarray:
    """Return IA values aligned with the evaluator's GO column order."""
    aligned = np.asarray(
        [float(information_accretion.get(_go_string(go_id), 0.0)) for go_id in go_ids],
        dtype=np.float64,
    )
    if not np.isfinite(aligned).all() or np.any(aligned < 0.0):
        raise ValueError("Aligned IA vector contains invalid values")
    if not np.any(aligned > 0.0):
        raise ValueError("Evaluation GO space has no positive IA weights")
    return aligned


def _normalize_parent_map(
        go_ids: Sequence[object],
        dag_parents: Mapping[object, Iterable[object]] | None,
) -> list[list[int]]:
    """Convert child-to-parent GO IDs into column indices.

    Parent entries may be raw IDs or tuples such as ``(parent_id, relation)``.
    Parents outside the current evaluation matrix are ignored, matching a
    branch-specific matrix slice.
    """
    ids = [_go_string(x) for x in go_ids]
    id_to_col = {go_id: col for col, go_id in enumerate(ids)}
    normalized: list[list[int]] = [[] for _ in ids]
    if not dag_parents:
        return normalized

    raw_by_id: dict[str, Iterable[object]] = {}
    for child, parents in dag_parents.items():
        raw_by_id[_go_string(child)] = parents or []

    for child_col, child_id in enumerate(ids):
        parent_cols = set()
        for entry in raw_by_id.get(child_id, []):
            parent_raw = entry[0] if isinstance(entry, (tuple, list)) else entry
            parent_col = id_to_col.get(_go_string(parent_raw))
            if parent_col is not None:
                parent_cols.add(parent_col)
        normalized[child_col] = sorted(parent_cols)
    return normalized


def _topological_children_first(parent_cols: Sequence[Sequence[int]]) -> list[int]:
    """Return child-before-parent order and reject ontology cycles."""
    n_terms = len(parent_cols)
    indegree = np.zeros(n_terms, dtype=np.int64)
    for parents in parent_cols:
        for parent in parents:
            indegree[parent] += 1
    queue = [int(i) for i in np.flatnonzero(indegree == 0)]
    order: list[int] = []
    head = 0
    while head < len(queue):
        child = queue[head]
        head += 1
        order.append(child)
        for parent in parent_cols[child]:
            indegree[parent] -= 1
            if indegree[parent] == 0:
                queue.append(parent)
    if len(order) != n_terms:
        raise ValueError("dag_parents contains a cycle")
    return order


def propagate_max_to_parents(
        matrix: np.ndarray,
        go_ids: Sequence[object],
        dag_parents: Mapping[object, Iterable[object]] | None,
) -> np.ndarray:
    """Propagate scores or binary truth to ancestors using CAFA ``max``."""
    arr = np.asarray(matrix).copy()
    if arr.ndim != 2 or arr.shape[1] != len(go_ids):
        raise ValueError(f"Expected [N,{len(go_ids)}] matrix, got {arr.shape}")
    parent_cols = _normalize_parent_map(go_ids, dag_parents)
    for child in _topological_children_first(parent_cols):
        if not parent_cols[child]:
            continue
        child_values = arr[:, child]
        for parent in parent_cols[child]:
            arr[:, parent] = np.maximum(arr[:, parent], child_values)
    return arr


def _to_confidences(y_score: np.ndarray) -> np.ndarray:
    score = np.asarray(y_score, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("Prediction scores contain NaN or infinity")

    if score.size == 0:
        return score

    score_min = float(score.min())
    score_max = float(score.max())

    if score_max <= score_min:
        return np.zeros_like(score, dtype=np.float64)

    # Retrieval logits are not calibrated probabilities.
    # Global affine normalization preserves their ranking while mapping
    # the complete evaluation score range to CAFA thresholds [0, 1].
    score = ((score - score_min) / (score_max - score_min))
    return np.clip(score, 0.0, 1.0)


def compute_gor2023_wfmax(
        y_true: np.ndarray,
        y_score: np.ndarray,
        go_ids: Sequence[object],
        information_accretion: Mapping[str, float] | str | Path,
        dag_parents: Mapping[object, Iterable[object]] | None = None,
        threshold_step: float = 0.01,
        propagate: bool = True,
        eps: float = 1e-12,
) -> dict[str, float]:
    """Compute CAFA-normalized weighted Fmax for one GO branch.

    This matches ``BioComputingUP/CAFA-evaluator`` with ``-norm cafa``,
    ``-prop max`` and the default inclusion of ontology orphans. Terms with IA
    equal to zero are excluded from weighted evaluation, as in its ``toi_ia``.
    """
    truth = np.asarray(y_true) > 0
    score = _to_confidences(y_score)
    if truth.ndim != 2 or score.ndim != 2 or truth.shape != score.shape:
        raise ValueError(f"Expected matching [N,G] arrays, got {truth.shape} and {score.shape}")
    if truth.shape[1] != len(go_ids):
        raise ValueError(f"GO ID count {len(go_ids)} does not match matrix width {truth.shape[1]}")
    if truth.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty protein set")
    if not (0.0 < float(threshold_step) < 1.0):
        raise ValueError("threshold_step must lie strictly between 0 and 1")

    ia_map = (
        load_information_accretion(information_accretion)
        if isinstance(information_accretion, (str, Path))
        else dict(information_accretion)
    )
    ia = align_information_accretion(go_ids, ia_map)

    if propagate:
        truth = propagate_max_to_parents(truth, go_ids, dag_parents).astype(bool, copy=False)
        score = propagate_max_to_parents(score, go_ids, dag_parents)

    positive_ia = ia > 0.0
    truth = truth[:, positive_ia]
    score = score[:, positive_ia]
    weights = ia[positive_ia]
    weighted_true = truth * weights[None, :]
    true_weight = weighted_true.sum(axis=1)
    n_proteins = truth.shape[0]

    thresholds = np.arange(threshold_step, 1.0, threshold_step, dtype=np.float64)
    best = {
        "wfmax": 0.0,
        "threshold": float(thresholds[0]),
        "weighted_precision": 0.0,
        "weighted_recall": 0.0,
        "coverage": 0.0,
    }

    for threshold in thresholds:
        predicted = score >= threshold
        predicted_weight = (predicted * weights[None, :]).sum(axis=1)
        intersection_weight = ((predicted & truth) * weights[None, :]).sum(axis=1)
        has_prediction = predicted_weight > 0.0

        weighted_precision = (
            float(np.mean(intersection_weight[has_prediction] / predicted_weight[has_prediction]))
            if np.any(has_prediction)
            else 0.0
        )
        # CAFA normalization divides recall sum by the number of GT proteins.
        per_protein_recall = np.divide(
            intersection_weight,
            true_weight,
            out=np.zeros_like(intersection_weight, dtype=np.float64),
            where=true_weight > 0.0,
        )
        weighted_recall = float(per_protein_recall.sum() / n_proteins)
        wf = (2.0 * weighted_precision * weighted_recall) / max(
            eps, weighted_precision + weighted_recall
        )
        coverage = float(np.count_nonzero(has_prediction) / n_proteins)

        if wf > best["wfmax"]:
            best = {
                "wfmax": float(wf),
                "threshold": float(threshold),
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "coverage": coverage,
            }

    best["proteins"] = int(n_proteins)
    best["go_terms"] = int(len(go_ids))
    best["positive_ia_terms"] = int(np.count_nonzero(positive_ia))
    best["zero_ia_terms"] = int(np.count_nonzero(~positive_ia))
    return best

