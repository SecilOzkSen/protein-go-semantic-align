"""Frozen TRAIN-only GO weights and a fixed positive-exposure normalizer.

The weighted objective retains the original mean over each protein's positive
labels, then divides by one constant computed from effective training proteins.
It intentionally changes relative protein contributions: a singleton positive
keeps its GO weight rather than having that weight cancel within the protein.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Integral

import torch


FREQUENCY_INFONCE_METHOD = "train_positive_exposure_mean_v1"


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def build_frequency_weights(
        active_ids, train_counts, train_positive_sets,
        power=0.5, minimum=0.5, maximum=2.0,
):
    """Return (CPU float32 weights [G], JSON-compatible provenance metadata).

    All GO IDs are integer IDs in the scorer's active-column order. Counts must
    agree with the effective TRAIN positive sets actually supplied to the loss.
    Repeated labels within a protein are deduplicated, matching its boolean
    positive mask. Empty proteins and labels outside the active universe fail.

    For observed GO terms, inverse-frequency weights are divided by their
    observed-term mean and clipped. Unobserved terms get the neutral weight 1.
    Let e_g = mean_train_i(1[g in P_i] / |P_i|). The returned constant is
    C = sum_g e_g * w_g. The loss uses w_g / C, never a per-protein weight sum.
    Thus the average TRAIN coefficient is one while rare singleton proteins
    can receive more weight. Neither validation counts nor batch composition
    participates in building the weights or C.
    """
    power, minimum, maximum = float(power), float(minimum), float(maximum)
    if not math.isfinite(power) or power < 0:
        raise ValueError("frequency power must be finite and nonnegative")
    if not (math.isfinite(minimum) and math.isfinite(maximum)
            and 0 < minimum <= maximum):
        raise ValueError("frequency bounds must be finite, positive and ordered")
    ids = [_integer(g, "Active GO ID") for g in active_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("active_ids must be nonempty and unique")
    if not isinstance(train_counts, Mapping):
        raise ValueError("train_counts must map GO IDs to TRAIN protein counts")
    lookup = {g: col for col, g in enumerate(ids)}
    supplied = {}
    for gid, value in train_counts.items():
        gid = _integer(gid, "TRAIN count GO ID")
        count = _integer(value, "TRAIN GO frequency")
        if gid not in lookup and count:
            raise ValueError(f"Positive TRAIN count for inactive GO ID {gid}")
        supplied[gid] = count
    counts = [supplied.get(g, 0) for g in ids]

    observed_counts = [0] * len(ids)
    # Integer tallies by positive-set size keep exposure sums deterministic
    # when the same training proteins are enumerated in a different order.
    cardinality_tallies = [Counter() for _ in ids]
    positive_sets_hash = hashlib.sha256()
    n_proteins = 0
    n_singletons = 0
    for row in train_positive_sets:
        positives = sorted({_integer(g, "TRAIN positive GO ID") for g in row})
        if not positives:
            raise ValueError("Effective TRAIN protein has no positive GO labels")
        unknown = set(positives) - lookup.keys()
        if unknown:
            raise ValueError(f"Effective TRAIN positives contain inactive GO IDs: {sorted(unknown)[:10]}")
        cardinality = len(positives)
        n_proteins += 1
        n_singletons += int(cardinality == 1)
        positive_sets_hash.update(json.dumps(positives, separators=(",", ":")).encode("utf-8"))
        positive_sets_hash.update(b"\n")
        for gid in positives:
            col = lookup[gid]
            observed_counts[col] += 1
            cardinality_tallies[col][cardinality] += 1
    if n_proteins == 0:
        raise ValueError("Effective TRAIN positive sets must not be empty")
    if observed_counts != counts:
        mismatches = [ids[j] for j in range(len(ids)) if observed_counts[j] != counts[j]]
        raise ValueError(f"TRAIN counts differ from effective positive sets for GO IDs: {mismatches[:10]}")

    frequencies = torch.tensor(counts, dtype=torch.float64)
    observed = frequencies > 0
    if not bool(observed.any()):
        raise ValueError("No observed TRAIN GO terms")
    raw = frequencies[observed].pow(-power)
    raw_mean = raw.mean()
    if not bool(torch.isfinite(raw).all()) or not bool(raw_mean > 0):
        raise ValueError("Invalid inverse-frequency weights; check frequency power")
    weights = torch.ones(len(ids), dtype=torch.float32)
    weights[observed] = (raw / raw_mean).clamp(minimum, maximum).to(torch.float32)
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("Frequency weights must be finite and positive in float32")

    exposures = [
        math.fsum(count / cardinality for cardinality, count in sorted(tally.items())) / n_proteins
        for tally in cardinality_tallies
    ]
    weight_values = weights.tolist()
    normalizer = math.fsum(e * w for e, w in zip(exposures, weight_values))
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("TRAIN positive-exposure normalizer must be finite and positive")
    observed_weights = weights[observed]
    metadata = {
        "method": FREQUENCY_INFONCE_METHOD,
        "count_policy": "effective_training_protein_positive_sets_only",
        "exposure_definition": "mean_train_protein(positive_indicator / positive_count)",
        "loss_normalization": "fixed_train_exposure_mean",
        "per_protein_positive_reduction": "mean_by_positive_count",
        "singleton_policy": "weighted_by_go_weight_divided_by_fixed_train_normalizer",
        "zero_frequency_weight": 1.0,
        "power": power,
        "minimum": minimum,
        "maximum": maximum,
        "normalizer": normalizer,
        "n_active_go": len(ids),
        "n_observed_train_go": int(observed.sum()),
        "n_train_proteins": n_proteins,
        "n_single_positive_train_proteins": n_singletons,
        "n_train_positive_pairs": sum(counts),
        "weight_min": float(observed_weights.min()),
        "weight_mean": float(observed_weights.mean()),
        "weight_max": float(observed_weights.max()),
        "train_exposure_weighted_mean": normalizer,
        "active_go_ids_sha256": _json_hash(ids),
        "train_counts_sha256": _json_hash({"go_ids": ids, "counts": counts}),
        "weights_sha256": _json_hash({"go_ids": ids, "float32_weights": weight_values}),
        "exposures_sha256": _json_hash({"go_ids": ids, "positive_exposures": exposures}),
        "train_positive_sets_sha256": positive_sets_hash.hexdigest(),
    }
    return weights, metadata
