#!/usr/bin/env python3
"""
StarGO-style full-vocabulary protein permutation diagnostic.

Purpose
-------
Compare normal protein-to-GO retrieval against a protein-identity permutation
while preserving the exact same GO vocabulary and StarGO-style ranking metrics:

    MRR, Hits@1, Hits@5, Hits@10

The script also reports Recall@K and mean rank because protein function
prediction is multi-positive, unlike StarGO's single subClassOf target setup.

Expected inputs
---------------
1) protein embeddings:
   [N, D] NumPy array, one projected/normalized protein vector per validation sample.

2) GO embeddings:
   [G, D] NumPy array, one projected/normalized GO vector per vocabulary item.

3) GO ids:
   JSON list or text file with G GO identifiers, in the same row order as go_embeddings.

4) positives:
   JSON mapping:
       {"protein_id": ["GO:...", ...], ...}
   OR JSON list aligned with protein rows:
       [["GO:...", ...], ["GO:...", ...], ...]

5) protein ids:
   Optional JSON/text file aligned with protein rows. Required when positives is a mapping.

Optional ancestor filtering
---------------------------
StarGO removes inferred ancestors of the query GO term before ranking its target.
For protein function prediction, removing ancestors of true labels can be enabled
with --filter_true_ancestors and an ancestor mapping:

    {"GO:child": ["GO:parent", "GO:grandparent", ...], ...}

Important:
- True labels themselves are never removed.
- Filtering is applied independently per protein.
- Normal and permutation conditions use exactly the same masks.

Example
-------
python -m tests.diagnostics.protein_permutation_stargo \
  --protein_embeddings /workspace/results/general_val/protein_z.float16.npy \
  --go_embeddings /workspace/results/general_val/go_z.float16.npy \
  --go_ids /workspace/results/general_val/eval_go_ids.json \
  --protein_ids /workspace/results/general_val/protein_ids.json \
  --positives /workspace/data/val_pid_to_positives.json \
  --ancestors_json /workspace/data/go_ancestors.json \
  --filter_true_ancestors \
  --branch all \
  --go_vocab_json /workspace/data/go_vocab.json \
  --num_permutations 10 \
  --seed 42 \
  --batch_size 256 \
  --out_json /workspace/results/diagnostics/general_protein_permutation.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


BRANCH_ALIASES = {
    "all": "all",
    "mf": "molecular_function",
    "mfo": "molecular_function",
    "molecular_function": "molecular_function",
    "bp": "biological_process",
    "bpo": "biological_process",
    "biological_process": "biological_process",
    "cc": "cellular_component",
    "cco": "cellular_component",
    "cellular_component": "cellular_component",
}


def canonical_go_id(value: Any) -> str:
    """Normalize common GO id forms to GO:0000000."""
    s = str(value).strip()
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    s = s.replace("GO_", "GO:")
    if s.startswith("GO:"):
        suffix = s.split(":", 1)[1]
        if suffix.isdigit():
            return f"GO:{int(suffix):07d}"
    if s.isdigit():
        return f"GO:{int(s):07d}"
    return s


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_id_list(path: Path) -> List[str]:
    """Load ids from JSON list, .npy, or one-id-per-line text."""
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return [canonical_go_id(x) for x in arr.tolist()]

    if path.suffix == ".json":
        obj = load_json(path)
        if not isinstance(obj, list):
            raise ValueError(f"{path} must contain a JSON list.")
        return [canonical_go_id(x) for x in obj]

    with path.open("r", encoding="utf-8") as handle:
        return [canonical_go_id(line) for line in handle if line.strip()]


def load_protein_ids(path: Optional[Path], n: int) -> List[str]:
    if path is None:
        return [str(i) for i in range(n)]

    if path.suffix == ".json":
        obj = load_json(path)
        if not isinstance(obj, list):
            raise ValueError(f"{path} must contain a JSON list.")
        ids = [str(x) for x in obj]
    elif path.suffix == ".npy":
        ids = [str(x) for x in np.load(path, allow_pickle=True).tolist()]
    else:
        with path.open("r", encoding="utf-8") as handle:
            ids = [line.strip() for line in handle if line.strip()]

    if len(ids) != n:
        raise ValueError(f"protein_ids length {len(ids)} != number of proteins {n}")
    return ids


def load_positive_sets(
    path: Path,
    protein_ids: Sequence[str],
) -> List[Set[str]]:
    """
    Supports:
      - JSON dict: protein_id -> [GO ids]
      - JSON list aligned with rows
      - .npy object array aligned with rows
    """
    if path.suffix == ".npy":
        raw = np.load(path, allow_pickle=True).tolist()
    else:
        raw = load_json(path)

    if isinstance(raw, Mapping):
        result = []
        missing = []
        for pid in protein_ids:
            values = raw.get(pid)
            if values is None:
                missing.append(pid)
                values = []
            result.append({canonical_go_id(x) for x in values})
        if missing:
            print(f"[warning] {len(missing)} protein ids had no positives entry.")
        return result

    if isinstance(raw, list):
        if len(raw) != len(protein_ids):
            raise ValueError(
                f"positives length {len(raw)} != number of proteins {len(protein_ids)}"
            )
        return [{canonical_go_id(x) for x in row} for row in raw]

    raise ValueError("Unsupported positives format.")


def load_ancestors(path: Optional[Path]) -> Dict[str, Set[str]]:
    if path is None:
        return {}
    obj = load_json(path)
    if not isinstance(obj, Mapping):
        raise ValueError("ancestors_json must be a mapping GO -> list[ancestor GO].")
    return {
        canonical_go_id(k): {canonical_go_id(x) for x in values}
        for k, values in obj.items()
    }


def infer_namespace(record: Any) -> Optional[str]:
    if isinstance(record, str):
        value = record
    elif isinstance(record, Mapping):
        value = (
            record.get("namespace")
            or record.get("branch")
            or record.get("aspect")
            or record.get("ont")
        )
    else:
        value = None

    if value is None:
        return None

    key = str(value).strip().lower()
    aliases = {
        "mf": "molecular_function",
        "mfo": "molecular_function",
        "molecular function": "molecular_function",
        "molecular_function": "molecular_function",
        "f": "molecular_function",
        "bp": "biological_process",
        "bpo": "biological_process",
        "biological process": "biological_process",
        "biological_process": "biological_process",
        "p": "biological_process",
        "cc": "cellular_component",
        "cco": "cellular_component",
        "cellular component": "cellular_component",
        "cellular_component": "cellular_component",
        "c": "cellular_component",
    }
    return aliases.get(key, key)


def load_namespace_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}

    obj = load_json(path)
    namespace_map: Dict[str, str] = {}

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            go_id = canonical_go_id(key)
            namespace = infer_namespace(value)
            if namespace:
                namespace_map[go_id] = namespace
        return namespace_map

    if isinstance(obj, list):
        for record in obj:
            if not isinstance(record, Mapping):
                continue
            raw_id = record.get("id") or record.get("go_id") or record.get("go")
            if raw_id is None:
                continue
            namespace = infer_namespace(record)
            if namespace:
                namespace_map[canonical_go_id(raw_id)] = namespace
        return namespace_map

    raise ValueError("Unsupported go_vocab_json format.")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


@dataclass
class MetricAccumulator:
    reciprocal_rank_sum: float = 0.0
    hits1_sum: float = 0.0
    hits5_sum: float = 0.0
    hits10_sum: float = 0.0
    recall10_sum: float = 0.0
    recall50_sum: float = 0.0
    recall200_sum: float = 0.0
    mean_best_rank_sum: float = 0.0
    num_samples: int = 0
    num_positive_labels: int = 0

    def update(self, ranked_go_ids: Sequence[str], positives: Set[str]) -> None:
        ranks = [
            idx + 1
            for idx, go_id in enumerate(ranked_go_ids)
            if go_id in positives
        ]
        if not ranks:
            return

        best_rank = min(ranks)
        self.reciprocal_rank_sum += 1.0 / best_rank
        self.hits1_sum += float(best_rank <= 1)
        self.hits5_sum += float(best_rank <= 5)
        self.hits10_sum += float(best_rank <= 10)
        self.mean_best_rank_sum += best_rank

        positives_count = len(positives)
        self.recall10_sum += sum(r <= 10 for r in ranks) / positives_count
        self.recall50_sum += sum(r <= 50 for r in ranks) / positives_count
        self.recall200_sum += sum(r <= 200 for r in ranks) / positives_count
        self.num_samples += 1
        self.num_positive_labels += positives_count

    def finalize(self) -> Dict[str, float]:
        n = self.num_samples
        if n == 0:
            return {
                "n_samples": 0,
                "n_positive_labels": 0,
                "mrr": float("nan"),
                "hits@1": float("nan"),
                "hits@5": float("nan"),
                "hits@10": float("nan"),
                "recall@10": float("nan"),
                "recall@50": float("nan"),
                "recall@200": float("nan"),
                "mean_best_rank": float("nan"),
            }

        return {
            "n_samples": n,
            "n_positive_labels": self.num_positive_labels,
            "mrr": self.reciprocal_rank_sum / n,
            "hits@1": self.hits1_sum / n,
            "hits@5": self.hits5_sum / n,
            "hits@10": self.hits10_sum / n,
            "recall@10": self.recall10_sum / n,
            "recall@50": self.recall50_sum / n,
            "recall@200": self.recall200_sum / n,
            "mean_best_rank": self.mean_best_rank_sum / n,
        }


def build_sample_mask(
    base_mask: np.ndarray,
    positives: Set[str],
    go_to_col: Mapping[str, int],
    ancestors: Mapping[str, Set[str]],
    filter_true_ancestors: bool,
) -> np.ndarray:
    mask = base_mask.copy()
    if not filter_true_ancestors:
        return mask

    removable: Set[str] = set()
    for true_go in positives:
        removable.update(ancestors.get(true_go, set()))

    # Do not remove another true annotation, even when it is an ancestor.
    removable.difference_update(positives)

    for go_id in removable:
        col = go_to_col.get(go_id)
        if col is not None:
            mask[col] = False
    return mask


def evaluate_embeddings(
    protein_embeddings: np.ndarray,
    go_embeddings: np.ndarray,
    go_ids: Sequence[str],
    positives_per_protein: Sequence[Set[str]],
    base_go_mask: np.ndarray,
    ancestors: Mapping[str, Set[str]],
    filter_true_ancestors: bool,
    batch_size: int,
) -> Dict[str, float]:
    """
    Full-vocabulary evaluation.

    For each protein:
      1) score every GO term
      2) apply branch/ancestor mask
      3) stably rank descending
      4) compute StarGO metrics from the first true term
      5) compute multi-positive Recall@K
    """
    accumulator = MetricAccumulator()
    go_ids_array = np.asarray(go_ids, dtype=object)
    go_to_col = {go_id: idx for idx, go_id in enumerate(go_ids)}

    n = protein_embeddings.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores_batch = protein_embeddings[start:end] @ go_embeddings.T

        for local_idx, scores in enumerate(scores_batch):
            sample_idx = start + local_idx
            positives = {
                go for go in positives_per_protein[sample_idx]
                if go in go_to_col and base_go_mask[go_to_col[go]]
            }
            if not positives:
                continue

            sample_mask = build_sample_mask(
                base_mask=base_go_mask,
                positives=positives,
                go_to_col=go_to_col,
                ancestors=ancestors,
                filter_true_ancestors=filter_true_ancestors,
            )

            valid_cols = np.flatnonzero(sample_mask)
            valid_scores = scores[valid_cols]

            # Stable descending sort, matching StarGO's stable ranking intent.
            order = np.argsort(-valid_scores, kind="stable")
            ranked_go_ids = go_ids_array[valid_cols[order]].tolist()
            accumulator.update(ranked_go_ids, positives)

    return accumulator.finalize()


def summarize_permutations(
    normal: Mapping[str, float],
    permutation_results: Sequence[Mapping[str, float]],
) -> Dict[str, Any]:
    metric_names = [
        "mrr", "hits@1", "hits@5", "hits@10",
        "recall@10", "recall@50", "recall@200", "mean_best_rank",
    ]
    summary: Dict[str, Any] = {
        "normal": dict(normal),
        "permutations": list(permutation_results),
        "permutation_summary": {},
        "delta_normal_minus_permuted": {},
    }

    for metric in metric_names:
        values = np.asarray([x[metric] for x in permutation_results], dtype=np.float64)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        summary["permutation_summary"][metric] = {
            "mean": mean,
            "std": std,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }
        summary["delta_normal_minus_permuted"][metric] = float(normal[metric] - mean)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StarGO-style full-vocabulary protein permutation diagnostic."
    )
    parser.add_argument("--protein_embeddings", type=Path, required=True)
    parser.add_argument("--go_embeddings", type=Path, required=True)
    parser.add_argument("--go_ids", type=Path, required=True)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--protein_ids", type=Path)
    parser.add_argument("--ancestors_json", type=Path)
    parser.add_argument("--go_vocab_json", type=Path)
    parser.add_argument(
        "--branch",
        default="all",
        choices=sorted(BRANCH_ALIASES),
        help="Evaluate all GO terms or one branch within the general model.",
    )
    parser.add_argument(
        "--filter_true_ancestors",
        action="store_true",
        help="Remove ancestors of true annotations from candidate ranking, preserving true labels.",
    )
    parser.add_argument("--num_permutations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Do not L2-normalize embeddings before dot-product scoring.",
    )
    parser.add_argument("--out_json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_permutations < 1:
        raise ValueError("--num_permutations must be >= 1")

    protein_z = np.load(args.protein_embeddings, mmap_mode="r")
    go_z = np.load(args.go_embeddings, mmap_mode="r")

    if protein_z.ndim != 2 or go_z.ndim != 2:
        raise ValueError("protein_embeddings and go_embeddings must both be 2D arrays.")
    if protein_z.shape[1] != go_z.shape[1]:
        raise ValueError(
            f"Embedding dimensions differ: protein {protein_z.shape}, GO {go_z.shape}"
        )

    protein_z = np.asarray(protein_z, dtype=np.float32)
    go_z = np.asarray(go_z, dtype=np.float32)

    if not args.no_normalize:
        protein_z = l2_normalize(protein_z)
        go_z = l2_normalize(go_z)

    go_ids = load_id_list(args.go_ids)
    if len(go_ids) != go_z.shape[0]:
        raise ValueError(f"go_ids length {len(go_ids)} != GO rows {go_z.shape[0]}")

    protein_ids = load_protein_ids(args.protein_ids, protein_z.shape[0])
    positives = load_positive_sets(args.positives, protein_ids)
    ancestors = load_ancestors(args.ancestors_json)

    branch = BRANCH_ALIASES[args.branch]
    if branch == "all":
        base_go_mask = np.ones(len(go_ids), dtype=bool)
    else:
        namespace_map = load_namespace_map(args.go_vocab_json)
        if not namespace_map:
            raise ValueError(
                "--go_vocab_json is required for branch-specific breakdown of the general model."
            )
        base_go_mask = np.asarray(
            [namespace_map.get(go_id) == branch for go_id in go_ids],
            dtype=bool,
        )
        if not np.any(base_go_mask):
            raise ValueError(f"No GO terms found for branch {branch}.")

    print(
        f"[normal] proteins={len(protein_ids)} GO={int(base_go_mask.sum())} "
        f"branch={branch} ancestor_filter={args.filter_true_ancestors}"
    )
    normal = evaluate_embeddings(
        protein_embeddings=protein_z,
        go_embeddings=go_z,
        go_ids=go_ids,
        positives_per_protein=positives,
        base_go_mask=base_go_mask,
        ancestors=ancestors,
        filter_true_ancestors=args.filter_true_ancestors,
        batch_size=args.batch_size,
    )
    print(json.dumps(normal, indent=2))

    rng = np.random.default_rng(args.seed)
    permutation_results: List[Dict[str, float]] = []
    permutation_indices: List[List[int]] = []

    for run_idx in range(args.num_permutations):
        permutation = rng.permutation(protein_z.shape[0])

        # Avoid accidental fixed points where practical.
        if protein_z.shape[0] > 1:
            fixed = np.flatnonzero(permutation == np.arange(protein_z.shape[0]))
            for idx in fixed:
                swap_idx = (idx + 1) % protein_z.shape[0]
                permutation[idx], permutation[swap_idx] = (
                    permutation[swap_idx],
                    permutation[idx],
                )

        print(f"[permutation {run_idx + 1}/{args.num_permutations}]")
        metrics = evaluate_embeddings(
            protein_embeddings=protein_z[permutation],
            go_embeddings=go_z,
            go_ids=go_ids,
            positives_per_protein=positives,
            base_go_mask=base_go_mask,
            ancestors=ancestors,
            filter_true_ancestors=args.filter_true_ancestors,
            batch_size=args.batch_size,
        )
        metrics["run"] = run_idx
        metrics["seed"] = args.seed
        permutation_results.append(metrics)
        permutation_indices.append(permutation.astype(int).tolist())
        print(json.dumps(metrics, indent=2))

    output = summarize_permutations(normal, permutation_results)
    output["metadata"] = {
        "protein_embeddings": str(args.protein_embeddings),
        "go_embeddings": str(args.go_embeddings),
        "go_ids": str(args.go_ids),
        "positives": str(args.positives),
        "protein_ids": str(args.protein_ids) if args.protein_ids else None,
        "branch": branch,
        "filter_true_ancestors": args.filter_true_ancestors,
        "ancestors_json": str(args.ancestors_json) if args.ancestors_json else None,
        "num_permutations": args.num_permutations,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "normalized": not args.no_normalize,
        "protocol": "StarGO-style full-vocabulary stable ranking",
    }

    # Saving permutations makes the diagnostic fully reproducible.
    output["permutation_indices"] = permutation_indices

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=True)

    print(f"\nSaved: {args.out_json}")
    print("\nNormal minus permutation mean:")
    print(json.dumps(output["delta_normal_minus_permuted"], indent=2))


if __name__ == "__main__":
    main()
