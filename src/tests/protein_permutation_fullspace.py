#!/usr/bin/env python3
"""
Checkpoint-based full-space Protein Identity Permutation Diagnostic.

Designed for:
  SecilOzkSen/protein-go-semantic-align
  branch: b1-v1-refinement

How it works
------------
1. Reuses src.main.py unchanged for:
   - YAML/config parsing
   - data stores and validation loader
   - GO text/cache construction
   - model construction
   - checkpoint loading

2. Temporarily replaces OppTrainer.eval_epoch with a diagnostic evaluator.

3. At the exact retriever scoring interface:
      Zp = model.encode_protein_for_scoring(H, mask)

   it collects every validation protein representation.

4. It scores those representations against every GO term in main.py's
   eval_id_list ("observed" or "seen", controlled by the YAML config).

5. It globally permutes Zp across validation proteins and scores again.

Candidate dumps, top-1000 candidate filtering, SemExp, ESM-kNN and rerankers
are not involved.

Run from repository root:
python -m tests.diagnostics.protein_identity_permutation \
  --config src/p3a_eval.yaml \
  --checkpoint /workspace/.../checkpoint_step83945.pt \
  --num-permutations 10 \
  --output /workspace/results/diagnostics/p3a_permutation.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-evaluation-space protein identity permutation diagnostic."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-permutations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--num-thresholds", type=int, default=101)
    parser.add_argument(
        "--save-representations",
        type=Path,
        default=None,
        help="Optional directory for Zp, y_true and eval GO IDs.",
    )
    return parser.parse_args()


def load_config_with_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    temp_dir: Path,
) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML root in {config_path}")

    config = copy.deepcopy(config)
    training = config.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("YAML field 'training' must be a mapping.")

    training["resume"] = str(checkpoint_path)
    training["eval_only"] = True

    # Diagnostic does not need online logging.
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")

    temp_config = temp_dir / "permutation_eval_config.yaml"
    with temp_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return temp_config


def protein_centric_fmax(
    y_true: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int,
) -> Dict[str, float]:
    """
    CAFA-style protein-centric Fmax over the score range.

    Precision:
      averaged over proteins receiving at least one prediction.

    Recall:
      averaged over proteins having at least one true label.
    """
    true_counts = y_true.sum(axis=1)
    valid_truth = true_counts > 0

    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return {
            "protein_fmax": float("nan"),
            "protein_fmax_threshold": float("nan"),
            "protein_precision_at_fmax": float("nan"),
            "protein_recall_at_fmax": float("nan"),
        }

    score_min = float(finite_scores.min())
    score_max = float(finite_scores.max())
    if score_min == score_max:
        thresholds = np.asarray([score_min], dtype=np.float32)
    else:
        thresholds = np.linspace(
            score_min, score_max, num_thresholds, dtype=np.float32
        )

    best_f = -1.0
    best_threshold = float("nan")
    best_precision = float("nan")
    best_recall = float("nan")

    for threshold in thresholds:
        predicted = scores >= float(threshold)
        pred_counts = predicted.sum(axis=1)
        overlaps = np.logical_and(predicted, y_true > 0).sum(axis=1)

        precision_mask = valid_truth & (pred_counts > 0)
        if not np.any(precision_mask):
            continue

        precision = float(
            np.mean(overlaps[precision_mask] / pred_counts[precision_mask])
        )
        recall = float(
            np.mean(overlaps[valid_truth] / true_counts[valid_truth])
        )

        denominator = precision + recall
        f_score = (
            0.0
            if denominator <= 0
            else 2.0 * precision * recall / denominator
        )

        if f_score > best_f:
            best_f = f_score
            best_threshold = float(threshold)
            best_precision = precision
            best_recall = recall

    return {
        "protein_fmax": float(best_f),
        "protein_fmax_threshold": best_threshold,
        "protein_precision_at_fmax": best_precision,
        "protein_recall_at_fmax": best_recall,
    }


def micro_fmax(
    y_true: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int,
) -> Dict[str, float]:
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return {
            "micro_fmax": float("nan"),
            "micro_fmax_threshold": float("nan"),
        }

    score_min = float(finite_scores.min())
    score_max = float(finite_scores.max())
    thresholds = (
        np.asarray([score_min], dtype=np.float32)
        if score_min == score_max
        else np.linspace(score_min, score_max, num_thresholds, dtype=np.float32)
    )

    best_f = -1.0
    best_threshold = float("nan")

    truth = y_true.astype(bool)
    for threshold in thresholds:
        pred = scores >= float(threshold)
        tp = float(np.logical_and(pred, truth).sum())
        fp = float(np.logical_and(pred, ~truth).sum())
        fn = float(np.logical_and(~pred, truth).sum())

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f_score = 2.0 * precision * recall / (precision + recall + 1e-12)

        if f_score > best_f:
            best_f = f_score
            best_threshold = float(threshold)

    return {
        "micro_fmax": float(best_f),
        "micro_fmax_threshold": best_threshold,
    }


def ranking_metrics(
    scores: np.ndarray,
    y_true: np.ndarray,
    ks: Sequence[int] = (1, 5, 10, 50, 100, 200, 500, 1000),
) -> Dict[str, float]:
    reciprocal_ranks: List[float] = []
    recalls: Dict[int, List[float]] = {int(k): [] for k in ks}
    hits: Dict[int, List[float]] = {int(k): [] for k in ks}
    best_ranks: List[int] = []

    for row_index in range(scores.shape[0]):
        positive_columns = np.flatnonzero(y_true[row_index] > 0)
        if positive_columns.size == 0:
            continue

        ordering = np.argsort(-scores[row_index], kind="stable")
        inverse_rank = np.empty_like(ordering)
        inverse_rank[ordering] = np.arange(1, len(ordering) + 1)

        positive_ranks = inverse_rank[positive_columns]
        best_rank = int(positive_ranks.min())

        best_ranks.append(best_rank)
        reciprocal_ranks.append(1.0 / best_rank)

        for k in ks:
            effective_k = min(int(k), scores.shape[1])
            recalls[int(k)].append(
                float(np.mean(positive_ranks <= effective_k))
            )
            hits[int(k)].append(float(best_rank <= effective_k))

    output: Dict[str, float] = {
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_best_rank": float(np.mean(best_ranks)),
    }

    for k in ks:
        output[f"hits@{k}"] = float(np.mean(hits[int(k)]))
        output[f"recall@{k}"] = float(np.mean(recalls[int(k)]))

    return output


def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    # Macro AUPR is undefined for columns with no validation positives.
    active_columns = np.flatnonzero(y_true.sum(axis=0) > 0)
    if active_columns.size == 0:
        return {
            "micro_aupr": float("nan"),
            "macro_aupr": float("nan"),
            "roc_auc": float("nan"),
            "num_active_go_terms": 0,
        }

    active_truth = y_true[:, active_columns]
    active_scores = scores[:, active_columns]

    micro_aupr = float(
        average_precision_score(active_truth, active_scores, average="micro")
    )
    macro_aupr = float(
        average_precision_score(active_truth, active_scores, average="macro")
    )

    try:
        roc_auc = float(
            roc_auc_score(active_truth.ravel(), active_scores.ravel())
        )
    except ValueError:
        roc_auc = float("nan")

    return {
        "micro_aupr": micro_aupr,
        "macro_aupr": macro_aupr,
        "roc_auc": roc_auc,
        "num_active_go_terms": int(active_columns.size),
    }


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int,
) -> Dict[str, Any]:
    return {
        **protein_centric_fmax(y_true, scores, num_thresholds),
        **micro_fmax(y_true, scores, num_thresholds),
        **classification_metrics(y_true, scores),
        "ranking": ranking_metrics(scores, y_true),
    }


def score_all_go_terms(
    trainer: Any,
    encoded_proteins: torch.Tensor,
    raw_go_vectors: torch.Tensor,
    score_batch_size: int,
) -> np.ndarray:
    """
    Score pre-encoded proteins against every GO term.

    encoded_proteins:
      pooled models: [N,Dz]
      slot models:   [N,S,Dz]

    raw_go_vectors:
      pooled GO evaluation: [G,Dg]

    The current P3a/B1 segmented-GO path produces pooled raw GO vectors.
    """
    if raw_go_vectors.ndim != 2:
        raise RuntimeError(
            "This diagnostic currently expects pooled GO vectors [G,Dg]. "
            f"Observed {tuple(raw_go_vectors.shape)}. "
            "For token-align/multivector-token models, use the token-specific "
            "exhaustive path instead."
        )

    model = trainer.model
    model.eval()
    device = trainer.device
    scale = trainer.logit_scale_tensor().detach().float()

    output_parts: List[torch.Tensor] = []
    num_proteins = encoded_proteins.shape[0]

    with torch.no_grad():
        for start in range(0, num_proteins, score_batch_size):
            end = min(start + score_batch_size, num_proteins)

            z_batch = encoded_proteins[start:end].to(
                device, non_blocking=True
            )
            batch_size = z_batch.shape[0]

            go_batch = raw_go_vectors.to(
                device, non_blocking=True
            ).unsqueeze(0).expand(batch_size, -1, -1)

            score_batch = model.score_from_encoded_protein(
                z_batch,
                go_batch,
                go_mask=None,
            )
            score_batch = score_batch.float() * scale
            output_parts.append(score_batch.cpu())

    return torch.cat(output_parts, dim=0).numpy().astype(np.float32)


def make_derangement(
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    if n < 2:
        raise ValueError("Permutation diagnostic requires at least two proteins.")

    # Rejection is cheap because expected fixed points are ~1.
    for _ in range(100):
        permutation = rng.permutation(n)
        if np.all(permutation != np.arange(n)):
            return permutation

    # Deterministic fallback: cyclic shift by a random non-zero offset.
    shift = int(rng.integers(1, n))
    return np.roll(np.arange(n), shift)


def summarize_permutations(
    normal: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    scalar_paths = [
        ("protein_fmax",),
        ("micro_fmax",),
        ("micro_aupr",),
        ("macro_aupr",),
        ("roc_auc",),
        ("ranking", "mrr"),
        ("ranking", "hits@1"),
        ("ranking", "hits@5"),
        ("ranking", "hits@10"),
        ("ranking", "recall@10"),
        ("ranking", "recall@50"),
        ("ranking", "recall@200"),
        ("ranking", "recall@500"),
        ("ranking", "recall@1000"),
    ]

    def get_value(
        item: Mapping[str, Any],
        path: Tuple[str, ...],
    ) -> float:
        value: Any = item
        for key in path:
            value = value[key]
        return float(value)

    summary: Dict[str, Any] = {}
    delta: Dict[str, float] = {}

    for path in scalar_paths:
        metric_name = ".".join(path)
        values = np.asarray(
            [get_value(run, path) for run in runs],
            dtype=np.float64,
        )
        mean_value = float(np.nanmean(values))

        summary[metric_name] = {
            "mean": mean_value,
            "std": (
                float(np.nanstd(values, ddof=1))
                if len(values) > 1
                else 0.0
            ),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }
        delta[metric_name] = (
            get_value(normal, path) - mean_value
        )

    return summary, delta


def install_diagnostic_eval_patch(options: argparse.Namespace) -> None:
    from src.training.trainer import OppTrainer

    @torch.no_grad()
    def diagnostic_eval_epoch(
        trainer: Any,
        loader: Any,
        epoch_idx: int,
    ) -> Dict[str, float]:
        if getattr(trainer, "_use_token_align", False):
            raise RuntimeError(
                "This script targets the pooled-GO P3a/B1 retriever. "
                "The loaded checkpoint is configured for token-align."
            )

        trainer.model.eval()
        device = trainer.device

        # Build the same complete evaluation space used by normal eval_epoch.
        trainer._refresh_eval_go_cache(chunk=trainer.cfg.eval_go_bs)
        trainer._eval_cache_ready = False
        trainer._ensure_eval_cache_v2(chunk=trainer.cfg.eval_go_bs)

        encoded_parts: List[torch.Tensor] = []
        truth_parts: List[torch.Tensor] = []
        protein_ids: List[str] = []
        raw_go_vectors: torch.Tensor | None = None

        for batch_index, batch in enumerate(loader):
            H = batch["prot_emb_pad"].to(device, non_blocking=True)
            if trainer.to_f32 is not None:
                H = trainer.to_f32(H)

            attention_valid, _ = trainer._valid_and_pad_masks(batch)
            G_eval, y_true = trainer._build_eval_space(batch)

            # Encode each protein once. Permutation is applied to this exact
            # representation, after protein pooling/projection/normalization.
            Zp = trainer.model.encode_protein_for_scoring(
                H,
                attention_valid,
            )
            encoded_parts.append(Zp.detach().float().cpu())
            truth_parts.append(y_true.detach().to(torch.int8).cpu())

            batch_protein_ids = batch.get("protein_ids")
            if batch_protein_ids is None:
                protein_ids.extend(
                    [
                        f"row_{len(protein_ids) + offset}"
                        for offset in range(H.shape[0])
                    ]
                )
            else:
                protein_ids.extend(str(pid) for pid in batch_protein_ids)

            # G_eval has shape [B,G,Dg] and is identical across batch rows.
            current_go = G_eval[0].detach().float().cpu().contiguous()
            if raw_go_vectors is None:
                raw_go_vectors = current_go
            else:
                if current_go.shape != raw_go_vectors.shape:
                    raise RuntimeError(
                        "Evaluation GO-space shape changed across batches: "
                        f"{tuple(raw_go_vectors.shape)} vs {tuple(current_go.shape)}"
                    )
                # Check a small deterministic slice; avoid a full costly compare.
                rows_to_check = min(8, current_go.shape[0])
                if not torch.allclose(
                    current_go[:rows_to_check],
                    raw_go_vectors[:rows_to_check],
                    atol=1e-5,
                    rtol=1e-4,
                ):
                    raise RuntimeError(
                        "G_eval contents changed across validation batches."
                    )

        if raw_go_vectors is None:
            raise RuntimeError("Validation loader yielded no batches.")

        encoded_proteins = torch.cat(encoded_parts, dim=0).contiguous()
        y_true_tensor = torch.cat(truth_parts, dim=0).contiguous()
        y_true = y_true_tensor.numpy().astype(np.int8)

        valid_rows = y_true.sum(axis=1) > 0
        if not np.any(valid_rows):
            raise RuntimeError(
                "No validation proteins have positives in eval_id_list."
            )

        valid_indices = np.flatnonzero(valid_rows)
        encoded_proteins = encoded_proteins.index_select(
            0,
            torch.as_tensor(valid_indices, dtype=torch.long),
        )
        y_true = y_true[valid_rows]
        protein_ids = [
            protein_ids[index] for index in valid_indices.tolist()
        ]

        eval_ids_tensor = getattr(trainer, "_eval_ids_cpu", None)
        if eval_ids_tensor is None:
            raise RuntimeError(
                "trainer._eval_ids_cpu was not created by _ensure_eval_cache_v2."
            )
        eval_go_ids = [
            int(value) for value in eval_ids_tensor.tolist()
        ]

        if len(eval_go_ids) != raw_go_vectors.shape[0]:
            raise RuntimeError(
                "GO ID count does not match G_eval rows: "
                f"{len(eval_go_ids)} vs {raw_go_vectors.shape[0]}"
            )

        print(
            "\n[Permutation diagnostic]\n"
            f"proteins={encoded_proteins.shape[0]}\n"
            f"GO terms={raw_go_vectors.shape[0]}\n"
            f"protein representation={tuple(encoded_proteins.shape)}\n"
            f"GO representation={tuple(raw_go_vectors.shape)}\n"
            f"eval_space={getattr(trainer.ctx, 'eval_id_list', None) is not None}"
        )

        if options.save_representations is not None:
            save_dir = options.save_representations
            save_dir.mkdir(parents=True, exist_ok=True)

            np.save(
                save_dir / "protein_z.float32.npy",
                encoded_proteins.numpy().astype(np.float32),
            )
            np.save(
                save_dir / "y_true.int8.npy",
                y_true,
            )
            np.save(
                save_dir / "go_raw.float32.npy",
                raw_go_vectors.numpy().astype(np.float32),
            )
            with (save_dir / "protein_ids.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(protein_ids, handle)
            with (save_dir / "eval_go_ids.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(eval_go_ids, handle)

        print("\nEvaluating normal protein identities...")
        normal_scores = score_all_go_terms(
            trainer=trainer,
            encoded_proteins=encoded_proteins,
            raw_go_vectors=raw_go_vectors,
            score_batch_size=options.score_batch_size,
        )
        normal_metrics = evaluate_scores(
            y_true=y_true,
            scores=normal_scores,
            num_thresholds=options.num_thresholds,
        )
        print(json.dumps(normal_metrics, indent=2))

        rng = np.random.default_rng(options.seed)
        permutation_runs: List[Dict[str, Any]] = []
        permutation_indices: List[List[int]] = []

        for run_index in range(options.num_permutations):
            permutation = make_derangement(
                rng,
                encoded_proteins.shape[0],
            )
            permuted_z = encoded_proteins.index_select(
                0,
                torch.as_tensor(permutation, dtype=torch.long),
            )

            print(
                f"\nPermutation "
                f"{run_index + 1}/{options.num_permutations}..."
            )
            permuted_scores = score_all_go_terms(
                trainer=trainer,
                encoded_proteins=permuted_z,
                raw_go_vectors=raw_go_vectors,
                score_batch_size=options.score_batch_size,
            )
            run_metrics = evaluate_scores(
                y_true=y_true,
                scores=permuted_scores,
                num_thresholds=options.num_thresholds,
            )
            run_metrics["run"] = run_index
            permutation_runs.append(run_metrics)
            permutation_indices.append(
                permutation.astype(int).tolist()
            )
            print(json.dumps(run_metrics, indent=2))

        permutation_summary, delta = summarize_permutations(
            normal_metrics,
            permutation_runs,
        )

        output = {
            "normal": normal_metrics,
            "permutation_runs": permutation_runs,
            "permutation_summary": permutation_summary,
            "delta_normal_minus_permuted": delta,
            "metadata": {
                "config": str(options.config),
                "checkpoint": str(options.checkpoint),
                "num_proteins": int(encoded_proteins.shape[0]),
                "num_go_terms": int(raw_go_vectors.shape[0]),
                "protein_representation_shape": list(
                    encoded_proteins.shape
                ),
                "go_representation_shape": list(
                    raw_go_vectors.shape
                ),
                "num_permutations": int(options.num_permutations),
                "seed": int(options.seed),
                "score_batch_size": int(options.score_batch_size),
                "protocol": (
                    "Global permutation of encoded protein representations "
                    "against the complete main.py evaluation GO space"
                ),
                "candidate_filtering": False,
                "semexp": False,
                "reranker": False,
            },
            "eval_go_ids": eval_go_ids,
            "permutation_indices": permutation_indices,
        }

        options.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with options.output.open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                output,
                handle,
                indent=2,
                allow_nan=True,
            )

        print(f"\nSaved diagnostic: {options.output}")
        print("\nNormal minus mean permuted:")
        print(json.dumps(delta, indent=2))

        # src.main expects a flat numeric dict for logging.
        flat_logs = {
            "perm_normal_protein_fmax": float(
                normal_metrics["protein_fmax"]
            ),
            "perm_mean_protein_fmax": float(
                permutation_summary["protein_fmax"]["mean"]
            ),
            "perm_delta_protein_fmax": float(
                delta["protein_fmax"]
            ),
            "perm_normal_mrr": float(
                normal_metrics["ranking"]["mrr"]
            ),
            "perm_mean_mrr": float(
                permutation_summary["ranking.mrr"]["mean"]
            ),
            "perm_delta_mrr": float(
                delta["ranking.mrr"]
            ),
        }
        return flat_logs

    OppTrainer.eval_epoch = diagnostic_eval_epoch


def main() -> None:
    options = parse_cli()

    if not options.config.exists():
        raise FileNotFoundError(options.config)
    if not options.checkpoint.exists():
        raise FileNotFoundError(options.checkpoint)
    if options.num_permutations < 1:
        raise ValueError("--num-permutations must be at least 1.")
    if options.score_batch_size < 1:
        raise ValueError("--score-batch-size must be positive.")

    # Install patch before importing and running src.main.main().
    install_diagnostic_eval_patch(options)

    from src import main as training_main

    with tempfile.TemporaryDirectory(
        prefix="protein_go_permutation_"
    ) as temporary_directory:
        temp_config = load_config_with_checkpoint(
            config_path=options.config,
            checkpoint_path=options.checkpoint,
            temp_dir=Path(temporary_directory),
        )

        previous_argv = sys.argv[:]
        try:
            sys.argv = [
                "src.main",
                "--config",
                str(temp_config),
            ]
            training_main.main()
        finally:
            sys.argv = previous_argv


if __name__ == "__main__":
    main()
