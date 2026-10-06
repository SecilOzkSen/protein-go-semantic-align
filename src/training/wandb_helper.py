from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None


class RetrieverWandbLogger:
    """
    W&B logging/visualization for Retriever v2.

    This class does NOT compute scientific metrics such as PBC, MRR, nDCG,
    Recall@K, or oracle F. Those belong to trainer/evaluation code.

    It only:
      - formats scalar metrics for W&B,
      - reads gradients after backward(),
      - turns trainer diagnostics into W&B Tables/plots,
      - records static run metadata.

    Safe to instantiate with run=None. All methods then become no-ops.
    """

    CARDINALITY_ORDER = (
        "1_5",
        "6_10",
        "11_20",
        "21_40",
        "41_80",
        "81_160",
        "161plus",
    )

    DEFAULT_CDF_RANKS = (1, 5, 10, 50, 100, 200, 500, 1000)

    GRADIENT_GROUPS = {
        "protein_pool": "protein_mean_attn_gate_pool.",
        "protein_ln": "protein_ln.",
        "proj_p": "proj_p.",
        "go_segment_gate": "go_segment_gate.",
        "go_ln": "go_ln.",
        "proj_g": "proj_g.",
    }

    def __init__(
            self,
            run,
            cfg,
            *,
            go_ids: Optional[Sequence[int]] = None,
            segment_names: Optional[Sequence[str]] = None,
            cdf_ranks: Optional[Sequence[int]] = None,
    ):
        self.run = run
        self.cfg = cfg
        self.go_ids = list(go_ids or [])
        self.segment_names = list(segment_names or [])
        self.cdf_ranks = tuple(
            int(x) for x in (cdf_ranks or self.DEFAULT_CDF_RANKS)
        )

        if self.enabled and wandb is None:
            raise RuntimeError(
                "A W&B run was supplied but the wandb package is unavailable."
            )

    @property
    def enabled(self) -> bool:
        return self.run is not None

    @staticmethod
    def _scalar(value: Any) -> Optional[float]:
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            return value if np.isfinite(value) else None

        if torch.is_tensor(value) and value.numel() == 1:
            value = float(value.detach().item())
            return value if np.isfinite(value) else None

        return None

    def _log(self, payload: Dict[str, Any], step: int) -> None:
        if not self.enabled or not payload:
            return
        payload = dict(payload)
        payload["trainer_step"] = int(step)
        self.run.log(payload, step=int(step))

    # ------------------------------------------------------------------
    # Static run metadata
    # ------------------------------------------------------------------

    def log_run_metadata(
            self,
            *,
            model,
            active_go_ids: Sequence[int],
            protein_pooling: str,
            go_segments: Sequence[str],
            objective: str = "full_go_infonce+pbr",
    ) -> None:
        if not self.enabled:
            return

        trainable_total = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        frozen_total = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

        go_encoder = getattr(model, "go_encoder", None)
        go_text_trainable = 0
        if go_encoder is not None:
            go_text_trainable = sum(
                p.numel() for p in go_encoder.parameters() if p.requires_grad
            )

        self.run.summary["architecture/protein_mode"] = "single_vector"
        self.run.summary["architecture/protein_pooling"] = str(protein_pooling)
        self.run.summary["architecture/go_segments"] = ",".join(go_segments)
        self.run.summary["architecture/go_universe_size"] = int(
            len(active_go_ids)
        )
        self.run.summary["architecture/objective"] = objective

        self.run.summary["params/trainable_total"] = int(trainable_total)
        self.run.summary["params/frozen_total"] = int(frozen_total)
        self.run.summary["params/go_text_encoder_trainable"] = int(
            go_text_trainable
        )

        # Retriever-v2 invariant.
        if go_text_trainable != 0:
            raise RuntimeError(
                "Retriever v2 invariant violated: GO text encoder has "
                f"{go_text_trainable} trainable parameters."
            )

    # ------------------------------------------------------------------
    # Training scalars
    # ------------------------------------------------------------------

    def log_train(
            self,
            *,
            losses: Mapping[str, Any],
            lr: float,
            step: int,
    ) -> None:
        if not self.enabled:
            return

        key_map = {
            "total": "train/total_loss",
            "contrastive": "train/infonce_loss",
            "pbr": "train/pbr_loss",
            "pbr_weighted": "train/pbr_weighted",
            "positive_score_mean": "scores/positive_mean",
            "negative_score_mean": "scores/negative_mean",
            "pos_neg_score_gap": "scores/pos_neg_gap",
            "pbr_violation_sum_per_positive":
                "pbr/negative_invasion_sum_per_positive",
            "pbr_pair_violation_mean":
                "pbr/pair_violation_mean",
            "pbr_margin_violation_fraction":
                "pbr/margin_violation_fraction",
            "pbr_margin_satisfied_fraction":
                "pbr/margin_satisfied_fraction",
            "positive_block_coverage":
                "pbr/positive_block_coverage_batch",
            "positive_block_violation":
                "pbr/positive_block_violation_batch",
            "positive_rank_mean":
                "pbr/positive_rank_mean_batch",
            "positive_rank_median":
                "pbr/positive_rank_median_batch",
            "positive_rank_worst_mean":
                "pbr/positive_rank_worst_mean_batch",
        }

        payload: Dict[str, Any] = {
            "train/lr": float(lr),
        }

        for source, target in key_map.items():
            if source not in losses:
                continue
            value = self._scalar(losses[source])
            if value is not None:
                payload[target] = value

        # Quantiles, e.g. positive_rank_q50/q75/q90/q95.
        for key, raw in losses.items():
            if key.startswith("positive_rank_q"):
                value = self._scalar(raw)
                if value is not None:
                    payload[f"pbr/{key}_batch"] = value

            if key.startswith("go_segment_weight/"):
                value = self._scalar(raw)
                if value is not None:
                    name = key.split("/", 1)[1]
                    payload[f"go_segments/{name}_mean_train"] = value

        self._log(payload, step)

    # ------------------------------------------------------------------
    # Gradient diagnostics
    # ------------------------------------------------------------------

    def gradient_norms(self, model) -> Dict[str, float]:
        """
        Read gradients after loss.backward() and before clipping/optimizer.step().

        Group norm is the L2 norm over all parameter gradients in the module,
        not the sum of individual tensor norms.
        """
        result: Dict[str, float] = {}

        for label, prefix in self.GRADIENT_GROUPS.items():
            sq_sum = 0.0
            found = False

            for name, param in model.named_parameters():
                if not name.startswith(prefix) or param.grad is None:
                    continue

                grad = param.grad.detach().float()
                sq_sum += float(torch.sum(grad * grad).item())
                found = True

            result[f"grad/{label}"] = (
                float(np.sqrt(sq_sum)) if found else 0.0
            )

        return result

    def log_gradients(
            self,
            *,
            model,
            step: int,
    ) -> Dict[str, float]:
        norms = self.gradient_norms(model)
        if self.enabled:
            self._log(norms, step)
        return norms

    # ------------------------------------------------------------------
    # Validation scalars
    # ------------------------------------------------------------------

    def log_validation(
            self,
            *,
            metrics: Mapping[str, Any],
            diagnostics: Optional[Mapping[str, Any]],
            step: int,
            epoch: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return

        payload: Dict[str, Any] = {}

        for key, raw in metrics.items():
            value = self._scalar(raw)
            if value is not None:
                payload[f"val/{key}"] = value

        if epoch is not None:
            payload["epoch"] = int(epoch)

        self._log(payload, step)

        if not diagnostics:
            return

        if getattr(self.cfg, "log_positive_rank_cdf", True):
            self.log_positive_rank_cdf(
                diagnostics.get("positive_ranks"),
                step=step,
            )

        if getattr(self.cfg, "log_cardinality_metrics", True):
            self.log_cardinality_diagnostics(
                diagnostics.get("cardinality_table"),
                step=step,
            )

        if getattr(self.cfg, "log_go_segment_weights", True):
            self.log_go_segment_diagnostics(
                go_ids=diagnostics.get("go_ids", self.go_ids),
                segment_names=diagnostics.get(
                    "segment_names",
                    self.segment_names,
                ),
                weights=diagnostics.get("go_segment_weights"),
                step=step,
            )

    # ------------------------------------------------------------------
    # Positive-rank CDF
    # ------------------------------------------------------------------

    def log_positive_rank_cdf(
            self,
            positive_ranks,
            *,
            step: int,
    ) -> None:
        if not self.enabled or positive_ranks is None:
            return

        ranks = np.asarray(positive_ranks, dtype=np.float64).reshape(-1)
        ranks = ranks[np.isfinite(ranks)]
        if ranks.size == 0:
            return

        x_values = sorted(
            {
                int(x)
                for x in self.cdf_ranks
                if int(x) > 0
            }
            | {int(np.max(ranks))}
        )

        rows = [
            [
                int(rank_cutoff),
                float(np.mean(ranks <= rank_cutoff)),
            ]
            for rank_cutoff in x_values
        ]

        table = wandb.Table(
            columns=["rank", "fraction_positive_retrieved"],
            data=rows,
        )

        plot = wandb.plot.line(
            table,
            x="rank",
            y="fraction_positive_retrieved",
            title="Positive Rank CDF",
        )

        self._log(
            {
                "diagnostics/positive_rank_cdf": plot,
                "diagnostics/positive_rank_cdf_table": table,
            },
            step,
        )

    # ------------------------------------------------------------------
    # Cardinality table / heatmap
    # ------------------------------------------------------------------

    def log_cardinality_diagnostics(
            self,
            rows,
            *,
            step: int,
    ) -> None:
        if not self.enabled or not rows:
            return

        row_by_bin = {
            str(row["cardinality_bin"]): dict(row)
            for row in rows
        }

        ordered = [
            row_by_bin[label]
            for label in self.CARDINALITY_ORDER
            if label in row_by_bin
        ]
        if not ordered:
            return

        # Stable column order.
        recall_cols = sorted(
            {
                key
                for row in ordered
                for key in row.keys()
                if key.startswith("R@")
            },
            key=lambda x: int(x.split("@", 1)[1]),
        )

        columns = [
            "cardinality_bin",
            "n_proteins",
            *recall_cols,
            "PBC",
            "PBV",
            "positive_rank_median",
            "positive_rank_p90",
        ]

        table_data = [
            [row.get(col, float("nan")) for col in columns]
            for row in ordered
        ]

        table = wandb.Table(
            columns=columns,
            data=table_data,
        )

        payload: Dict[str, Any] = {
            "diagnostics/cardinality_table": table,
        }

        # W&B native heatmap accepts one matrix with categorical axes.
        heat_metrics = [
                           col for col in recall_cols
                           if col in {"R@50", "R@100", "R@200", "R@500", "R@1000"}
                       ] + ["PBC"]

        matrix = []
        y_labels = []
        for row in ordered:
            y_labels.append(str(row["cardinality_bin"]))
            matrix.append(
                [
                    float(row.get(metric, np.nan))
                    for metric in heat_metrics
                ]
            )

        if heat_metrics and matrix:
            payload["diagnostics/cardinality_retrieval_heatmap"] = (
                wandb.plots.HeatMap(
                    x_labels=heat_metrics,
                    y_labels=y_labels,
                    matrix_values=matrix,
                    show_text=True,
                )
            )

        self._log(payload, step)

    # ------------------------------------------------------------------
    # GO segment table / heatmap
    # ------------------------------------------------------------------

    def log_go_segment_diagnostics(
            self,
            *,
            go_ids,
            segment_names,
            weights,
            step: int,
    ) -> None:
        if not self.enabled or weights is None:
            return

        weights = np.asarray(weights, dtype=np.float64)
        go_ids = list(go_ids or [])
        segment_names = list(segment_names or [])

        if weights.ndim != 2:
            raise ValueError(
                "GO segment weights must have shape [G,S], got "
                f"{weights.shape}"
            )

        G, S = weights.shape
        if len(go_ids) != G:
            raise ValueError(
                f"GO id count {len(go_ids)} != weight rows {G}"
            )
        if len(segment_names) != S:
            raise ValueError(
                f"Segment-name count {len(segment_names)} != weight columns {S}"
            )

        # Mean segment weights as simple scalars.
        payload: Dict[str, Any] = {}
        means = np.nanmean(weights, axis=0)
        for i, name in enumerate(segment_names):
            payload[f"go_segments/{name}_mean_val"] = float(means[i])

        # Full GO x segment table, useful for sorting/filtering in W&B.
        table = wandb.Table(
            columns=["go_id", *segment_names],
            data=[
                [
                    self._format_go_id(go_ids[g]),
                    *[float(x) for x in weights[g].tolist()],
                ]
                for g in range(G)
            ],
        )
        payload["diagnostics/go_segment_weights_table"] = table

        # A 1943-row heatmap is unreadable. Show the GO terms with the largest
        # departure from uniform segment weighting, while keeping the full table.
        uniform = np.full((1, S), 1.0 / max(1, S), dtype=np.float64)
        deviation = np.abs(weights - uniform).sum(axis=1)
        n_show = min(50, G)
        selected = np.argsort(-deviation)[:n_show]

        heat_matrix = weights[selected].tolist()
        heat_y = [self._format_go_id(go_ids[i]) for i in selected]

        payload["diagnostics/go_segment_weights_heatmap"] = (
            wandb.plots.HeatMap(
                x_labels=segment_names,
                y_labels=heat_y,
                matrix_values=heat_matrix,
                show_text=True,
            )
        )

        self._log(payload, step)

    @staticmethod
    def _format_go_id(go_id: Any) -> str:
        text = str(go_id)
        if text.upper().startswith("GO:"):
            return text
        try:
            return f"GO:{int(go_id):07d}"
        except (TypeError, ValueError):
            return text

    # ------------------------------------------------------------------
    # End of run
    # ------------------------------------------------------------------

    def finalize(
            self,
            *,
            best_metrics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        if best_metrics:
            for key, raw in best_metrics.items():
                value = self._scalar(raw)
                if value is not None:
                    self.run.summary[f"best/{key}"] = value

        self.run.finish()
