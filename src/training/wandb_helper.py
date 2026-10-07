from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt

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
            fig, ax = plt.subplots(
                figsize=(max(7.0, 1.25 * len(heat_metrics)), 5.5)
            )
            arr = np.asarray(matrix, dtype=np.float64)
            im = ax.imshow(arr, aspect="auto")
            ax.set_xticks(np.arange(len(heat_metrics)))
            ax.set_xticklabels(heat_metrics, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels)
            ax.set_xlabel("Metric")
            ax.set_ylabel("Protein positive cardinality")
            ax.set_title("Cardinality x Retrieval / PBC")

            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if np.isfinite(arr[i, j]):
                        ax.text(
                            j, i, f"{arr[i, j]:.3f}",
                            ha="center", va="center", fontsize=8,
                        )

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            payload["diagnostics/cardinality_retrieval_heatmap"] = wandb.Image(fig)
            plt.close(fig)

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

        fig, ax = plt.subplots(
            figsize=(
                max(6.0, 1.6 * len(segment_names)),
                max(8.0, 0.28 * n_show),
            )
        )
        arr = np.asarray(heat_matrix, dtype=np.float64)
        im = ax.imshow(arr, aspect="auto")
        ax.set_xticks(np.arange(len(segment_names)))
        ax.set_xticklabels(segment_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(heat_y)))
        ax.set_yticklabels(heat_y, fontsize=7)
        ax.set_xlabel("GO text segment")
        ax.set_ylabel("GO term")
        ax.set_title("GO Segment Weights, Top 50 Most Non-uniform Terms")

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(
                        j, i, f"{arr[i, j]:.2f}",
                        ha="center", va="center", fontsize=6,
                    )

        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        payload["diagnostics/go_segment_weights_heatmap"] = wandb.Image(fig)
        plt.close(fig)

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


class RerankerV2WandbLogger:
    """
    W&B logging for Experiment B:
    frozen Retriever-v2 dumps + low-rank interaction prediction head + ASL.

    Scientific metrics are computed elsewhere. This class only formats and
    logs training diagnostics, gradients, adapter movement, validation metrics,
    and run metadata.

    Safe to instantiate with run=None. All methods then become no-ops.
    """

    GRADIENT_GROUPS = {
        "protein_adapter_down": "protein_adapter.down.",
        "protein_adapter_up": "protein_adapter.up.",
        "go_adapter_down": "go_adapter.down.",
        "go_adapter_up": "go_adapter.up.",
        "predictor": "predictor.",
    }

    def __init__(self, run, cfg=None):
        self.run = run
        self.cfg = cfg

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
            go_universe_size: int,
            embedding_dim: int,
            adapter_rank: int,
            hidden_dim: int,
            dropout: float,
            gamma_pos: float,
            gamma_neg: float,
            asl_clip: float,
            train_dump: Optional[str] = None,
            val_dump: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        trainable_total = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        frozen_total = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

        summary = self.run.summary
        summary["architecture/experiment"] = "B"
        summary["architecture/model"] = "low_rank_interaction_prediction_head"
        summary["architecture/retriever"] = "frozen_retriever_v2_dump"
        summary["architecture/go_universe_size"] = int(go_universe_size)
        summary["architecture/embedding_dim"] = int(embedding_dim)
        summary["architecture/adapter_rank"] = int(adapter_rank)
        summary["architecture/hidden_dim"] = int(hidden_dim)
        summary["architecture/dropout"] = float(dropout)
        summary["architecture/ontology_size_independent"] = True
        summary["architecture/interaction"] = "adapted_hadamard+retriever_similarity"

        summary["objective/name"] = "asymmetric_loss"
        summary["objective/gamma_pos"] = float(gamma_pos)
        summary["objective/gamma_neg"] = float(gamma_neg)
        summary["objective/clip"] = float(asl_clip)

        summary["params/trainable_total"] = int(trainable_total)
        summary["params/frozen_total"] = int(frozen_total)

        if train_dump is not None:
            summary["data/train_dump"] = str(train_dump)
        if val_dump is not None:
            summary["data/val_dump"] = str(val_dump)

    # ------------------------------------------------------------------
    # ASL / prediction diagnostics
    # ------------------------------------------------------------------

    def log_train(
            self,
            *,
            diagnostics: Mapping[str, Any],
            lr: float,
            step: int,
    ) -> None:
        if not self.enabled:
            return

        key_map = {
            "loss": "train/asl_loss",
            "positive_loss_mean": "train/asl_positive_mean",
            "negative_loss_mean": "train/asl_negative_mean",
            "positive_probability_mean": "prediction/positive_probability_mean",
            "negative_probability_mean": "prediction/negative_probability_mean",
            "probability_gap": "prediction/probability_gap",
            "suppressed_negative_fraction": "asl/suppressed_negative_fraction",
            "easy_negative_fraction": "asl/easy_negative_fraction",
            "medium_negative_fraction": "asl/medium_negative_fraction",
            "hard_negative_fraction": "asl/hard_negative_fraction",
            "num_positive_pairs": "batch/num_positive_pairs",
            "num_negative_pairs": "batch/num_negative_pairs",
        }

        payload: Dict[str, Any] = {"train/lr": float(lr)}
        for source, target in key_map.items():
            if source not in diagnostics:
                continue
            value = self._scalar(diagnostics[source])
            if value is not None:
                payload[target] = value

        self._log(payload, step)

    # ------------------------------------------------------------------
    # Gradient diagnostics
    # ------------------------------------------------------------------

    def gradient_norms(self, model) -> Dict[str, float]:
        """
        Read gradients after loss.backward() and before clipping/optimizer.step().

        With zero-initialized adapter up projections, adapter down gradients may
        legitimately be zero on the first optimization step. Up projections and
        predictor should receive gradient immediately.
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
            total_before_clip: Optional[Any] = None,
    ) -> Dict[str, float]:
        norms = self.gradient_norms(model)

        if total_before_clip is not None:
            value = self._scalar(total_before_clip)
            if value is not None:
                norms["grad/total_before_clip"] = value

        if self.enabled:
            self._log(norms, step)

        return norms

    # ------------------------------------------------------------------
    # Residual-adapter movement
    # ------------------------------------------------------------------

    def log_adapter_state(
            self,
            *,
            protein_max_delta: Any,
            go_max_delta: Any,
            step: int,
    ) -> None:
        if not self.enabled:
            return

        payload: Dict[str, Any] = {}
        p = self._scalar(protein_max_delta)
        g = self._scalar(go_max_delta)

        if p is not None:
            payload["adapter/protein_max_delta"] = p
        if g is not None:
            payload["adapter/go_max_delta"] = g

        self._log(payload, step)

    # ------------------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------------------

    def log_validation(
            self,
            *,
            metrics: Mapping[str, Any],
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
