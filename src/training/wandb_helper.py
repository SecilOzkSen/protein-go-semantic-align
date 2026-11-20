from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None


_LOG = logging.getLogger(__name__)


def _configure_trainer_axis(run) -> None:
    """Use trainer_step as x-axis without conflating it with W&B history rows."""
    if run is not None:
        run.define_metric("trainer_step")
        run.define_metric("*", step_metric="trainer_step")


class RetrieverWandbLogger:
    """Common BP/CC/MF logger, with no ontology- or model-specific implementation.

    The trainer owns scientific metrics. This helper formats supplied scalars and
    diagnostics, logs unclipped gradients, and exports local CSV/JSON/PNG copies.
    Every event carries ``trainer_step``. W&B history indices advance normally,
    so gradients, train scalars and validation can share the same training step.
    A complete validation report is committed as one W&B event.

    With run=None, local diagnostic exports remain available if output_dir is set.
    The old constructor and public methods remain supported.
    """

    CARDINALITY_ORDER = ("1_5", "6_10", "11_20", "21_40", "41_80", "81_160", "161plus")
    DEFAULT_CDF_RANKS = (1, 5, 10, 50, 100, 200, 500, 1000)
    GRADIENT_GROUPS = {
        "protein_pool": "protein_mean_attn_gate_pool.",
        "protein_ln": "protein_ln.",
        "proj_p": "proj_p.",
        "proj_local": "proj_local.",
        "fusion_gate": "fusion_gate.",
        "go_segment_gate": "go_segment_gate.",
        "go_ln": "go_ln.",
        "proj_g": "proj_g.",
    }

    def __init__(self, run, cfg, *, go_ids=None, segment_names=None,
                 cdf_ranks=None, output_dir=None):
        self.run, self.cfg = run, cfg
        self.go_ids = [] if go_ids is None else list(go_ids)
        self.segment_names = [] if segment_names is None else list(segment_names)
        self.cdf_ranks = tuple(int(x) for x in (
            self.DEFAULT_CDF_RANKS if cdf_ranks is None else cdf_ranks))
        output_dir = output_dir if output_dir is not None else self._cfg("output_dir", None)
        self.local_dir = (Path(output_dir) / "diagnostics") if (
            output_dir is not None and self._cfg("save_diagnostics_local", True)) else None
        self.max_table_rows = max(1, int(self._cfg("diagnostics_max_table_rows", 200)))
        self.heatmap_rows = max(1, int(self._cfg("diagnostics_heatmap_rows", 50)))
        self.diagnostic_interval = max(1, int(self._cfg("diagnostics_log_every_epochs", 1)))
        self._collecting = None
        self._current_epoch = None
        self._unverified_weights_warned = False
        self._validation_history = []
        if self.enabled and wandb is None:
            raise RuntimeError("A W&B run was supplied but wandb is unavailable.")
        _configure_trainer_axis(run)

    def _cfg(self, name, default):
        if isinstance(self.cfg, Mapping):
            return self.cfg.get(name, default)
        return getattr(self.cfg, name, default)

    @property
    def enabled(self):
        return self.run is not None

    @property
    def active(self):
        return self.enabled or self.local_dir is not None

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value) and value.numel() == 1:
            value = value.detach().item()
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            return value if np.isfinite(value) else None
        return None

    @staticmethod
    def _array(value, dtype=None):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def _json_value(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return RetrieverWandbLogger._json_value(value.tolist())
        if isinstance(value, np.generic):
            return RetrieverWandbLogger._json_value(value.item())
        if isinstance(value, Mapping):
            return {str(k): RetrieverWandbLogger._json_value(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [RetrieverWandbLogger._json_value(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value

    def _folder(self, step):
        if self.local_dir is None:
            return None
        label = (f"epoch_{int(self._current_epoch):04d}_" if self._current_epoch is not None else "")
        folder = self.local_dir / f"{label}step_{int(step):09d}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _save_json(self, name, data, step):
        folder = self._folder(step)
        if folder is not None:
            (folder / f"{name}.json").write_text(
                json.dumps(self._json_value(data), indent=2, allow_nan=False), encoding="utf-8")

    def _save_csv(self, name, columns, rows, step):
        folder = self._folder(step)
        if folder is not None:
            with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerows([[self._json_value(x) for x in row] for row in rows])

    def _save_figure(self, name, fig, step):
        folder = self._folder(step)
        if folder is not None:
            fig.savefig(folder / f"{name}.png", dpi=130, bbox_inches="tight")

    def _log(self, payload, step):
        if not payload:
            return
        payload = dict(payload)
        payload["trainer_step"] = int(step)
        if self._collecting is not None:
            self._collecting.update(payload)
        elif self.enabled:
            # No explicit W&B history step. Repeated trainer_step values are valid.
            self.run.log(payload, commit=True)

    @staticmethod
    def _format_go_id(go_id):
        text = str(go_id)
        if text.upper().startswith("GO:"):
            return text
        try:
            return f"GO:{int(go_id):07d}"
        except (TypeError, ValueError):
            return text

    def log_run_metadata(self, *, model, active_go_ids, protein_pooling,
                         go_segments, objective="full_go_infonce+pbr"):
        if not self.active:
            return
        metadata_fn = getattr(model, "architecture_metadata", None)
        architecture = dict(metadata_fn()) if callable(metadata_fn) else {}
        if "protein_mode" not in architecture:
            mode = getattr(model, "protein_mode", None)
            variant = getattr(model, "global_pool_variant", None)
            architecture["protein_mode"] = mode or (f"global_{variant}" if variant else "unspecified")
        # Configured legacy pooling can differ from the actual scoring path.
        architecture["configured_protein_pooling"] = str(protein_pooling)
        architecture.setdefault("protein_pooling", architecture.get("protein_mode", "unspecified"))
        architecture["go_bank_segments"] = list(go_segments)
        architecture["go_segments"] = ",".join(
            architecture.get("score_segments") or self.segment_names or list(go_segments))
        architecture["go_universe_size"] = len(active_go_ids)
        architecture["objective"] = str(objective)
        encoder = getattr(model, "go_encoder", None)
        go_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad) if encoder is not None else 0
        if go_trainable:
            raise RuntimeError(f"Frozen GO encoder invariant violated: {go_trainable} trainable parameters")
        parameter_stats = {
            "trainable_total": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_total": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "go_text_encoder_trainable": go_trainable,
        }
        if self.enabled:
            for key, value in architecture.items():
                self.run.summary[f"architecture/{key}"] = self._json_value(value)
            for key, value in parameter_stats.items():
                self.run.summary[f"params/{key}"] = int(value)
            self.run.summary["logging/history_axis"] = "trainer_step"
            self.run.summary["logging/diagnostics_max_table_rows"] = self.max_table_rows
            self.run.summary["logging/local_diagnostics"] = str(self.local_dir) if self.local_dir else None
        if self.local_dir is not None:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            (self.local_dir / "run_metadata.json").write_text(json.dumps(
                self._json_value({"architecture": architecture, "parameters": parameter_stats}),
                indent=2, allow_nan=False), encoding="utf-8")

    def log_train(self, *, losses, lr, step):
        if not self.active:
            return
        key_map = {
            "total": "train/total_loss", "contrastive": "train/infonce_loss",
            "pbr": "train/pbr_loss", "pbr_weighted": "train/pbr_weighted",
            "pbr_enabled": "train/pbr_enabled",
            "local_pbr": "train/local_pbr_loss", "local_pbr_weighted": "train/local_pbr_weighted",
            "positive_score_mean": "scores/positive_mean", "negative_score_mean": "scores/negative_mean",
            "pos_neg_score_gap": "scores/pos_neg_gap",
            "pbr_violation_sum_per_positive": "pbr/negative_invasion_sum_per_positive",
            "pbr_pair_violation_mean": "pbr/pair_violation_mean",
            "pbr_margin_violation_fraction": "pbr/margin_violation_fraction",
            "pbr_margin_satisfied_fraction": "pbr/margin_satisfied_fraction",
            "positive_block_coverage": "pbr/positive_block_coverage_batch",
            "positive_block_violation": "pbr/positive_block_violation_batch",
            "positive_rank_mean": "pbr/positive_rank_mean_batch",
            "positive_rank_median": "pbr/positive_rank_median_batch",
            "positive_rank_worst_mean": "pbr/positive_rank_worst_mean_batch",
        }
        payload = {"train/lr": float(lr)}
        for source, target in key_map.items():
            value = self._scalar(losses.get(source))
            if value is not None:
                payload[target] = value
        if "train/pbr_loss" not in payload:
            value = self._scalar(losses.get("ranking_loss"))
            if value is not None:
                payload["train/pbr_loss"] = value
        if "train/pbr_weighted" not in payload and "train/local_pbr_weighted" in payload:
            payload["train/pbr_weighted"] = payload["train/local_pbr_weighted"]
        for key, value in losses.items():
            scalar = self._scalar(value)
            if scalar is None:
                continue
            if key.startswith("positive_rank_q"):
                payload[f"pbr/{key}_batch"] = scalar
            elif key.startswith("go_segment_weight/"):
                payload[f"go_segments/{key.split('/', 1)[1]}_mean_train"] = scalar
            elif key.startswith(("h/", "aggregation/")):
                payload[key] = scalar
        self._log(payload, step)

    def gradient_norms(self, model):
        result = {}
        named = list(model.named_parameters())
        for label, prefix in self.GRADIENT_GROUPS.items():
            params = [p for n, p in named if n.startswith(prefix)]
            grads = [p.grad.detach().float() for p in params if p.grad is not None]
            # One GPU synchronization per group instead of per parameter.
            squared = torch.stack([g.square().sum() for g in grads]).sum() if grads else None
            result[f"grad/{label}"] = float(squared.sqrt().item()) if squared is not None else 0.0
            result[f"grad_status/{label}_trainable"] = int(any(p.requires_grad for p in params))
            result[f"grad_status/{label}_has_gradient"] = int(bool(grads))
        return result

    def log_gradients(self, *, model, step):
        norms = self.gradient_norms(model)
        self._log(norms, step)
        return norms

    def log_validation(self, *, metrics, diagnostics, step, epoch=None):
        if not self.active:
            return
        if self._collecting is not None:
            raise RuntimeError("Nested validation logging is not supported")
        self._current_epoch = epoch
        scalar_metrics = {k: value for k, raw in metrics.items()
                          if (value := self._scalar(raw)) is not None}
        record = {"epoch": epoch, "trainer_step": int(step), **scalar_metrics}
        self._validation_history.append(record)
        self._save_json("validation_metrics", record, step)
        if self.local_dir is not None:
            with (self.local_dir / "validation_history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(self._json_value(record), allow_nan=False) + "\n")
        payload = {f"val/{k}": value for k, value in scalar_metrics.items()}
        if epoch is not None:
            payload["epoch"] = int(epoch)
        self._collecting = payload
        availability, errors = {}, {}
        diagnostics = {} if diagnostics is None else diagnostics
        should_log = epoch is None or int(epoch) % self.diagnostic_interval == 0
        tasks = [
            ("positive_rank_cdf", self._cfg("log_positive_rank_cdf", True), diagnostics.get("positive_ranks"),
             lambda: self.log_positive_rank_cdf(diagnostics.get("positive_ranks"), step=step)),
            ("cardinality", self._cfg("log_cardinality_metrics", True), diagnostics.get("cardinality_table"),
             lambda: self.log_cardinality_diagnostics(diagnostics.get("cardinality_table"), step=step)),
            ("go_segment_weights", self._cfg("log_go_segment_weights", True), diagnostics.get("go_segment_weights"),
             lambda: self.log_go_segment_diagnostics(
                 go_ids=diagnostics.get("go_ids", self.go_ids),
                 segment_names=diagnostics.get("segment_names", self.segment_names),
                 weights=diagnostics.get("go_segment_weights"),
                 present=diagnostics.get("go_segment_present"),
                 source=diagnostics.get("go_segment_weight_source", "unverified"),
                 aggregation=diagnostics.get("segment_aggregation", "unspecified"),
                 gate_conditioning=diagnostics.get("gate_conditioning", "unspecified"), step=step)),
            ("go_term_table", self._cfg("log_go_term_metrics", self._cfg("log_go_term_table", True)), diagnostics.get("go_term_table"),
             lambda: self.log_go_term_diagnostics(diagnostics.get("go_term_table"), step=step)),
        ]
        try:
            for name, enabled, supplied, fn in tasks:
                if not enabled:
                    availability[name] = "disabled"
                elif supplied is None:
                    availability[name] = "not_supplied_by_trainer"
                elif hasattr(supplied, "__len__") and len(supplied) == 0:
                    availability[name] = "empty"
                elif not should_log:
                    availability[name] = "skipped_by_interval"
                else:
                    try:
                        fn()
                        availability[name] = "logged"
                    except Exception as exc:
                        # Visualization must not discard a completed GPU epoch.
                        _LOG.exception("Could not log diagnostic %s at trainer_step=%d", name, step)
                        availability[name] = "error"
                        errors[name] = f"{type(exc).__name__}: {exc}"
            if should_log:
                history_cols = ["epoch", "trainer_step", *scalar_metrics]
                history_data = [[r.get(k) for k in history_cols] for r in self._validation_history]
                self._save_csv("epoch_summary", history_cols, history_data, step)
                if self.enabled:
                    payload["diagnostics/epoch_summary"] = wandb.Table(
                        columns=history_cols, data=history_data[-self.max_table_rows:])
            status = {"availability": availability, "errors": errors,
                      "table_max_rows": self.max_table_rows, "heatmap_max_rows": self.heatmap_rows}
            self._save_json("diagnostic_status", status, step)
            if self.enabled:
                status_rows = [[name, value, errors.get(name)] for name, value in availability.items()]
                payload["diagnostics/availability_table"] = wandb.Table(
                    columns=["diagnostic", "status", "error"], data=status_rows)
                for key, value in availability.items():
                    self.run.summary[f"diagnostics/status/{key}"] = value
                self.run.summary["diagnostics/last_error_count"] = len(errors)
                payload["diagnostics/error_count"] = len(errors)
        finally:
            self._collecting = None
            self._current_epoch = None
        self._log(payload, step)

    def log_positive_rank_cdf(self, positive_ranks, *, step):
        if not self.active or positive_ranks is None:
            return
        ranks = self._array(positive_ranks, np.float64).reshape(-1)
        ranks = ranks[np.isfinite(ranks)]
        if not ranks.size:
            return
        cutoffs = sorted({x for x in self.cdf_ranks if x > 0} | {int(np.max(ranks))})
        columns = ["rank", "fraction_positive_retrieved"]
        rows = [[int(k), float(np.mean(ranks <= k))] for k in cutoffs]
        self._save_csv("positive_rank_cdf", columns, rows, step)
        if self.enabled:
            table = wandb.Table(columns=columns, data=rows)
            self._log({"diagnostics/positive_rank_cdf_table": table,
                       "diagnostics/positive_rank_cdf": wandb.plot.line(
                           table, x=columns[0], y=columns[1], title="Positive Rank CDF")}, step)

    def log_cardinality_diagnostics(self, rows, *, step):
        if not self.active or rows is None or len(rows) == 0:
            return
        lookup = {str(row["cardinality_bin"]): dict(row) for row in rows}
        order = [x for x in self.CARDINALITY_ORDER if x in lookup]
        order += sorted(set(lookup) - set(order))
        ordered = [lookup[x] for x in order]
        recall_cols = sorted({k for r in ordered for k in r if k.startswith("R@")},
                             key=lambda x: int(x.split("@", 1)[1]))
        columns = ["cardinality_bin", "n_proteins", *recall_cols, "PBC", "PBV",
                   "positive_rank_median", "positive_rank_p90"]
        data = [[self._json_value(row.get(col)) for col in columns] for row in ordered]
        self._save_csv("cardinality", columns, data, step)
        payload = {}
        if self.enabled:
            payload["diagnostics/cardinality_table"] = wandb.Table(columns=columns, data=data)
        heat_cols = recall_cols + ["PBC"]
        arr = np.asarray([[row.get(k, np.nan) for k in heat_cols] for row in ordered], dtype=float)
        fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(heat_cols)), max(3.0, 0.6 * len(ordered))))
        try:
            im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(heat_cols)), heat_cols, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(order)), order)
            ax.set(xlabel="Metric", ylabel="Positive GO count", title="Cardinality and retrieval")
            for i, j in np.ndindex(arr.shape):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            self._save_figure("cardinality_retrieval_heatmap", fig, step)
            if self.enabled:
                payload["diagnostics/cardinality_retrieval_heatmap"] = wandb.Image(fig)
        finally:
            plt.close(fig)
        self._log(payload, step)

    def log_go_segment_diagnostics(self, *, go_ids, segment_names, weights, step,
                                   present=None, source="unverified", aggregation="unspecified",
                                   gate_conditioning="unspecified"):
        if not self.active or weights is None:
            return
        weights = self._array(weights, np.float64)
        go_ids = [] if go_ids is None else list(go_ids)
        names = [] if segment_names is None else list(segment_names)
        if weights.ndim != 2 or weights.shape != (len(go_ids), len(names)):
            raise ValueError(f"Expected [{len(go_ids)},{len(names)}] scoring weights, got {weights.shape}")
        if not weights.size:
            return
        if not np.isfinite(weights).all() or np.any(weights < -1e-6):
            raise ValueError("Segment weights must be finite and nonnegative")
        if not np.allclose(weights.sum(1), 1, rtol=1e-4, atol=1e-4):
            raise ValueError("Segment weights do not sum to one")
        presence = None if present is None else self._array(present, bool)
        if presence is not None:
            if presence.shape != weights.shape or not presence.any(1).all():
                raise ValueError("Segment presence mask has invalid shape or an all-missing row")
            if np.any(np.abs(weights[~presence]) > 1e-5):
                raise ValueError("Missing segment has nonzero scoring weight")
            uniform = presence / presence.sum(1, keepdims=True)
        else:
            uniform = np.full(weights.shape, 1.0 / weights.shape[1])
        verified = source == "scoring"
        if not verified and not self._unverified_weights_warned:
            _LOG.warning("Segment weights lack scoring provenance. They are logged as unverified; "
                         "legacy presence-normalized weights are not evidence of learned attention.")
            self._unverified_weights_warned = True
        deviation = np.abs(weights - uniform).sum(1)
        selected = np.argsort(-deviation, kind="stable")
        columns = ["go_id", *names, *[f"{name}_present" for name in names],
                   "deviation_from_available_uniform", "weight_source"]
        rows = [[self._format_go_id(go_ids[i]), *weights[i].tolist(),
                 *(presence[i].astype(int).tolist() if presence is not None else [None] * len(names)),
                 float(deviation[i]), str(source)] for i in range(len(go_ids))]
        self._save_csv("go_segment_weights", columns, rows, step)
        self._save_json("go_segment_semantics", {
            "source": source, "aggregation": aggregation, "gate_conditioning": gate_conditioning,
            "segments": names, "n_terms": len(go_ids),
            "parent_note": "Parent correction is not a learned segment weight unless explicitly supplied.",
            "selection": "largest L1 departure from uniform over available segments",
        }, step)
        payload = {f"go_segments/{name}_mean_val": float(weights[:, j].mean())
                   for j, name in enumerate(names)}
        payload.update({"diagnostics/go_segment_rows_total": len(rows),
                        "diagnostics/go_segment_rows_displayed": min(len(rows), self.max_table_rows),
                        "diagnostics/go_segment_weights_verified": int(verified)})
        if self.enabled:
            payload["diagnostics/go_segment_weights_table"] = wandb.Table(
                columns=columns, data=[rows[i] for i in selected[:self.max_table_rows]])
            self.run.summary["diagnostics/go_segment_weight_source"] = str(source)
            self.run.summary["diagnostics/segment_aggregation"] = str(aggregation)
        top = selected[:self.heatmap_rows]
        arr = weights[top]
        fig, ax = plt.subplots(figsize=(max(6.0, len(names) * 1.8), max(4.0, len(top) * 0.25)))
        try:
            im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(top)), [self._format_go_id(go_ids[i]) for i in top], fontsize=7)
            title = f"{aggregation} scoring weights" if verified else "Reported segment weights, unverified source"
            ax.set(xlabel="Scored GO text segment", ylabel="GO term", title=title)
            for i, j in np.ndindex(arr.shape):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            fig.tight_layout()
            self._save_figure("go_segment_weights_heatmap", fig, step)
            if self.enabled:
                payload["diagnostics/go_segment_weights_heatmap"] = wandb.Image(fig)
        finally:
            plt.close(fig)
        self._log(payload, step)

    def log_go_term_diagnostics(self, rows, *, step):
        """Display trainer-computed term recall/frequencies; export all supplied rows."""
        if not self.active or rows is None or len(rows) == 0:
            return
        rows = [dict(row) for row in rows]
        preferred = ["go_id", "train_frequency", "validation_positive_count", "has_validation_positive",
                     "name_weight", "definition_weight", "definition_present"]
        keys = {k for row in rows for k in row}
        columns = [k for k in preferred if k in keys] + sorted(keys - set(preferred))
        data = [[self._json_value(row.get(col)) for col in columns] for row in rows]
        if "go_id" in columns:
            ix = columns.index("go_id")
            for row in data:
                row[ix] = self._format_go_id(row[ix])
        self._save_csv("go_term_metrics", columns, data, step)
        # Frequency-spanning deterministic sample, not a top-performing-term sample.
        order = sorted(range(len(rows)), key=lambda i: (
            float(rows[i].get("train_frequency") or 0), str(rows[i].get("go_id", ""))))
        positions = np.linspace(0, len(order) - 1, min(len(order), self.max_table_rows), dtype=int)
        sample = [data[order[i]] for i in positions]
        payload = {"diagnostics/go_term_rows_total": len(rows),
                   "diagnostics/go_term_rows_displayed": len(sample)}
        if self.enabled:
            table = wandb.Table(columns=columns, data=sample)
            payload["diagnostics/go_term_table"] = table
            recall_key = next((k for k in ("recall@500", "R@500", "term_recall@500") if k in columns), None)
            if recall_key and "train_frequency" in columns:
                fi, ri = columns.index("train_frequency"), columns.index(recall_key)
                scatter_rows = [r for r in sample if r[fi] is not None and r[ri] is not None]
                if scatter_rows:
                    scatter_table = wandb.Table(columns=columns, data=scatter_rows)
                    payload["diagnostics/go_recall_vs_frequency"] = wandb.plot.scatter(
                        scatter_table, x="train_frequency", y=recall_key,
                        title="Validation term recall vs training frequency (sampled terms)")
        self._log(payload, step)

    def finalize(self, *, best_metrics=None):
        if best_metrics:
            values = {key: scalar for key, raw in best_metrics.items()
                      if (scalar := self._scalar(raw)) is not None}
            if self.enabled:
                for key, value in values.items():
                    self.run.summary[f"best/{key}"] = value
            if self.local_dir is not None:
                self.local_dir.mkdir(parents=True, exist_ok=True)
                (self.local_dir / "best_metrics.json").write_text(
                    json.dumps(values, indent=2), encoding="utf-8")
        if self.enabled:
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
        _configure_trainer_axis(run)

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
        self.run.log(payload, commit=True)

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

