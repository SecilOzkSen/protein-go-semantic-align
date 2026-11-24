"""Resolve the MF B3-PC YAML; no architecture selection via environment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


# This package adds only protein conditioning to the paper plain-InfoNCE B3.
PACKAGE_VARIANT = "B3-PC"
VARIANT_SETTINGS = {
    "B0": {"protein_mode": "global_mean", "go_representation": "single_text", "segment_aggregation": "fixed"},
    "B1": {"protein_mode": "global_mean", "go_representation": "segments", "segment_aggregation": "fixed"},
    "B2": {"protein_mode": "local", "go_representation": "segments", "segment_aggregation": "fixed"},
    "B3": {"protein_mode": "local", "go_representation": "segments", "segment_aggregation": "learned"},
    "B3-PC": {"protein_mode": "local", "go_representation": "segments",
              "segment_aggregation": "learned", "protein_conditioning": True},
}


# Each entry is YAML key -> (runtime attribute, default). The schema is deliberately
# explicit: a typo must not silently leave a requested ablation disabled.
SCHEMA = {
    "experiment": {"ablation_id": ("ablation_id", None)},
    "mzsgo": {
        "branch": ("branch", "CC"), "root": ("root", "/workspace/MZSGO_temporal"),
        "split_dir": ("split_dir", None), "train_ids_path": ("train_ids_path", None),
        "val_ids_path": ("val_ids_path", None), "pid2pos_path": ("pid2pos", None),
        "candidate_go_ids_path": ("branch_go_ids_path", None),
        "temporal_go_ids_path": ("temporal_go_ids_path", None),
        "go_text_path": ("go_text_path", None), "frozen_bank": ("mzsgo_frozen_bank", None),
        "original_train_ids_path": ("original_train_ids_path", None),
        "original_valid_ids_path": ("original_valid_ids_path", None),
        "temporal_test_ids_path": ("temporal_test_ids_path", None),
    },
    "data": {"overlap": ("overlap", 256), "max_len": ("max_len", 1000),
             "rare_lt": ("rare_lt", 20)},
    "stores": {"embed_dir_res": ("embed_dir_res", None)},
    "model": {
        "protein_mode": ("protein_mode", "local"),
        "go_representation": ("go_representation", "segments"),
        "segment_aggregation": ("segment_aggregation", "learned"),
        "protein_conditioning": ("protein_conditioning", True),
        "align_dim": ("align_dim", 768), "temperature": ("temperature", 0.07),
        "window_length": ("window_length", 128), "window_stride": ("window_stride", 64),
        "mil_tau": ("mil_tau", 0.1), "parent_strength": ("parent_strength", 0.15),
        "go_chunk_size": ("go_chunk_size", 512),
        "encoder_name": ("encoder_name", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
        "go_encoder_inner_pooling": ("go_encoder_inner_pooling", "mean"),
        "go_segment_max_len": ("go_segment_max_len", 64),
        "go_text_store_max_len": ("go_text_store_max_len", 128),
        "go_single_text_max_len": ("go_single_text_max_len", 128),
        "single_text_bank_path": ("single_text_bank_path", None),
    },
    "training": {
        "epochs": ("epochs", 30), "batch_size": ("batch_size", 64),
        "eval_batch_size": ("eval_batch_size", 64), "num_workers": ("num_workers", 0),
        "fp16": ("fp16", True), "device": ("device", "cuda:0"), "seed": ("seed", 42),
        "output_dir": ("output_dir", None), "keep_last_n": ("keep_last_n", 3),
        "resume": ("resume", None), "eval_only": ("eval_only", False),
        "log_every": ("log_every", 500), "log_level": ("log_level", "INFO"),
        "monitor_metric": ("monitor_metric", "protein_wise_map"),
        "secondary_monitor_metric": ("secondary_monitor_metric", "macro_term_recall@500"),
        "monitor_mode": ("monitor_mode", "max"),
        "early_stop_patience": ("early_stop_patience", 3),
        "eval_go_bs": ("eval_go_bs", 256), "torch_threads": ("torch_threads", 4),
    },
    "optim": {"lr": ("lr", 1e-5), "weight_decay": ("weight_decay", 0.0),
              "grad_clip": ("grad_clip", 1.0)},
    "loss": {
        "lambda_con": ("lambda_con", 1.0), "pbr_lambda": ("pbr_lambda", 0.0),
        "pbr_mode": ("pbr_mode", "local"), "pbr_local_k": ("pbr_local_k", 128),
        "pbr_margin": ("pbr_margin", 0.0), "pbr_tau": ("pbr_tau", 0.05),
        "frequency_aware_local": ("frequency_aware_local", False),
        "frequency_weight_power": ("frequency_weight_power", 0.5),
        "frequency_weight_min": ("frequency_weight_min", 0.5),
        "frequency_weight_max": ("frequency_weight_max", 2.0),
    },
    "evaluation": {
        "retrieval_eval_ks": ("retrieval_eval_ks", [50, 100, 200, 500, 1000]),
        "positive_rank_quantiles": ("positive_rank_quantiles", [0.5, 0.75, 0.9, 0.95]),
        "validation_fixed_threshold": ("validation_fixed_threshold", None),
    },
    "diagnostics": {
        "log_gradient_norms": ("log_gradient_norms", True),
        "log_positive_rank_cdf": ("log_positive_rank_cdf", True),
        "log_cardinality_metrics": ("log_cardinality_metrics", True),
        "log_go_segment_weights": ("log_go_segment_weights", True),
        "log_go_term_metrics": ("log_go_term_metrics", True),
        "log_every_steps": ("diagnostics_log_every_steps", 500),
        "log_every_epochs": ("diagnostics_log_every_epochs", 1),
        "max_table_rows": ("diagnostics_max_table_rows", 200),
        "heatmap_rows": ("diagnostics_heatmap_rows", 50),
        "save_local": ("save_diagnostics_local", True),
    },
    "wandb": {
        "enabled": ("wandb", True), "project": ("wandb_project", "protein-go-align-mzsgo"),
        "entity": ("wandb_entity", None), "run_name": ("wandb_run_name", None),
        "mode": ("wandb_mode", "online"),
    },
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--branch", type=str.upper, choices=("BP", "CC", "MF"))
    p.add_argument("--root")
    p.add_argument("--split-dir")
    p.add_argument("--protein-mode", choices=("local", "global_mean"))
    p.add_argument("--aggregation", dest="segment_aggregation", choices=("fixed", "learned"))
    p.add_argument("--go-representation", choices=("segments", "single_text"))
    p.add_argument("--device")
    p.add_argument("--output-dir")
    p.add_argument("--resume")
    p.add_argument("--eval-only", action="store_true", default=None)
    p.add_argument("--train-ids", dest="train_ids_path")
    p.add_argument("--val-ids", dest="val_ids_path")
    p.add_argument("--annotations", dest="pid2pos")
    p.add_argument("--go-bank", dest="mzsgo_frozen_bank")
    p.add_argument("--embed-dir", dest="embed_dir_res")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--eval-batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    p.add_argument("--run-name", dest="wandb_run_name")
    p.add_argument("--print-config", action="store_true",
                   help="Print resolved configuration without loading data or a model")
    return p


def resolve_config(path: Path, overrides: dict[str, Any] | None = None) -> argparse.Namespace:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("YAML must contain a mapping")
    unknown = set(raw) - set(SCHEMA)
    if unknown:
        raise ValueError(f"Unknown YAML sections: {sorted(unknown)}; use the included example YAML")
    values = {}
    for section, keys in SCHEMA.items():
        entries = raw.get(section, {})
        if entries is None:
            entries = {}
        if not isinstance(entries, dict):
            raise ValueError(f"{section} must be a mapping")
        unknown = set(entries) - set(keys)
        if unknown:
            raise ValueError(f"Unknown {section} keys: {sorted(unknown)}")
        for key, (attr, default) in keys.items():
            values[attr] = entries.get(key, default)
    for key, value in (overrides or {}).items():
        if key in values and value is not None:
            values[key] = value
    a = argparse.Namespace(**values)
    a.branch = str(a.branch).upper()
    if a.branch not in {"BP", "CC", "MF"}:
        raise ValueError("mzsgo.branch must be BP, CC or MF")
    a.root = Path(a.root).expanduser().resolve()
    a.config_path = path
    branch = a.branch.lower()
    fmt = {"root": str(a.root), "branch": branch, "BRANCH": a.branch,
           "protein_mode": a.protein_mode, "aggregation": a.segment_aggregation,
           "ablation_id": a.ablation_id, "go_representation": a.go_representation}

    def render(value):
        return str(value).format_map(fmt)

    # Legacy BP used a differently named directory; this is only a path fallback.
    # Supplying split_dir/train_ids_path/val_ids_path always takes precedence.
    default_split = a.root / f"mzsgo_graphs_{branch}"
    if branch == "bp" and not default_split.is_dir():
        default_split = a.root / "mzsgo_graphs"
    a.split_dir = Path(render(a.split_dir or default_split)).expanduser().resolve()
    fmt["split_dir"] = str(a.split_dir)
    proc = a.root / "processed" / branch
    defaults = {
        "train_ids_path": a.split_dir / "train_homology30_v2.txt",
        "val_ids_path": a.split_dir / "valid_homology30_v2.txt",
        "pid2pos": proc / f"pid_to_go_train_{branch}.json",
        "branch_go_ids_path": proc / f"candidate_go_ids_{branch}.json",
        "temporal_go_ids_path": proc / f"temporal_go_ids_{branch}.json",
        "go_text_path": proc / "go_texts_canonical_segmented_dual_2023_2025.jsonl",
        "mzsgo_frozen_bank": a.root / "go_cache" / branch / "frozen_segment_parent_bank.pt",
        "original_train_ids_path": proc / "train.txt",
        "original_valid_ids_path": proc / "valid.txt",
        "temporal_test_ids_path": proc / "test.txt",
        "embed_dir_res": a.root / "protein_embeddings" / "esm2_residue_overlap",
        "output_dir": a.root / "outputs" / branch / f"{a.protein_mode}_infonce_{a.segment_aggregation}_aggregation",
    }
    for name, default in defaults.items():
        setattr(a, name, Path(render(getattr(a, name) or default)).expanduser().resolve())
    fmt["output_dir"] = str(a.output_dir)
    a.single_text_bank_path = Path(render(
        a.single_text_bank_path or (a.output_dir / "go_single_text_bank.pt")
    )).expanduser().resolve()
    if a.resume:
        a.resume = Path(render(a.resume)).expanduser().resolve()
    if a.wandb_run_name:
        a.wandb_run_name = render(a.wandb_run_name)
    else:
        a.wandb_run_name = f"MZSGO-{a.branch}-{a.protein_mode}-InfoNCE-{a.segment_aggregation}Aggregation"
    a.enabled_segments = ["name", "definition", "is_a"]  # frozen bank layout, not learned channels
    a.phase = -2
    a.go_pool_type = "mean"
    a.go_encoder_output_mode = "segment_pooled"
    a.go_segment_representation_mode = "segments_only"
    a.protein_pooling_strategy = "mean_attn_gate"  # legacy constructor compatibility only
    a.warmstart_path = None  # controlled experiment starts from scratch; resume is exact continuation
    validate_config(a)
    return a


def validate_config(a):
    def finite_number(name):
        raw = getattr(a, name)
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be a finite number, not a boolean")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        setattr(a, name, value)
        return value

    if a.protein_mode not in {"local", "global_mean"}:
        raise ValueError("model.protein_mode must be local or global_mean")
    if a.segment_aggregation not in {"fixed", "learned"}:
        raise ValueError("model.segment_aggregation must be fixed or learned")
    if a.go_representation not in {"segments", "single_text"}:
        raise ValueError("model.go_representation must be segments or single_text")
    for name in ("epochs", "batch_size", "eval_batch_size", "keep_last_n", "log_every",
                 "align_dim", "window_length", "window_stride", "go_chunk_size", "eval_go_bs",
                 "max_len", "rare_lt", "go_segment_max_len", "go_text_store_max_len", "go_single_text_max_len",
                 "diagnostics_log_every_steps", "diagnostics_log_every_epochs", "torch_threads",
                 "diagnostics_max_table_rows", "diagnostics_heatmap_rows", "pbr_local_k"):
        value = getattr(a, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("num_workers", "early_stop_patience", "overlap", "seed"):
        value = getattr(a, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if a.overlap >= a.max_len:
        raise ValueError("data.overlap must be smaller than data.max_len")
    for name in ("fp16", "wandb", "eval_only", "frequency_aware_local", "protein_conditioning", "log_gradient_norms",
                 "log_positive_rank_cdf", "log_cardinality_metrics", "log_go_segment_weights",
                 "log_go_term_metrics", "save_diagnostics_local"):
        if not isinstance(getattr(a, name), bool):
            raise ValueError(f"{name} must be a YAML boolean, not a quoted string")
    for name in ("lr", "temperature", "mil_tau", "pbr_tau", "lambda_con",
                 "frequency_weight_power", "frequency_weight_min", "frequency_weight_max"):
        value = finite_number(name)
        if value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "grad_clip", "parent_strength", "pbr_lambda", "pbr_margin"):
        value = finite_number(name)
        if value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if a.pbr_mode not in {"pure", "local"}:
        raise ValueError("loss.pbr_mode must be pure or local")
    if a.pbr_mode == "local" and a.pbr_margin != 0:
        raise ValueError("Local PBR requires pbr_margin=0")
    if a.frequency_aware_local:
        if a.pbr_lambda <= 0 or a.pbr_mode != "local":
            raise ValueError("frequency_aware_local weights Local PBR only; it requires pbr_lambda>0 and pbr_mode=local")
        if not (0 < a.frequency_weight_power and 0 < a.frequency_weight_min <= a.frequency_weight_max):
            raise ValueError("Invalid frequency weight settings")
    if a.monitor_metric != "protein_wise_map" or a.monitor_mode != "max":
        raise ValueError("This ablation selects checkpoints by validation protein_wise_map (max)")
    if (not isinstance(a.retrieval_eval_ks, (list, tuple)) or not a.retrieval_eval_ks
            or any(isinstance(k, bool) or not isinstance(k, int) or k < 1
                   for k in a.retrieval_eval_ks)):
        raise ValueError("retrieval_eval_ks must be a list of positive integers")
    a.retrieval_eval_ks = tuple(a.retrieval_eval_ks)
    a.positive_rank_quantiles = tuple(float(q) for q in a.positive_rank_quantiles)
    if any(not 0 < q <= 1 for q in a.positive_rank_quantiles):
        raise ValueError("positive_rank_quantiles must be in (0,1]")
    if a.secondary_monitor_metric not in {f"macro_term_recall@{k}" for k in a.retrieval_eval_ks}:
        raise ValueError("secondary_monitor_metric must be a configured macro_term_recall@K")
    if a.validation_fixed_threshold is not None:
        finite_number("validation_fixed_threshold")
    if a.eval_only and not a.resume:
        raise ValueError("--eval-only requires a package checkpoint via --resume")
    if a.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline or disabled")
    if a.wandb_mode == "disabled":
        a.wandb = False
    if a.go_representation == "single_text":
        if a.protein_mode != "global_mean" or a.segment_aggregation != "fixed" or a.parent_strength != 0:
            raise ValueError("single_text requires global_mean, fixed aggregation and parent_strength=0")
        if a.go_encoder_inner_pooling != "mean":
            raise ValueError("single_text requires frozen GO mean pooling")
    validate_ablation(a)


def validate_ablation(a):
    """A package must not silently become a different experiment after overrides."""
    if PACKAGE_VARIANT is not None and a.ablation_id != PACKAGE_VARIANT:
        raise ValueError(f"This package is locked to {PACKAGE_VARIANT}; received ablation_id={a.ablation_id!r}")
    if a.ablation_id is None:
        return
    if a.ablation_id not in VARIANT_SETTINGS:
        raise ValueError("experiment.ablation_id must name a supported B0/B1/B2/B3/B3-PC variant")
    expected = dict(VARIANT_SETTINGS[a.ablation_id])
    expected.update(branch="MF", parent_strength=0.0, lambda_con=1.0,
                    pbr_lambda=0.0, frequency_aware_local=False,
                    go_encoder_inner_pooling="mean")
    if expected["protein_mode"] == "local":
        expected.update(window_length=128, window_stride=64, mil_tau=0.1)
    mismatches = [f"{key}={getattr(a, key)!r}, expected {value!r}"
                  for key, value in expected.items() if getattr(a, key) != value]
    if mismatches:
        raise ValueError(f"{a.ablation_id} ablation contract violated: " + "; ".join(mismatches))


def json_config(a) -> dict:
    return json.loads(json.dumps(vars(a), default=str))


def get_args(argv=None):
    cli = parser().parse_args(argv)
    args = resolve_config(cli.config, vars(cli))
    args.print_config = cli.print_config
    return args
