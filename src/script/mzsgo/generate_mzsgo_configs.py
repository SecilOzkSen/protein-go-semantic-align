#!/usr/bin/env python3
"""Generate branch-specific retriever and reranker YAMLs for MZSGO temporal evaluation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


BRANCHES = ("mf", "bp", "cc")


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def find_base(base_dir: Path, branch: str) -> Path:
    candidates = (
        base_dir / f"gor2023_{branch}.yaml",
        base_dir / "configs" / f"gor2023_{branch}.yaml",
        base_dir / "src" / "configs" / f"gor2023_{branch}.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No GOR2023 {branch.upper()} YAML under {base_dir}")


def retriever_config(base: dict, root: Path, branch: str, temporal_eval: bool) -> dict:
    cfg = copy.deepcopy(base)
    upper = branch.upper()
    branch_dir = root / "processed" / branch
    pf = cfg.setdefault("pfresgo", {})
    pf.update({
        "branch": upper,
        "benchmark_protocol": "standard",
        "evaluation_space": "benchmark",
        "branch_go_ids_path": str(branch_dir / f"candidate_go_ids_{branch}.json"),
        "go_text_path": str(root / "processed" / "go_texts_canonical.jsonl"),
        "test_ids_path": str(branch_dir / "test.txt"),
        "evaluation_split": "test" if temporal_eval else "valid",
        "rare_lt": 20,
        "enabled_segments": ["name", "definition", "is_a"],
    })
    pf.pop("ia_weights_path", None)
    if temporal_eval:
        pf["allow_checkpoint_for_eval"] = True
    else:
        pf.pop("allow_checkpoint_for_eval", None)

    cfg.setdefault("general", {})["ablation_id"] = f"MZSGO-{upper}-temporal-global-local"
    training = cfg.setdefault("training", {})
    training.update({
        "output_dir": str(root / "outputs" / branch / "retriever"),
        "monitor_metric": "oracle_microF@200",
        "secondary_monitor_metric": "macro_term_recall@100",
        "eval_only": bool(temporal_eval),
        "warmstart_path": str(root / "outputs" / branch / "retriever" / "best.pt") if temporal_eval else None,
        "eval_space": "observed",
    })
    cfg.setdefault("wandb", {}).update({
        "project": "protein-go-align-mzsgo-temporal",
        "wandb_run_name": f"MZSGO-{upper}-temporal-retriever",
        "enabled": not temporal_eval,
    })
    stores = cfg.setdefault("stores", {})
    stores.update({
        "train_ids_path": str(branch_dir / "train.txt"),
        "val_ids_path": str(branch_dir / ("test.txt" if temporal_eval else "valid.txt")),
        "go_basic_json": str(root / "processed" / "go_vocab.json"),
        "pid2pos_path": str(branch_dir / f"pid_to_positives_{branch}.json"),
        "zero_shot_path": None,
        "few_shot_path": None,
        "go_path_seen": None,
        "go_path_observed": None,
        "embed_dir_res": str(root / "protein_embeddings" / "esm1b_residue"),
        "go_text_folder": str(root / "processed"),
        "go_cache_path": str(root / "go_cache" / branch / "go_text_embeddings_canonical.npy"),
        "seq_len_lookup": str(root / "protein_embeddings" / "esm1b_residue" / "seq_len_lookup.pkl"),
        "logs": str(root / "logs" / branch / ("temporal_eval" if temporal_eval else "train")),
    })
    return cfg


def reranker_config(root: Path, branch: str) -> dict:
    branch_dir = root / "processed" / branch
    return {
        "model": {
            "model_kind": "hiercross",
            "text_model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "protein_dim": 1280,
            "reranker_hidden_dim": 512,
            "reranker_dropout": 0.1,
            "freeze_text_encoder": True,
            "use_protein_ln": True,
            "use_go_ln": True,
            "cross_dim": 256,
            "cross_heads": 4,
            "cross_dropout": 0.1,
            "candidate_chunk_size": 16,
            "use_retriever_features": True,
            "use_cls_residual": True,
        },
        "training": {
            "retriever_ckpt": str(root / "outputs" / branch / "retriever" / "best.pt"),
            "device": "cuda",
            "fp16": True,
            "topk": 200,
            "batch_size": 4,
            "lr": 0.00005,
            "weight_decay": 0.01,
            "epochs": 5,
            "log_every": 100,
            "eval_every": 500,
            "eval_every_steps": 1000,
            "save_metric": "fmax_full",
            "out_dir": str(root / "outputs" / branch / "reranker"),
            "use_dag_loss": False,
            "lambda_dag": 0.0,
            "pos_weight": 20.0,
            "inject_true_positives": 0,
            "use_candidate_dump": True,
            "train_candidate_dump": str(root / "candidate_dumps" / branch / "train_top200"),
            "val_candidate_dump": str(root / "candidate_dumps" / branch / "valid_top200"),
            "score_feature_dropout": 0.0,
            "rank_feature_dropout": 0.0,
            "max_eval_batches": 0,
        },
        "stores": {
            "train_ids_path": str(branch_dir / "train.txt"),
            "val_ids_path": str(branch_dir / "valid.txt"),
            "pid2pos_path": str(branch_dir / f"pid_to_positives_{branch}.json"),
            "embed_dir_res": str(root / "protein_embeddings" / "esm1b_residue"),
            "embed_dir_fused": "",
            "seq_len_lookup_dir": str(root / "protein_embeddings" / "esm1b_residue"),
            "go_text_folder": str(root / "processed"),
            "dag_parents_path": "",
            "go_cache_path": str(root / "go_cache" / branch / "go_text_embeddings_canonical.npy"),
            "go_basic_json": str(root / "processed" / "go_vocab.json"),
            "zero_shot_path": str(branch_dir / "empty_go_ids.pkl"),
            "few_shot_path": str(branch_dir / "empty_go_ids.pkl"),
            "go_path_seen": str(branch_dir / f"seen_go_ids_{branch}.pkl"),
            "go_path_observed": str(branch_dir / f"candidate_go_ids_{branch}.pkl"),
        },
        "data": {"protein_max_len": 1024, "overlap": 256, "fs_target_ratio": 0.0},
    }


def reranker_temporal_eval_config(root: Path, branch: str) -> dict:
    cfg = reranker_config(root, branch)
    branch_dir = root / "processed" / branch
    cfg["stores"]["val_ids_path"] = str(branch_dir / "test.txt")
    cfg["training"]["val_candidate_dump"] = str(
        root / "candidate_dumps" / branch / "temporal_top200"
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config-dir", type=Path, required=True)
    parser.add_argument("--mzsgo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.mzsgo_root.resolve()
    for branch in BRANCHES:
        base = load_yaml(find_base(args.base_config_dir, branch))
        save_yaml(args.out_dir / f"mzsgo_{branch}_train.yaml", retriever_config(base, root, branch, False))
        save_yaml(args.out_dir / f"mzsgo_{branch}_temporal_eval.yaml", retriever_config(base, root, branch, True))
        save_yaml(args.out_dir / f"mzsgo_{branch}_reranker.yaml", reranker_config(root, branch))
        save_yaml(
            args.out_dir / f"mzsgo_{branch}_reranker_temporal_eval.yaml",
            reranker_temporal_eval_config(root, branch),
        )
    print(f"Wrote 12 configs to {args.out_dir}")


if __name__ == "__main__":
    main()
