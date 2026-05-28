from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import yaml

from src.training.reranker_trainer_p3a import P3aRerankerConfig, P3aRerankerTrainer


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Accept either flat yaml or {p3a_reranker: {...}}.
    if "p3a_reranker" in data and isinstance(data["p3a_reranker"], dict):
        data = data["p3a_reranker"]
    if "reranker" in data and isinstance(data["reranker"], dict):
        data = data["reranker"]
    return data


def _filter_dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def parse_args():
    p = argparse.ArgumentParser("Train P3a branch-aware pairwise candidate reranker.")
    default_cfg = Path(__file__).resolve().parent / "reranker_p3a.yaml"
    p.add_argument("--config", type=str, default=str(default_cfg))
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    print(f"[main] loading config: {cfg_path}")
    data = _load_yaml(cfg_path)
    cfg = P3aRerankerConfig(**_filter_dataclass_kwargs(P3aRerankerConfig, data))
    trainer = P3aRerankerTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
