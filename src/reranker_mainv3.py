from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import yaml

from src.training.reranker_trainerv3 import P3aSemExpRerankerConfig, P3aSemExpRerankerTrainer


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Accept flat yaml or nested keys.
    for key in ["p3a_semexp_reranker", "p3a_reranker", "reranker"]:
        if key in data and isinstance(data[key], dict):
            return data[key]
    return data


def _filter_dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def parse_args():
    p = argparse.ArgumentParser("Train P3a semantic-expansion reranker.")
    default_cfg = Path(__file__).resolve().parent / "rerankerv3.yaml"
    p.add_argument("--config", type=str, default=str(default_cfg))
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    print(f"[main] loading config: {cfg_path}")
    data = _load_yaml(cfg_path)
    cfg = P3aSemExpRerankerConfig(**_filter_dataclass_kwargs(P3aSemExpRerankerConfig, data))
    trainer = P3aSemExpRerankerTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
