import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.training.reranker_union_trainer import SourceAwareRerankerConfig, SourceAwareRerankerTrainer


DEFAULT_CONFIG = "/workspace/protein-go-semantic-align/src/reranker_union.yaml"


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "reranker" in data:
        data = data["reranker"] or {}
    return dict(data)


def parse_args():
    p = argparse.ArgumentParser("Train source-aware union reranker.")
    p.add_argument("--config", type=str, default=None)
    for f in fields(SourceAwareRerankerConfig):
        name = f.name
        default = None
        arg_type = type(f.default) if f.default is not None else str
        if name in {"train_dump", "val_dump", "out_dir"}:
            p.add_argument(f"--{name}", type=str, default=None)
        elif name == "test_dump":
            p.add_argument(f"--{name}", type=str, default=None)
        elif arg_type is int:
            p.add_argument(f"--{name}", type=int, default=None)
        elif arg_type is float:
            p.add_argument(f"--{name}", type=float, default=None)
        else:
            p.add_argument(f"--{name}", type=str, default=None)
    return p.parse_args()


def build_config(cli) -> SourceAwareRerankerConfig:
    config_path = cli.config
    if config_path is None:
        config_path = DEFAULT_CONFIG
        print(f"[main] No --config passed, defaulting to {config_path}")

    data = read_yaml(config_path)
    valid_keys = {f.name for f in fields(SourceAwareRerankerConfig)}
    data = {k: v for k, v in data.items() if k in valid_keys}

    for k in valid_keys:
        v = getattr(cli, k, None)
        if v is not None:
            data[k] = v

    missing = [k for k in ["train_dump", "val_dump", "out_dir"] if not data.get(k)]
    if missing:
        raise RuntimeError(f"Missing required config values: {missing}")

    # dataclass requires test_dump key
    data.setdefault("test_dump", None)
    return SourceAwareRerankerConfig(**data)


def main():
    cli = parse_args()
    cfg = build_config(cli)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
    trainer = SourceAwareRerankerTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
