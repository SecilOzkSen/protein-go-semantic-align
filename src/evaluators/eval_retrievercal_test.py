'''
python -m src.script.eval_retrievercal_test \
  --checkpoint /workspace/protein-go-align/outputs/retrievercal/embsetcal_name_def_isa_step52929_top500/best_stargo_fmax.pt \
  --test_dump /workspace/candidate_dumps/name_def_isa_step52929_test_top500 \
  --test_embedding_dump /workspace/candidate_dumps/name_def_isa_step52929_test_top500 \
  --go_obo /workspace/stargo/datasets/pfresgo/go.obo \
  --ontology bp \
  --device cuda:0 \
  --out_json /workspace/protein-go-align/outputs/retrievercal/embsetcal_name_def_isa_step52929_top500/test_metrics_bp.json
'''

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import fields
from pathlib import Path

import torch

from src.training.retrievercal_trainer import (
    CalibConfig,
    CalibTrainer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate EmbSetCal on a candidate dump"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_dump", required=True)
    parser.add_argument(
        "--test_embedding_dump",
        default="",
    )
    parser.add_argument("--go_obo", required=True)
    parser.add_argument("--ontology", default="bp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out_json",
        default="test_metrics.json",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "config" not in ckpt:
        raise KeyError(
            f"Checkpoint has no config: {checkpoint_path}"
        )

    saved_cfg = dict(ckpt["config"])

    # Ignore fields removed from the current CalibConfig.
    allowed = {
        field.name
        for field in fields(CalibConfig)
    }
    saved_cfg = {
        key: value
        for key, value in saved_cfg.items()
        if key in allowed
    }

    # Reuse the original training dump so score statistics and
    # model construction stay consistent.
    # Only replace validation data with the test dump.
    saved_cfg["val_dump"] = args.test_dump
    saved_cfg["val_embedding_dump"] = (
        args.test_embedding_dump
        if args.test_embedding_dump
        else args.test_dump
    )

    saved_cfg["device"] = args.device
    saved_cfg["use_stargo_eval"] = True
    saved_cfg["stargo_ontology"] = args.ontology
    saved_cfg["stargo_go_obo"] = args.go_obo

    cfg = CalibConfig(**saved_cfg)

    trainer = CalibTrainer(cfg)

    print("\n=== NORMALIZATION CHECK ===")
    print(
        "fused:",
        trainer.score_mean,
        trainer.score_std,
        "| ckpt:",
        ckpt.get("score_mean"),
        ckpt.get("score_std"),
    )

    if cfg.use_expert_scores:
        print(
            "global:",
            trainer.global_score_mean,
            trainer.global_score_std,
            "| ckpt:",
            ckpt.get("global_score_mean"),
            ckpt.get("global_score_std"),
        )

        print(
            "local:",
            trainer.local_score_mean,
            trainer.local_score_std,
            "| ckpt:",
            ckpt.get("local_score_mean"),
            ckpt.get("local_score_std"),
        )

    load_result = trainer.model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    if load_result.missing_keys:
        raise RuntimeError(
            f"Missing model keys: {load_result.missing_keys}"
        )

    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected model keys: "
            f"{load_result.unexpected_keys}"
        )

    trainer.model.eval()

    metrics = trainer.evaluate()

    print("\n=== BP TEST RESULTS ===")
    print(json.dumps(metrics, indent=2))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logging.info(
        "Test metrics saved to %s",
        out_path,
    )


if __name__ == "__main__":
    main()