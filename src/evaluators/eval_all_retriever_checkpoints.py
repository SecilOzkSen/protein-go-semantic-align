from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

FOCUS_METRICS = [
    "oracle_microF@500",
    "cand_coverage@500",
    "oracle_proteinF@500",
    "align_R@500",
    "align_R@200",
    "align_R@1000",
    "expert_fused_R@500",
    "expert_global_R@500",
    "expert_local_R@500",
    "obs_fmax",
    "obs_aupr",
    "obs_protein_fmax",
    "debug_margin",
]


def parse_args():
    p = argparse.ArgumentParser(
        "Evaluate all retriever checkpoints on PFresGO test split"
    )

    p.add_argument(
        "--config",
        required=True,
        help="Original training YAML for this retriever run.",
    )

    p.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing retriever .pt checkpoints.",
    )

    p.add_argument(
        "--out_dir",
        required=True,
        help="Directory where per-checkpoint evals and summary are saved.",
    )

    p.add_argument(
        "--pattern",
        default="*.pt",
        help="Checkpoint glob pattern. Default: *.pt",
    )

    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse an existing metrics.json instead of rerunning that checkpoint.",
    )

    return p.parse_args()


def checkpoint_sort_key(path: Path):
    """
    Prefer numeric step order when filename contains e.g.
      checkpoint_step70572.pt
    """
    m = re.search(
        r"(?:step|checkpoint[_-]?)(\d+)",
        path.stem,
        flags=re.IGNORECASE,
    )

    if m:
        return (0, int(m.group(1)), path.name)

    nums = re.findall(r"\d+", path.stem)

    if nums:
        return (1, int(nums[-1]), path.name)

    return (2, 0, path.name)


def discover_checkpoints(
        checkpoint_dir: Path,
        pattern: str,
):
    paths = [
        p
        for p in checkpoint_dir.glob(pattern)
        if p.is_file()
    ]

    # Deduplicate resolved paths.
    uniq = {}
    for p in paths:
        uniq[str(p.resolve())] = p.resolve()

    paths = sorted(
        uniq.values(),
        key=checkpoint_sort_key,
    )

    return paths


def parse_eval_line(log_path: Path):
    """
    Parse the last line containing:

      [eval_only] metric: value | metric: value | ...
    """

    if not log_path.exists():
        raise FileNotFoundError(
            f"Missing evaluation log: {log_path}"
        )

    lines = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    eval_lines = [
        x
        for x in lines
        if "[eval_only]" in x
    ]

    if not eval_lines:
        raise RuntimeError(
            f"No [eval_only] line found in {log_path}"
        )

    line = eval_lines[-1]

    payload = line.split(
        "[eval_only]",
        1,
    )[1].strip()

    metrics = {}

    for item in payload.split("|"):
        item = item.strip()

        if ":" not in item:
            continue

        key, value = item.split(
            ":",
            1,
        )

        key = key.strip()
        value = value.strip()

        try:
            metrics[key] = float(value)
        except ValueError:
            continue

    return metrics


def checkpoint_metadata(
        checkpoint_path: Path,
) -> dict[str, Any]:
    """
    Best-effort extraction only.

    Different retriever checkpoints may have slightly different
    top-level structures, so failure here must not stop evaluation.
    """

    out: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
    }

    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        out["checkpoint_metadata_error"] = str(exc)
        return out

    if not isinstance(ckpt, dict):
        return out

    for key in [
        "epoch",
        "step",
        "global_step",
        "best_metric",
        "best_score",
    ]:
        value = ckpt.get(key)

        if isinstance(
                value,
                (int, float, str, bool),
        ):
            out[key] = value

    # Some checkpoint formats save validation metrics.
    metric_block = None

    for key in [
        "metrics",
        "val_metrics",
        "validation_metrics",
        "logs",
    ]:
        value = ckpt.get(key)

        if isinstance(value, dict):
            metric_block = value
            break

    if metric_block is not None:
        for key, value in metric_block.items():
            if isinstance(value, (int, float)):
                out[
                    f"saved_val_{key}"
                ] = float(value)

    return out


def build_test_config(
        original_config: Path,
        checkpoint_path: Path,
        eval_dir: Path,
):
    with original_config.open(
            "r",
            encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    pf = cfg.setdefault(
        "pfresgo",
        {},
    )

    pf["evaluation_split"] = "test"
    pf["allow_checkpoint_for_eval"] = True

    if not pf.get("test_ids_path"):
        raise RuntimeError(
            "pfresgo.test_ids_path is missing from config."
        )

    training = cfg.setdefault(
        "training",
        {},
    )

    training["eval_only"] = True
    training["resume"] = str(
        checkpoint_path
    )
    training["warmstart_path"] = None
    training["output_dir"] = str(
        eval_dir
    )

    # No checkpoint creation during evaluation.
    training["save_every"] = 10 ** 12

    # Avoid any early-stop weirdness in eval-only.
    training["early_stop_patience"] = 10 ** 9

    wandb = cfg.setdefault(
        "wandb",
        {},
    )

    wandb["enabled"] = False
    wandb["mode"] = "disabled"

    eval_config_path = (
            eval_dir
            / "test_eval_config.yaml"
    )

    with eval_config_path.open(
            "w",
            encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
        )

    return eval_config_path


def evaluate_checkpoint(
        original_config: Path,
        checkpoint_path: Path,
        eval_dir: Path,
        skip_existing: bool,
):
    eval_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_json = (
            eval_dir
            / "metrics.json"
    )

    if (
            skip_existing
            and metrics_json.exists()
    ):
        print(
            f"[SKIP] {checkpoint_path.name}, "
            "using existing metrics.json"
        )

        return json.loads(
            metrics_json.read_text(
                encoding="utf-8"
            )
        )

    eval_config_path = build_test_config(
        original_config=original_config,
        checkpoint_path=checkpoint_path,
        eval_dir=eval_dir,
    )

    cmd = [
        sys.executable,
        "-m",
        "src.main_pfresgo",
        "--config",
        str(eval_config_path),
    ]

    print()
    print("=" * 80)
    print(
        f"EVALUATING: {checkpoint_path.name}"
    )
    print("=" * 80)
    print(
        " ".join(cmd)
    )
    print()

    subprocess.run(
        cmd,
        check=True,
    )

    log_path = (
            eval_dir
            / "train.log"
    )

    metrics = parse_eval_line(
        log_path
    )

    with metrics_json.open(
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            sort_keys=True,
        )

    return metrics


def save_summary(
        rows: list[dict[str, Any]],
        out_dir: Path,
):
    json_path = (
            out_dir
            / "all_checkpoint_test_metrics.json"
    )

    with json_path.open(
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            rows,
            f,
            indent=2,
            sort_keys=True,
        )

    all_keys = set()

    for row in rows:
        all_keys.update(row.keys())

    preferred = [
        "checkpoint_name",
        "checkpoint",
        "epoch",
        "step",
        "global_step",
    ]

    metric_cols = [
        key
        for key in FOCUS_METRICS
        if key in all_keys
    ]

    saved_val_cols = sorted(
        key
        for key in all_keys
        if key.startswith(
            "saved_val_"
        )
    )

    other_cols = sorted(
        all_keys
        - set(preferred)
        - set(metric_cols)
        - set(saved_val_cols)
    )

    fieldnames = (
            [
                x
                for x in preferred
                if x in all_keys
            ]
            + saved_val_cols
            + metric_cols
            + other_cols
    )

    csv_path = (
            out_dir
            / "all_checkpoint_test_metrics.csv"
    )

    with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    return json_path, csv_path


def fmt(value):
    if value is None:
        return "NA"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def print_summary(
        rows: list[dict[str, Any]],
):
    print()
    print()
    print("=" * 100)
    print("ALL CHECKPOINTS, TEST RETRIEVAL")
    print("=" * 100)

    header = (
        f"{'Checkpoint':<32}"
        f"{'Epoch':>8}"
        f"{'Step':>10}"
        f"{'Val oracle500':>16}"
        f"{'Test oracle500':>17}"
        f"{'Test R@500':>13}"
        f"{'Test R@200':>13}"
        f"{'Margin':>10}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        val_oracle = row.get(
            "saved_val_oracle_microF@500"
        )

        print(
            f"{row['checkpoint_name'][:31]:<32}"
            f"{fmt(row.get('epoch')):>8}"
            f"{fmt(row.get('step', row.get('global_step'))):>10}"
            f"{fmt(val_oracle):>16}"
            f"{fmt(row.get('oracle_microF@500')):>17}"
            f"{fmt(row.get('align_R@500')):>13}"
            f"{fmt(row.get('align_R@200')):>13}"
            f"{fmt(row.get('debug_margin')):>10}"
        )

    valid_test_oracle = [
        row
        for row in rows
        if isinstance(
            row.get("oracle_microF@500"),
            (int, float),
        )
    ]

    if valid_test_oracle:
        best_test = max(
            valid_test_oracle,
            key=lambda x: x[
                "oracle_microF@500"
            ],
        )

        print()
        print(
            "Diagnostic best TEST oracle_microF@500:"
        )
        print(
            f"  {best_test['checkpoint_name']}"
            f" -> "
            f"{best_test['oracle_microF@500']:.4f}"
        )

    valid_test_r500 = [
        row
        for row in rows
        if isinstance(
            row.get("align_R@500"),
            (int, float),
        )
    ]

    if valid_test_r500:
        best_r500 = max(
            valid_test_r500,
            key=lambda x: x[
                "align_R@500"
            ],
        )

        print(
            "Diagnostic best TEST R@500:"
        )
        print(
            f"  {best_r500['checkpoint_name']}"
            f" -> "
            f"{best_r500['align_R@500']:.4f}"
        )

    print()
    print(
        "IMPORTANT: test-best checkpoint is diagnostic only. "
        "Do not select the final reported model from test performance."
    )


def main():
    args = parse_args()

    config_path = Path(
        args.config
    ).resolve()

    checkpoint_dir = Path(
        args.checkpoint_dir
    ).resolve()

    out_dir = Path(
        args.out_dir
    ).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            config_path
        )

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            checkpoint_dir
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints = discover_checkpoints(
        checkpoint_dir=checkpoint_dir,
        pattern=args.pattern,
    )

    if not checkpoints:
        raise RuntimeError(
            f"No checkpoints found in "
            f"{checkpoint_dir} "
            f"with pattern {args.pattern}"
        )

    print(
        f"Found {len(checkpoints)} checkpoint(s):"
    )

    for p in checkpoints:
        print(
            "  ",
            p.name,
        )

    rows = []

    for i, checkpoint_path in enumerate(
            checkpoints,
            start=1,
    ):
        safe_name = checkpoint_path.stem

        eval_dir = (
                out_dir
                / safe_name
        )

        print(
            f"\n[{i}/{len(checkpoints)}] "
            f"{checkpoint_path.name}"
        )

        metadata = checkpoint_metadata(
            checkpoint_path
        )

        try:
            metrics = evaluate_checkpoint(
                original_config=config_path,
                checkpoint_path=checkpoint_path,
                eval_dir=eval_dir,
                skip_existing=args.skip_existing,
            )

            row = {
                **metadata,
                **metrics,
            }

            row["status"] = "ok"

        except Exception as exc:
            print(
                f"[ERROR] {checkpoint_path.name}: {exc}"
            )

            row = {
                **metadata,
                "status": "error",
                "error": str(exc),
            }

        rows.append(row)

        # Save incrementally so a later crash does not
        # throw away completed evaluations.
        save_summary(
            rows,
            out_dir,
        )

    json_path, csv_path = save_summary(
        rows,
        out_dir,
    )

    print_summary(
        rows
    )

    print()
    print(
        "JSON:",
        json_path,
    )
    print(
        "CSV :",
        csv_path,
    )


if __name__ == "__main__":
    main()