import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

CARD_BINS = [
    "1_5",
    "6_10",
    "11_20",
    "21_40",
    "41_80",
    "81_160",
    "161plus",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        required=True,
        help="Training YAML used for the global/local run.",
    )

    p.add_argument(
        "--checkpoint",
        required=True,
        help="Best retriever checkpoint.",
    )

    p.add_argument(
        "--out_dir",
        required=True,
        help="Directory for test-eval artifacts.",
    )

    return p.parse_args()


def parse_eval_line(log_path: Path):
    """
    Parses:

      [eval_only] metric: value | metric: value | ...

    from the final eval-only line.
    """

    lines = log_path.read_text(
        encoding="utf-8"
    ).splitlines()

    eval_lines = [
        line
        for line in lines
        if "[eval_only]" in line
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
            pass

    return metrics


def get(metrics, key):
    value = metrics.get(key)

    if value is None:
        return "NA"

    return f"{value:.4f}"


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==========================================================
    # LOAD ORIGINAL TRAINING CONFIG
    # ==========================================================

    with config_path.open(
            "r",
            encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    # ==========================================================
    # TEST SPLIT
    # ==========================================================

    pf = cfg.setdefault(
        "pfresgo",
        {},
    )

    pf["evaluation_split"] = "test"

    # main_pfresgo.py validation explicitly allows a checkpoint
    # for evaluation when this flag and eval_only are both true.
    pf["allow_checkpoint_for_eval"] = True

    if not pf.get("test_ids_path"):
        raise RuntimeError(
            "pfresgo.test_ids_path is missing."
        )

    # ==========================================================
    # EVAL-ONLY CHECKPOINT
    # ==========================================================

    training = cfg.setdefault(
        "training",
        {},
    )

    training["eval_only"] = True

    # run_training() loads `resume` before entering eval_only.
    training["resume"] = str(
        checkpoint_path
    )

    training["warmstart_path"] = None

    training["output_dir"] = str(
        out_dir
    )

    # No need for checkpoint creation during eval.
    training["save_every"] = 10 ** 12

    # ==========================================================
    # DISABLE W&B FOR TEST SCRIPT
    # ==========================================================

    wandb = cfg.setdefault(
        "wandb",
        {},
    )

    wandb["enabled"] = False
    wandb["mode"] = "disabled"

    # ==========================================================
    # WRITE TEMP CONFIG
    # ==========================================================

    eval_config = (
            out_dir
            / "test_eval_config.yaml"
    )

    with eval_config.open(
            "w",
            encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
        )

    print(
        "\n======================================="
    )
    print("RETRIEVER TEST EVALUATION")
    print(
        "======================================="
    )

    print("Config     :", eval_config)
    print("Checkpoint :", checkpoint_path)
    print("Output      :", out_dir)
    print()

    # ==========================================================
    # RUN EXISTING EVALUATOR
    # ==========================================================

    cmd = [
        sys.executable,
        "-m",
        "src.main_pfresgo",
        "--config",
        str(eval_config),
    ]

    print(
        "Running:\n",
        " ".join(cmd),
        "\n",
    )

    subprocess.run(
        cmd,
        check=True,
    )

    # ==========================================================
    # PARSE METRICS
    # ==========================================================

    log_path = (
            out_dir
            / "train.log"
    )

    if not log_path.exists():
        raise RuntimeError(
            f"Expected log was not produced: {log_path}"
        )

    metrics = parse_eval_line(
        log_path
    )

    json_path = (
            out_dir
            / "test_metrics_experts.json"
    )

    with json_path.open(
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            sort_keys=True,
        )

    # ==========================================================
    # FOCUSED SUMMARY
    # ==========================================================

    print(
        "\n\n======================================="
    )
    print("OVERALL TEST RETRIEVAL")
    print(
        "=======================================\n"
    )

    print(
        f"{'Branch':<12}"
        f"{'R@50':>10}"
        f"{'R@100':>10}"
        f"{'R@200':>10}"
        f"{'R@500':>10}"
        f"{'R@1000':>10}"
    )

    for branch in [
        "global",
        "local",
        "fused",
    ]:
        print(
            f"{branch:<12}"
            f"{get(metrics, f'expert_{branch}_R@50'):>10}"
            f"{get(metrics, f'expert_{branch}_R@100'):>10}"
            f"{get(metrics, f'expert_{branch}_R@200'):>10}"
            f"{get(metrics, f'expert_{branch}_R@500'):>10}"
            f"{get(metrics, f'expert_{branch}_R@1000'):>10}"
        )

    print(
        "\nFused standard align_R@500:",
        get(
            metrics,
            "align_R@500",
        ),
    )

    print(
        "Global fusion weight:",
        get(
            metrics,
            "expert_global_weight",
        ),
    )

    # ==========================================================
    # CARDINALITY BREAKDOWN
    # ==========================================================

    print(
        "\n\n======================================="
    )
    print("TEST R@500 BY LABEL CARDINALITY")
    print(
        "=======================================\n"
    )

    print(
        f"{'Cardinality':<14}"
        f"{'N':>8}"
        f"{'Global':>12}"
        f"{'Local':>12}"
        f"{'Fused':>12}"
    )

    display_names = {
        "1_5": "1-5",
        "6_10": "6-10",
        "11_20": "11-20",
        "21_40": "21-40",
        "41_80": "41-80",
        "81_160": "81-160",
        "161plus": "161+",
    }

    for card in CARD_BINS:
        n_key = (
            f"expert_fused_N_card_{card}"
        )

        n = metrics.get(
            n_key,
            0,
        )

        print(
            f"{display_names[card]:<14}"
            f"{int(n):>8}"
            f"{get(metrics, f'expert_global_R@500_card_{card}'):>12}"
            f"{get(metrics, f'expert_local_R@500_card_{card}'):>12}"
            f"{get(metrics, f'expert_fused_R@500_card_{card}'):>12}"
        )

    # ==========================================================
    # CANDIDATE CEILING
    # ==========================================================

    print(
        "\n\n======================================="
    )
    print("FUSED CANDIDATE CEILING")
    print(
        "=======================================\n"
    )

    for k in [
        50,
        100,
        200,
        500,
        1000,
    ]:
        print(
            f"K={k:<4} "
            f"coverage={get(metrics, f'cand_coverage@{k}')}  "
            f"oracle_microF={get(metrics, f'oracle_microF@{k}')}  "
            f"oracle_proteinF={get(metrics, f'oracle_proteinF@{k}')}"
        )

    # ==========================================================
    # RETRIEVER-LEVEL FMAX
    # ==========================================================

    print(
        "\n\n======================================="
    )
    print("FUSED RETRIEVER PERFORMANCE")
    print(
        "=======================================\n"
    )

    print(
        "obs_fmax               :",
        get(metrics, "obs_fmax"),
    )

    print(
        "obs_aupr               :",
        get(metrics, "obs_aupr"),
    )

    print(
        "obs_protein_fmax       :",
        get(metrics, "obs_protein_fmax"),
    )

    print(
        "obs_protein_fmax thresh:",
        get(
            metrics,
            "obs_protein_fmax_threshold",
        ),
    )

    print(
        "\nSaved metrics:",
        json_path,
    )


if __name__ == "__main__":
    main()