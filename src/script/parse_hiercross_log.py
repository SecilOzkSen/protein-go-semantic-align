from __future__ import annotations

"""
Parse HierCross training logs containing lines like:
  [val@step75000] {'fmax_topk': ..., ...}
  [val] epoch 4 :: Fmax@200=... AUPR@200=... Fmax@Full=...

Recommended use:
  python -m src.script.parse_hiercross_log \
    --log_file /workspace/logs/hiercross.log \
    --out_tsv /workspace/results/hiercross_learning_curve.tsv \
    --out_json /workspace/results/hiercross_learning_curve.json
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List


MID_RE = re.compile(r"\[val@step(\d+)\]\s+(\{.*\})")
EPOCH_RE = re.compile(
    r"\[val\]\s+epoch\s+(\d+)\s+::\s+"
    r"Fmax@(?:200|Cand|TopK)=([0-9.]+)\s+"
    r"AUPR@(?:200|Cand|TopK)=([0-9.]+)\s+"
    r"Fmax@Full=([0-9.]+)\s+"
    r"AUPR@Full=([0-9.]+)\s+"
    r"R@K=([0-9.]+)\s+"
    r"hits@1=([0-9.]+)\s+"
    r"hits@5=([0-9.]+)\s+"
    r"hits@10=([0-9.]+)"
)


def parse_log(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = MID_RE.search(line)
            if m:
                step = int(m.group(1))
                try:
                    metrics = ast.literal_eval(m.group(2))
                except Exception:
                    continue
                if not isinstance(metrics, dict):
                    continue
                row = {"type": "mid", "step": step, "epoch": None}
                row.update(metrics)
                rows.append(row)
                continue

            m = EPOCH_RE.search(line)
            if m:
                epoch = int(m.group(1))
                row = {
                    "type": "epoch",
                    "epoch": epoch,
                    "step": None,
                    "fmax_topk": float(m.group(2)),
                    "aupr_topk": float(m.group(3)),
                    "fmax_full": float(m.group(4)),
                    "aupr_full": float(m.group(5)),
                    "retrieval_recall@K": float(m.group(6)),
                    "hits@1": float(m.group(7)),
                    "hits@5": float(m.group(8)),
                    "hits@10": float(m.group(9)),
                }
                rows.append(row)
    return rows


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser("Parse P3a-HierCross logs into learning-curve files.")
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--out_tsv", type=str, required=True)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    rows = parse_log(args.log_file)
    if not rows:
        raise RuntimeError(f"No validation rows parsed from {args.log_file}")

    keys = [
        "type", "epoch", "step",
        "fmax_topk", "aupr_topk", "fmax_full", "aupr_full",
        "hits@1", "hits@5", "hits@10",
        "retrieval_recall@K", "oracle_microF@K",
        "bce", "dag_loss", "loss",
    ]

    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r.get(k)) for k in keys) + "\n")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    best = max(rows, key=lambda r: float(r.get("fmax_full", -1.0)))
    print("parsed rows:", len(rows))
    print("best fmax_full:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
