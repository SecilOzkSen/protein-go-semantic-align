import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple, Optional

def read_lengths(path: str) -> Dict[str, int]:
    """
    Read a mapping of sequence_id -> length from JSON or CSV.
    - JSON: {"seq1": 123, "seq2": 456, ...}
    - CSV: header with columns id,length (order doesn't matter); separator=","
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate
        out = {}
        for k, v in data.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out
    elif ext == ".pkl":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Validate
        out = {}
        for k, v in data.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .json or .csv.")

def basic_stats(lengths: List[int]) -> Dict[str, float]:
    import numpy as np
    arr = np.array(lengths, dtype=float)
    stats = {
        "count": int(arr.size),
        "min": float(np.min(arr)) if arr.size else math.nan,
        "max": float(np.max(arr)) if arr.size else math.nan,
        "mean": float(np.mean(arr)) if arr.size else math.nan,
        "median": float(np.median(arr)) if arr.size else math.nan,
        "p90": float(np.percentile(arr, 90)) if arr.size else math.nan,
        "p95": float(np.percentile(arr, 95)) if arr.size else math.nan,
        "p99": float(np.percentile(arr, 99)) if arr.size else math.nan,
        "std": float(np.std(arr)) if arr.size else math.nan,
    }
    return stats

def tail_proportions(lengths: List[int], thresholds=(1024, 2048, 4096, 8192, 16384, 32768)) -> Dict[str, float]:
    n = len(lengths)
    out = {}
    if n == 0:
        return {f">={t}": 0.0 for t in thresholds}
    for t in thresholds:
        out[f">={t}"] = sum(1 for L in lengths if L >= t) / n
    return out

def recommend_strategy(stats: Dict[str, float], tails: Dict[str, float]) -> Dict[str, object]:
    """
    Heuristic rules for choosing processing strategy:
    - If max <= 2048 and p99 <= 2048: FULL-LENGTH (masked) recommended.
    - If 2048 < max <= 8192 or >=1024 tail > 0.2: SEGMENTATION with win=1024, stride=256.
    - If max > 8192 or >=4096 tail > 0.1: SEGMENTATION with win=2048, stride=512; consider multi-pass.
    - If max > 20000: enable streaming windows + shard-by-protein to keep memory in check.
    Truncation is discouraged; only as coarse prefilter for rough experiments.
    """
    mx = stats.get("max", 0) or 0
    p99 = stats.get("p99", 0) or 0
    tail_1k = tails.get(">=1024", 0.0)
    tail_4k = tails.get(">=4096", 0.0)

    rec = {
        "strategy": "full_length_masked",
        "window_len": None,
        "stride": None,
        "notes": [],
        "attribution": {
            "delta_y_method": "gradient_surrogate_training__maskout_eval_topk",
            "topk_per_window": 64,
            "entropy_reg_lambda_alpha": 0.05,
            "entropy_reg_lambda_window": 0.01
        }
    }

    if mx <= 2048 and p99 <= 2048:
        rec["strategy"] = "full_length_masked"
        rec["notes"].append("All sequences comfortably fit under 2k; use masked softmax over real tokens only.")
    elif (2048 < mx <= 8192) or (tail_1k > 0.2):
        rec["strategy"] = "segmentation"
        rec["window_len"] = 1024
        rec["stride"] = 256
        rec["notes"].append("Use overlapping windows to preserve long-range signals; aggregate with window-level attention.")
    elif (mx > 8192) or (tail_4k > 0.1):
        rec["strategy"] = "segmentation"
        rec["window_len"] = 2048
        rec["stride"] = 512
        rec["notes"].append("Very long sequences present; prefer larger windows and potentially two-stage batching.")
    if mx > 20000:
        rec["notes"].append("Enable streaming windowing and per-protein sharding to avoid OOM.")
    # Truncation hint
    rec["notes"].append("Avoid truncation; only use for coarse ablations or sanity checks.")
    return rec

def render_report(stats: Dict[str, float], tails: Dict[str, float], rec: Dict[str, object]) -> str:
    lines = []
    lines.append("# Sequence Length Analysis Report")
    lines.append("")
    lines.append("## Basic Stats")
    for k in ["count", "min", "max", "mean", "median", "p90", "p95", "p99", "std"]:
        v = stats.get(k, float("nan"))
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.2f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Tail Proportions")
    for k, v in tails.items():
        lines.append(f"- proportion of sequences {k}: {v*100:.2f}%")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- strategy: **{rec['strategy']}**")
    if rec.get("window_len"):
        lines.append(f"- window_len: {rec['window_len']}")
    if rec.get("stride"):
        lines.append(f"- stride: {rec['stride']}")
    lines.append("- notes:")
    for n in rec.get("notes", []):
        lines.append(f"  - {n}")
    lines.append("")
    lines.append("## Attribution Settings (suggested)")
    att = rec.get("attribution", {})
    lines.append(f"- Δŷ method (train): {att.get('delta_y_method', 'gradient_surrogate_training__maskout_eval_topk')}")
    lines.append(f"- Top-K per window (eval/mask-out): {att.get('topk_per_window', 64)}")
    lines.append(f"- λ_entropy (α): {att.get('entropy_reg_lambda_alpha', 0.05)}")
    lines.append(f"- λ_entropy (window): {att.get('entropy_reg_lambda_window', 0.01)}")
    return "\n".join(lines)

def save_histogram(lengths: List[int], out_png: str, bins: Optional[int] = 60):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib not available, skipping histogram: {e}", file=sys.stderr)
        return
    import numpy as np
    arr = np.array(lengths, dtype=float)
    plt.figure()
    plt.hist(arr, bins=bins if bins else 60)
    plt.xlabel("Sequence length (aa)")
    plt.ylabel("Count")
    plt.title("Sequence Length Distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Analyze sequence lengths and recommend processing strategy.")
    ap.add_argument("--input", required=True, help="Path to JSON or CSV with mapping id->length.")
    ap.add_argument("--outdir", default="./seq_length_report", help="Output directory for report files.")
    ap.add_argument("--hist", action="store_true", help="Also save a histogram PNG.")
    ap.add_argument("--bins", type=int, default=60, help="Histogram bin count.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = read_lengths(args.input)
    lengths = list(data.values())

    if not lengths:
        print("No lengths found. Check the input file.", file=sys.stderr)
        sys.exit(2)

    stats = basic_stats(lengths)
    tails = tail_proportions(lengths)
    rec = recommend_strategy(stats, tails)

    # Save report (markdown and json), print to stdout
    report_md = render_report(stats, tails, rec)
    md_path = os.path.join(args.outdir, "length_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    json_path = os.path.join(args.outdir, "length_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "stats": stats,
            "tails": tails,
            "recommendation": rec
        }, f, ensure_ascii=False, indent=2)

    if args.hist:
        png_path = os.path.join(args.outdir, "length_hist.png")
        save_histogram(lengths, png_path, bins=args.bins)

    # Console output
    print(report_md)
    print("\nSaved files:")
    print(f"- {md_path}")
    print(f"- {json_path}")
    if args.hist:
        print(f"- {png_path}")

if __name__ == "__main__":
    main()
