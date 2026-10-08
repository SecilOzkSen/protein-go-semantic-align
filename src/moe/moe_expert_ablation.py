"""Experiment D: checkpoint expert-only and direct-support ablation. No training."""
import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.moe.dataset import MoEDataset
from src.moe.model import MoEConfig, NeighbourGuidedMoE
from src.metrics.stargo_pfresgo_metrics import StarGOPFresGOEvaluator


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--val_dump", type=Path, required=True)
    p.add_argument("--evidence_dir", type=Path, required=True)
    p.add_argument("--neighbour_dir", type=Path, required=True)
    p.add_argument("--go_graph_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--ontology", default="bp", choices=("bp", "mf", "cc"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--save_predictions", action="store_true")
    return p.parse_args()


def as_float(x):
    return float(x.item()) if isinstance(x, np.generic) else float(x)


def main():
    a = args_parser()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    ds = MoEDataset(a.val_dump, a.evidence_dir, a.neighbour_dir, "val")
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=0)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {})
    if meta.get("go_ids") != ds.go_ids.tolist():
        raise RuntimeError("Checkpoint and validation GO ID order mismatch")
    model = NeighbourGuidedMoE(MoEConfig(hidden_dim=a.hidden_dim, dropout=a.dropout))
    model.load_state_dict(ck["model"], strict=True)
    device = torch.device(a.device)
    model.to(device).eval()
    predictions = {k: [] for k in ("moe", "retriever_expert", "neighbour_expert", "direct_5nn")}
    labels = []
    gates = []
    with torch.inference_mode():
        for s, d, e, n, y in dl:
            out = model(s.to(device), d.to(device), e.to(device), n.to(device), return_details=True)
            for name, tensor in (
                    ("moe", out["probability"]),
                    ("retriever_expert", out["retriever_probability"]),
                    ("neighbour_expert", out["neighbour_probability"]),
                    ("direct_5nn", d),
            ):
                arr = tensor.detach().cpu().numpy().astype(np.float32)
                if not np.isfinite(arr).all():
                    raise RuntimeError(f"Non-finite predictions in {name}")
                predictions[name].append(arr)
            labels.append(y.numpy().astype(np.int8))
            gates.append(out["retriever_weight"].detach().cpu().numpy().astype(np.float32))
    labels = np.concatenate(labels)
    gates = np.concatenate(gates)
    predictions = {k: np.concatenate(v) for k, v in predictions.items()}
    assert labels.shape == (len(ds), len(ds.go_ids))
    assert all(x.shape == labels.shape for x in predictions.values())
    for name, x in predictions.items():
        print(
            f"[RANGE CHECK] {name}: "
            f"min={x.min():.6f}, "
            f"max={x.max():.6f}, "
            f"mean={x.mean():.6f}, "
            f"finite={np.isfinite(x).all()}"
        )

        if not np.isfinite(x).all():
            raise RuntimeError(f"Non-finite predictions: {name}")

        if x.min() < -1e-6 or x.max() > 1.000001:
            raise RuntimeError(
                f"{name} outside probability range: "
                f"[{x.min():.6f}, {x.max():.6f}]"
            )
    evaluator = StarGOPFresGOEvaluator(goterms=ds.go_ids, ontology=a.ontology, go_graph_path=a.go_graph_path)
    results = {}
    for name, probs in predictions.items():
        metrics = evaluator.evaluate(labels, probs)
        results[name] = {k: as_float(v) for k, v in metrics.items()}
        logging.info("%-20s Fmax=%.6f @ %.3f macroAUPR=%.6f microAUPR=%.6f AUC=%.6f",
                     name, results[name]["protein_fmax"], results[name]["protein_fmax_threshold"],
                     results[name]["macro_aupr"], results[name]["micro_aupr"], results[name]["auc"])
    mean_gate = float(gates.mean())
    logging.info("Gate retriever mean=%.6f median=%.6f p10=%.6f p90=%.6f",
                 mean_gate, float(np.median(gates)), float(np.quantile(gates, .1)), float(np.quantile(gates, .9)))
    best = float(meta.get("best_fmax", float("nan")))
    diff = abs(results["moe"]["protein_fmax"] - best)
    if not np.isfinite(best) or diff > 0.001:
        raise RuntimeError(f"Checkpoint parity FAILED: stored={best:.6f}, recomputed={results['moe']['protein_fmax']:.6f}; no report saved")
    report = {
        "checkpoint": str(a.checkpoint), "checkpoint_epoch": meta.get("epoch"),
        "checkpoint_best_epoch": meta.get("best_epoch"), "checkpoint_best_fmax": best,
        "val_dump": str(a.val_dump), "n_proteins": len(ds), "n_go": len(ds.go_ids),
        "gate_retriever_mean": mean_gate,
        "gate_retriever_p10": float(np.quantile(gates, .1)),
        "gate_retriever_p90": float(np.quantile(gates, .9)),
        "metrics": results,
        "note": "Experts are evaluated as jointly-trained components, not independently retrained baselines."
    }
    (a.output_dir / "expert_ablation.json").write_text(json.dumps(report, indent=2))
    columns = ["model", "protein_fmax", "protein_fmax_threshold", "macro_aupr", "micro_aupr", "auc"]
    with (a.output_dir / "expert_ablation.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for name, metric in results.items():
            w.writerow([name] + [metric.get(k, "") for k in columns[1:]])
    if a.save_predictions:
        for name, probs in predictions.items():
            np.save(a.output_dir / f"{name}_probabilities.float32.npy", probs)
        np.save(a.output_dir / "labels.int8.npy", labels)
        np.save(a.output_dir / "retriever_gate.float32.npy", gates)
        (a.output_dir / "protein_ids.json").write_text(json.dumps(ds.ids))
        np.save(a.output_dir / "go_ids.int64.npy", ds.go_ids)
    logging.info("PASSED checkpoint parity; reports saved to %s", a.output_dir)


if __name__ == "__main__":
    main()
