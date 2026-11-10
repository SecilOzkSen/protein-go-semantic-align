"""Post-process saved V2/V3 temporal RAW scores with GO True Path Rule.

Uses MZSGO's compute_propagated_metrics on the 34-temporal-GO induced
ancestor graph. No GPU, ESM encoding, or checkpoint reload required.

Important: this mirrors the *unseen-only* propagation scope of
MZSGO/utils/test_zero.py, NOT propagation over the full GO ontology.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def go_id(x):
    s = str(x)
    return "GO:" + str(int(s.split(":")[-1])).zfill(7)


def load_original_util(repo):
    path = Path(repo) / "utils" / "util.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("mzsgo_original_util_tpr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "compute_propagated_metrics"):
        raise RuntimeError("MZSGO util.py missing compute_propagated_metrics")
    return module


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Existing v2_raw_temporal or v3_pure_temporal output")
    parser.add_argument("--out-dir", required=True, help="New directory for propagated results")
    parser.add_argument("--root", default="/workspace/MZSGO_temporal")
    parser.add_argument("--mzsgo-repo", default="/workspace/MZSGO")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out = Path(args.out_dir)
    if out.resolve() == raw_dir.resolve():
        raise ValueError("Output must be separate from raw results")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory not empty: {out}")

    proc = Path(args.root) / "processed" / "bp"
    temporal_path = proc / "temporal_go_ids_bp.json"
    labels_path = proc / "pid_to_go_temporal_bp.json"
    ontology_path = proc / "go_texts_canonical_segmented_dual_2023_2025.jsonl"

    raw_scores_path = raw_dir / "scores_01.npy"
    ids_path = raw_dir / "protein_ids.json"
    go_ids_path = raw_dir / "go_ids.json"
    for path in [temporal_path, labels_path, ontology_path, raw_scores_path, ids_path, go_ids_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    protein_ids = json.loads(ids_path.read_text())
    all_go_ids = [go_id(g) for g in json.loads(go_ids_path.read_text())]
    unseen = sorted({go_id(g) for g in json.loads(temporal_path.read_text())})
    if len(unseen) != 34 or len(all_go_ids) != 20345 or len(protein_ids) != 3687:
        raise RuntimeError("Unexpected BP temporal protocol dimensions")
    if len(set(all_go_ids)) != len(all_go_ids) or len(set(protein_ids)) != len(protein_ids):
        raise RuntimeError("Duplicate protein or GO IDs")
    if not set(unseen) <= set(all_go_ids):
        raise RuntimeError("Missing temporal GO IDs in scored candidate universe")

    raw_scores = np.load(raw_scores_path, mmap_mode="r")
    if raw_scores.shape != (len(protein_ids), len(all_go_ids)):
        raise RuntimeError(f"Score matrix shape mismatch: {raw_scores.shape}")
    cols = {g: i for i, g in enumerate(all_go_ids)}
    scores = np.asarray(raw_scores[:, [cols[g] for g in unseen]], dtype=np.float32)
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise RuntimeError("Expected finite [0,1] scores_01.npy")

    annotations = json.loads(labels_path.read_text())
    if set(protein_ids) != set(annotations):
        raise RuntimeError("Protein IDs do not match temporal annotations")
    unseen_cols = {g: i for i, g in enumerate(unseen)}
    gold = np.zeros((len(protein_ids), len(unseen)), dtype=np.float32)
    for i, protein in enumerate(protein_ids):
        for g in annotations[protein]:
            normalized = go_id(g)
            if normalized not in unseen_cols:
                raise RuntimeError(f"Non-temporal ground truth: {normalized}")
            gold[i, unseen_cols[normalized]] = 1
    if not np.all(gold.sum(axis=1) > 0):
        raise RuntimeError("Protein without temporal positives")

    parents = {}
    with ontology_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            parents[go_id(row["go_id"])] = {
                go_id(p) for p in row.get("is_a_parent_ids", [])
            }
    if not set(unseen) <= set(parents):
        raise RuntimeError("Temporal GO missing from ontology JSONL")

    # Transitive ancestor closure, restricted to unseen labels.
    # adj[child, parent] = 1 is MZSGO's expected direction.
    adjacency = np.zeros((len(unseen), len(unseen)), dtype=np.float32)
    for child in unseen:
        stack = list(parents[child])
        visited = set()
        while stack:
            parent = stack.pop()
            if parent in visited:
                continue
            visited.add(parent)
            if parent == child:
                raise RuntimeError(f"GO ontology cycle at {child}")
            if parent in unseen_cols:
                adjacency[unseen_cols[child], unseen_cols[parent]] = 1
            stack.extend(parents.get(parent, ()))

    util = load_original_util(args.mzsgo_repo)
    y = torch.from_numpy(gold)
    s = torch.from_numpy(scores.copy())
    adj = torch.from_numpy(adjacency)
    raw = util.evaluate_annotations(y, s, "temporal_raw_recheck")
    prop = util.compute_propagated_metrics(y, s, adj)
    prop_scores = prop[5].detach().cpu().numpy()
    if np.any(prop_scores + 1e-7 < scores):
        raise RuntimeError("Propagation unexpectedly lowered a score")

    def metrics(values):
        f, precision, recall, aupr, threshold = values[:5]
        return {
            "Fmax": float(f), "precision": float(precision),
            "recall": float(recall), "AUPR": float(aupr),
            "threshold": float(threshold)
        }

    report = {
        "scope": "34 temporal GO terms only, MZSGO unseen adjacency subgraph",
        "ground_truth_propagated": False,
        "prediction_propagation": "max(child, ancestor), transitive is_a closure",
        "n_proteins": len(protein_ids),
        "n_temporal_go": len(unseen),
        "n_unseen_ancestor_edges": int(adjacency.sum()),
        "n_changed_scores": int(np.count_nonzero(prop_scores > scores + 1e-7)),
        "raw": metrics(raw),
        "propagated": metrics(prop),
        "delta_Fmax": float(prop[0] - raw[0]),
        "delta_AUPR": float(prop[3] - raw[3]),
        "source_raw_dir": str(raw_dir),
        "source_scores_sha256": sha256(raw_scores_path),
        "ontology_sha256": sha256(ontology_path),
        "note": "Not full-ontology TPR; only temporal-only scores are propagated. Full-universe ranking metrics are unchanged and not recomputed.",
    }
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "temporal_scores_propagated_01.npy", prop_scores.astype(np.float32))
    (out / "temporal_go_ids.json").write_text(json.dumps(unseen, indent=2))
    (out / "metrics_tpr.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
