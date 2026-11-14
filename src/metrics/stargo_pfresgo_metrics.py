from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence
import networkx as nx
import numpy as np
import obonet
from sklearn.metrics import auc, average_precision_score, roc_curve

ROOTS = {"bp": "GO:0008150", "mf": "GO:0003674", "cc": "GO:0005575"}


def normalize_go_id(x):
    if isinstance(x, (int, np.integer)): return f"GO:{int(x):07d}"
    s = str(x).strip()
    if s.startswith("GO:"): return s
    if s.isdigit(): return f"GO:{int(s):07d}"
    raise ValueError(f"Invalid GO id: {x!r}")


class StarGOPFresGOEvaluator:
    """Vectorized reproduction of boun-tabi-lifelu/stargo pfresgo_eval.py."""

    def __init__(self, *, goterms: Sequence, ontology: str, go_graph_path: str | Path):
        self.ontology = ontology.lower()
        if self.ontology not in ROOTS: raise ValueError("ontology must be bp/mf/cc")
        self.root = ROOTS[self.ontology]
        self.goterms = np.asarray([normalize_go_id(x) for x in goterms], dtype=object)
        self.go2idx = {g: i for i, g in enumerate(self.goterms.tolist())}
        self.go_graph = obonet.read_obo(str(go_graph_path))
        universe = set(self.goterms.tolist())
        self.closure = []
        for go in self.goterms:
            rel = {go}
            if go in self.go_graph:
                rel.update(universe.intersection(nx.descendants(self.go_graph, go)))
            self.closure.append(np.asarray([self.go2idx[x] for x in rel], dtype=np.int64))
        self.root_idx = self.go2idx.get(self.root)

    def propagate_predictions(self, y_pred):
        out = np.asarray(y_pred, dtype=np.float32).copy()
        for src, dst in enumerate(self.closure):
            out[:, dst] = np.maximum(out[:, dst], out[:, src, None])
        return out

    def propagate_labels(self, y_true):
        y = np.asarray(y_true)
        out = np.zeros_like(y, dtype=bool)
        for src in np.where(y.sum(axis=0) > 0)[0]:
            rows = np.where(y[:, src] > 0)[0]
            if rows.size:
                out[np.ix_(rows, self.closure[src])] = True
        if self.root_idx is not None: out[:, self.root_idx] = False
        return out

    def protein_fmax(self, y_true, pred_prop) -> Dict[str, float]:
        true_prop = self.propagate_labels(y_true)
        true_n = true_prop.sum(axis=1)
        best = None
        for t in np.linspace(0.0, 0.99, 100, dtype=np.float32):
            pred = pred_prop > float(t)
            if self.root_idx is not None: pred[:, self.root_idx] = False
            pred_n = pred.sum(axis=1)
            overlap = np.logical_and(pred, true_prop).sum(axis=1)
            valid = (pred_n > 0) & (true_n > 0)
            m = int(valid.sum())
            if not m: continue
            P = float(np.mean(overlap[valid] / pred_n[valid]))
            R = float(np.sum(overlap[valid] / true_n[valid])) / y_true.shape[0]
            if P + R <= 0: continue
            F = 2 * P * R / (P + R)
            if best is None or F > best[0]: best = (F, float(t), P, R)
        if best is None: raise ValueError("No valid Fmax threshold")
        return {"protein_fmax": best[0], "protein_fmax_threshold": best[1],
                "protein_precision_at_fmax": best[2], "protein_recall_at_fmax": best[3]}

    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        if y_true.shape != y_pred.shape: raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
        if y_true.ndim != 2 or y_true.shape[1] != len(self.goterms): raise ValueError("GO dimension mismatch")
        pred_prop = self.propagate_predictions(y_pred)
        out = self.protein_fmax(y_true, pred_prop)
        keep = np.where(y_true.sum(axis=0) > 0)[0]
        if keep.size == 0: raise ValueError("No positive GO columns")
        yt = y_true[:, keep];
        yp = pred_prop[:, keep]
        out["micro_aupr"] = float(average_precision_score(yt, yp, average="micro"))
        out["macro_aupr"] = float(average_precision_score(yt, yp, average="macro"))
        fpr, tpr, _ = roc_curve(y_true.ravel(), pred_prop.ravel(), pos_label=1)
        out["auc"] = float(auc(fpr, tpr))
        return out
