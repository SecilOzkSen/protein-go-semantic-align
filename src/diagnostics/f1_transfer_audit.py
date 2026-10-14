"""Offline F1 transfer audit: threshold, per-GO discrimination, GO geometry.
Requires scores.npz from f1_collect_scores.py for valid and test.
No training. Test-optimal threshold is DIAGNOSTIC ONLY.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr


def load(folder):
    a = np.load(Path(folder) / "scores.npz")
    return {k: a[k] for k in a.files}


def fmetrics(s, y, t):
    pred = s >= t
    tp = np.sum(pred & y, axis=1).astype(float)
    npred = pred.sum(axis=1)
    ntrue = y.sum(axis=1)
    # Protein-centric CAFA-like: average precision over proteins with predictions;
    # average recall over all annotated proteins. Explicitly report coverage.
    precision = float(np.mean(tp[npred > 0] / npred[npred > 0])) if np.any(npred > 0) else 0.
    recall = float(np.mean(tp[ntrue > 0] / ntrue[ntrue > 0])) if np.any(ntrue > 0) else 0.
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    microtp = float(tp.sum());
    microfp = float(npred.sum() - microtp);
    microfn = float(ntrue.sum() - microtp)
    micro = 2 * microtp / (2 * microtp + microfp + microfn) if (2 * microtp + microfp + microfn) > 0 else 0.
    return {"protein_F": f, "micro_F": micro, "precision": precision, "recall": recall,
            "prediction_coverage": float(np.mean(npred > 0)),
            "mean_predicted_labels": float(np.mean(npred))}


def threshold_audit(v, t):
    sv, yv = v["scores"], v["labels"].astype(bool)
    st, yt = t["scores"], t["labels"].astype(bool)
    grid = np.linspace(float(min(sv.min(), st.min())), float(max(sv.max(), st.max())), 401)
    fv = [fmetrics(sv, yv, x)["protein_F"] for x in grid]
    i = int(np.argmax(fv));
    chosen = float(grid[i])
    ft = [fmetrics(st, yt, x)["protein_F"] for x in grid]
    j = int(np.argmax(ft))
    result = {"validation_selected_threshold": chosen,
              "validation_at_selected": fmetrics(sv, yv, chosen),
              "test_at_validation_threshold": fmetrics(st, yt, chosen),
              "test_oracle_threshold_DIAGNOSTIC_ONLY": float(grid[j]),
              "test_oracle_DIAGNOSTIC_ONLY": fmetrics(st, yt, float(grid[j])),
              "test_oracle_minus_fixed_protein_F": float(ft[j] - ft[i]),
              "note": "Test oracle threshold must never be used to choose models or deployment thresholds. Protein F definition is explicit here; verify against src.metrics.cafa before calling it benchmark parity."}
    return result


def go_discrimination(v, t, out, min_pos):
    sv, yv = v["scores"], v["labels"].astype(bool)
    st, yt = t["scores"], t["labels"].astype(bool)
    rows = []
    for k, g in enumerate(v["go_ids"]):
        a = yv[:, k];
        b = yt[:, k]
        nva = int(a.sum());
        nte = int(b.sum())
        if nva < min_pos or nte < min_pos or nva == len(a) or nte == len(b):
            continue
        pva = float(sv[a, k].mean());
        nva_score = float(sv[~a, k].mean())
        pte = float(st[b, k].mean());
        nte_score = float(st[~b, k].mean())
        apv = float(average_precision_score(a, sv[:, k]))
        apt = float(average_precision_score(b, st[:, k]))
        av = float(roc_auc_score(a, sv[:, k]))
        at = float(roc_auc_score(b, st[:, k]))
        rows.append(dict(go_id=int(g), val_pos=nva, test_pos=nte,
                         val_prevalence=float(a.mean()), test_prevalence=float(b.mean()),
                         val_positive_score=pva, test_positive_score=pte,
                         val_negative_score=nva_score, test_negative_score=nte_score,
                         val_gap=pva - nva_score, test_gap=pte - nte_score,
                         val_AP=apv, test_AP=apt, delta_AP=apt - apv,
                         val_AUC=av, test_AUC=at, delta_AUC=at - av))
    if not rows: raise RuntimeError("No shared GO terms have sufficient positives in both splits")
    with (out / "per_go_discrimination.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));
        w.writeheader();
        w.writerows(rows)
    return {"n_comparable_GO": len(rows), "minimum_positives_each_split": min_pos,
            "macro_AP_val_common": float(np.mean([r["val_AP"] for r in rows])),
            "macro_AP_test_common": float(np.mean([r["test_AP"] for r in rows])),
            "macro_AUC_val_common": float(np.mean([r["val_AUC"] for r in rows])),
            "macro_AUC_test_common": float(np.mean([r["test_AUC"] for r in rows])),
            "mean_score_gap_val": float(np.mean([r["val_gap"] for r in rows])),
            "mean_score_gap_test": float(np.mean([r["test_gap"] for r in rows])),
            "median_delta_AP": float(np.median([r["delta_AP"] for r in rows])),
            "note": "AP depends on prevalence. AUC and positive-negative score gap help separate discrimination changes from prevalence shifts. Unannotated labels are treated as negatives for this diagnostic."}


def geometry(folder, out):
    a = np.load(Path(folder) / "go_geometry.npz")
    x = a["pretrained_segment_mean"].astype(np.float64)
    z = a["learned_projected"].astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    sx = x @ x.T;
    sz = z @ z.T
    upper = np.triu_indices(len(x), 1)
    rho = float(spearmanr(sx[upper], sz[upper]).statistic)
    k = 10
    np.fill_diagonal(sx, -np.inf);
    np.fill_diagonal(sz, -np.inf)
    ix = np.argpartition(-sx, k, axis=1)[:, :k]
    iz = np.argpartition(-sz, k, axis=1)[:, :k]
    overlap = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ix, iz)]))
    return {"GO_pairwise_cos_spearman": rho, "GO_top10_neighbor_overlap": overlap,
            "n_GO": len(x),
            "note": "Pretrained input is mean of frozen BioMedBERT segment vectors, not raw token embedding. Learned output includes segment gating and projection. Geometry change alone is not evidence of harmful overfitting."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["threshold", "discrimination", "geometry", "all"], required=True)
    ap.add_argument("--min_pos", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out);
    out.mkdir(parents=True, exist_ok=True)
    v = load(args.valid_dir);
    t = load(args.test_dir)
    if not np.array_equal(v["go_ids"], t["go_ids"]):
        raise ValueError("GO universe or order differs")
    for split, d in [("valid", v), ("test", t)]:
        if len(set(map(str, d["protein_ids"]))) != len(d["protein_ids"]):
            raise ValueError(f"Duplicate protein IDs in {split}")
    if set(map(str, v["protein_ids"])) & set(map(str, t["protein_ids"])):
        raise ValueError("Validation/test protein overlap")
    result = {}
    if args.mode in ("threshold", "all"):
        result["threshold"] = threshold_audit(v, t)
    if args.mode in ("discrimination", "all"):
        result["discrimination"] = go_discrimination(v, t, out, args.min_pos)
    if args.mode in ("geometry", "all"):
        result["geometry"] = geometry(args.valid_dir, out)
    (out / f"{args.mode}_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("SAVED", out / f"{args.mode}_audit.json")


if __name__ == "__main__":
    main()
