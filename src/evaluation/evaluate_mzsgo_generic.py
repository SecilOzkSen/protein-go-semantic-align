#!/usr/bin/env python3
"""Model-independent MZSGO-style overall/seen/unseen evaluation.

Inputs are an NPZ with protein_ids, go_ids, labels and optionally scores,
plus a train protein->GO annotation JSON. Scores can instead be supplied as NPY.
The NPZ need not originate from any particular model.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def normalize_go(x):
    s = str(x).strip()
    if s.startswith('GO:'):
        return 'GO:' + str(int(s[3:])).zfill(7)
    return 'GO:' + str(int(s)).zfill(7)


def parse_annotation_json(path):
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, dict):
        raise ValueError('Expected training annotations as {protein_id: [GO IDs]} JSON')
    terms = set()
    for protein, value in obj.items():
        if isinstance(value, dict):
            for key in ('go_ids', 'gos', 'labels', 'annotations', 'pos_go_ids'):
                if key in value:
                    value = value[key]
                    break
            else:
                raise ValueError(f'No GO list found for protein {protein}')
        if not isinstance(value, (list, tuple)):
            raise ValueError(f'GO annotations for {protein} are not a list')
        for item in value:
            terms.add(normalize_go(item))
    return terms


def mzsgo_metrics(y, scores):
    """Exact evaluation algorithm in MZSGO utils/util.py evaluate_annotations.

    Drop proteins with no positive annotations in the selected GO columns;
    threshold grid 0..1 step .01; protein-averaged precision over proteins
    with >=1 prediction, recall over all retained proteins; trapezoidal AUPR
    after sorting PR points by recall.
    """
    keep = np.asarray(y.sum(axis=1) > 0).reshape(-1)
    y, scores = y[keep], scores[keep]
    if not len(y):
        return None
    y = y.astype(bool)
    counts = y.sum(axis=1)
    precisions, recalls, fs = [], [], []
    for t in np.linspace(0, 1, 101):
        pred = scores >= t
        tp = (y & pred).sum(axis=1)
        pred_count = pred.sum(axis=1)
        has_pred = pred_count > 0
        precision = float((tp[has_pred] / pred_count[has_pred]).mean()) if has_pred.any() else 0.
        recall = float((tp / counts).mean())
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        precisions.append(precision); recalls.append(recall); fs.append(f)
    best = int(np.argmax(fs))
    order = np.argsort(recalls)
    return dict(Fmax=float(fs[best]), precision=float(precisions[best]),
                recall=float(recalls[best]), AUPR=float(np.trapz(np.asarray(precisions)[order], np.asarray(recalls)[order])),
                threshold=float(np.linspace(0, 1, 101)[best]), proteins_with_positive=int(len(y)),
                positive_pairs=int(y.sum()))


def main():
    p = argparse.ArgumentParser(description='Model-independent MZSGO seen/unseen evaluation')
    p.add_argument('--data-npz', required=True, help='NPZ with labels, go_ids, protein_ids; optionally scores')
    p.add_argument('--scores-npy', help='Optional [proteins, GO] scores .npy; otherwise data NPZ scores')
    p.add_argument('--train-labels-json', required=True, help='Protein -> list of GO annotations, training set only')
    p.add_argument('--out', help='Optional JSON output')
    a = p.parse_args()
    with np.load(a.data_npz, allow_pickle=False) as z:
        for key in ('labels','go_ids','protein_ids'):
            if key not in z: raise ValueError(f'Missing {key} in {a.data_npz}')
        labels = z['labels'].copy()
        go_ids = z['go_ids'].copy()
        protein_ids = z['protein_ids'].copy()
        if a.scores_npy:
            scores = np.load(a.scores_npy, allow_pickle=False)
        else:
            if 'scores' not in z: raise ValueError('Missing scores in NPZ; pass --scores-npy')
            scores = z['scores'].copy()
    if labels.ndim != 2 or labels.shape != scores.shape:
        raise ValueError(f'labels {labels.shape} and scores {scores.shape} must match [N,G]')
    if len(protein_ids) != labels.shape[0] or len(go_ids) != labels.shape[1]:
        raise ValueError('Protein/GO ID length mismatch')
    if not np.isin(labels, (0,1)).all(): raise ValueError('labels must be binary')
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise ValueError('MZSGO evaluator requires finite scores in [0,1]. Apply a predeclared transform before evaluation, never fit on test.')
    normalized_go = [normalize_go(x) for x in go_ids]
    if len(set(normalized_go)) != len(normalized_go): raise ValueError('Duplicate GO IDs')
    if len(set(map(str,protein_ids))) != len(protein_ids): raise ValueError('Duplicate protein IDs')
    train_go = parse_annotation_json(a.train_labels_json)
    test_pos = labels.sum(axis=0) > 0
    seen = np.array([(go in train_go) and bool(test_pos[j]) for j,go in enumerate(normalized_go)])
    unseen = np.array([(go not in train_go) and bool(test_pos[j]) for j,go in enumerate(normalized_go)])
    train_only = len(train_go - set(normalized_go))
    overall = mzsgo_metrics(labels, scores)
    seen_result = mzsgo_metrics(labels[:,seen],scores[:,seen])
    unseen_result = mzsgo_metrics(labels[:,unseen],scores[:,unseen])
    h = None
    if seen_result is not None and unseen_result is not None:
        s,u=seen_result['AUPR'],unseen_result['AUPR']
        h=2*s*u/(s+u) if s+u else 0.
    result = {'n_proteins':int(len(protein_ids)), 'n_candidates':int(len(go_ids)),
              'n_train_go':len(train_go), 'n_seen_test_go':int(seen.sum()),
              'n_unseen_test_go':int(unseen.sum()), 'n_train_only_go_not_in_candidates':train_only,
              'overall':overall,'seen':seen_result,'unseen':unseen_result,'H':h,
              'unseen_go_ids':[normalized_go[i] for i in np.flatnonzero(unseen)]}
    print(json.dumps(result, indent=2))
    if a.out:
        out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+'\n')
        print(f'[SAVED] {out}')

if __name__ == '__main__': main()
