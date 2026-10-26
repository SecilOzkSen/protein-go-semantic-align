"""Validation-selected residual blend, using exact PFresGO metric semantics.

Requires existing src.evaluation.pfresgo_fast_exact and pfresgo_comparison.
No model training; no test-set hyperparameter selection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import obonet
import torch

from src.evaluation.pfresgo_comparison import read_npz, load_head, sigmoid, go_string
from src.evaluation.pfresgo_fast_exact import (
    FullAncestors, go_ancestors, metrics, parity, load_original,
    propagate_original_scores, f_curve, ROOTS,
)

GRID = np.linspace(0.0, 0.99, 100, dtype=np.float32)


def annotated_labels(data, path, branch):
    """Read original annot.tsv, aligning both protein and GO IDs; fail if missing."""
    if not path:
        return data['labels'].astype(np.uint8)
    cols = {'mf': 1, 'bp': 2, 'cc': 3}
    annotations = {}
    in_table = False
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if fields[0].startswith('### PDB-chain'):
                in_table = True
                continue
            if not in_table or len(fields) < 4 or not fields[0].strip():
                continue
            pid = fields[0].strip()
            if pid in annotations:
                raise ValueError(f'Duplicate annotation protein: {pid}')
            annotations[pid] = set(x.strip() for x in fields[cols[branch]].split(',') if x.strip())
    if not annotations:
        raise ValueError('Could not parse annotation table')
    gos = [go_string(x) for x in data['go_ids']]
    idx = {g: j for j, g in enumerate(gos)}
    if len(idx) != len(gos):
        raise ValueError('Duplicate GO IDs')
    y = np.zeros(data['scores'].shape, dtype=np.uint8)
    for i, protein in enumerate(map(str, data['protein_ids'])):
        if protein not in annotations:
            raise ValueError(f'Protein absent from annot.tsv: {protein}')
        for go in annotations[protein]:
            j = idx.get(go)
            if j is not None:
                y[i, j] = 1
    return y


def score_metrics(y, prob, anc, full_anc, root):
    # The input is already probability-space. Do not apply sigmoid twice.
    return metrics(y, prob, anc, None, full_anc, root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--valid', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--head-dir', required=True, help='V1 balanced calibration head directory')
    p.add_argument('--annotations', required=True, help='Original PFresGO annot.tsv')
    p.add_argument('--obo', required=True)
    p.add_argument('--original-eval', required=True)
    p.add_argument('--branch', choices=['bp', 'mf', 'cc'], default='bp')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--lambdas', default='0,0.25,0.5,0.75,1')
    p.add_argument('--min-retention', type=float, default=0.98)
    p.add_argument('--parity-n', type=int, default=20)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    valid, test = read_npz(a.valid), read_npz(a.test)
    if not np.array_equal(valid['go_ids'], test['go_ids']):
        raise ValueError('Validation/test GO order mismatch')
    if set(map(str, valid['protein_ids'])) & set(map(str, test['protein_ids'])):
        raise ValueError('Validation/test protein overlap')
    lambdas = [float(x.strip()) for x in a.lambdas.split(',')]
    if len(set(lambdas)) != len(lambdas) or not all(0 <= x <= 1 for x in lambdas):
        raise ValueError('Lambda values must be unique and within [0,1]')
    if 0.0 not in lambdas or 1.0 not in lambdas:
        raise ValueError('Include both 0 and 1 as baselines')
    gos = [go_string(x) for x in valid['go_ids']]
    graph = obonet.read_obo(a.obo)
    anc = go_ancestors(graph, gos)
    full_anc = FullAncestors(graph, gos)
    original = load_original(a.original_eval)
    root = ROOTS[a.branch]
    datasets = {}
    for name, data in [('valid', valid), ('test', test)]:
        y = annotated_labels(data, a.annotations, a.branch)
        cal, _ = load_head('v1', Path(a.head_dir), data, torch.device(a.device))
        raw = sigmoid(data['scores'])
        calibrated = sigmoid(cal)
        if raw.shape != calibrated.shape or not np.isfinite(calibrated).all():
            raise ValueError('Invalid calibrated score matrix')
        datasets[name] = (data, y, raw, calibrated)
    # Validate exact implementation on representative validation and test subsets.
    for name, (data, y, raw, cal) in datasets.items():
        n = min(a.parity_n, len(y))
        for tag, prob in [('raw', raw), ('v1', cal)]:
            parity(original, y[:n], prob[:n], data['protein_ids'][:n], gos,
                   a.obo, a.branch, anc, None, full_anc,
                   out / f'_parity_{name}_{tag}.pckl')
    val_rows = []
    vdata, vy, vr, vc = datasets['valid']
    for lam in lambdas:
        prob = (1 - lam) * vr + lam * vc
        m, (curve, _, _, grid) = score_metrics(vy, prob, anc, full_anc, root)
        best = int(np.argmax(curve))
        row = {'lambda': lam, 'valid_fmax': float(curve[best]),
               'valid_threshold': float(grid[best]),
               'valid_macro_aupr': m['macro_aupr'],
               'valid_micro_aupr': m['micro_aupr'],
               'valid_auroc': m['flattened_auc']}
        val_rows.append(row)
        print('VALID', row, flush=True)
    raw_row = next(r for r in val_rows if r['lambda'] == 0)
    keys = ['macro_aupr', 'micro_aupr', 'auroc']
    for r in val_rows:
        r['feasible'] = all(r[f'valid_{key}'] >= a.min_retention * raw_row[f'valid_{key}'] for key in keys)
    feasible = [r for r in val_rows if r['feasible']]
    # Choose only from validation; raw is always feasible.
    chosen = max(feasible, key=lambda r: (r['valid_fmax'], -r['lambda']))
    print('VALIDATION-SELECTED', chosen, flush=True)
    # Test once, with a locked validation threshold and lambda.
    tdata, ty, tr, tc = datasets['test']
    final_prob = (1 - chosen['lambda']) * tr + chosen['lambda'] * tc
    test_metrics, (test_curve, _, _, _) = score_metrics(ty, final_prob, anc, full_anc, root)
    ix = int(np.argmin(np.abs(GRID - chosen['valid_threshold'])))
    if abs(float(GRID[ix]) - chosen['valid_threshold']) > 1e-6:
        raise ValueError('Validation threshold not in PFresGO 100-threshold grid')
    result = {
        'protocol': 'PFresGO original semantics, parity checked, annotation-aligned, 100 threshold grid',
        'selection': 'validation only: maximize Fmax with >= retention constraint on macro AP, micro AP and flattened AUROC',
        'selected_lambda': chosen['lambda'],
        'validation_threshold_probability': chosen['valid_threshold'],
        'test_F_at_validation_threshold': float(test_curve[ix]),
        'test_Fmax_oracle_diagnostic_only': test_metrics['fmax_test_oracle_diagnostic'],
        'test_macro_aupr': test_metrics['macro_aupr'],
        'test_micro_aupr': test_metrics['micro_aupr'],
        'test_flattened_auroc': test_metrics['flattened_auc'],
        'n_test_proteins': len(ty), 'n_go_terms': len(gos),
        'min_retention': a.min_retention,
        'valid_candidates': val_rows,
        'original_eval_sha256': hashlib.sha256(Path(a.original_eval).read_bytes()).hexdigest(),
    }
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    with (out / 'validation_grid.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(val_rows[0]))
        writer.writeheader()
        writer.writerows(val_rows)
    print(json.dumps({k: v for k, v in result.items() if k != 'valid_candidates'}, indent=2))
    print('Saved:', out / 'results.json')

if __name__ == '__main__':
    main()
