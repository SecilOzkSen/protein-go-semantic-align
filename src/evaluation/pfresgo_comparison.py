"""Compare raw/V1/V2/V3 scores using a common PFresGO-style evaluation.

This follows the previous eval_pfresgo.py adapter (including label propagation).
It is not a byte-for-byte reproduction of PFresGO's original label handling.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def go_string(x):
    s = str(x)
    return f'GO:{int(s):07d}' if s.isdigit() else s


def sigmoid(x):
    return (1 / (1 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -80, 80)))).astype(np.float32)


def read_npz(path):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    for key in ('scores', 'labels', 'go_ids', 'protein_ids'):
        if key not in d:
            raise ValueError(f'{path}: missing {key}')
    if d['scores'].shape != d['labels'].shape or d['scores'].shape != (len(d['protein_ids']), len(d['go_ids'])):
        raise ValueError(f'{path}: inconsistent dimensions')
    if not np.isfinite(d['scores']).all():
        raise ValueError(f'{path}: non-finite scores')
    return d


def load_head(version, directory, data, device):
    if version == 'v1':
        from src.protein_calibration.model import CardinalityGuidedCalibrator as Model
        from src.protein_calibration.io import hash_go_ids
        model = Model()
    elif version == 'v2':
        from src.protein_calibrationv2.model import CardinalityGuidedCalibratorV2 as Model
        from src.protein_calibrationv2.io import hash_go_ids
        model = Model(protein_dim=data['protein_embeddings'].shape[1])
    else:
        from src.protein_calibrationv3.model import ProteinCalibrationV3 as Model
        from src.protein_calibrationv3.io import hash_go_ids
        model = Model(protein_dim=data['protein_embeddings'].shape[1])
    ckpt = torch.load(directory/'head.pt', map_location='cpu', weights_only=False)
    report = json.loads((directory/'validation_results.json').read_text())
    go_hash = hash_go_ids(data['go_ids'])
    if ckpt['go_hash'] != go_hash or report['go_id_hash'] != go_hash:
        raise ValueError(f'{version}: GO ID hash mismatch')
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device).eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(data['scores']), 128):
            scores = torch.as_tensor(data['scores'][i:i+128], dtype=torch.float32, device=device)
            if version == 'v1':
                out = model(scores)
            else:
                emb = torch.as_tensor(data['protein_embeddings'][i:i+128], dtype=torch.float32, device=device)
                out = model(scores, emb)
            chunks.append(out['scores'].cpu().numpy())
    return np.concatenate(chunks), report


def ancestors_from_obo(obo, ids):
    import networkx as nx
    import obonet
    graph = obonet.read_obo(str(obo))
    mapping = {go_string(g): i for i, g in enumerate(ids)}
    return [np.array([mapping[a] for a in nx.descendants(graph, g) if a in mapping], dtype=np.int64)
            if g in graph else np.array([], dtype=np.int64)
            for g in map(go_string, ids)]


def propagate(mat, ancestors):
    original = np.asarray(mat)
    out = original.copy()
    for child, parents in enumerate(ancestors):
        if len(parents):
            out[:, parents] = np.maximum(out[:, parents], original[:, child, None])
    return out


def f_at(y, scores, t):
    # PFresGO-style: strict >; precision over predicted-and-annotated proteins,
    # recall sum divided by ALL evaluated proteins.
    selected = scores > t
    tp = (selected & (y > 0)).sum(axis=1)
    pred_n = selected.sum(axis=1)
    true_n = (y > 0).sum(axis=1)
    mask = (pred_n > 0) & (true_n > 0)
    if not mask.any():
        return 0.
    precision = float(np.mean(tp[mask] / pred_n[mask]))
    recall = float(np.sum(tp[mask] / true_n[mask]) / len(y))
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.


def metrics(labels, logits, ancestors, root_index, fixed_logit_threshold=None):
    y = propagate(labels.astype(np.uint8), ancestors)
    scores = propagate(sigmoid(logits), ancestors)
    if root_index is not None:
        y[:, root_index] = 0
        scores[:, root_index] = -np.inf
    grid = np.linspace(0, .99, 100, dtype=np.float32)
    curve = np.array([f_at(y, scores, t) for t in grid])
    best = int(curve.argmax())
    term_mask = y.sum(axis=0) > 0
    if root_index is not None:
        term_mask[root_index] = False
    if not term_mask.any():
        raise ValueError('No GO terms with positive labels')
    safe_scores = scores.copy()
    if root_index is not None:
        safe_scores[:, root_index] = 0
    result = {
        'fmax_oracle_diagnostic': float(curve[best]),
        'oracle_threshold_probability_diagnostic': float(grid[best]),
        'macro_aupr': float(average_precision_score(y[:, term_mask], safe_scores[:, term_mask], average='macro')),
        'micro_aupr': float(average_precision_score(y[:, term_mask], safe_scores[:, term_mask], average='micro')),
        'flattened_auroc': float(roc_auc_score(y.ravel(), safe_scores.ravel())) if np.unique(y).size == 2 else None,
        'n_proteins': len(y), 'n_go_terms': y.shape[1], 'n_terms_with_positives': int(term_mask.sum()),
    }
    if fixed_logit_threshold is not None:
        prob_t = float(sigmoid([fixed_logit_threshold])[0])
        result['validation_selected_threshold_logit'] = float(fixed_logit_threshold)
        result['validation_selected_threshold_probability'] = prob_t
        result['test_F_at_validation_threshold'] = f_at(y, scores, prob_t)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test', required=True, help='V2/V3 NPZ, including protein embeddings')
    p.add_argument('--obo', required=True)
    p.add_argument('--branch', default='bp', choices=['bp','mf','cc'])
    p.add_argument('--out-dir', required=True)
    p.add_argument('--v1-standard')
    p.add_argument('--v1-balanced')
    p.add_argument('--v2')
    p.add_argument('--v3')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    d = read_npz(a.test)
    if any([a.v2, a.v3]) and 'protein_embeddings' not in d:
        raise ValueError('V2/V3 require protein_embeddings in NPZ')
    ancestors = ancestors_from_obo(a.obo, d['go_ids'])
    mapping = {go_string(x): i for i, x in enumerate(d['go_ids'])}
    root = {'bp':'GO:0008150','mf':'GO:0003674','cc':'GO:0005575'}[a.branch]
    root_index = mapping.get(root)
    results = {'protocol': 'PFresGO-style 100-threshold oracle, sigmoid, ancestor max propagation; propagated labels',
               'branch': a.branch, 'root_present': root_index is not None, 'models': {}}
    results['models']['raw'] = metrics(d['labels'], d['scores'], ancestors, root_index)
    for name, version, path in [('v1_standard','v1',a.v1_standard), ('v1_balanced','v1',a.v1_balanced),
                                ('v2','v2',a.v2), ('v3','v3',a.v3)]:
        if not path:
            continue
        calibrated, report = load_head(version, Path(path), d, torch.device(a.device))
        results['models'][name] = metrics(d['labels'], calibrated, ancestors, root_index,
                                           report['calibrated_valid_threshold'])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'results.json').write_text(json.dumps(results, indent=2))
    columns = ['model','fmax_oracle_diagnostic','test_F_at_validation_threshold','macro_aupr','micro_aupr',
               'flattened_auroc','n_proteins','n_go_terms','n_terms_with_positives']
    with (out/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name, row in results['models'].items():
            writer.writerow({'model':name, **{k:row.get(k) for k in columns[1:]}})
    print(json.dumps(results, indent=2))
    print('Saved:', out/'results.csv')
    print('NOTE: oracle Fmax is diagnostic; validation-selected threshold results are separate.')


if __name__ == '__main__':
    main()
