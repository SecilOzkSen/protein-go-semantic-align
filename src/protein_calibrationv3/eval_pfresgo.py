"""PFresGO-compatible diagnostic evaluation for protein_calibrationv3.
Run from repository root: python -m src.protein_calibrationv3.eval_pfresgo ...
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from src.protein_calibrationv3.io import load_matrix, hash_go_ids
from src.protein_calibrationv3.model import ProteinCalibrationV3
from src.protein_calibrationv3.fit import predict


def normalize_go(x):
    if isinstance(x, (int, np.integer)) or str(x).isdigit():
        return 'GO:%07d' % int(x)
    return str(x)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return (1 / (1 + np.exp(-np.clip(x, -80, 80)))).astype(np.float32)


def propagate(mat, ancestors):
    # Graph-based ancestor max propagation, same as PFresGO.
    out = np.asarray(mat).copy()
    # Using original child values is sufficient for transitive ancestor closure.
    for child, parents in enumerate(ancestors):
        if len(parents):
            out[:, parents] = np.maximum(out[:, parents], mat[:, child, None])
    return out


def evaluate(y, s, ancestors, root_index, prefix):
    from sklearn.metrics import average_precision_score
    yp = propagate(y.astype(np.uint8), ancestors)
    sp = propagate(s.astype(np.float32), ancestors)
    if root_index is not None:
        yp[:, root_index] = 0
        sp[:, root_index] = -np.inf
    n = len(yp)
    true_count = yp.sum(axis=1).astype(np.float64)
    valid_true = true_count > 0
    # Exact PFresGO threshold grid, strict >, precision denominator m, recall denominator n.
    grid = np.linspace(0., .99, 100, dtype=np.float32)
    fcurve = []
    for t in grid:
        selected = sp > t
        pred_count = selected.sum(axis=1)
        intersection = (selected & (yp > 0)).sum(axis=1)
        mask = (pred_count > 0) & valid_true
        m = int(mask.sum())
        precision = (intersection[mask] / pred_count[mask]).mean() if m else 0.
        recall = float((intersection[mask] / true_count[mask]).sum() / n) if m else 0.
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        fcurve.append(float(f))
    # PFresGO's function-centric AP uses terms with >=1 positive.
    valid_terms = yp.sum(axis=0) > 0
    if root_index is not None: valid_terms[root_index] = False
    finite_scores = sp.copy()
    if root_index is not None: finite_scores[:, root_index] = 0
    macro_ap = float(average_precision_score(yp[:, valid_terms], finite_scores[:, valid_terms], average='macro'))
    micro_ap = float(average_precision_score(yp[:, valid_terms], finite_scores[:, valid_terms], average='micro'))
    # PFresGO AUC flattens all GO terms, root included in its implementation.
    auc = float(roc_auc_score(yp.ravel(), finite_scores.ravel())) if np.unique(yp).size == 2 else None
    i = int(np.argmax(fcurve))
    return {'pfresgo_fmax_oracle_diagnostic': fcurve[i], 'pfresgo_best_threshold_diagnostic': float(grid[i]), 'pfresgo_macro_aupr': macro_ap,
            'pfresgo_micro_aupr': micro_ap, 'pfresgo_flattened_auc': auc, 'n_proteins': n, 'n_go_terms': int(y.shape[1]),
            'n_terms_with_positives': int(valid_terms.sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test', required=True)
    p.add_argument('--head-dir', required=True)
    p.add_argument('--obo', required=True)
    p.add_argument('--branch', default='bp', choices=['bp', 'mf', 'cc'])
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    try:
        import obonet, networkx as nx
    except ImportError as e:
        raise SystemExit('Install: pip install obonet networkx') from e
    d = load_matrix(a.test)
    directory = Path(a.head_dir)
    ckpt = torch.load(directory / 'head.pt', map_location='cpu', weights_only=False)
    report = json.loads((directory / 'validation_results.json').read_text())
    if ckpt['go_hash'] != hash_go_ids(d['go_ids']) or ckpt['go_hash'] != report['go_id_hash']:
        raise ValueError('GO order mismatch')
    model = ProteinCalibrationV3(protein_dim=d['protein_embeddings'].shape[1]).to(a.device)
    model.load_state_dict(ckpt['model'], strict=True)
    cal = predict(model, d['scores'].astype('float32'), d['protein_embeddings'].astype('float32'), torch.device(a.device))['scores']
    graph = obonet.read_obo(a.obo)
    go_ids = [normalize_go(x) for x in d['go_ids']]
    mapping = {g: i for i, g in enumerate(go_ids)}
    ancestors = []
    for g in go_ids:
        ancestors.append(
            np.array([mapping[x] for x in nx.descendants(graph, g) if x in mapping], dtype=np.int64) if g in graph else np.array([], dtype=np.int64))
    root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}[a.branch]
    root_index = mapping.get(root)
    result = {'protocol': 'PFresGO 100 thresholds, GO ancestor max propagation, sigmoid scores', 'root': root, 'root_present': root_index is not None,
              'raw': evaluate(d['labels'], sigmoid(d['scores']), ancestors, root_index, 'raw'),
              'v3': evaluate(d['labels'], sigmoid(cal), ancestors, root_index, 'v3')}
    out = Path(a.out);
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print('WARNING: PFresGO fmax here is test-oracle, diagnostic only; no validation-selected threshold transfer is reported.')


if __name__ == '__main__': main()
