"""No-training homology-stratified expert ablation on validation predictions."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from src.metrics.stargo_pfresgo_metrics import StarGOPFresGOEvaluator

BUCKETS = [('0-30%', 0, .30), ('30-50%', .30, .50), ('50-70%', .50, .70), ('70-90%', .70, .90), ('90-99%', .90, .99), ('99-100%', .99, 1.000001)]
MODES = ('direct_5nn', 'retriever_expert', 'neighbour_expert', 'moe')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation_dir', type=Path, required=True)
    ap.add_argument('--homology_dir', type=Path, required=True)
    ap.add_argument('--go_graph_path', type=Path, required=True)
    ap.add_argument('--out_dir', type=Path, required=True)
    ap.add_argument('--ontology', default='bp')
    a = ap.parse_args()
    d = a.ablation_dir
    ids = list(map(str, json.loads((d / 'protein_ids.json').read_text())))
    if len(ids) != len(set(ids)): raise RuntimeError('Duplicate prediction IDs')
    labels = np.load(d / 'labels.int8.npy', mmap_mode='r')
    go_ids = np.load(d / 'go_ids.int64.npy')
    if labels.shape != (len(ids), len(go_ids)): raise RuntimeError('Label shape mismatch')
    probs = {m: np.load(d / f'{m}_probabilities.float32.npy', mmap_mode='r') for m in MODES}
    for m, x in probs.items():
        if x.shape != labels.shape or not np.isfinite(x).all(): raise RuntimeError(f'Invalid {m} prediction matrix')
    identity = {pid: [] for pid in ids}
    with (a.homology_dir / 'pairs.csv').open() as f:
        for r in csv.DictReader(f):
            if r['query_id'] in identity:
                identity[r['query_id']].append((int(r['rank']), float(r['global_identity'])))
    bad = [pid for pid, values in identity.items() if len(values) != 5 or sorted(x[0] for x in values) != [1, 2, 3, 4, 5]]
    if bad: raise RuntimeError(f'Missing or duplicate homology pairs: {bad[:10]}')
    max5 = np.array([max(v for _, v in identity[pid]) for pid in ids])
    evaluator = StarGOPFresGOEvaluator(goterms=go_ids, ontology=a.ontology, go_graph_path=a.go_graph_path)
    rows = []
    for label, lo, hi in [('overall', 0, 1.000001)] + BUCKETS:
        mask = (max5 >= lo) & (max5 < hi)
        for m in MODES:
            row = {'identity_bucket': label, 'model': m, 'n': int(mask.sum())}
            if mask.any():
                metrics = evaluator.evaluate(np.asarray(labels[mask]), np.asarray(probs[m][mask]))
                row.update({k: float(v) for k, v in metrics.items()})
            rows.append(row)
            if mask.any(): print(f'{label:>9} {m:>19} n={mask.sum():4d} Fmax={row["protein_fmax"]:.6f}')
    overall = {r['model']: r for r in rows if r['identity_bucket'] == 'overall'}
    reference = json.loads((d / 'expert_ablation.json').read_text())['metrics']
    for m in MODES:
        delta = abs(overall[m]['protein_fmax'] - float(reference[m]['protein_fmax']))
        if delta > 1e-5: raise RuntimeError(f'Overall parity FAILED for {m}: delta={delta}')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / 'identity_bucket_ablation.json').write_text(json.dumps(
        {'identity_definition': 'max global alignment identity among ESM top-5 neighbours', 'n_proteins': len(ids), 'n_go': len(go_ids), 'results': rows},
        indent=2))
    columns = ['identity_bucket', 'model', 'n', 'protein_fmax', 'protein_fmax_threshold', 'protein_precision_at_fmax', 'protein_recall_at_fmax', 'macro_aupr',
               'micro_aupr', 'auc']
    with (a.out_dir / 'identity_bucket_ablation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore');
        writer.writeheader();
        writer.writerows(rows)
    print('PASSED: overall parity and identity-bucket ablation; outputs:', a.out_dir)


if __name__ == '__main__': main()
