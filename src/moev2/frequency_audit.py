"""Experiment E: train-GO-frequency-stratified validation/test diagnostic. No training."""
import argparse, csv, json, logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from src.moev2.data import load_dump, DumpDataset, graph_matrices
from src.moev2.model import Config, SemanticOntologyModel


def main():
    ap = argparse.ArgumentParser()
    for name in ('train_dump', 'val_dump', 'test_dump', 'checkpoint', 'test_predictions_dir', 'go_graph_path', 'out_dir'):
        ap.add_argument('--' + name, required=True, type=Path)
    ap.add_argument('--device', default='cuda:0');
    ap.add_argument('--batch_size', type=int, default=64)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    tr, va, te = [load_dump(p) for p in (a.train_dump, a.val_dump, a.test_dump)]
    for d in (va, te):
        if not np.array_equal(tr['go'], d['go']): raise RuntimeError('GO ID order mismatch')
    for x, y in ((tr, va), (tr, te), (va, te)):
        if set(x['ids']) & set(y['ids']): raise RuntimeError('Protein ID overlap')
    train_y = np.asarray(tr['labels']) if 'labels' in tr else np.load(a.train_dump / 'labels.int8.npy')
    if train_y.shape != (len(tr['ids']), len(tr['go'])): raise RuntimeError('Train labels shape mismatch')
    freq = (train_y > 0).sum(axis=0)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if not np.array_equal(np.asarray(ck['meta']['go_ids']), tr['go']): raise RuntimeError('Checkpoint GO mismatch')
    par, ch = graph_matrices(tr['go'], a.go_graph_path)
    model = SemanticOntologyModel((par, ch), Config(dim=tr['z'].shape[1]))
    model.load_state_dict(ck['model'], strict=True)
    device = torch.device(a.device);
    model = model.to(device).eval();
    gz = torch.from_numpy(tr['z']).to(device)
    dl = DataLoader(DumpDataset(va), batch_size=a.batch_size, shuffle=False, num_workers=0)
    vy = [];
    vp = {'final': [], 'semantic': []}
    with torch.inference_mode():
        for p, s, y in dl:
            out = model(p.to(device), gz, s.to(device), True)
            vy.append(y.numpy())
            for name in vp: vp[name].append(torch.sigmoid(out[name]).cpu().numpy())
    vy = np.concatenate(vy);
    vp = {k: np.concatenate(v) for k, v in vp.items()}
    td = a.test_predictions_dir
    ty = np.load(td / 'labels.npy')
    test_ids = json.loads((td / 'protein_ids.json').read_text())
    if test_ids != te['ids']: raise RuntimeError('Test prediction protein order mismatch')
    if not np.array_equal(np.load(td / 'go_ids.npy'), tr['go']): raise RuntimeError('Test prediction GO order mismatch')
    tp = {k: np.load(td / f'{k}_probabilities.npy') for k in vp}
    if vy.shape != (len(va['ids']), len(tr['go'])) or ty.shape != (len(te['ids']), len(tr['go'])): raise RuntimeError('Label shape mismatch')
    for name in vp:
        if vp[name].shape != vy.shape or tp[name].shape != ty.shape: raise RuntimeError('Prediction shape mismatch')
        if not np.isfinite(vp[name]).all() or not np.isfinite(tp[name]).all(): raise RuntimeError('Nonfinite predictions')
    # Fixed, interpretable training-count buckets. No validation/test-dependent bucket selection.
    groups = {'unseen_train': freq == 0, 'rare_1_10': (freq >= 1) & (freq <= 10),
              'medium_11_100': (freq >= 11) & (freq <= 100), 'frequent_101_plus': freq >= 101}
    rows = []
    for split, y, pr in (('validation', vy, vp), ('test', ty, tp)):
        for name, pred in pr.items():
            for bucket, mask in groups.items():
                cols = np.where(mask)[0]
                aps = [];
                aucs = [];
                pos_terms = 0
                for j in cols:
                    yy = y[:, j]
                    if yy.sum() == 0: continue
                    pos_terms += 1
                    aps.append(float(average_precision_score(yy, pred[:, j])))
                    if yy.sum() < len(yy): aucs.append(float(roc_auc_score(yy, pred[:, j])))
                row = {'split': split, 'model': name, 'frequency_bucket': bucket, 'n_go': len(cols),
                       'n_go_with_positives': pos_terms, 'n_positive_pairs': int(y[:, cols].sum()),
                       'macro_aupr_observed_terms': float(np.mean(aps)) if aps else None,
                       'macro_auc_observed_nonconstant_terms': float(np.mean(aucs)) if aucs else None,
                       'micro_aupr': float(average_precision_score(y[:, cols].ravel(), pred[:, cols].ravel())) if len(cols) and y[:, cols].sum() > 0 else None}
                rows.append(row)
                logging.info('%s %-8s %-18s GO=%4d observed=%4d macroAP=%s microAP=%s', split, name, bucket, len(cols), pos_terms,
                             f'{row["macro_aupr_observed_terms"]:.4f}' if aps else 'NA',
                             f'{row["micro_aupr"]:.4f}' if row['micro_aupr'] is not None else 'NA')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / 'frequency_audit.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));
        w.writeheader();
        w.writerows(rows)
    (a.out_dir / 'frequency_audit.json').write_text(json.dumps({'buckets': 'GO training-positive count: 0, 1-10, 11-100, >=101',
                                                                'metric_note': 'Per-term AP excludes GO terms with no positives in that split. Micro AP pools pairs within each frequency bucket. These are diagnostics, not StarGO Fmax.',
                                                                'results': rows}, indent=2))
    logging.info('DONE %s', a.out_dir)


if __name__ == '__main__': main()
