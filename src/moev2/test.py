"""Experiment E one-shot test. No optimizer; validation parity before test."""
import argparse, json, logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.moev2.data import load_dump, DumpDataset, graph_matrices
from src.moev2.model import Config, SemanticOntologyModel
from src.metrics.stargo_pfresgo_metrics import StarGOPFresGOEvaluator


def native(v):
    if isinstance(v, dict): return {k: native(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [native(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    return v


def fixed_protein_f1(y, scores, threshold):
    """Explicit fixed-threshold protein-centric P/R/F1 on dump label universe.
    This does not implement additional ancestor propagation; report separately
    from the project's StarGO evaluator sweep.
    """
    y = y.astype(bool);
    pred = scores >= threshold
    tp = (pred & y).sum(axis=1);
    n_pred = pred.sum(axis=1);
    n_true = y.sum(axis=1)
    annotated = n_true > 0
    with_predictions = annotated & (n_pred > 0)
    precision = float(np.mean(tp[with_predictions] / n_pred[with_predictions])) if with_predictions.any() else 0.0
    recall = float(np.mean(tp[annotated] / n_true[annotated])) if annotated.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(threshold=float(threshold), protein_f1=f1, precision=precision,
                recall=recall, n_annotated=int(annotated.sum()), n_predicted=int(with_predictions.sum()))


def main():
    ap = argparse.ArgumentParser()
    for k in ('train_dump', 'val_dump', 'test_dump', 'checkpoint', 'go_graph_path', 'out_dir'):
        ap.add_argument('--' + k, type=Path, required=True)
    ap.add_argument('--device', default='cuda:0');
    ap.add_argument('--ontology', default='bp')
    ap.add_argument('--batch_size', type=int, default=64);
    ap.add_argument('--parity_tol', type=float, default=.001)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    tr, va, te = [load_dump(p) for p in (a.train_dump, a.val_dump, a.test_dump)]
    for name, d in [('validation', va), ('test', te)]:
        if not np.array_equal(tr['go'], d['go']): raise RuntimeError(f'{name} GO ID order mismatch')
        if not np.allclose(tr['z'], d['z'], atol=.003): raise RuntimeError(f'{name} GO embeddings mismatch')
        if len(set(d['ids'])) != len(d['ids']): raise RuntimeError(f'{name} duplicate IDs')
    for x, y, name in [(tr, va, 'train/val'), (tr, te, 'train/test'), (va, te, 'val/test')]:
        if set(x['ids']) & set(y['ids']): raise RuntimeError(f'{name} overlap')
    if len(te['ids']) != 3416: raise RuntimeError(f'Expected 3416 test proteins, got {len(te["ids"])}')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    meta = ck['meta']
    if not np.array_equal(np.asarray(meta['go_ids'], dtype=np.int64), tr['go']):
        raise RuntimeError('Checkpoint GO IDs mismatch')
    parents, children = graph_matrices(tr['go'], a.go_graph_path)
    model = SemanticOntologyModel((parents, children), Config(dim=tr['z'].shape[1]))
    model.load_state_dict(ck['model'], strict=True)
    device = torch.device(a.device);
    model = model.to(device).eval()
    gz = torch.from_numpy(tr['z']).to(device)
    evaluator = StarGOPFresGOEvaluator(goterms=tr['go'], ontology=a.ontology, go_graph_path=a.go_graph_path)

    def infer(d):
        dl = DataLoader(DumpDataset(d), batch_size=a.batch_size, shuffle=False, num_workers=0)
        pred = {'final': [], 'semantic': []};
        labels = []
        with torch.inference_mode():
            for p, s, y in dl:
                out = model(p.to(device), gz, s.to(device), True)
                for name in pred:
                    z = torch.sigmoid(out[name]).cpu().numpy().astype(np.float32)
                    if not np.isfinite(z).all(): raise RuntimeError(f'Nonfinite {name}')
                    pred[name].append(z)
                labels.append(y.numpy())
        return np.concatenate(labels), {k: np.concatenate(v) for k, v in pred.items()}

    logging.info('Running VALIDATION checkpoint parity before opening test')
    vy, vp = infer(va)
    report = {'checkpoint': str(a.checkpoint), 'epoch': ck['epoch'], 'validation': {}, 'test': {},
              'fixed_f1_definition': 'Protein-centric precision over proteins with predictions, recall over annotated proteins, threshold on saved probabilities; no extra DAG propagation.'}
    for name in ('final', 'semantic'):
        expected = meta[f'{name}_metrics'];
        actual = native(evaluator.evaluate(vy, vp[name]))
        if abs(actual['protein_fmax'] - expected['protein_fmax']) > a.parity_tol:
            raise RuntimeError(f'{name} checkpoint parity failed: {actual["protein_fmax"]} vs {expected["protein_fmax"]}')
        if 'protein_fmax_threshold' not in expected:
            raise RuntimeError(f'{name} validation checkpoint lacks selected threshold')
        threshold = float(expected['protein_fmax_threshold'])
        report['validation'][name] = {'metrics': actual, 'selected_threshold': threshold}
        logging.info('VAL parity PASS %s Fmax=%.6f threshold=%.3f', name, actual['protein_fmax'], threshold)
    logging.info('One-shot TEST evaluation, no optimizer steps')
    ty, tp = infer(te)
    for name in ('final', 'semantic'):
        sweep = native(evaluator.evaluate(ty, tp[name]))
        threshold = report['validation'][name]['selected_threshold']
        fixed = fixed_protein_f1(ty, tp[name], threshold)
        report['test'][name] = {'stargo_threshold_sweep': sweep, 'validation_fixed_threshold': fixed}
        logging.info('TEST %s sweep_Fmax=%.6f @ %.3f fixed_F1=%.6f @ %.3f',
                     name, sweep['protein_fmax'], sweep['protein_fmax_threshold'], fixed['protein_f1'], threshold)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / 'test_metrics.json').write_text(json.dumps(native(report), indent=2))
    (a.out_dir / 'protein_ids.json').write_text(json.dumps(te['ids']))
    np.save(a.out_dir / 'go_ids.npy', tr['go']);
    np.save(a.out_dir / 'labels.npy', ty)
    for name, prob in tp.items(): np.save(a.out_dir / f'{name}_probabilities.npy', prob)
    logging.info('DONE %s', a.out_dir)


if __name__ == '__main__': main()
