"""Compare raw Retriever-v2 scores and Experiment E heads across valid/test.
No training, threshold tuning, or checkpoint selection. Requires existing src.moev2.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from src.moev2.data import load_dump, DumpDataset, graph_matrices
from src.moev2.model import Config, SemanticOntologyModel
from src.metrics.stargo_pfresgo_metrics import StarGOPFresGOEvaluator


def metrics(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    if y.shape != s.shape or not np.isfinite(s).all():
        raise RuntimeError('Shape or finite-value check failed')
    observed = np.flatnonzero(y.sum(axis=0) > 0)
    two_class = np.flatnonzero((y.sum(axis=0) > 0) & (y.sum(axis=0) < len(y)))
    macro_ap = float(np.mean([average_precision_score(y[:, j], s[:, j]) for j in observed])) if len(observed) else None
    macro_auc = float(np.mean([roc_auc_score(y[:, j], s[:, j]) for j in two_class])) if len(two_class) else None
    return {'macro_aupr_observed': macro_ap,
            'micro_aupr': float(average_precision_score(y.ravel(), s.ravel())),
            'micro_auroc': float(roc_auc_score(y.ravel(), s.ravel())) if y.min() != y.max() else None,
            'macro_auroc_two_class': macro_auc,
            'n_observed_go': int(len(observed)), 'n_two_class_go': int(len(two_class)),
            'n_proteins': int(len(y)), 'n_positive_pairs': int(y.sum())}


def main():
    ap = argparse.ArgumentParser()
    for k in ('train_dump', 'val_dump', 'test_dump', 'checkpoint', 'go_graph_path', 'out_dir'):
        ap.add_argument('--' + k, type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--ontology', default='bp')
    ap.add_argument('--parity_tol', type=float, default=0.001)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    tr, va, te = [load_dump(x) for x in (a.train_dump, a.val_dump, a.test_dump)]
    for name, d in (('valid', va), ('test', te)):
        if not np.array_equal(tr['go'], d['go']) or not np.allclose(tr['z'], d['z'], atol=.003):
            raise RuntimeError(name + ': GO ID/embedding mismatch')
        if len(d['ids']) != len(set(d['ids'])):
            raise RuntimeError(name + ': duplicate protein IDs')
    if set(tr['ids']) & set(va['ids']) or set(tr['ids']) & set(te['ids']) or set(va['ids']) & set(te['ids']):
        raise RuntimeError('Split overlap')
    if len(va['ids']) != 2625 or len(te['ids']) != 3416:
        raise RuntimeError(f'Unexpected sizes valid={len(va["ids"])} test={len(te["ids"])}')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if not np.array_equal(np.asarray(ck['meta']['go_ids']), tr['go']):
        raise RuntimeError('Checkpoint GO IDs differ')
    parents, children = graph_matrices(tr['go'], a.go_graph_path)
    model = SemanticOntologyModel((parents, children), Config(dim=tr['z'].shape[1]))
    model.load_state_dict(ck['model'], strict=True)
    device = torch.device(a.device)
    model = model.to(device).eval()
    gz = torch.from_numpy(tr['z']).to(device)
    evaluator = StarGOPFresGOEvaluator(goterms=tr['go'], ontology=a.ontology, go_graph_path=a.go_graph_path)
    results = {}
    for name, d in (('validation', va), ('test', te)):
        loader = DataLoader(DumpDataset(d), batch_size=a.batch_size, shuffle=False, num_workers=0)
        yy, rr, ss, ff = [], [], [], []
        with torch.inference_mode():
            for protein, score, y in loader:
                output = model(protein.to(device), gz, score.to(device), True)
                yy.append(y.numpy())
                rr.append(score.numpy())
                ss.append(torch.sigmoid(output['semantic']).cpu().numpy())
                ff.append(torch.sigmoid(output['final']).cpu().numpy())
        y = np.concatenate(yy)
        scores = {'retriever_raw': np.concatenate(rr), 'semantic': np.concatenate(ss), 'final': np.concatenate(ff)}
        results[name] = {k: metrics(y, v) for k, v in scores.items()}
        # Parity against the checkpoint's recorded validation Fmax, before accepting test comparisons.
        if name == 'validation':
            for key, stored_key in (('semantic', 'semantic_metrics'), ('final', 'final_metrics')):
                got = evaluator.evaluate(y, scores[key])['protein_fmax']
                expected = ck['meta'][stored_key]['protein_fmax']
                if abs(float(got) - float(expected)) > a.parity_tol:
                    raise RuntimeError(f'{key} validation parity FAILED: {got} vs {expected}')
                logging.info('VAL parity %s Fmax=%.6f PASS', key, got)
        for key, m in results[name].items():
            logging.info('%s %-14s macroAP=%.5f microAP=%.5f microAUC=%.5f observedGO=%d',
                         name, key, m['macro_aupr_observed'], m['micro_aupr'], m['micro_auroc'], m['n_observed_go'])
    comparison = {}
    for key in ('retriever_raw', 'semantic', 'final'):
        v, t = results['validation'][key], results['test'][key]
        comparison[key] = {
            'macro_aupr_drop': v['macro_aupr_observed'] - t['macro_aupr_observed'],
            'micro_aupr_drop': v['micro_aupr'] - t['micro_aupr'],
            'micro_auroc_drop': v['micro_auroc'] - t['micro_auroc'],
        }
        logging.info('DROP %-14s macroAP=%.5f microAP=%.5f microAUC=%.5f', key, *comparison[key].values())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / 'pipeline_transfer_audit.json').write_text(json.dumps({
        'checkpoint': str(a.checkpoint),
        'note': 'Ranking metrics only; no threshold optimization. Macro AP uses GO terms observed in each split, so differing observed GO sets can affect comparison.',
        'results': results, 'validation_minus_test': comparison}, indent=2))
    logging.info('DONE %s', a.out_dir / 'pipeline_transfer_audit.json')


if __name__ == '__main__':
    main()
