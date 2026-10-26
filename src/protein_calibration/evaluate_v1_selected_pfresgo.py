"""Evaluate the LOCKED validation-selected V1 checkpoint on PFresGO exact test.
Run: python -m src.protein_calibration.evaluate_selected_pfresgo ...
Never reselects epoch, lambda, or threshold on test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import obonet
from .io import load_matrix, verify_compatible, hash_go_ids
from .model import CardinalityGuidedCalibrator
from .fit import predict, annotated_labels
from src.evaluation.pfresgo_comparison import sigmoid, go_string
from src.evaluation.pfresgo_fast_exact import FullAncestors, go_ancestors, metrics, ROOTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test', required=True)
    p.add_argument('--head-dir', required=True)
    p.add_argument('--obo', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--branch', choices=['bp','mf','cc'], default='bp')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    data = load_matrix(a.test)
    folder = Path(a.head_dir)
    report = json.loads((folder/'validation_results.json').read_text())
    raw_val = report['raw_validation']
    selected = report['best_validation']
    lam = float(selected['lambda_blend'])
    cal_threshold = float(selected['threshold'])
    raw_threshold = float(raw_val['threshold'])
    grid = np.linspace(0,.99,100,dtype=np.float32)
    def threshold_index(th):
        i = int(np.argmin(abs(grid-th)))
        if abs(float(grid[i])-th)>1e-6: raise ValueError(f'Not on PFresGO grid: {th}')
        return i
    raw_ix, cal_ix = threshold_index(raw_threshold), threshold_index(cal_threshold)
    raw_prob = sigmoid(data['scores'])
    if report['raw_fallback']:
        if lam != 0: raise ValueError('Fallback must have lambda=0')
        final_prob = raw_prob
    else:
        ckpt = torch.load(folder/'head.pt',map_location='cpu',weights_only=False)
        if hash_go_ids(data['go_ids']) != ckpt['go_hash']:
            raise ValueError('Test GO ordering mismatch')
        if int(ckpt['epoch']) != int(report['best_epoch']):
            raise ValueError('Checkpoint epoch does not match validation selection')
        if abs(float(ckpt['selected_lambda'])-lam)>1e-8:
            raise ValueError('Checkpoint lambda mismatch')
        model = CardinalityGuidedCalibrator().to(a.device)
        model.load_state_dict(ckpt['model'],strict=True)
        cal_logits = predict(model,data['scores'].astype('float32'),torch.device(a.device))['scores']
        final_prob = (1-lam)*raw_prob + lam*sigmoid(cal_logits)
    if not np.isfinite(final_prob).all(): raise ValueError('Non-finite predictions')
    gos = [go_string(g) for g in data['go_ids']]
    graph = obonet.read_obo(a.obo)
    anc = go_ancestors(graph,gos)
    full = FullAncestors(graph,gos)
    labels = annotated_labels(data,a.annotations,a.branch)
    root = ROOTS[a.branch]
    raw_m,(raw_curve,*_) = metrics(labels,raw_prob,anc,None,full,root)
    final_m,(final_curve,*_) = metrics(labels,final_prob,anc,None,full,root)
    result = {
        'protocol':'PFresGO exact, annotation-aligned, test locked to validation-selected checkpoint/lambda/threshold',
        'selected_epoch':int(report['best_epoch']), 'selected_lambda':lam,
        'raw_validation_threshold':raw_threshold, 'calibrated_validation_threshold':cal_threshold,
        'raw_test_F_at_validation_threshold':float(raw_curve[raw_ix]),
        'calibrated_test_F_at_validation_threshold':float(final_curve[cal_ix]),
        'delta_test_F_at_validation_threshold':float(final_curve[cal_ix]-raw_curve[raw_ix]),
        'raw_test_macro_aupr':float(raw_m['macro_aupr']),
        'calibrated_test_macro_aupr':float(final_m['macro_aupr']),
        'raw_test_micro_aupr':float(raw_m['micro_aupr']),
        'calibrated_test_micro_aupr':float(final_m['micro_aupr']),
        'raw_test_auroc':float(raw_m['flattened_auc']),
        'calibrated_test_auroc':float(final_m['flattened_auc']),
        'raw_test_oracle_Fmax_diagnostic_only':float(max(raw_curve)),
        'calibrated_test_oracle_Fmax_diagnostic_only':float(max(final_curve)),
        'n_test_proteins':len(labels),'n_go_terms':len(gos)
    }
    dest = Path(a.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    print('Saved:',dest)

if __name__=='__main__':main()
