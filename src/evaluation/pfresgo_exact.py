"""Run the ORIGINAL stargo/pfresgo_eval.py Method without reimplementing metrics.

Original file must be available locally, preferably a pinned Git commit.
Scores are adapted to its pickle schema; any optional baseline comparison uses
exactly the same protein and GO universe or fails closed.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src.evaluation.pfresgo_comparison import read_npz, load_head, sigmoid, go_string


def load_original(path):
    spec = importlib.util.spec_from_file_location('stargo_pfresgo_original_eval', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Disable noisy progress bars without changing any metric code.
    module.trange = range
    return module


def load_baseline(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def strict_reorder(data, reference, what):
    values = [str(v) for v in data]
    wanted = [str(v) for v in reference]
    if len(set(values)) != len(values) or len(set(wanted)) != len(wanted):
        raise ValueError(f'Duplicate {what} IDs')
    if set(values) != set(wanted):
        missing = list(set(wanted) - set(values))[:8]
        extra = list(set(values) - set(wanted))[:8]
        raise ValueError(f'{what} universe differs; cannot claim exact parity. Missing={missing}, extra={extra}')
    idx = {v:i for i,v in enumerate(values)}
    return np.asarray([idx[v] for v in wanted], dtype=int)


def read_seq_csv(path, column):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    selected = []
    for row in rows[1:]:
        if len(row) <= column + 1:
            raise ValueError('Sequence identity CSV lacks requested column')
        if int(row[column+1]) == 1:
            selected.append(row[0])
    return selected


def original_metrics(original, labels, scores, proteins, gos, obo, branch, output_pickle):
    payload = {'Y_true': labels.copy(), 'Y_pred': scores.copy(),
               'goterms': np.asarray(gos), 'proteins': np.asarray(proteins)}
    with output_pickle.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    obj = original.Method('our_model', str(output_pickle), branch, str(obo))
    all_idx = np.arange(len(proteins))
    # All metric calculations below call original PFresGO functions directly.
    f = float(obj.fmax(all_idx))
    macro = float(obj.macro_aupr(all_idx))
    micro = float(obj.micro_aupr(all_idx))
    roc = float(obj.AUC(all_idx))
    return {'fmax_test_oracle_diagnostic':f, 'macro_aupr':macro,
            'micro_aupr':micro, 'flattened_auc':roc}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test',required=True)
    p.add_argument('--obo',required=True)
    p.add_argument('--original-eval',required=True,help='Local copy of unmodified stargo/pfresgo_eval.py')
    p.add_argument('--branch',choices=['bp','mf','cc'],default='bp')
    p.add_argument('--baseline-pickle',help='Original PFresGO prediction pickle for strict protein/GO parity')
    p.add_argument('--sequence-csv',help='PFresGO sequence identity CSV')
    p.add_argument('--sequence-column',type=int,default=4,help='Zero-based identity indicator index, default 4 as in PFresGO main')
    p.add_argument('--v1-standard'); p.add_argument('--v1-balanced'); p.add_argument('--v2'); p.add_argument('--v3')
    p.add_argument('--out-dir',required=True); p.add_argument('--device',default='cpu')
    a=p.parse_args()
    d=read_npz(a.test)
    proteins=np.asarray([str(v) for v in d['protein_ids']])
    gos=np.asarray([go_string(v) for v in d['go_ids']])
    row=np.arange(len(proteins)); col=np.arange(len(gos))
    if a.baseline_pickle:
        baseline=load_baseline(a.baseline_pickle)
        row=strict_reorder(proteins, baseline['proteins'], 'protein')
        col=strict_reorder(gos, [go_string(v) for v in baseline['goterms']], 'GO')
    if a.sequence_csv:
        selected=read_seq_csv(a.sequence_csv, a.sequence_column)
        current=proteins[row]
        # Exact intersection only; refuse missing IDs, as it invalidates paper parity.
        mapping={x:i for i,x in enumerate(current)}
        missing=[x for x in selected if x not in mapping]
        if missing:
            raise ValueError(f'{len(missing)} sequence-selected proteins missing from test data; first={missing[:5]}')
        row=row[np.asarray([mapping[x] for x in selected],dtype=int)]
    if not a.baseline_pickle:
        print('WARNING: no original baseline pickle supplied; metrics use original code, but published baseline universe parity is NOT established.')
    if not a.sequence_csv:
        print('WARNING: no sequence identity CSV supplied; using all selected test proteins, not PFresGO main() sequence subset.')
    original_path=Path(a.original_eval)
    original=load_original(original_path)
    output=Path(a.out_dir); output.mkdir(parents=True,exist_ok=True)
    models={'raw':d['scores']}
    for name, version, folder in [('v1_standard','v1',a.v1_standard),('v1_balanced','v1',a.v1_balanced),('v2','v2',a.v2),('v3','v3',a.v3)]:
        if folder:
            models[name],_=load_head(version,Path(folder),d,torch.device(a.device))
    results={'protocol':'UNMODIFIED stargo pfresgo_eval.py Method',
             'original_eval_sha256':hashlib.sha256(original_path.read_bytes()).hexdigest(),
             'branch':a.branch,'baseline_pickle':a.baseline_pickle,'sequence_csv':a.sequence_csv,
             'n_proteins':len(row),'n_go_terms':len(col),'models':{}}
    for name, logits in models.items():
        # Input conversion only, not a modification to PFresGO metric code.
        # PFresGO expects [0,1] prediction scores. We use sigmoid on logits.
        scores=sigmoid(logits)[np.ix_(row,col)]
        labels=d['labels'][np.ix_(row,col)]
        path=output/f'{name}_pfresgo_input.pckl'
        print('Evaluating original PFresGO:',name,flush=True)
        results['models'][name]=original_metrics(original,labels,scores,proteins[row],gos[col],a.obo,a.branch,path)
    (output/'results.json').write_text(json.dumps(results,indent=2))
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['model','fmax_test_oracle_diagnostic','macro_aupr','micro_aupr','flattened_auc'])
        for name,m in results['models'].items():
            writer.writerow([name,m['fmax_test_oracle_diagnostic'],m['macro_aupr'],m['micro_aupr'],m['flattened_auc']])
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
