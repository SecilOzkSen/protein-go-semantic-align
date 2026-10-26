"""Locked test evaluation. Never selects alpha, checkpoint or threshold on test."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import obonet
from .io import load_matrix,hash_go_ids
from .model import ResidualV3
from .fit import predict,original_annotations
from src.evaluation.pfresgo_fast_exact import FullAncestors,go_ancestors,metrics,parity,load_original,ROOTS
from src.evaluation.pfresgo_comparison import go_string

def main():
    p=argparse.ArgumentParser()
    for k in ('test','head-dir','obo','annotations','original-eval','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--branch',choices=['bp','mf','cc'],default='bp')
    p.add_argument('--device',default='cpu')
    a=p.parse_args();d=load_matrix(a.test);head=Path(a.head_dir)
    sel=json.loads((head/'selection.json').read_text())
    ckpt=None
    if not sel['raw_fallback']:
        ckpt=torch.load(head/'head.pt',map_location='cpu',weights_only=False)
        if hash_go_ids(d['go_ids'])!=ckpt['go_hash']:raise ValueError('GO ordering mismatch')
        model=ResidualV3(ckpt['protein_dim']).to(a.device)
        model.load_state_dict(ckpt['model'],strict=True)
        prob=predict(model,d,torch.device(a.device))
        threshold=ckpt['valid_threshold']
    else:
        prob=1/(1+np.exp(-np.clip(d['scores'],-80,80)))
        threshold=sel['raw_valid_threshold']
    gos=[go_string(x) for x in d['go_ids']]
    graph=obonet.read_obo(a.obo);anc=go_ancestors(graph,gos);full=FullAncestors(graph,gos)
    y=original_annotations(d,a.annotations,a.branch)
    original=load_original(a.original_eval)
    parity(original,y[:min(20,len(y))],prob[:min(20,len(y))],d['protein_ids'][:min(20,len(y))],gos,a.obo,a.branch,anc,None,full,head/'parity_test.pckl')
    m,(curve,_,_,grid)=metrics(y,prob,anc,None,full,ROOTS[a.branch])
    ix=int(np.argmin(abs(np.asarray(grid)-threshold)))
    if abs(float(grid[ix])-threshold)>1e-6:raise ValueError('Validation threshold not on grid')
    result={'test_F_at_validation_threshold':float(curve[ix]),'validation_threshold':threshold,
            'test_Fmax_oracle_diagnostic_only':float(max(curve)),
            'test_macro_aupr':float(m['macro_aupr']),'test_micro_aupr':float(m['micro_aupr']),
            'test_auroc':float(m['flattened_auc']),'n_test_proteins':len(y)}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
