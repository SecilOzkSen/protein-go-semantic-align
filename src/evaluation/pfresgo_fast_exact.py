"""Fast, parity-checked implementation of STARGO pfresgo_eval.py metrics.

Requires the unmodified original for a small-sample parity test before full evaluation.
Keeps original metric semantics: in-place sequential score propagation,
ancestor-set expansion for Fmax, 100 strict thresholds, original denominators,
raw Y_true for macro/micro AP and flattened AUROC.
"""
import argparse, csv, hashlib, importlib.util, json, pickle, time
from pathlib import Path
import numpy as np
import networkx as nx
import obonet
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from src.evaluation.pfresgo_comparison import read_npz, load_head, sigmoid, go_string

ROOTS={'bp':'GO:0008150','mf':'GO:0003674','cc':'GO:0005575'}

def load_original(path):
    spec=importlib.util.spec_from_file_location('original_pfresgo',str(path))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.trange=range
    return mod

def go_ancestors(graph, gos):
    mapping={g:i for i,g in enumerate(gos)}
    return [np.asarray(sorted(mapping[p] for p in nx.descendants(graph,g) if p in mapping),dtype=np.int32)
            if g in graph else np.empty(0,dtype=np.int32) for g in gos]

def propagate_original_scores(scores, anc):
    # Match original PFresGO sequential, in-place update (order matters).
    out=np.asarray(scores).copy()
    for i,parents in enumerate(anc):
        if len(parents):
            out[:,parents]=np.maximum(out[:,i,None],out[:,parents])
    return out

def close_sets(scores, anc):
    # Exact set union of all ancestors of each selected GO. For max scores,
    # closure is equivalent to threshold -> ancestor union at every threshold.
    out=np.asarray(scores).copy()
    original=np.asarray(scores)
    for i,parents in enumerate(anc):
        if len(parents):
            out[:,parents]=np.maximum(out[:,parents],original[:,i,None])
    return out

def f_curve(y, propagated_scores, full_anc, root):
    # Expand into the FULL ontology ancestor universe, not only scored GO terms.
    # Original PFresGO builds sets using all nx.descendants(graph, GO).
    n, g = y.shape
    universe = sorted(set().union(*full_anc) | set(full_anc.original_gos))
    universe = [x for x in universe if x != root]
    mapping = {x:i for i,x in enumerate(universe)}
    y_closure = np.zeros((n,len(universe)),dtype=bool)
    score_closure = np.full((n,len(universe)),-np.inf,dtype=propagated_scores.dtype)
    for i,go in enumerate(full_anc.original_gos):
        idx = [mapping[x] for x in (full_anc[i] | {go}) if x in mapping]
        if idx:
            y_closure[:,idx] |= y[:,i,None].astype(bool)
            score_closure[:,idx] = np.maximum(score_closure[:,idx], propagated_scores[:,i,None])
    true_n=y_closure.sum(axis=1)
    thresholds=np.linspace(0.,.99,num=100,dtype=np.float32)
    fs=[]; precs=[]; recs=[]
    n=len(y)
    for t in thresholds:
        pred=score_closure>float(t)
        pred_n=pred.sum(axis=1)
        valid=(pred_n>0)&(true_n>0)
        if not np.any(valid):
            fs.append(0.); precs.append(0.); recs.append(0.); continue
        tp=np.count_nonzero(pred&y_closure,axis=1)
        precision=float(np.mean(tp[valid]/pred_n[valid]))
        recall=float(np.sum(tp[valid]/true_n[valid])/n)
        f=2*precision*recall/(precision+recall) if precision+recall>0 else 0.
        fs.append(f); precs.append(precision); recs.append(recall)
    return np.asarray(fs),np.asarray(precs),np.asarray(recs),thresholds

def metrics(y, scores, anc, root_idx, full_anc, root):
    propagated=propagate_original_scores(scores,anc)
    f,p,r,grid=f_curve(y,propagated,full_anc,root)
    best=int(np.argmax(f))
    keep=np.where(y.sum(axis=0)>0)[0]
    # original uses unpropagated labels and already-propagated prediction scores
    return {'fmax_test_oracle_diagnostic':float(f[best]),
            'best_threshold_diagnostic':float(grid[best]),
            'macro_aupr':float(average_precision_score(y[:,keep],propagated[:,keep],average='macro')),
            'micro_aupr':float(average_precision_score(y[:,keep],propagated[:,keep],average='micro')),
            'flattened_auc':float(roc_auc_score(y.ravel(),propagated.ravel())),
            'n_terms_with_positives':int(len(keep))},(f,p,r,grid)

def parity(original, y, scores, proteins, gos, obo, branch, anc, root_idx, full_anc, tmp, tolerance=1e-9):
    payload={'Y_true':y,'Y_pred':scores,'goterms':np.asarray(gos),'proteins':np.asarray(proteins)}
    with tmp.open('wb') as f:pickle.dump(payload,f)
    obj=original.Method('parity',str(tmp),branch,str(obo))
    ref_f,ref_r,ref_p,ref_t=obj._protein_centric_fmax()
    fast, (fs,ps,rs,ts)=metrics(y,scores,anc,root_idx,full_anc,ROOTS[branch])
    # Original omits thresholds where no protein has a valid prediction.
    reference=np.zeros(100,dtype=float)
    grid=np.linspace(0.,.99,100,dtype=np.float32)
    for v,t in zip(ref_f,ref_t):
        inds=np.flatnonzero(grid==t)
        if len(inds)!=1:raise AssertionError('Threshold grid mismatch')
        reference[inds[0]]=v
    checks={'f_curve':np.max(np.abs(fs-reference)),
            'macro_aupr':abs(fast['macro_aupr']-obj.macro_aupr()),
            'micro_aupr':abs(fast['micro_aupr']-obj.micro_aupr()),
            'flattened_auc':abs(fast['flattened_auc']-obj.AUC())}
    for key,error in checks.items():
        if error>tolerance:raise AssertionError(f'PARITY FAILED {key}: error={error}')
    print('PARITY PASS',checks,flush=True)
    return checks

class FullAncestors(list):
    def __init__(self, graph, gos):
        super().__init__([set(nx.descendants(graph,g)) if g in graph else set() for g in gos])
        self.original_gos = list(gos)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--test',required=True);p.add_argument('--obo',required=True)
    p.add_argument('--original-eval',required=True);p.add_argument('--branch',choices=['bp','mf','cc'],default='bp')
    p.add_argument('--sequence-csv');p.add_argument('--sequence-column',type=int,default=4)
    p.add_argument('--v1-standard');p.add_argument('--v1-balanced');p.add_argument('--v2');p.add_argument('--v3')
    p.add_argument('--out-dir',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--parity-n',type=int,default=20)
    a=p.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    d=read_npz(a.test)
    gos=np.asarray([go_string(x) for x in d['go_ids']]);proteins=np.asarray([str(x) for x in d['protein_ids']])
    if len(set(gos))!=len(gos) or len(set(proteins))!=len(proteins):raise ValueError('Duplicate IDs')
    selected=np.arange(len(proteins))
    if a.sequence_csv:
        import csv
        with open(a.sequence_csv,newline='') as f:rows=list(csv.reader(f))
        chosen=[r[0] for r in rows[1:] if int(r[a.sequence_column+1])==1]
        lookup={v:i for i,v in enumerate(proteins)}
        missing=set(chosen)-set(lookup)
        if missing:raise ValueError(f'Sequence IDs missing: {list(missing)[:5]}')
        selected=np.asarray([lookup[v] for v in chosen])
    graph=obonet.read_obo(a.obo)
    anc=go_ancestors(graph,gos.tolist())
    full_anc=FullAncestors(graph,gos.tolist())
    root_idx=gos.tolist().index(ROOTS[a.branch]) if ROOTS[a.branch] in gos else None
    original=load_original(a.original_eval)
    models={'raw':d['scores']}
    for name,version,folder in [('v1_standard','v1',a.v1_standard),('v1_balanced','v1',a.v1_balanced),('v2','v2',a.v2),('v3','v3',a.v3)]:
        if folder:models[name],_=load_head(version,Path(folder),d,torch.device(a.device))
    results={'protocol':'Fast exact PFresGO metrics, small-sample original-code parity checked',
             'original_eval_sha256':hashlib.sha256(Path(a.original_eval).read_bytes()).hexdigest(),
             'test':a.test,'n_proteins':len(selected),'n_go_terms':len(gos),'models':{}}
    for name,logits in models.items():
        print('Evaluating:',name,flush=True)
        scores=sigmoid(logits[selected]);y=d['labels'][selected].astype(np.uint8)
        n=min(a.parity_n,len(y))
        if n<1:raise ValueError('No selected proteins')
        parity(original,y[:n],scores[:n],proteins[selected][:n],gos,a.obo,a.branch,anc,root_idx,full_anc,out/f'_parity_{name}.pckl')
        start=time.time()
        result,_=metrics(y,scores,anc,root_idx,full_anc,ROOTS[a.branch])
        results['models'][name]=result
        print(name,result,'elapsed_sec',round(time.time()-start,2),flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model','fmax_test_oracle_diagnostic','best_threshold_diagnostic','macro_aupr','micro_aupr','flattened_auc'])
        for name,r in results['models'].items():w.writerow([name]+[r[k] for k in ['fmax_test_oracle_diagnostic','best_threshold_diagnostic','macro_aupr','micro_aupr','flattened_auc']])
    print('Saved',out/'results.csv',flush=True)

if __name__=='__main__':main()
