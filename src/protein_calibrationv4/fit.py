"""Train residual V3 and select checkpoint by exact PFresGO validation Fmax.
Run as python -m src.residual_v3_pfresgo.fit from repository root.
Requires existing src.evaluation.pfresgo_fast_exact, original PFresGO evaluator, and annot.tsv.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from .model import ResidualV3,pfresgo_soft_f
from .io import load_matrix,verify_compatible,hash_go_ids
from src.evaluation.pfresgo_fast_exact import FullAncestors,go_ancestors,metrics,parity,load_original,ROOTS
from src.evaluation.pfresgo_comparison import go_string
import obonet


def original_annotations(data,path,branch):
    cols={'mf':1,'bp':2,'cc':3}; ann={}; active=False
    for line in Path(path).read_text(encoding='utf8').splitlines():
        f=line.split('\t')
        if f[0].startswith('### PDB-chain'): active=True;continue
        if active and len(f)>=4 and f[0].strip():
            if f[0].strip() in ann: raise ValueError('Duplicate protein in annotations')
            ann[f[0].strip()]=set(x.strip() for x in f[cols[branch]].split(',') if x.strip())
    if not ann: raise ValueError('Annotation file could not be parsed')
    ids={go_string(x):j for j,x in enumerate(data['go_ids'])}
    y=np.zeros_like(data['labels'],dtype=np.uint8)
    for i,p in enumerate(map(str,data['protein_ids'])):
        if p not in ann: raise ValueError(f'Missing annotation: {p}')
        for g in ann[p]:
            if g in ids:y[i,ids[g]]=1
    return y

@torch.no_grad()
def predict(model,data,device,batch=128):
    model.eval();result=[]
    for start in range(0,len(data['scores']),batch):
        s=torch.as_tensor(data['scores'][start:start+batch],device=device,dtype=torch.float32)
        p=torch.as_tensor(data['protein_embeddings'][start:start+batch],device=device,dtype=torch.float32)
        result.append(model(s,p)['prob'].cpu().numpy())
    return np.concatenate(result)


@torch.no_grad()
def calibration_diagnostics(model, data, device, batch=128):
    """Full-validation diagnostics, no labels and no gradient tracking."""
    model.eval()
    accum = {'boundary': [], 'scale': [], 'residual_abs': []}
    for start in range(0, len(data['scores']), batch):
        scores = torch.as_tensor(data['scores'][start:start+batch], device=device, dtype=torch.float32)
        emb = torch.as_tensor(data['protein_embeddings'][start:start+batch], device=device, dtype=torch.float32)
        result = model(scores, emb)
        raw = torch.sigmoid(scores)
        accum['boundary'].append(result['boundary'].flatten().cpu().numpy())
        accum['scale'].append(result['scale'].flatten().cpu().numpy())
        accum['residual_abs'].append((result['prob'] - raw).abs().flatten().cpu().numpy())
    b, sc, r = (np.concatenate(accum[k]) for k in ('boundary', 'scale', 'residual_abs'))
    return {
        'boundary_mean': float(b.mean()), 'boundary_std': float(b.std()),
        'boundary_min': float(b.min()), 'boundary_max': float(b.max()),
        'scale_mean': float(sc.mean()), 'scale_std': float(sc.std()),
        'scale_min': float(sc.min()), 'scale_max': float(sc.max()),
        'residual_mean_abs': float(r.mean()), 'residual_max_abs': float(r.max()),
    }


def grad_norm(params):
    values = [p.grad.detach().float().square().sum() for p in params if p.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.0


def main():
    ap=argparse.ArgumentParser()
    for key in ('calibration','valid','obo','annotations','original-eval','out-dir'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--branch',choices=['bp','mf','cc'],default='bp')
    ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--temperature',type=float,default=.08)
    ap.add_argument('--sweep-temperature',type=float,default=.05)
    ap.add_argument('--patience',type=int,default=12)
    ap.add_argument('--min-retention',type=float,default=.98)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--parity-n',type=int,default=20)
    args=ap.parse_args();torch.manual_seed(args.seed);np.random.seed(args.seed)
    train=load_matrix(args.calibration);valid=load_matrix(args.valid)
    verify_compatible(train,valid)
    device=torch.device(args.device)
    graph=obonet.read_obo(args.obo)
    gos=[go_string(x) for x in train['go_ids']]
    if len(gos)!=len(set(gos)):raise ValueError('Duplicate GO IDs')
    anc=go_ancestors(graph,gos);full=FullAncestors(graph,gos)
    root=ROOTS[args.branch]
    root_idx=gos.index(root) if root in gos else None
    edges=[torch.as_tensor(a,dtype=torch.long,device=device) for a in anc]
    original=load_original(args.original_eval)
    vy=original_annotations(valid,args.annotations,args.branch)
    ty=train['labels'].astype(np.uint8)  # calibration training labels, never validation/test labels
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    raw=1/(1+np.exp(-np.clip(valid['scores'],-80,80)))
    # Verify exact evaluator implementation before selection.
    n=min(args.parity_n,len(vy))
    parity(original,vy[:n],raw[:n],valid['protein_ids'][:n],gos,args.obo,args.branch,anc,None,full,out/'parity_raw.pckl')
    raw_m,(raw_curve,*_)=metrics(vy,raw,anc,None,full,root)
    raw_ix=int(np.argmax(raw_curve));raw_f=float(raw_curve[raw_ix]);raw_thr=float(np.linspace(0,.99,100,dtype=np.float32)[raw_ix])
    model=ResidualV3(train['protein_embeddings'].shape[1]).to(device)
    model.set_protein_normalization(train['protein_embeddings'])
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    ds=TensorDataset(torch.tensor(train['scores'],dtype=torch.float32),torch.tensor(train['protein_embeddings'],dtype=torch.float32),torch.tensor(ty,dtype=torch.float32))
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True)
    best=raw_f;best_epoch=-1;wait=0;history=[]
    for epoch in range(args.epochs):
        model.train();losses=[];grad_logs={'grad_alpha': [], 'grad_boundary': [], 'grad_scale': []}
        for scores,emb,labels in loader:
            scores=scores.to(device);emb=emb.to(device);labels=labels.to(device)
            outp=model(scores,emb)
            loss=pfresgo_soft_f(outp['prob'],labels,edges,root_idx,args.temperature,sweep_temperature=args.sweep_temperature)
            # Mild regularization toward raw predictions, not a separate ranking objective.
            loss=loss+1e-3*(outp['prob']-torch.sigmoid(scores)).square().mean()
            optim.zero_grad(set_to_none=True);loss.backward()
            # Pre-clipping gradient norms, collected per batch and averaged per epoch.
            grad_logs['grad_alpha'].append(grad_norm([model.alpha_logit]))
            grad_logs['grad_boundary'].append(grad_norm(list(model.boundary_head.parameters()) + [model.global_boundary]))
            grad_logs['grad_scale'].append(grad_norm(model.scale_head.parameters()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optim.step();losses.append(float(loss.item()))
        vp=predict(model,valid,device)
        diagnostics=calibration_diagnostics(model,valid,device)
        m,(curve,_,_,grid)=metrics(vy,vp,anc,None,full,root)
        ix=int(np.argmax(curve));vf=float(curve[ix]);thr=float(grid[ix])
        feasible=all(m[k]>=args.min_retention*raw_m[k] for k in ('macro_aupr','micro_aupr','flattened_auc'))
        rec={'epoch':epoch,'train_loss':float(np.mean(losses)),'valid_fmax':vf,'valid_threshold':thr,
             'valid_macro_aupr':float(m['macro_aupr']),'valid_micro_aupr':float(m['micro_aupr']),
             'valid_auroc':float(m['flattened_auc']),'alpha':float(torch.sigmoid(model.alpha_logit).item()),'feasible':bool(feasible),
             **diagnostics, **{k: float(np.mean(v)) for k,v in grad_logs.items()}}
        history.append(rec);print('EPOCH',json.dumps(rec),flush=True)
        # Persist diagnostics every epoch, including interrupted runs.
        (out/'history.json').write_text(json.dumps(history,indent=2))
        if feasible and vf>best+1e-7:
            best=vf;best_epoch=epoch;wait=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'go_hash':hash_go_ids(train['go_ids']),
                        'protein_dim':train['protein_embeddings'].shape[1],'valid_threshold':thr},out/'head.pt')
        else:
            # Do not stop before any feasible improvement is found.
            if best_epoch >= 0:
                wait += 1
        if best_epoch >= 0 and wait >= args.patience:
            break
    result={'selection':'validation PFresGO exact Fmax with 98% AUPR/AUROC retention',
            'raw_valid_fmax':raw_f,'raw_valid_threshold':raw_thr,'raw_valid_macro_aupr':float(raw_m['macro_aupr']),
            'raw_valid_micro_aupr':float(raw_m['micro_aupr']),'raw_valid_auroc':float(raw_m['flattened_auc']),
            'best_epoch':best_epoch,'best_valid_fmax':best,'raw_fallback':best_epoch<0,
            'epochs_completed':len(history),'soft_thresholds':[round(x/10,2) for x in range(1,10)],
            'n_valid':len(vy),'n_go_terms':len(gos),
            'original_eval_sha256':hashlib.sha256(Path(args.original_eval).read_bytes()).hexdigest()}
    (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'selection.json').write_text(json.dumps(result,indent=2))
    print('SELECTION',json.dumps(result,indent=2),flush=True)
    if best_epoch<0:print('No feasible improvement. Use raw Combined scores.',flush=True)

if __name__=='__main__':main()
