
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .metrics import load_scores, exact_fmax, protein_f
from .model import features, OffsetHead, normalize_features

def validate_ids(a,b):
    if a is not None and b is not None and not np.array_equal(a,b):
        raise ValueError("GO term order differs between files")

def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    vs,vy,vids=load_scores(args.valid)
    if args.test:
        ts,ty,tids=load_scores(args.test)
        validate_ids(vids,tids)
        if ts.shape[1]!=vs.shape[1]: raise ValueError("GO dimension mismatch")
    n=len(vs)
    if n<20: raise ValueError("Need >=20 validation proteins for split")
    perm=np.random.default_rng(args.seed).permutation(n)
    nfit=int(n*args.fit_fraction)
    fit_idx=perm[:nfit]
    select_idx=perm[nfit:]
    if not len(select_idx): raise ValueError("Empty threshold-selection split")

    vf=features(vs)
    mu=vf[fit_idx].mean(0)
    std=vf[fit_idx].std(0)
    vf=normalize_features(vf,mu,std)
    x=torch.from_numpy(vf[fit_idx])
    y=torch.from_numpy(vy[fit_idx])
    scores=torch.from_numpy(vs[fit_idx])
    head=OffsetHead(hidden=args.hidden,max_offset=args.max_offset)
    optimizer=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=1e-4)

    # Fit only on fit half. Threshold selected only on disjoint validation half.
    for step in range(args.steps):
        head.train()
        optimizer.zero_grad()
        offset=head(x)
        logits=(scores+offset[:,None])*args.logit_scale
        # Equal weight per protein, GO labels are multi-label.
        bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none').mean(1).mean()
        loss=bce+args.offset_penalty*offset.square().mean()
        loss.backward()
        optimizer.step()

    head.eval()
    with torch.no_grad():
        v_offsets=head(torch.from_numpy(vf)).numpy()
    vcal=vs+v_offsets[:,None]
    base_f,base_t=exact_fmax(vy[select_idx],vs[select_idx])
    cal_f,cal_t=exact_fmax(vy[select_idx],vcal[select_idx])

    result={
        'protocol':'validation split: fit offset on first subset; select thresholds on disjoint subset',
        'seed':args.seed,'n_fit':len(fit_idx),'n_threshold_select':len(select_idx),
        'baseline_valid_select':{'fmax':base_f,'threshold':base_t},
        'calibrated_valid_select':{'fmax':cal_f,'threshold':cal_t},
        'mean_abs_offset_valid':float(np.abs(v_offsets).mean()),
        'note':'Test oracle is diagnostic only. Test labels never used to fit head or select threshold.'
    }
    if args.test:
        tf=normalize_features(features(ts),mu,std)
        with torch.no_grad():
            toff=head(torch.from_numpy(tf)).numpy()
        tcal=ts+toff[:,None]
        result['test']={
            'baseline_F_at_valid_threshold':protein_f(ty,ts,base_t),
            'calibrated_F_at_valid_threshold':protein_f(ty,tcal,cal_t),
            'baseline_oracle_Fmax_DIAGNOSTIC':exact_fmax(ty,ts)[0],
            'calibrated_oracle_Fmax_DIAGNOSTIC':exact_fmax(ty,tcal)[0],
            'mean_abs_offset':float(np.abs(toff).mean())
        }

    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':head.state_dict(),'feature_mean':mu,'feature_std':std,
                'hidden':args.hidden,'max_offset':args.max_offset,
                'logit_scale':args.logit_scale,'threshold':cal_t},out/'calibrator.pt')
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description='Frozen-retriever protein-conditioned offset calibration')
    p.add_argument('--valid',required=True,help='scores.npz with scores,labels,go_ids')
    p.add_argument('--test',help='Optional test scores.npz')
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fit-fraction',type=float,default=.5)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--lr',type=float,default=.005)
    p.add_argument('--hidden',type=int,default=16)
    p.add_argument('--max-offset',type=float,default=.5)
    p.add_argument('--offset-penalty',type=float,default=.1)
    p.add_argument('--logit-scale',type=float,default=10.)
    a=p.parse_args()
    if not 0<a.fit_fraction<1: p.error('fit-fraction must be in (0,1)')
    run(a)

if __name__=='__main__': main()
