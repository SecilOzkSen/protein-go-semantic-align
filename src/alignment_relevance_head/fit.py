"""Train score-only and alignment-aware residual heads on split validation proteins.
Frozen retriever. Never reads test labels.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class Head(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,16),nn.Tanh(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        # Retain retriever score as a fixed anchor; learn bounded residual.
        return 4.0*x[:,0]+0.5*torch.tanh(self.net(x).squeeze(-1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--negatives-per-positive',type=int,default=5)
    ap.add_argument('--max-train-pairs',type=int,default=500000)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args(); rng=np.random.default_rng(args.seed)
    data=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    feat=np.load(data/'features.npy',mmap_mode='r'); score=np.load(data/'scores.npy',mmap_mode='r')
    pids=json.loads((data/'protein_ids.json').read_text()); gids=json.loads((data/'go_ids.json').read_text())
    lab=json.loads((data/'labels.json').read_text()); gindex={g:i for i,g in enumerate(gids)}
    n,g,_=feat.shape
    if len(pids)!=n: raise ValueError('Mismatched protein IDs')
    perm=rng.permutation(n); split=int(.8*n); tr=perm[:split]; va=perm[split:]
    # Sampling is only for head training, never for validation metrics.
    rows=[]; cols=[]; ys=[]
    for i in tr:
        pos=np.array([gindex[x] for x in lab[pids[i]] if x in gindex],dtype=np.int64)
        if not len(pos): continue
        # Hard negatives from highest retriever scores plus random negatives.
        scores=np.asarray(score[i]); hard_n=min(len(pos)*args.negatives_per_positive//2,g)
        pool=np.argpartition(scores,-min(g,hard_n+len(pos)+64))[-min(g,hard_n+len(pos)+64):]
        pool=pool[~np.isin(pool,pos)]; hard=pool[np.argsort(scores[pool])[-hard_n:]]
        rand=rng.integers(0,g,size=len(pos)*args.negatives_per_positive-hard_n)
        rand=rand[~np.isin(rand,pos)]
        neg=np.unique(np.concatenate([hard,rand])); c=np.concatenate([pos,neg])
        rows.extend([int(i)]*len(c)); cols.extend(c.tolist()); ys.extend([1]*len(pos)+[0]*len(neg))
    pick=rng.permutation(len(rows))[:args.max_train_pairs]
    rows=np.asarray(rows)[pick]; cols=np.asarray(cols)[pick]; ys=np.asarray(ys,dtype='float32')[pick]
    X=np.asarray(feat[rows,cols],dtype='float32'); X[:,0]=np.asarray(score[rows,cols],dtype='float32')
    # Fixed feature standardization fitted only on training samples; preserve raw score.
    mu=X[:,1:].mean(0); sd=X[:,1:].std(0).clip(1e-4); X[:,1:]=(X[:,1:]-mu)/sd
    device=torch.device(args.device); xx=torch.from_numpy(X).to(device); yy=torch.from_numpy(ys).to(device)
    models={}
    for name,cols_used in [('score_only',[0]),('alignment',list(range(feat.shape[-1])))]:
        model=Head(len(cols_used)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
        for ep in range(args.epochs):
            ix=torch.randperm(len(xx),device=device)
            for sub in ix.split(8192):
                logit=model(xx[sub][:,cols_used]); loss=nn.functional.binary_cross_entropy_with_logits(logit,yy[sub])
                # Keep residual modest; prevents free replacement of retriever ranking.
                loss=loss+1e-4*sum(p.square().sum() for p in model.parameters())
                opt.zero_grad(); loss.backward(); opt.step()
            print(f'[{name}] epoch={ep+1} train_bce={loss.item():.5f}',flush=True)
        models[name]=(model.eval(),cols_used)
        torch.save({'state_dict':model.cpu().state_dict(),'columns':cols_used,'mean':mu,'std':sd},out/(name+'.pt'))
        model.to(device)
    # Evaluation: streaming TP/FP/FN across all GO columns and held-out proteins.
    # Fmax is protein-macro F1 at a common threshold; AUPR uses 201-bin PR integration.
    thresholds=np.linspace(0,1,201)
    results={}
    for name in ['raw','score_only','alignment']:
        sum_p=np.zeros(201); sum_r=np.zeros(201); sum_tp=np.zeros(201); sum_fp=np.zeros(201); sum_fn=np.zeros(201)
        for i in va:
            positives=np.array([gindex[x] for x in lab[pids[i]] if x in gindex],dtype=np.int64)
            y=np.zeros(g,dtype=bool); y[positives]=True
            if name=='raw':
                pred=np.clip((np.asarray(score[i],dtype='float32')+1)/2,0,1)
            else:
                model,cols_used=models[name]
                pieces=[]
                with torch.inference_mode():
                    for st in range(0,g,8192):
                        ed=min(g,st+8192)
                        x=np.asarray(feat[i,st:ed],dtype='float32').copy()
                        x[:,0]=np.asarray(score[i,st:ed],dtype='float32')
                        x[:,1:]=(x[:,1:]-mu)/sd
                        logit=model(torch.from_numpy(x[:,cols_used]).to(device))
                        pieces.append(torch.sigmoid(logit).cpu().numpy())
                pred=np.concatenate(pieces)
            # Bin to threshold grid; vectorized histograms avoid huge [201,G] arrays.
            bins=np.minimum((pred*200).astype(np.int32),200)
            hist_pos=np.bincount(bins[y],minlength=201)[::-1].cumsum()[::-1]
            hist_neg=np.bincount(bins[~y],minlength=201)[::-1].cumsum()[::-1]
            tp=hist_pos; fp=hist_neg; fn=y.sum()-tp
            p=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)>0)
            r=tp/max(1,y.sum())
            sum_p+=p; sum_r+=r; sum_tp+=tp; sum_fp+=fp; sum_fn+=fn
        # Protein-macro precision and recall, as an approximate diagnostic only.
        P=sum_p/len(va); R=sum_r/len(va)
        f=np.divide(2*P*R,P+R,out=np.zeros_like(P),where=(P+R)>0)
        best=int(f.argmax())
        # PR AUC based on 201 thresholds, not identical to original MZSGO AUPR.
        auc=float(np.trapz(P[::-1],R[::-1]))
        results[name]={'heldout_proteins':len(va),'protein_macro_fmax_201grid':float(f[best]),'threshold':float(thresholds[best]),'protein_macro_pr_auc_201grid':auc}
        print(name,results[name],flush=True)
    result={'seed':args.seed,'n_train_proteins':len(tr),'n_holdout_proteins':len(va),'n_sampled_pairs':len(ys),
            'warning':'Exploratory heldout validation only. 201-bin macro PR metrics are NOT MZSGO paper metrics. No test set used.',
            'results':results}
    (out/'results.json').write_text(json.dumps(result,indent=2)); print('[DONE]',out,flush=True)
if __name__=='__main__': main()
