"""Train shared cross-attention relevance model, evaluate all GO labels on validation."""
import argparse, json, math, random, importlib.util, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from .model import FullSpaceCrossScorer

def smooth(x,valid,tau=.1):
    x=x.masked_fill(~valid[:,None],-1e4)
    return tau*(torch.logsumexp(x/tau,dim=0)-valid.sum().float().log())

@torch.no_grad()
def base_scores(w,valid,seg,present,parents,parent_present):
    # w [W,D], seg [G,2,D], parents [G,P,D]
    sn=smooth(w@seg[:,0].T,valid)
    sd=smooth(w@seg[:,1].T,valid)
    child=torch.where(present[:,1],.5*(sn+sd),sn)
    if parent_present.any():
        ps=torch.einsum('wd,gpd->wgp',w,parents)
        ps=ps.masked_fill(~valid[:,None,None],-1e4)
        pmil=.1*(torch.logsumexp(ps/.1,dim=0)-valid.sum().float().log())
        pmil=pmil.masked_fill(~parent_present,0)
        avg=pmil.sum(-1)/parent_present.sum(-1).clamp_min(1)
        child=child-.15*parent_present.any(-1)*torch.relu(child-avg)
    return child

def fetch(root,split,i,device):
    x=torch.load(root/split/f'{i:06d}.pt',map_location='cpu',weights_only=False)
    return x['windows'].float().to(device),x['valid'].bool().to(device),x['positives']

def evaluate(model,root,go,indices,device,chunk,limit=0):
    model.eval()
    gids=go['go_ids']; gidx={g:i for i,g in enumerate(gids)}
    seg=go['segments'].float().to(device)
    present=go['present'].bool().to(device)
    parents=go['parents']; pp=go['parent_present']
    aps=[]; raw_aps=[]; fprec=np.zeros(201); frec=np.zeros(201)
    if limit: indices=indices[:limit]
    with torch.inference_mode():
        for j,i in enumerate(indices):
            w,v,pos=fetch(root,'val',i,device)
            y=np.zeros(len(gids),dtype=np.bool_)
            for g in pos:
                if g in gidx:y[gidx[g]]=True
            if not y.any(): continue
            outputs=[]; baselines=[]
            for st in range(0,len(gids),chunk):
                ed=min(st+chunk,len(gids))
                p=parents[st:ed].float().to(device)
                pm=pp[st:ed].bool().to(device)
                b=base_scores(w,v,seg[st:ed],present[st:ed],p,pm)
                z=model(w[None],v[None],seg[st:ed],present[st:ed],b[None])[0]
                outputs.append(torch.sigmoid(z).cpu())
                baselines.append(b.cpu())
            pred=torch.cat(outputs).numpy(); raw=torch.cat(baselines).numpy()
            aps.append(average_precision_score(y,pred))
            raw_aps.append(average_precision_score(y,raw))
            # Threshold diagnostic, macro precision/recall as in MZSGO Fmax.
            bins=np.clip(np.floor(pred*200).astype(np.int32),0,200)
            hp=np.bincount(bins[y],minlength=201)[::-1].cumsum()[::-1]
            hn=np.bincount(bins[~y],minlength=201)[::-1].cumsum()[::-1]
            fprec+=np.divide(hp,hp+hn,out=np.zeros(201),where=(hp+hn)>0)
            frec+=hp/y.sum()
            if (j+1)%50==0:print(f'[VAL] {j+1}/{len(indices)}',flush=True)
    n=len(aps)
    if not n: raise RuntimeError('No validation positives')
    p=fprec/n; r=frec/n
    f=np.divide(2*p*r,p+r,out=np.zeros(201),where=(p+r)>0)
    j=int(f.argmax())
    return {'exact_protein_map':float(np.mean(aps)),
            'raw_exact_protein_map':float(np.mean(raw_aps)),
            'macro_fmax_201grid':float(f[j]),
            'threshold':j/200,'n_proteins':n}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--patience',type=int,default=5)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--train-proteins-per-epoch',type=int,default=3000)
    ap.add_argument('--positives-per-protein',type=int,default=16)
    ap.add_argument('--negatives-per-protein',type=int,default=64)
    ap.add_argument('--val-proteins',type=int,default=200)
    ap.add_argument('--go-chunk',type=int,default=128)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=Path(args.cache);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    go=torch.load(root/'go.pt',map_location='cpu',weights_only=False)
    gids=go['go_ids']; gidx={g:i for i,g in enumerate(gids)}
    seg=go['segments'].float().to(device)
    present=go['present'].bool().to(device)
    parents=go['parents'];pp=go['parent_present']
    train_ids=json.loads((root/'train_ids.json').read_text())
    val_ids=json.loads((root/'val_ids.json').read_text())
    rng=np.random.default_rng(args.seed)
    # Fixed protein-disjoint early-stop and held-out partitions.
    perm=rng.permutation(len(val_ids))
    n_hold=max(1,int(.2*len(perm)))
    heldout=perm[-n_hold:].tolist()
    early=perm[:-n_hold].tolist()
    if args.val_proteins:early=early[:args.val_proteins]
    (out/'split.json').write_text(json.dumps({'early':early,'heldout':heldout},indent=2))
    model=FullSpaceCrossScorer().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-3)
    best=-1.;best_epoch=0;bad=0
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; steps=0
        order=rng.permutation(len(train_ids))[:args.train_proteins_per_epoch]
        for j,i in enumerate(order):
            w,v,pos=fetch(root,'train',int(i),device)
            pos_idx=np.array([gidx[g] for g in pos if g in gidx],dtype=np.int64)
            if not len(pos_idx):continue
            selected_pos=rng.choice(pos_idx,size=min(len(pos_idx),args.positives_per_protein),replace=False)
            # Mix hard negatives (retriever similarity) and random negatives.
            rand=rng.choice(len(gids),size=min(len(gids),args.negatives_per_protein*5),replace=False)
            rand=rand[~np.isin(rand,pos_idx)]
            with torch.no_grad():
                ri=torch.tensor(rand,device=device)
                rb=base_scores(w,v,seg[ri],present[ri],
                               parents[ri.cpu()].float().to(device),
                               pp[ri.cpu()].bool().to(device))
                hard=rand[rb.topk(min(args.negatives_per_protein//2,len(rand))).indices.cpu().numpy()]
            easy=rng.choice(rand,size=min(args.negatives_per_protein-len(hard),len(rand)),replace=False)
            neg=np.unique(np.concatenate([hard,easy]))
            c=np.concatenate([selected_pos,neg])
            idx=torch.tensor(c,device=device)
            base=base_scores(w,v,seg[idx],present[idx],
                             parents[idx.cpu()].float().to(device),
                             pp[idx.cpu()].bool().to(device))
            logits=model(w[None],v[None],seg[idx],present[idx],base[None])[0]
            target=torch.tensor(np.concatenate([np.ones(len(selected_pos)),np.zeros(len(neg))]),device=device,dtype=torch.float32)
            # Pairwise BCE learns relevance, with balanced positive/negative weights.
            weights=torch.where(target>0,.5/target.sum().clamp_min(1),
                                .5/(1-target).sum().clamp_min(1))
            loss=(nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*weights).sum()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();total+=float(loss.item());steps+=1
            if (j+1)%500==0:print(f'[TRAIN] epoch={epoch} proteins={j+1}/{len(order)} loss={total/steps:.5f}',flush=True)
        metrics=evaluate(model,root,go,early,device,args.go_chunk)
        m=metrics['exact_protein_map']
        print(f'[EPOCH {epoch}] BCE={total/max(steps,1):.5f} val_exact_mAP={m:.6f} raw={metrics["raw_exact_protein_map"]:.6f}',flush=True)
        if m>best:
            best=m;best_epoch=epoch;bad=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'val_exact_map':best,
                        'retriever_cache':str(root),'architecture':'FullSpaceCrossScorer'},
                       out/'best.pt')
        else:bad+=1
        if bad>=args.patience:break
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(ck['model'])
    result=evaluate(model,root,go,heldout,device,args.go_chunk)
    result.update({'best_epoch':best_epoch,'best_early_stop_map':best,
                   'note':'Held-out validation diagnostic; macro Fmax uses 201 threshold grid, not official MZSGO test evaluation. BCE sampling makes sigmoid scores uncalibrated.'})
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print('[DONE]',json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
