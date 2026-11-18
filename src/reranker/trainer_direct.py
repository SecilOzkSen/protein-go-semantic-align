import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .model_direct import EmbSetCal
from .metrics import evaluate

def load(path, labels=True):
    with np.load(path,allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
    needed={'scores','protein_z','go_z','protein_ids','go_ids'} | ({'labels'} if labels else set())
    if not needed.issubset(d): raise ValueError(f'{path} missing {needed-set(d)}')
    n,g=d['scores'].shape
    if d['protein_z'].shape[0]!=n or d['go_z'].shape[0]!=g or len(d['protein_ids'])!=n or len(d['go_ids'])!=g: raise ValueError('Shape mismatch')
    if not np.isfinite(d['scores']).all() or d['scores'].min()<0 or d['scores'].max()>1: raise ValueError('Scores outside [0,1]')
    if labels and (d['labels'].shape!=(n,g) or not np.isin(d['labels'],[0,1]).all()): raise ValueError('Invalid labels')
    return d

def predict(model,data,k,device,batch):
    model.eval(); base=data['scores'].astype(np.float32); result=base.copy()
    with torch.inference_mode():
        for start in range(0,len(base),batch):
            end=min(len(base),start+batch); ids=np.argsort(-base[start:end],axis=1,kind='stable')[:,:k]
            p=torch.tensor(data['protein_z'][start:end],device=device,dtype=torch.float32)
            g=torch.tensor(data['go_z'][ids],device=device,dtype=torch.float32)
            s=torch.tensor(np.take_along_axis(base[start:end],ids,axis=1),device=device,dtype=torch.float32)
            result[np.arange(start,end)[:,None],ids]=torch.sigmoid(model(p,g,s)).cpu().numpy()
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-npz',type=Path,required=True);ap.add_argument('--val-npz',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--mzsgo-repo',default='/workspace/MZSGO')
    ap.add_argument('--k',type=int,default=500);ap.add_argument('--epochs',type=int,default=25)
    ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--device',default='cuda')
    args=ap.parse_args();torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    train=load(args.train_npz);val=load(args.val_npz)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {args.out_dir}')
    if not np.array_equal(train['go_ids'],val['go_ids']) or not np.allclose(train['go_z'],val['go_z']):raise ValueError('Train/val GO universe mismatch')
    if set(train['protein_ids'].tolist()) & set(val['protein_ids'].tolist()):raise ValueError('Train/val protein leakage')
    if args.k>train['scores'].shape[1] or args.k>val['scores'].shape[1] or args.k<1:raise ValueError('Invalid K')
    device=torch.device(args.device)
    model=EmbSetCal(train['protein_z'].shape[1],train['go_z'].shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-3)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    baseline=evaluate(val['labels'],val['scores'],args.mzsgo_repo,'embsetcal_baseline')
    print('[BASELINE]',json.dumps(baseline),flush=True)
    best=-1
    for epoch in range(1,args.epochs+1):
        model.train();order=np.random.default_rng(args.seed+epoch).permutation(len(train['scores']));losses=[]
        for start in range(0,len(order),args.batch_size):
            rows=order[start:start+args.batch_size];scores=train['scores'][rows]
            ids=np.argsort(-scores,axis=1,kind='stable')[:,:args.k]
            p=torch.tensor(train['protein_z'][rows],device=device,dtype=torch.float32)
            g=torch.tensor(train['go_z'][ids],device=device,dtype=torch.float32)
            s=torch.tensor(np.take_along_axis(scores,ids,axis=1),device=device,dtype=torch.float32)
            y=torch.tensor(np.take_along_axis(train['labels'][rows],ids,axis=1),device=device,dtype=torch.float32)
            logits=model(p,g,s)
            # Balanced within-batch loss. Calibration quality must be assessed on validation.
            pos=y.sum();neg=y.numel()-pos
            weights=torch.where(y>0,neg.clamp_min(1)/pos.clamp_min(1),torch.ones_like(y))
            loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(float(loss.detach().cpu()))
        scores=predict(model,val,args.k,device,args.batch_size)
        m=evaluate(val['labels'],scores,args.mzsgo_repo,f'embsetcal_epoch_{epoch}')
        print(f'[EPOCH {epoch}] loss={np.mean(losses):.5f} metrics={json.dumps(m)}',flush=True)
        if m['Fmax']>best:
            best=m['Fmax']
            torch.save({'model':model.state_dict(),'config':{'protein_dim':train['protein_z'].shape[1],'go_dim':train['go_z'].shape[1],'k':args.k,'scoring_mode':'direct'},'go_ids':train['go_ids'].tolist(),'epoch':epoch,'metrics':m},args.out_dir/'best.pt')
            np.save(args.out_dir/'best_val_scores_01.npy',scores)
            (args.out_dir/'best_metrics.json').write_text(json.dumps({'baseline':baseline,'best':m,'epoch':epoch},indent=2))
    print('[DONE] best original MZSGO Fmax',best)

if __name__=='__main__':main()
