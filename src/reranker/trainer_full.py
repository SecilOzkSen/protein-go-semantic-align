"""Full-universe Direct EmbSetCal. Requires existing .metrics.evaluate module."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .model_full import EmbSetCal
from .metrics import evaluate


def load(path, require_labels=True):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    required = {'scores','protein_z','go_z','protein_ids','go_ids'}
    if require_labels: required.add('labels')
    if not required.issubset(d): raise ValueError(f'Missing {required-set(d)} in {path}')
    n,g = d['scores'].shape
    if d['protein_z'].shape[0] != n or d['go_z'].shape[0] != g or len(d['protein_ids']) != n or len(d['go_ids']) != g: raise ValueError('NPZ dimensions mismatch')
    if not np.isfinite(d['scores']).all() or d['scores'].min()<0 or d['scores'].max()>1: raise ValueError('Scores not in [0,1]')
    if not np.isfinite(d['protein_z']).all() or not np.isfinite(d['go_z']).all(): raise ValueError('Nonfinite embeddings')
    if 'labels' in d and (d['labels'].shape != (n,g) or not np.isin(d['labels'],(0,1)).all()): raise ValueError('Invalid labels')
    if len(set(d['go_ids'].tolist())) != g: raise ValueError('Duplicate GO IDs')
    return d


def predict(model, data, device, batch_size=1, amp=True):
    model.eval()
    base = data['scores']; n,g = base.shape
    out = np.empty((n,g), dtype=np.float32)
    go = torch.as_tensor(data['go_z'], dtype=torch.float32, device=device)
    with torch.inference_mode():
        for start in range(0,n,batch_size):
            end = min(n,start+batch_size)
            p = torch.as_tensor(data['protein_z'][start:end],dtype=torch.float32,device=device)
            s = torch.as_tensor(base[start:end],dtype=torch.float32,device=device)
            gg = go.unsqueeze(0).expand(end-start,-1,-1)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
                logits = model(p,gg,s)
            out[start:end] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-npz',type=Path,required=True)
    ap.add_argument('--val-npz',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--mzsgo-repo',default='/workspace/MZSGO')
    ap.add_argument('--epochs',type=int,default=25)
    ap.add_argument('--batch-size',type=int,default=1)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--no-amp',action='store_true')
    args=ap.parse_args()
    if args.batch_size<1 or args.epochs<1: raise ValueError('Invalid batch size/epochs')
    if args.out_dir.exists() and any(args.out_dir.iterdir()): raise FileExistsError(f'Output dir not empty: {args.out_dir}')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    train=load(args.train_npz);val=load(args.val_npz)
    if not np.array_equal(train['go_ids'],val['go_ids']) or not np.allclose(train['go_z'],val['go_z']): raise ValueError('Train/val GO universe mismatch')
    if set(train['protein_ids'].tolist()) & set(val['protein_ids'].tolist()): raise ValueError('Protein leakage')
    device=torch.device(args.device)
    model=EmbSetCal(train['protein_z'].shape[1],train['go_z'].shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-3)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    baseline=evaluate(val['labels'],val['scores'],args.mzsgo_repo,'embsetcal_full_baseline')
    print('[BASELINE]',json.dumps(baseline),flush=True)
    go=torch.as_tensor(train['go_z'],dtype=torch.float32,device=device)
    best=-1.0
    for epoch in range(1,args.epochs+1):
        model.train();rng=np.random.default_rng(args.seed+epoch);order=rng.permutation(len(train['scores']));losses=[]
        for start in range(0,len(order),args.batch_size):
            rows=order[start:start+args.batch_size]
            p=torch.as_tensor(train['protein_z'][rows],dtype=torch.float32,device=device)
            s=torch.as_tensor(train['scores'][rows],dtype=torch.float32,device=device)
            y=torch.as_tensor(train['labels'][rows],dtype=torch.float32,device=device)
            gg=go.unsqueeze(0).expand(len(rows),-1,-1)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=not args.no_amp and device.type=='cuda'):
                logits=model(p,gg,s)
                # Same batch-balanced BCE as Direct Top-500 baseline.
                pos=y.sum();neg=y.numel()-pos
                weights=torch.where(y>0,neg.clamp_min(1)/pos.clamp_min(1),torch.ones_like(y))
                loss=(F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none')*weights).mean()
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(float(loss.detach().cpu()))
        scores=predict(model,val,device,args.batch_size,amp=not args.no_amp)
        m=evaluate(val['labels'],scores,args.mzsgo_repo,f'embsetcal_full_epoch_{epoch}')
        print(f'[EPOCH {epoch}] loss={np.mean(losses):.5f} metrics={json.dumps(m)}',flush=True)
        if m['Fmax']>best:
            best=m['Fmax']
            torch.save({'model':model.state_dict(),'config':{'protein_dim':train['protein_z'].shape[1],'go_dim':train['go_z'].shape[1],'k':int(train['scores'].shape[1]),'scoring_mode':'direct_full','amp':not args.no_amp},'go_ids':train['go_ids'].tolist(),'epoch':epoch,'metrics':m},args.out_dir/'best.pt')
            np.save(args.out_dir/'best_val_scores_01.npy',scores)
            (args.out_dir/'best_metrics.json').write_text(json.dumps({'baseline':baseline,'best':m,'epoch':epoch},indent=2))
    print('[DONE] best Fmax',best,flush=True)

if __name__=='__main__': main()
