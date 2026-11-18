"""Full-universe inference and optional original MZSGO evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .trainer_full import load,predict
from .model_full import EmbSetCal
from .metrics import evaluate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--batch-size',type=int,default=1)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--mzsgo-repo',default='/workspace/MZSGO')
    ap.add_argument('--no-eval',action='store_true')
    ap.add_argument('--no-amp',action='store_true')
    args=ap.parse_args()
    if args.batch_size<1: raise ValueError('Invalid batch size')
    data=load(args.npz,require_labels=False)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=ck['config']
    if cfg.get('scoring_mode')!='direct_full': raise ValueError('Not a full-universe direct checkpoint')
    if data['protein_z'].shape[1]!=cfg['protein_dim'] or data['go_z'].shape[1]!=cfg['go_dim']: raise ValueError('Embedding dimension mismatch')
    # GO universe can change at inference; all candidates get a model score.
    out=args.out_dir
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Output dir not empty: {out}')
    device=torch.device(args.device)
    model=EmbSetCal(cfg['protein_dim'],cfg['go_dim']).to(device)
    model.load_state_dict(ck['model'],strict=True)
    scores=predict(model,data,device,args.batch_size,amp=not args.no_amp)
    if not np.isfinite(scores).all() or scores.min()<0 or scores.max()>1: raise RuntimeError('Invalid prediction scores')
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'scores_01.npy',scores)
    (out/'protein_ids.json').write_text(json.dumps(data['protein_ids'].tolist()))
    (out/'go_ids.json').write_text(json.dumps(data['go_ids'].tolist()))
    if 'labels' in data and not args.no_eval:
        metrics=evaluate(data['labels'],scores,args.mzsgo_repo,'embsetcal_full_test')
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        print('[METRICS]',json.dumps(metrics),flush=True)
    print(f'[SAVED] {out} scores={scores.shape} checkpoint_epoch={ck.get("epoch")}',flush=True)

if __name__=='__main__': main()
