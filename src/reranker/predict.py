import argparse,json
from pathlib import Path
import numpy as np
import torch
from .trainer import load,predict
from .model import EmbSetCal

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--npz',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--device',default='cuda');ap.add_argument('--mzsgo-repo',default='/workspace/MZSGO')
    a=ap.parse_args();data=load(a.npz,labels=False);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    # General/temporal test candidate universes may differ from training.
    # Model is candidate-agnostic; only feature dimensions must match.
    model=EmbSetCal(ck['config']['protein_dim'],ck['config']['go_dim']).to(a.device);model.load_state_dict(ck['model'])
    s=predict(model,data,ck['config']['k'],torch.device(a.device),a.batch_size)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    np.save(out/'scores_01.npy',s)
    (out/'protein_ids.json').write_text(json.dumps(data['protein_ids'].tolist()))
    (out/'go_ids.json').write_text(json.dumps(data['go_ids'].tolist()))
    if 'labels' in data:
        from .metrics import evaluate
        metrics=evaluate(data['labels'],s,a.mzsgo_repo,'embsetcal_test')
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        print('[METRICS]',json.dumps(metrics),flush=True)
    print('[SAVED]',out)
if __name__=='__main__':main()
