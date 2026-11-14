"""Score EVERY GO candidate for cached proteins, with no Top-K filtering."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .model import FullSpaceCrossScorer
from .train import fetch,base_scores

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--split',choices=['train','val'],default='val')
    ap.add_argument('--go-chunk',type=int,default=128)
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    root=Path(args.cache);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    go=torch.load(root/'go.pt',map_location='cpu',weights_only=False)
    seg=go['segments'].float().to(device); present=go['present'].bool().to(device)
    model=FullSpaceCrossScorer().to(device)
    ck=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(ck['model']);model.eval()
    pids=json.loads((root/f'{args.split}_ids.json').read_text())
    g=len(go['go_ids'])
    scores=np.lib.format.open_memmap(out/'scores.npy',mode='w+',dtype='float32',shape=(len(pids),g))
    raw=np.lib.format.open_memmap(out/'retriever_scores.npy',mode='w+',dtype='float32',shape=(len(pids),g))
    with torch.inference_mode():
        for i in range(len(pids)):
            w,v,_=fetch(root,args.split,i,device)
            for st in range(0,g,args.go_chunk):
                ed=min(st+args.go_chunk,g)
                idx=slice(st,ed)
                b=base_scores(w,v,seg[idx],present[idx],
                              go['parents'][idx].float().to(device),
                              go['parent_present'][idx].bool().to(device))
                z=model(w[None],v[None],seg[idx],present[idx],b[None])[0]
                scores[i,idx]=torch.sigmoid(z).cpu().numpy()
                raw[i,idx]=b.cpu().numpy()
            if (i+1)%50==0:print(f'[PREDICT] {i+1}/{len(pids)}',flush=True)
    scores.flush();raw.flush()
    (out/'protein_ids.json').write_text(json.dumps(pids))
    (out/'go_ids.json').write_text(json.dumps(go['go_ids']))
    (out/'meta.json').write_text(json.dumps({'n_proteins':len(pids),'n_go':g,
            'score_type':'sigmoid relevance score, not probability-calibrated',
            'no_top_k_filtering':True},indent=2))
    print('[DONE]',out,flush=True)
if __name__=='__main__':main()
