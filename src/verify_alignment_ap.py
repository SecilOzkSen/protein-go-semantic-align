import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score

class Head(torch.nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(d,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
    def forward(self,x):
        return 4*x[:,0]+0.5*torch.tanh(self.net(x).squeeze(-1))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--fit',required=True)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    data=Path(a.data);fit=Path(a.fit)
    features=np.load(data/'features.npy',mmap_mode='r')
    scores=np.load(data/'scores.npy',mmap_mode='r')
    pids=json.loads((data/'protein_ids.json').read_text())
    gids=json.loads((data/'go_ids.json').read_text())
    labels=json.loads((data/'labels.json').read_text())
    gi={v:i for i,v in enumerate(gids)}
    n,g,_=features.shape
    rng=np.random.default_rng(a.seed)
    va=rng.permutation(n)[int(.8*n):]
    models={}
    for name in ('score_only','alignment'):
        ck=torch.load(fit/(name+'.pt'),map_location='cpu',weights_only=False)
        model=Head(len(ck['columns']))
        model.load_state_dict(ck['state_dict']);model.eval()
        models[name]=(model,ck)
    aps={k:[] for k in ('raw','score_only','alignment')}
    npos=0
    with torch.inference_mode():
        for count,i in enumerate(va,1):
            positives=[gi[x] for x in labels[pids[i]] if x in gi]
            y=np.zeros(g,dtype=np.uint8);y[positives]=1
            npos+=int(y.sum())
            raw=np.asarray(scores[i],dtype=np.float32)
            if not y.any():
                # sklearn AP undefined for no positives; skip for all three models.
                continue
            aps['raw'].append(average_precision_score(y,raw))
            for name,(model,ck) in models.items():
                parts=[]
                for st in range(0,g,8192):
                    ed=min(st+8192,g)
                    x=np.asarray(features[i,st:ed],dtype=np.float32).copy()
                    x[:,0]=raw[st:ed]
                    x[:,1:]=(x[:,1:]-ck['mean'])/ck['std']
                    z=model(torch.from_numpy(x[:,ck['columns']])).numpy()
                    parts.append(z)
                logits=np.concatenate(parts)
                aps[name].append(average_precision_score(y,logits))
            if count%100==0:print(f'{count}/{len(va)}',flush=True)
    result={'heldout_proteins':len(va),'evaluated_with_positives':len(aps['raw']),'positive_pairs':npos,
            'mean_exact_AP':{k:float(np.mean(v)) for k,v in aps.items()},
            'delta_alignment_minus_raw':float(np.mean(np.asarray(aps['alignment'])-np.asarray(aps['raw']))),
            'delta_alignment_minus_score_only':float(np.mean(np.asarray(aps['alignment'])-np.asarray(aps['score_only']))),
            'fraction_proteins_alignment_improves_over_raw':float(np.mean(np.asarray(aps['alignment'])>np.asarray(aps['raw'])))}
    print(json.dumps(result,indent=2))
    (fit/'exact_ap_verification.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
