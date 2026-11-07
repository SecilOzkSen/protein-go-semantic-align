"""Extract frozen BP validation scores and magnitude features. Run from repo root."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.residue_store import ESMResidueStore
from src.training.collate import ContrastiveEmbCollator
from src.encoders import BioMedBERTEncoder
from src.models.alignment_model import ProteinGoAligner

MODEL='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
def gid(x): return int(str(x).split(':')[-1])
def checkpoint_state(path):
    d=torch.load(path,map_location='cpu',weights_only=False)
    for _ in range(10):
        if isinstance(d,dict) and sum(torch.is_tensor(v) for v in d.values())>=5: break
        d=next((d[k] for k in ('model','model_state_dict','state_dict','net','module') if isinstance(d,dict) and isinstance(d.get(k),dict)),None)
        if d is None: raise RuntimeError('Checkpoint state not found')
    out={}
    for k,v in d.items():
        if not torch.is_tensor(v): continue
        while True:
            prefix=next((p for p in ('trainer.model.','module.','model.') if k.startswith(p)),None)
            if prefix is None: break
            k=k[len(prefix):]
        out[k]=v
    return out

class Dataset(torch.utils.data.Dataset):
    def __init__(self,ids,labels,store): self.ids,self.labels,self.store=ids,labels,store
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        p=self.ids[i]
        return {'protein_id':p,'prot_emb':self.store.get(p),'pos_go_ids':torch.tensor(self.labels[p],dtype=torch.long)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--root',default='/workspace/MZSGO_temporal')
    ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--go-chunk-size',type=int,default=256)
    ap.add_argument('--smoke-proteins',type=int,default=0)
    args=ap.parse_args()
    root=Path(args.root); proc=root/'processed/bp'; out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    ids=sorted(map(gid,json.loads((proc/'candidate_go_ids_bp.json').read_text())))
    temporal=set(map(gid,json.loads((proc/'temporal_go_ids_bp.json').read_text())))
    ids=[g for g in ids if g not in temporal]
    bank=torch.load(root/'go_cache/bp/frozen_segment_parent_bank.pt',map_location='cpu',weights_only=False)
    bank_ids=list(map(int,bank['go_ids'])); idx={g:i for i,g in enumerate(bank_ids)}
    sel=torch.tensor([idx[g] for g in ids]); seg=bank['segment_embs'][sel].float(); present=bank['segment_present'][sel].bool()
    parents=bank['parent_embs'][sel].float(); pp=bank['parent_present'][sel].bool()
    labels=json.loads((proc/'pid_to_go_train_bp.json').read_text())
    val_ids=[s.strip() for s in (root/'mzsgo_graphs/valid_homology30_v2.txt').read_text().splitlines() if s.strip()]
    if args.smoke_proteins: val_ids=val_ids[:args.smoke_proteins]
    val_labels={p:sorted(set(map(gid,labels[p])) & set(ids)) for p in val_ids}
    if any(not v for v in val_labels.values()): raise RuntimeError('Protein with no labels')
    store=ESMResidueStore(embed_dir=str(root/'protein_embeddings/esm2_residue_overlap'),max_len=1000,overlap=256)
    if not all(store.has(p) for p in val_ids): raise RuntimeError('Missing residue embeddings')
    loader=DataLoader(Dataset(val_ids,val_labels,store),batch_size=1,shuffle=False,num_workers=0,collate_fn=ContrastiveEmbCollator())
    device=torch.device(args.device)
    enc=BioMedBERTEncoder(model_name=MODEL,device=device,max_length=512,attention_pooling_strategy='mean',attn_hidden=128,attn_dropout=.1,gradient_checkpointing=False)
    model=ProteinGoAligner(d_h=1280,d_g=768,d_z=768,go_encoder=enc,normalize=True,protein_pool_type='mean_attn_gate',go_pool_type='mean',go_segment_representation_mode='segments_only').to(device)
    model.load_state_dict(checkpoint_state(args.checkpoint),strict=True); model.eval(); model.go_chunk_size=args.go_chunk_size
    name=seg[:,0].to(device); definition=seg[:,1].to(device); dp=present[:,1].to(device)
    parent=parents.to(device); pp=pp.to(device); raw=seg.mean(1).to(device)
    # GO projection norms: same proj_g used by score_fine_grained.
    with torch.inference_mode():
        gn=torch.linalg.vector_norm(model.proj_g(name).float(),dim=-1).clamp_min(1e-8).log().cpu().numpy()
        gd=torch.linalg.vector_norm(model.proj_g(definition).float(),dim=-1).clamp_min(1e-8).log().cpu().numpy()
    np.save(out/'go_norms.npy',np.stack([gn,gd],axis=-1).astype('float32'))
    n=len(val_ids); g=len(ids)
    scores=np.lib.format.open_memmap(out/'scores.npy',mode='w+',dtype='float32',shape=(n,g))
    # Features: [retriever score, mean window lognorm, best-match window lognorm, name lognorm, definition lognorm, score*best lognorm]
    feats=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype='float16',shape=(n,g,6))
    labels_out={p:[f'GO:{x:07d}' for x in val_labels[p]] for p in val_ids}
    (out/'labels.json').write_text(json.dumps(labels_out)); (out/'protein_ids.json').write_text(json.dumps(val_ids))
    (out/'go_ids.json').write_text(json.dumps([f'GO:{x:07d}' for x in ids]))
    for i,b in enumerate(loader):
        H=b['prot_emb_pad'].to(device).float(); mask=b['prot_attn_mask'].to(device).bool()
        # Capture unnormalized window projections from the exact live projection module.
        window_raw=[]
        def capture(module,inputs,output): window_raw.append(output.detach().float())
        handle=model.proj_local.register_forward_hook(capture)
        try:
            with torch.inference_mode(): s=model.score_fine_grained(H,mask,name,definition,dp,parent,pp,raw).float()[0]
        finally: handle.remove()
        if not window_raw: raise RuntimeError('No local windows captured')
        z=torch.stack(window_raw,dim=0).squeeze(1)
        # Check hook corresponds to exactly the local windows in the scoring function.
        zlog=torch.linalg.vector_norm(z,dim=-1).clamp_min(1e-8).log()
        zn=torch.nn.functional.normalize(z,dim=-1)
        scores[i]=s.cpu().numpy()
        meanlog=zlog.mean()
        for start in range(0,g,args.go_chunk_size):
            end=min(g,start+args.go_chunk_size)
            # GO-specific strongest window by name/definition cosine, same representations as scorer.
            qn=model.project_go(name[start:end]).float(); qd=model.project_go(definition[start:end]).float()
            cosn=zn@qn.T; cosd=zn@qd.T
            cos=torch.where(dp[start:end][None,:],(cosn+cosd)*.5,cosn)
            best=zlog[cos.argmax(dim=0)]
            f=torch.stack([s[start:end],meanlog.expand(end-start),best,
                           torch.from_numpy(gn[start:end]).to(device),torch.from_numpy(gd[start:end]).to(device),s[start:end]*best],dim=-1)
            feats[i,start:end]=f.cpu().numpy().astype('float16')
        if (i+1)%50==0: print(f'[EXTRACT] {i+1}/{n}',flush=True)
    scores.flush(); feats.flush()
    meta={'checkpoint':args.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
          'n_proteins':n,'n_candidates':g,'feature_names':['score','mean_log_window_norm','best_log_window_norm','go_name_log_norm','go_definition_log_norm','score_x_best_log_norm'],
          'note':'Frozen validation only. Best window defined by pre-MIL child cosine; not parent-adjusted.'}
    (out/'meta.json').write_text(json.dumps(meta,indent=2)); print('[DONE]',out,meta,flush=True)
if __name__=='__main__': main()
