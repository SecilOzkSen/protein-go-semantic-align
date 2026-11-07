"""Extract frozen BP validation alignment features. Run from repository root.
Requires same model and checkpoint as the retriever evaluation.
"""
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
    n=len(val_ids); g=len(ids)
    scores=np.lib.format.open_memmap(out/'scores.npy',mode='w+',dtype='float32',shape=(n,g))
    # score, name MIL, definition MIL, parent adjustment, definition present,
    # name-definition disagreement, parent-present
    feats=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype='float16',shape=(n,g,7))
    labels_out={p:[f'GO:{x:07d}' for x in val_labels[p]] for p in val_ids}
    (out/'labels.json').write_text(json.dumps(labels_out)); (out/'protein_ids.json').write_text(json.dumps(val_ids))
    (out/'go_ids.json').write_text(json.dumps([f'GO:{x:07d}' for x in ids]))
    # Project GO segments once, exactly as the model scorer does.
    with torch.inference_mode():
        gn=model.project_go(name).float()
        gd=model.project_go(definition).float()
    for i,b in enumerate(loader):
        H=b['prot_emb_pad'].to(device).float()
        mask=b['prot_attn_mask'].to(device).bool()
        with torch.inference_mode():
            # The frozen model remains the authority for the final score.
            s=model.score_fine_grained(H,mask,name,definition,dp,parent,pp,raw).float()[0]
            zw,valid=model.encode_local_windows(H,mask)
            scores[i]=s.cpu().numpy()
            for st in range(0,g,args.go_chunk_size):
                ed=min(g,st+args.go_chunk_size)
                sn=model.mil(torch.einsum('bkd,gd->bkg',zw,gn[st:ed]),valid)[0]
                sd=model.mil(torch.einsum('bkd,gd->bkg',zw,gd[st:ed]),valid)[0]
                dmask=dp[st:ed].bool()
                child=torch.where(dmask,0.5*(sn+sd),sn)
                pr=parent[st:ed]
                pm=pp[st:ed].bool()
                if pm.any():
                    projected=model.project_go(pr.reshape(-1,pr.shape[-1])).float()
                    projected=projected.reshape(ed-st,pr.shape[1],-1)
                    parent_scores=torch.einsum('bkd,cpd->bkcp',zw,projected)
                    C,P=pm.shape
                    parent_flat=parent_scores.permute(0,3,1,2).reshape(P,zw.shape[1],C)
                    parent_valid=valid[:,None,:].expand(1,P,valid.shape[1]).reshape(P,-1)
                    parent_mil=model.mil(parent_flat,parent_valid).reshape(P,C).permute(1,0)
                    parent_mil=parent_mil.masked_fill(~pm,0)
                    parent_avg=parent_mil.sum(-1)/pm.sum(-1).clamp_min(1)
                    adjustment=-model.parent_strength*pm.any(-1)*torch.relu(child-parent_avg)
                else:
                    adjustment=torch.zeros_like(child)
                # This identity must hold, otherwise feature extraction is invalid.
                reconstructed=child+adjustment
                err=(reconstructed-s[st:ed]).abs().max().item()
                if err>2e-4: raise RuntimeError(f'Alignment feature parity failed: {err}')
                f=torch.stack([s[st:ed],sn,sd,adjustment,dmask.float(),
                               (sn-sd).abs(),pm.any(-1).float()],dim=-1)
                feats[i,st:ed]=f.cpu().numpy().astype('float16')
        if (i+1)%50==0: print(f'[EXTRACT] {i+1}/{n}',flush=True)
    scores.flush(); feats.flush()
    meta={'checkpoint':args.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
          'n_proteins':n,'n_candidates':g,'feature_names':['score','name_mil','definition_mil','parent_adjustment','definition_present','name_definition_abs_gap','parent_present'],
          'note':'Frozen validation only. Alignment features reproduce the original scorer before FP16 storage.'}
    (out/'meta.json').write_text(json.dumps(meta,indent=2)); print('[DONE]',out,meta,flush=True)
if __name__=='__main__': main()
