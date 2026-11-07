"""Cache frozen retriever projected protein windows and GO text segments."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import Dataset, gid, checkpoint_state, MODEL
from src.datasets.residue_store import ESMResidueStore
from src.training.collate import ContrastiveEmbCollator
from src.encoders import BioMedBERTEncoder
from src.models.alignment_model import ProteinGoAligner

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--root',default='/workspace/MZSGO_temporal')
    ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--max-proteins',type=int,default=0)
    args=ap.parse_args()
    root=Path(args.root); out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Nonempty output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    proc=root/'processed/bp'
    ids=sorted(map(gid,json.loads((proc/'candidate_go_ids_bp.json').read_text())))
    temporal=set(map(gid,json.loads((proc/'temporal_go_ids_bp.json').read_text())))
    ids=[g for g in ids if g not in temporal]
    bank=torch.load(root/'go_cache/bp/frozen_segment_parent_bank.pt',map_location='cpu',weights_only=False)
    idx={int(g):i for i,g in enumerate(bank['go_ids'])}
    sel=torch.tensor([idx[g] for g in ids])
    seg=bank['segment_embs'][sel].float()
    present=bank['segment_present'][sel].bool()
    parent=bank['parent_embs'][sel].float()
    pp=bank['parent_present'][sel].bool()
    device=torch.device(args.device)
    enc=BioMedBERTEncoder(model_name=MODEL,device=device,max_length=512,attention_pooling_strategy='mean',
                          attn_hidden=128,attn_dropout=.1,gradient_checkpointing=False)
    retr=ProteinGoAligner(d_h=1280,d_g=768,d_z=768,go_encoder=enc,normalize=True,
                          protein_pool_type='mean_attn_gate',go_pool_type='mean',
                          go_segment_representation_mode='segments_only').to(device)
    retr.load_state_dict(checkpoint_state(args.checkpoint),strict=True)
    retr.eval()
    with torch.inference_mode():
        # Cache all required GO vectors, including parent evidence for exact baseline score.
        go=torch.stack([retr.project_go(seg[:,0].to(device)),
                        retr.project_go(seg[:,1].to(device))],dim=1).cpu().half()
        projected_parents=[]
        for part in parent.split(128):
            shape=part.shape
            projected_parents.append(retr.project_go(part.reshape(-1,shape[-1]).to(device)).reshape(shape[0],shape[1],-1).cpu().half())
        parents=torch.cat(projected_parents)
    torch.save({'go_ids':[f'GO:{x:07d}' for x in ids],
                'segments':go,'present':present[:,:2],
                'parents':parents,'parent_present':pp},out/'go.pt')
    labels=json.loads((proc/'pid_to_go_train_bp.json').read_text())
    store=ESMResidueStore(embed_dir=str(root/'protein_embeddings/esm2_residue_overlap'),max_len=1000,overlap=256)
    for split,filename in [('train','train_homology30_v2.txt'),('val','valid_homology30_v2.txt')]:
        pids=[p.strip() for p in (root/'mzsgo_graphs'/filename).read_text().splitlines() if p.strip()]
        if args.max_proteins: pids=pids[:args.max_proteins]
        (out/f'{split}_ids.json').write_text(json.dumps(pids))
        directory=out/split; directory.mkdir()
        for i,pid in enumerate(pids):
            if not store.has(pid): raise RuntimeError(f'Missing protein {pid}')
            H=store.get(pid)
            if isinstance(H,np.ndarray): H=torch.from_numpy(H)
            if H.ndim!=2: raise RuntimeError(f'Invalid embedding shape {pid}: {H.shape}')
            mask=torch.ones((1,H.shape[0]),dtype=torch.bool,device=device)
            with torch.inference_mode():
                windows,valid=retr.encode_local_windows(H[None].to(device).float(),mask)
            torch.save({'windows':windows[0].cpu().half(),'valid':valid[0].cpu(),
                        'positives':[f'GO:{gid(x):07d}' for x in labels[pid]]},directory/f'{i:06d}.pt')
            if (i+1)%200==0: print(f'[CACHE {split}] {i+1}/{len(pids)}',flush=True)
    meta={'checkpoint':str(args.checkpoint),
          'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
          'n_go':len(ids),'note':'Train/val projected windows, no fixed label outputs'}
    (out/'meta.json').write_text(json.dumps(meta,indent=2))
    print('[DONE] cache',out,meta,flush=True)
if __name__=='__main__': main()
