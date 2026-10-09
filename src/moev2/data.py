import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import obonet
from src.moev2.model import Config

def load_dump(path):
    path=Path(path)
    if not (path/'DONE').exists(): raise RuntimeError(f'Incomplete dump: {path}')
    ids=json.loads((path/'protein_ids.json').read_text())
    go=np.load(path/'eval_go_ids.int64.npy').astype(np.int64)
    z=np.load(path/'go_z.float16.npy').astype(np.float32)
    p=np.load(path/'protein_z.float16.npy',mmap_mode='r')
    sf=path/'retriever_scores.float32.npy'
    s=np.load(sf if sf.exists() else path/'retriever_scores.float16.npy',mmap_mode='r')
    y=np.load(path/'labels.int8.npy',mmap_mode='r')
    if len(ids)!=len(set(ids)) or len(go)!=len(set(go.tolist())): raise ValueError('Duplicate protein or GO ID')
    if p.shape[0]!=len(ids) or s.shape!=y.shape or s.shape!=(len(ids),len(go)) or z.shape[0]!=len(go) or z.shape[1]!=p.shape[1]: raise ValueError('Dump shape mismatch')
    for arr in (p,s,z):
        if not np.isfinite(arr).all(): raise ValueError('Nonfinite dump')
    return dict(ids=ids,go=go,z=z,p=p,s=s,y=y)

class DumpDataset(Dataset):
    def __init__(self,d): self.d=d
    def __len__(self): return len(self.d['ids'])
    def __getitem__(self,i):
        d=self.d
        return (torch.from_numpy(np.array(d['p'][i],dtype=np.float32)),torch.from_numpy(np.array(d['s'][i],dtype=np.float32)),torch.from_numpy(np.array(d['y'][i],dtype=np.float32)))

def graph_matrices(go_ids,obo):
    graph=obonet.read_obo(str(obo))
    ids=[f'GO:{int(i):07d}' for i in go_ids]
    idx={g:i for i,g in enumerate(ids)}
    G=len(ids); parents=np.zeros((G,G),dtype=np.float32)
    for child in ids:
        if child not in graph: continue
        for parent,edge_data in graph[child].items():
            # OBO graph child -> parent; only is_a edges
            attrs=edge_data.values() if isinstance(edge_data,dict) and any(isinstance(v,dict) for v in edge_data.values()) else [edge_data]
            if parent in idx and any(a.get('relation','is_a')=='is_a' for a in attrs): parents[idx[child],idx[parent]]=1
    return torch.from_numpy(parents),torch.from_numpy(parents.T.copy())
