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

