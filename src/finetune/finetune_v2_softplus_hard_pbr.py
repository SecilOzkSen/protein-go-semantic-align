"""EXPERIMENT 2: V2 pure -> normalized softplus hard-negative PBR-only BP fine-tuning.

Run from repository root: python -m src.finetune.finetune_v2_softplus_hard_pbr ...
Uses original 46,456 train proteins; reserves 5,162 validation proteins.
Does not train GO encoder, alter original checkpoint, or access temporal test labels.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.main import (load_structured_cfg, build_go_text_store, build_residue_store,
                      build_datasets, build_dataloaders, _read_ids, _go_int)
from src.configs.data_classes import TrainingContext, TrainerConfig, LoggingConfig
from src.encoders import BioMedBERTEncoder
from src.training.collate import ContrastiveEmbCollator
import src.training.trainer as trainer_module
from src.models.alignment_model_v2_original import ProteinGoAligner as OriginalV2


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def extract_state(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    for _ in range(10):
        if isinstance(state, dict) and sum(torch.is_tensor(v) for v in state.values()) >= 5:
            break
        state = next((state[k] for k in ('model','model_state_dict','state_dict','net','module')
                      if isinstance(state,dict) and isinstance(state.get(k),dict)), None)
        if state is None: raise RuntimeError('Cannot find model state in checkpoint')
    out = {}
    for k,v in state.items():
        if not torch.is_tensor(v): continue
        while True:
            pref = next((p for p in ('trainer.model.','module.','model.') if k.startswith(p)),None)
            if pref is None: break
            k=k[len(pref):]
        out[k]=v
    return out


def hard_pbr(scores, positives, k=128, margin=.05, tau=.05):
    """Per-protein mean of per-positive softplus violations against top-K negatives.
    Hard mining uses detached scores; gradients flow through selected scores.
    """
    losses=[]
    for i in range(scores.size(0)):
        pos=scores[i][positives[i]].float()
        neg=scores[i][~positives[i]].float()
        if pos.numel()==0 or neg.numel()==0: raise RuntimeError('Empty positive/negative set')
        hard=neg.topk(min(k,neg.numel()),sorted=False).values
        violations=torch.nn.functional.softplus((hard[None,:]-pos[:,None]+margin)/tau)
        losses.append(violations.mean())
    return torch.stack(losses).mean()


def score_v2(trainer,batch):
    H=batch['prot_emb_pad'].to(trainer.device,non_blocking=True).float()
    mask=batch['prot_attn_mask'].to(trainer.device,non_blocking=True).bool()
    names=trainer._segment_names
    seg=trainer._segment_embs_gpu
    present=trainer._segment_present_gpu
    return trainer.model.score_fine_grained(
        H,mask,seg[:,names.index('name')],seg[:,names.index('definition')],
        present[:,names.index('definition')],trainer._parent_bank_gpu,
        trainer._parent_present_gpu,seg.mean(1))


@torch.no_grad()
def validate(trainer, loader, ks=(100,200,500), max_batches=0):
    trainer.model.eval()
    n=0; ap_sum=0.; recall={k:0. for k in ks}
    term_total=np.zeros(len(trainer.eval_id_list),dtype=np.float64)
    term_hit={k:np.zeros_like(term_total) for k in ks}
    for j,batch in enumerate(loader):
        if max_batches and j>=max_batches: break
        scores=score_v2(trainer,batch)
        mask=trainer._positive_mask(batch)
        for i in range(scores.size(0)):
            positives=torch.where(mask[i])[0]
            ranks=torch.argsort(scores[i],descending=True,stable=True)
            inverse=torch.empty_like(ranks)
            inverse[ranks]=torch.arange(len(ranks),device=ranks.device)
            pr=(inverse[positives]+1).sort().values
            ap_sum+=float((torch.arange(1,len(pr)+1,device=pr.device)/pr).float().mean())
            pp=positives.cpu().numpy()
            term_total[pp]+=1
            for k in ks:
                hit=(inverse[positives]<k).cpu().numpy()
                recall[k]+=float(hit.mean())
                term_hit[k][pp]+=hit
            n+=1
    if n==0: raise RuntimeError('Empty validation')
    active=term_total>0
    out={'protein_wise_map':ap_sum/n,'n_validated':n}
    for k in ks:
        out[f'recall@{k}']=recall[k]/n
        out[f'macro_term_recall@{k}']=float(np.mean(term_hit[k][active]/term_total[active]))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='src/mzsgo_bp_infonce_pbr.yaml')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=5e-6)
    p.add_argument('--hard-k',type=int,default=128)
    p.add_argument('--margin',type=float,default=.05)
    p.add_argument('--tau',type=float,default=.05)
    p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--val-batch-size',type=int,default=1)
    p.add_argument('--val-max-batches',type=int,default=0,help='0 = full validation')
    p.add_argument('--smoke',action='store_true',help='Two training batches, 8 validation batches; no checkpoints')
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or a.hard_k<1 or a.tau<=0: p.error('Invalid hyperparameters')
    ckpt=Path(a.checkpoint).resolve(); out=Path(a.output_dir).resolve()
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    if out==ckpt.parent or out in ckpt.parents: raise RuntimeError('Output overlaps checkpoint directory')
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Output directory must be empty: {out}')
    cfg=load_structured_cfg(a.config)
    if not cfg.mzsgo_enabled or cfg.pfresgo_branch!='BP': raise RuntimeError('Only MZSGO BP supported')
    root=Path(cfg.mzsgo_root); proc=root/'processed'/'bp'; graph=root/'mzsgo_graphs'
    train_ids=set((graph/'train_homology30_v2.txt').read_text().split())
    val_ids=set((graph/'valid_homology30_v2.txt').read_text().split())
    if len(train_ids)!=46456 or len(val_ids)!=5162 or train_ids&val_ids:
        raise RuntimeError('Unexpected train/val split or overlap')
    annotations_raw=json.loads((proc/'pid_to_go_train_bp.json').read_text())
    annotations={pid: sorted({_go_int(g) for g in gos}) for pid,gos in annotations_raw.items()}
    print('[CHECK] Normalized protein GO labels to integer IDs',flush=True)
    candidate=set(_read_ids(proc/'candidate_go_ids_bp.json'))
    temporal=set(_read_ids(proc/'temporal_go_ids_bp.json'))
    active=sorted(candidate-temporal)
    if (len(candidate),len(temporal),len(active))!=(20345,34,20311): raise RuntimeError('GO universe mismatch')
    if any(not annotations.get(pid) or not set(map(_go_int,annotations[pid]))<=set(active)
           for pid in train_ids|val_ids): raise RuntimeError('Invalid protein GO annotations')
    cfg.train_ids_path=graph/'train_homology30_v2.txt'
    cfg.val_ids_path=graph/'valid_homology30_v2.txt'
    cfg.pid2pos=proc/'pid_to_go_train_bp.json'
    cfg.go_text_path=proc/'go_texts_canonical_segmented_dual_2023_2025.jsonl'
    cfg.branch_go_ids_path=proc/'candidate_go_ids_bp.json'
    cfg.mzsgo_frozen_bank=root/'go_cache'/'bp'/'frozen_segment_parent_bank.pt'
    cfg.embed_dir_res=root/'protein_embeddings'/'esm2_residue_overlap'
    cfg.batch_size=a.batch_size; cfg.eval_batch_size=a.val_batch_size
    cfg.num_workers=0; cfg.fp16=True
    cfg.use_dag_in_ds=False
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    bank=torch.load(cfg.mzsgo_frozen_bank,map_location='cpu',weights_only=False)
    if bank.get('format')!='mzsgo_frozen_segment_parent_v1': raise RuntimeError('Bank format')
    if set(map(int,bank['go_ids']))!=candidate: raise RuntimeError('Bank candidate mismatch')
    if bank.get('metadata',{}).get('canonical_jsonl_sha256')!=sha256(cfg.go_text_path):
        raise RuntimeError('Bank canonical hash mismatch')
    if list(bank['segment_names'])!=['name','definition','is_a']: raise RuntimeError('Bank segment mismatch')
    os.environ['EXPH_GO_JSONL']=str(cfg.go_text_path)
    os.environ['MZSGO_FROZEN_GO_BANK']=str(cfg.mzsgo_frozen_bank)
    encoder=BioMedBERTEncoder(model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        device=device,max_length=512,attention_pooling_strategy=cfg.go_encoder_inner_pooling,
        attn_hidden=128,attn_dropout=.1,gradient_checkpointing=False)
    for param in encoder.parameters(): param.requires_grad_(False)
    encoder.eval()
    texts=build_go_text_store(cfg,encoder)
    texts.materialize_tokens_once(batch_size=512,show_progress=True)
    store=build_residue_store(cfg)
    ds=build_datasets(cfg,store,texts,annotations,None,[],[])
    if (len(ds['train']),len(ds['val']))!=(46456,5162): raise RuntimeError('Dataset split mismatch')
    collate=ContrastiveEmbCollator()
    train=DataLoader(ds['train'],batch_size=a.batch_size,shuffle=True,num_workers=0,collate_fn=collate)
    val=DataLoader(ds['val'],batch_size=a.val_batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    go_cache=SimpleNamespace(embs=torch.empty((0,768)),id2row={int(g):i for i,g in enumerate(bank['go_ids'])})
    ctx=TrainingContext(go_cache=go_cache,device=device,go_text_store=texts,fp16_enabled=True,
        protein_pooling_strategy=cfg.protein_pooling_strategy,go_pool_type=cfg.go_pool_type,
        go_encoder_output_mode=cfg.go_encoder_output_mode,go_segment_representation_mode=cfg.go_segment_representation_mode,
        eval_id_list=active,eval_seen_go_ids=active,eval_unseen_ids=[],eval_rare_go_ids=[],logging=LoggingConfig())
    tc=TrainerConfig(d_h=1280,d_g=768,d_z=768,device=str(device),lr=a.lr,weight_decay=0.,
        grad_clip=a.grad_clip,batch_size=a.batch_size,eval_batch_size=a.val_batch_size,
        warmstart_path=None,protein_pooling_strategy=cfg.protein_pooling_strategy,
        go_pool_type=cfg.go_pool_type,go_encoder_output_mode=cfg.go_encoder_output_mode,
        go_segment_representation_mode=cfg.go_segment_representation_mode,eval_go_bs=256)
    # Trainer constructs original V2, never the newer V3 interaction model.
    trainer_module.ProteinGoAligner=OriginalV2
    trainer=trainer_module.OppTrainer(tc,ctx,encoder)
    state=extract_state(ckpt)
    trainer.model.load_state_dict(state,strict=True)
    if len(state)!=229: raise RuntimeError(f'Expected 229 V2 tensors, got {len(state)}')
    for n,param in trainer.model.named_parameters():
        if n.startswith('go_encoder.'): param.requires_grad_(False)
    trainer.opt=torch.optim.AdamW([v for v in trainer.model.parameters() if v.requires_grad],lr=a.lr,weight_decay=0.)
    print('[CHECK] V2 original strict load: 229 tensors',flush=True)
    print('[CHECK] checkpoint SHA256:',sha256(ckpt),flush=True)
    print('[CHECK] train=46456 valid=5162 seen_GO=20311 temporal_excluded=34',flush=True)
    print('[CHECK] objective=normalized top-K SOFTPLUS Hard-PBR only; InfoNCE=0; frequency weights=off',flush=True)
    print('[CHECK] optimizer freshly initialized; frozen GO encoder; original V2 scoring',flush=True)
    if not a.smoke: out.mkdir(parents=True,exist_ok=True)
    initial=validate(trainer,val,max_batches=8 if a.smoke else a.val_max_batches)
    print('[val] epoch=0',json.dumps(initial),flush=True)
    best=initial['protein_wise_map']
    if not a.smoke:
        (out/'run_config.json').write_text(json.dumps({**vars(a),'source_sha256':sha256(ckpt),
            'baseline_validation':initial,'objective':'topK_normalized_softplus_HardPBR_only'},indent=2))
    for epoch in range(1,a.epochs+1):
        trainer.model.train();trainer.model.go_encoder.eval()
        t0=time.time();loss_sum=0.;seen=0
        for j,batch in enumerate(train,1):
            trainer.opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                scores=score_v2(trainer,batch)
                positives=trainer._positive_mask(batch)
                loss=hard_pbr(scores,positives,a.hard_k,a.margin,a.tau)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite Softplus Hard-PBR loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([v for v in trainer.model.parameters() if v.requires_grad],a.grad_clip)
            trainer.opt.step()
            loss_sum+=float(loss.detach());seen+=1
            if j==1 or j%a.log_every==0:
                print(f'[train] epoch={epoch}/{a.epochs} batch={j}/{len(train)} '
                      f'loss_mean={loss_sum/seen:.6f} elapsed={time.time()-t0:.1f}s',flush=True)
            if a.smoke and j>=2: break
        metrics=validate(trainer,val,max_batches=8 if a.smoke else a.val_max_batches)
        print(f'[val] epoch={epoch} '+json.dumps(metrics),flush=True)
        if not a.smoke:
            path=out/f'checkpoint_epoch{epoch}.pt'
            torch.save({'model':trainer.model.state_dict(),'epoch':epoch,'validation':metrics,
                        'source_sha256':sha256(ckpt),'objective':'normalized_topK_softplus_hard_PBR_only'},path)
            print('[checkpoint] Saved',path,flush=True)
            if metrics['protein_wise_map']>best+1e-6:
                best=metrics['protein_wise_map']
                import shutil
                shutil.copy2(path,out/'best_primary.pt')
                print('[best] protein_wise_map',best,flush=True)
        if a.smoke: break
    print('[DONE] baseline validation mAP=',initial['protein_wise_map'],'best fine-tuned mAP=',best,flush=True)

if __name__=='__main__': main()
