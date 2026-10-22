"""Extract out-of-sample full-GO scores using the unchanged project trainer."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.main import (load_structured_cfg,_read_ids,_training_buckets,build_go_cache,
    build_residue_store,build_go_text_store,align_to_cache,build_datasets,
    _build_dag)
from src.encoders import BioMedBERTEncoder
from src.utils import load_raw_json
from src.utils.helpers import build_altid_map_from_go_terms,canonicalize_pid2pos,canonicalize_id_list
from src.configs.data_classes import TrainerConfig,TrainingContext,LoggingConfig
from src.training.collate import ContrastiveEmbCollator
from src.training.trainer import OppTrainer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--ids',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    args=load_structured_cfg(a.config)
    if args.evaluation_space!='benchmark':
        raise ValueError('Expected benchmark GO universe')
    if not Path(a.ids).is_file(): raise FileNotFoundError(a.ids)
    device=torch.device('cpu' if args.cpu else args.general_device)
    branch=_read_ids(args.branch_go_ids_path)
    seen,rare,zero,benchmark=_training_buckets(args.train_ids_path,args.pid2pos,branch,args.rare_lt)
    go_cache=build_go_cache(str(args.go_cache_path))
    vocab=load_raw_json(args.go_basic_json)
    alt=build_altid_map_from_go_terms(vocab) if vocab else {}
    raw=load_raw_json(args.pid2pos)
    labels=canonicalize_pid2pos(raw,alt) if alt else raw
    ids=canonicalize_id_list(benchmark,alt) if alt else benchmark
    seen=canonicalize_id_list(seen,alt) if alt else seen
    rare=canonicalize_id_list(rare,alt) if alt else rare
    zero=canonicalize_id_list(zero,alt) if alt else zero
    labels,ids,seen,zero,rare=align_to_cache(go_cache,labels,ids,seen,zero,rare)
    encoder=BioMedBERTEncoder(model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',device=device,max_length=512,
        attention_pooling_strategy=args.go_encoder_inner_pooling,attn_hidden=128,attn_dropout=.1,gradient_checkpointing=False)
    encoder.eval()
    for x in encoder.parameters(): x.requires_grad_(False)
    texts=build_go_text_store(args,encoder)
    texts.materialize_tokens_once(batch_size=512,show_progress=True)
    store=build_residue_store(args)
    # Only substitute validation IDs for dataset construction; no training is performed.
    args.val_ids_path=Path(a.ids)
    ds=build_datasets(args,store,texts,labels,_build_dag(args.go_basic_json)[0] if args.use_dag_in_ds else None,zero,rare)['val']
    loader=DataLoader(ds,batch_size=args.eval_batch_size,shuffle=False,num_workers=0,collate_fn=ContrastiveEmbCollator())
    if not len(ds): raise ValueError('Empty extraction dataset')
    example=ds[0]
    ctx=TrainingContext(go_cache=go_cache,device=device,go_text_store=texts,run_name='calibration_extract',fp16_enabled=args.fp16,
        protein_pooling_strategy=args.protein_pooling_strategy,go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,go_segment_representation_mode=args.go_segment_representation_mode,
        eval_id_list=list(ids),eval_seen_go_ids=list(seen),eval_unseen_ids=list(zero),eval_rare_go_ids=list(rare),
        logger=None,logging=LoggingConfig(log_every=args.log_every))
    cfg=TrainerConfig(d_h=int(example['prot_emb'].shape[1]),d_g=int(go_cache.embs.shape[1]),d_z=args.align_dim,device=str(device),
        lr=args.lr,weight_decay=args.weight_decay,grad_clip=args.grad_clip,max_epochs=args.epochs,batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,fp16=args.fp16,warmstart_path=None,monitor_metric=args.monitor_metric,
        secondary_monitor_metric=args.secondary_monitor_metric,monitor_mode=args.monitor_mode,
        protein_pooling_strategy=args.protein_pooling_strategy,use_lora=False,go_pooling=args.go_pooling,go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,go_segment_representation_mode=args.go_segment_representation_mode,
        eval_go_bs=args.eval_go_bs,temperature=args.temperature,lambda_con=args.lambda_con,pbr_lambda=args.pbr_lambda,
        pbr_margin=args.pbr_margin,pbr_tau=args.pbr_tau,retrieval_eval_ks=args.retrieval_eval_ks,
        positive_rank_quantiles=args.positive_rank_quantiles,log_gradient_norms=False,log_positive_rank_cdf=False,
        log_cardinality_metrics=False,log_go_segment_weights=False)
    trainer=OppTrainer(cfg=cfg,ctx=ctx,go_encoder=encoder)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if 'model' not in state: raise ValueError('Checkpoint has no model key')
    missing, unexpected = trainer.model.load_state_dict(
        state["model"]["model"], strict=True
    )
    trainer.model.eval()
    all_scores=[]; all_labels=[]; all_pids=[]
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bool(args.fp16 and device.type=='cuda')):
                s,_=trainer._score_full_go(batch)
            y=trainer._positive_mask(batch)
            all_scores.append(s.float().cpu().numpy());all_labels.append(y.cpu().numpy().astype(np.uint8))
            all_pids.extend(map(str,batch['protein_ids']))
    scores=np.concatenate(all_scores); ys=np.concatenate(all_labels)
    if scores.shape!=(len(all_pids),len(ids)): raise RuntimeError('Matrix shape mismatch')
    if not np.isfinite(scores).all(): raise RuntimeError('Non-finite scores')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,scores=scores,labels=ys,protein_ids=np.asarray(all_pids),go_ids=np.asarray(ids,dtype=np.int64))
    print(f'Saved {scores.shape} to {out}')

if __name__=='__main__': main()
