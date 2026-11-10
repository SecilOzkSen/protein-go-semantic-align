#!/usr/bin/env python3
"""MZSGO BP V2: fine-tune from best weights on original train + held-out valid.

Run from repository root with `python -m src.script.retrieverv2.finetune_v2_full_train ...`.
No validation, no test-label access, no best-checkpoint selection.
Requires the V2 architecture matching the supplied V2 checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.main import (load_structured_cfg, build_residue_store, build_go_text_store,
                      _read_ids, _go_int)
from src.configs.data_classes import FewZeroConfig, TrainerConfig, TrainingContext, LoggingConfig
from src.datasets.protein_dataset import ProteinEmbDataset
from src.encoders import BioMedBERTEncoder
from src.training.collate import ContrastiveEmbCollator
import src.training.trainer as trainer_module
from src.models.alignment_model_v2_original import ProteinGoAligner as V2ProteinGoAligner
from src.training.trainer import multi_positive_full_go_infonce, positive_block_rank_loss

# The current shared trainer has been modified for V3 BCE.
# Force the original V2 model and V2 objective, without editing the repository trainer.
trainer_module.ProteinGoAligner = V2ProteinGoAligner

class OppTrainer(trainer_module.OppTrainer):
    def _score_full_go(self, batch):
        H = batch["prot_emb_pad"].to(self.device, non_blocking=True)
        mask = batch["prot_attn_mask"].to(self.device, non_blocking=True).bool()
        names = self._segment_names
        seg = self._segment_embs_gpu
        present = self._segment_present_gpu
        parent = self._parent_bank_gpu
        parent_present = self._parent_present_gpu
        scores = self.model.score_fine_grained(
            H, mask,
            seg[:, names.index("name")],
            seg[:, names.index("definition")],
            present[:, names.index("definition")],
            parent, parent_present, None,
        )
        weights = present.float() / present.float().sum(1, keepdim=True).clamp_min(1)
        return scores, weights

    def step_losses(self, batch, epoch_idx, debug=False):
        self.model.train()
        self.model.go_encoder.eval()
        scores, _ = self._score_full_go(batch)
        pos_mask = self._positive_mask(batch)
        con = multi_positive_full_go_infonce(scores, pos_mask, self.cfg.temperature)
        pbr, _ = positive_block_rank_loss(
            scores, pos_mask, self.cfg.pbr_margin, self.cfg.pbr_tau
        )
        total = self.cfg.lambda_con * con + self.cfg.pbr_lambda * pbr
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite V2 loss")
        self._global_step += 1
        return {"total": total, "contrastive": con, "pbr": pbr}

from src.utils.checkpoint import save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Original V2 YAML, not V3 YAML')
    p.add_argument('--checkpoint', required=True, help='V2 best_primary.pt')
    p.add_argument('--output-dir', required=True, help='New, separate output folder')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def strict_load(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload
    for _ in range(10):
        if isinstance(state, dict) and len([v for v in state.values() if torch.is_tensor(v)]) >= 5:
            break
        for key in ('model', 'state_dict', 'model_state_dict', 'module', 'net'):
            if isinstance(state, dict) and isinstance(state.get(key), dict):
                state = state[key]
                break
        else:
            raise RuntimeError('Cannot unwrap model state_dict from checkpoint')
    cleaned = {}
    for key, value in state.items():
        name = key
        for _ in range(5):
            for prefix in ('module.', 'model.', 'trainer.model.'):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            else:
                break
        cleaned[name] = value
    model.load_state_dict(cleaned, strict=True)
    meta_cfg = payload.get('meta', {}).get('config', {})
    if isinstance(meta_cfg, dict):
        print('[CHECK] Original checkpoint objective:',
              {k: meta_cfg.get(k) for k in ('lambda_con', 'pbr_lambda', 'temperature')}, flush=True)
    print(f'[CHECK] Strict checkpoint load succeeded, {len(cleaned)} tensors', flush=True)
    return payload


def main():
    cli = parse_args()
    if cli.epochs <= 0 or cli.batch_size <= 0 or cli.lr <= 0:
        raise ValueError('epochs, batch-size and lr must be positive')
    args = load_structured_cfg(cli.config)
    if not args.mzsgo_enabled or args.pfresgo_branch != 'BP':
        raise ValueError('This script requires MZSGO BP config')
    # Original V2 scoring and objective are explicitly restored above.
    if 'interaction_bce' in str(cli.config).lower():
        raise ValueError('Do not pass V3 interaction BCE YAML')
    if cli.output_dir == args.output_dir or Path(cli.output_dir).resolve() == Path(cli.checkpoint).resolve().parent:
        raise ValueError('Fine-tune output must differ from original V2 output')
    output = Path(cli.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is not empty: {output}. Use a fresh directory.')
    if not Path(cli.checkpoint).is_file():
        raise FileNotFoundError(cli.checkpoint)

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cpu' if args.cpu else args.general_device)
    root = Path(args.mzsgo_root)
    proc = root / 'processed' / 'bp'
    graph = root / 'mzsgo_graphs'
    train_path = graph / 'train_homology30_v2.txt'
    valid_path = graph / 'valid_homology30_v2.txt'
    pid2pos_path = proc / 'pid_to_go_train_bp.json'
    go_text_path = proc / 'go_texts_canonical_segmented_dual_2023_2025.jsonl'
    bank_path = root / 'go_cache' / 'bp' / 'frozen_segment_parent_bank.pt'
    candidate_path = proc / 'candidate_go_ids_bp.json'
    temporal_path = proc / 'temporal_go_ids_bp.json'
    for path in (train_path, valid_path, pid2pos_path, go_text_path, bank_path, candidate_path, temporal_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_ids = [s.strip() for s in train_path.read_text().splitlines() if s.strip()]
    valid_ids = [s.strip() for s in valid_path.read_text().splitlines() if s.strip()]
    if (len(train_ids), len(valid_ids)) != (46456, 5162):
        raise RuntimeError(f'Unexpected split sizes: {len(train_ids)}, {len(valid_ids)}')
    if len(set(train_ids)) != len(train_ids) or len(set(valid_ids)) != len(valid_ids):
        raise RuntimeError('Duplicate protein IDs')
    if set(train_ids) & set(valid_ids):
        raise RuntimeError('Train/valid overlap')
    all_ids = train_ids + valid_ids
    assert len(all_ids) == 51618

    candidate = set(_read_ids(candidate_path))
    temporal = set(_read_ids(temporal_path))
    seen = sorted(candidate - temporal)
    if (len(candidate), len(temporal), len(seen)) != (20345, 34, 20311):
        raise RuntimeError('GO universe differs from locked MZSGO BP protocol')
    pid2pos_raw = json.loads(pid2pos_path.read_text())
    pid2pos = {pid: sorted({_go_int(g) for g in pid2pos_raw.get(pid, [])}) for pid in all_ids}
    for pid, labels in pid2pos.items():
        if not labels or not set(labels) <= set(seen):
            raise RuntimeError(f'Invalid or temporal positive labels: {pid}')
    print(f'[CHECK] train={len(train_ids)} valid={len(valid_ids)} total={len(all_ids)} seen_GO={len(seen)} temporal_GO_excluded={len(temporal)}', flush=True)

    bank = torch.load(bank_path, map_location='cpu', weights_only=False)
    if bank.get('format') != 'mzsgo_frozen_segment_parent_v1':
        raise RuntimeError('Unexpected frozen GO bank format')
    if set(map(int, bank['go_ids'])) != candidate:
        raise RuntimeError('Frozen bank GO IDs mismatch')
    if bank.get('metadata', {}).get('canonical_jsonl_sha256') != hashlib.sha256(go_text_path.read_bytes()).hexdigest():
        raise RuntimeError('Frozen GO bank text hash mismatch')
    if list(bank.get('segment_names', [])) != list(args.enabled_segments):
        raise RuntimeError('GO segments mismatch')
    d_g = int(bank['segment_embs'].shape[-1])
    del bank
    os.environ['MZSGO_FROZEN_GO_BANK'] = str(bank_path)
    os.environ['EXPH_GO_JSONL'] = str(go_text_path)
    args.go_text_path = go_text_path

    encoder = BioMedBERTEncoder(
        model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        device=device, max_length=512,
        attention_pooling_strategy=args.go_encoder_inner_pooling,
        attn_hidden=128, attn_dropout=0.1, gradient_checkpointing=False)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    go_store = build_go_text_store(args, encoder)
    go_store.materialize_tokens_once(batch_size=512, show_progress=True)
    residue_store = build_residue_store(args)
    missing = [pid for pid in all_ids if not residue_store.has(pid)]
    if missing:
        raise RuntimeError(f'{len(missing)} proteins missing residue embeddings, examples: {missing[:5]}')
    ds = ProteinEmbDataset(
        protein_ids=all_ids, pid2pos=pid2pos, go_text_store=go_store,
        fewzero=FewZeroConfig(zero_shot_terms=set(), few_shot_terms=set(), fs_target_ratio=0.0),
        dag_parents=None, store=residue_store)
    loader = DataLoader(ds, batch_size=cli.batch_size, shuffle=True,
                        num_workers=cli.num_workers, persistent_workers=cli.num_workers > 0,
                        pin_memory=True, drop_last=False, collate_fn=ContrastiveEmbCollator())
    sample = ds[0]
    ctx = TrainingContext(
        go_cache=SimpleNamespace(embs=torch.empty((0, d_g)), id2row={g:i for i,g in enumerate(sorted(candidate))}, row2id=sorted(candidate)),
        device=device, go_text_store=go_store, run_name='MZSGO-BP-V2-full-finetune',
        fp16_enabled=args.fp16, protein_pooling_strategy=args.protein_pooling_strategy,
        go_pool_type=args.go_pool_type, go_encoder_output_mode=args.go_encoder_output_mode,
        go_segment_representation_mode=args.go_segment_representation_mode,
        eval_id_list=seen, eval_seen_go_ids=seen, eval_unseen_ids=[], eval_rare_go_ids=[],
        logger=None, logging=LoggingConfig(log_every=cli.log_every))
    cfg = TrainerConfig(
        d_h=int(sample['prot_emb'].shape[1]), d_g=d_g, d_z=args.align_dim,
        device=str(device), lr=cli.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, max_epochs=cli.epochs, batch_size=cli.batch_size,
        eval_batch_size=cli.batch_size, fp16=args.fp16, warmstart_path=None,
        monitor_metric='protein_wise_map', secondary_monitor_metric='macro_term_recall@500',
        monitor_mode='max', protein_pooling_strategy=args.protein_pooling_strategy,
        use_lora=False, go_pooling=args.go_pooling, go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,
        go_segment_representation_mode=args.go_segment_representation_mode,
        eval_go_bs=args.eval_go_bs, temperature=args.temperature,
        lambda_con=args.lambda_con, pbr_lambda=args.pbr_lambda,
        pbr_margin=args.pbr_margin, pbr_tau=args.pbr_tau,
        retrieval_eval_ks=args.retrieval_eval_ks,
        positive_rank_quantiles=args.positive_rank_quantiles,
        log_gradient_norms=False, log_positive_rank_cdf=False,
        log_cardinality_metrics=False, log_go_segment_weights=False)
    trainer = OppTrainer(cfg=cfg, ctx=ctx, go_encoder=encoder, wandb_run=None)
    payload = strict_load(trainer.model, cli.checkpoint)
    print('[CHECK] Original V2 model and InfoNCE + PBR scoring active', flush=True)
    original_cfg = payload.get('meta', {}).get('config', {})
    if isinstance(original_cfg, dict):
        for key in ('lambda_con', 'pbr_lambda', 'pbr_margin', 'pbr_tau', 'temperature'):
            if original_cfg.get(key) is not None and not math.isclose(float(original_cfg[key]), float(getattr(args, key)), rel_tol=1e-7, abs_tol=1e-9):
                raise RuntimeError(f'V2 objective mismatch for {key}: checkpoint={original_cfg[key]} YAML={getattr(args, key)}')
    # New AdamW, deliberately NOT loading checkpoint optimizer or global step.
    trainer.opt = torch.optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=cli.lr, weight_decay=args.weight_decay)
    trainer._global_step = 0
    print('[CHECK] New optimizer, V2 model weights loaded, no validation', flush=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'finetune_manifest.json').write_text(json.dumps({
        'source_checkpoint': str(Path(cli.checkpoint).resolve()),
        'source_epoch': payload.get('meta', {}).get('epoch'),
        'config': str(Path(cli.config).resolve()),
        'train_proteins': len(train_ids), 'added_valid_proteins': len(valid_ids),
        'all_proteins': len(all_ids), 'seen_go': len(seen),
        'temporal_go_excluded': len(temporal), 'epochs': cli.epochs,
        'batch_size': cli.batch_size, 'lr': cli.lr,
        'lambda_con': args.lambda_con, 'pbr_lambda': args.pbr_lambda,
        'validation_used_for_training': True, 'validation_evaluation': False,
    }, indent=2))
    save_args = SimpleNamespace(**vars(args))
    save_args.output_dir = str(output)
    save_args.batch_size = cli.batch_size
    save_args.lr = cli.lr
    save_args.epochs = cli.epochs
    use_bf16 = bool(args.fp16 and device.type == 'cuda')
    for epoch in range(cli.epochs):
        trainer.model.train()
        trainer.model.go_encoder.eval()
        t0 = time.perf_counter()
        totals = {'total':0.0, 'contrastive':0.0, 'pbr':0.0}
        for idx, batch in enumerate(loader, 1):
            trainer.opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                losses = trainer.step_losses(batch, epoch)
                loss = losses['total']
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in trainer.model.parameters() if p.requires_grad], args.grad_clip)
            trainer.opt.step()
            for key in totals:
                if key in losses:
                    value = losses[key]
                    totals[key] += float(value.detach().item() if torch.is_tensor(value) else value)
            if idx == 1 or idx % cli.log_every == 0 or idx == len(loader):
                print(f'[train] epoch={epoch+1}/{cli.epochs} batch={idx}/{len(loader)} '
                      f'global_step={trainer._global_step} loss={totals["total"]/idx:.4f} '
                      f'InfoNCE={totals["contrastive"]/idx:.4f} '
                      f'PBR={totals["pbr"]/idx:.4f} elapsed={time.perf_counter()-t0:.1f}s', flush=True)
        save_checkpoint(str(output), f'epoch{epoch+1}', trainer, save_args, epoch, trainer._global_step)
        print(f'[epoch] {epoch+1}/{cli.epochs} finished in {(time.perf_counter()-t0)/60:.1f} min', flush=True)
    print(f'[DONE] Final checkpoint: {output / f"checkpoint_epoch{cli.epochs}.pt"}', flush=True)


if __name__ == '__main__':
    main()
