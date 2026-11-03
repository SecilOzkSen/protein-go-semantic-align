"""Frozen vs LoRA InfoNCE epoch-5 (zero-indexed) PFresGO diagnostic.

Run from repository root with `python compare_infonce_e5_pfresgo.py ...`.
Uses current src.main, src.training.trainer, and original PFresGO evaluator.
No training, optimizer restore, or checkpoint modification.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.main import (load_structured_cfg, _read_ids, _training_buckets,
                      build_go_cache, build_residue_store, build_go_text_store,
                      align_to_cache, build_datasets, _build_dag)
from src.encoders import BioMedBERTEncoder
from src.utils import load_raw_json
from src.utils.helpers import (build_altid_map_from_go_terms,
                               canonicalize_pid2pos, canonicalize_id_list)
from src.configs.data_classes import TrainerConfig, TrainingContext, LoggingConfig
from src.training.collate import ContrastiveEmbCollator
from src.training.trainer import OppTrainer


def load_original(path):
    spec = importlib.util.spec_from_file_location('pfresgo_original_for_comparison', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.trange = range
    return mod


def go_string(x):
    return f'GO:{int(str(x).split(":")[-1]):07d}'


def strict_order(actual, desired, label):
    a, b = list(map(str, actual)), list(map(str, desired))
    if len(a) != len(set(a)) or len(b) != len(set(b)) or set(a) != set(b):
        raise RuntimeError(f'{label} universe mismatch or duplicate IDs; refusing comparison')
    lookup = {v: i for i, v in enumerate(a)}
    return np.asarray([lookup[x] for x in b], dtype=np.int64)


def load_model_state(checkpoint, model, use_lora):
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = blob['model']
    if 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    has_lora = any('lora_' in k for k in state)
    if has_lora != use_lora:
        raise RuntimeError(f'LoRA mismatch for {checkpoint}: checkpoint={has_lora}, expected={use_lora}')
    model.load_state_dict(state, strict=True)
    print(f'[CHECKPOINT] {checkpoint} loaded strictly, {len(state)} tensors', flush=True)


def extract(config, checkpoint, test_ids, expected_lora, device, batch_size):
    args = load_structured_cfg(str(config))
    if bool(args.use_lora) != expected_lora:
        raise RuntimeError(f'{config}: general.use_lora should be {expected_lora}')
    if args.evaluation_space != 'benchmark' or args.pfresgo_branch != 'BP':
        raise RuntimeError('Requires BP benchmark GO universe')
    if args.lambda_con != 1.0 or args.pbr_lambda != 0.0:
        raise RuntimeError('Requires InfoNCE-only training config')
    args.val_ids_path = Path(test_ids)
    args.general_device = device
    args.cpu = device == 'cpu'
    dev = torch.device(device)

    branch = _read_ids(args.branch_go_ids_path)
    seen, rare, zero, benchmark = _training_buckets(
        args.train_ids_path, args.pid2pos, branch, args.rare_lt)
    cache = build_go_cache(str(args.go_cache_path))
    vocab = load_raw_json(args.go_basic_json)
    alt = build_altid_map_from_go_terms(vocab) if vocab else {}
    labels = load_raw_json(args.pid2pos)
    if alt:
        labels = canonicalize_pid2pos(labels, alt)
    canonical = lambda xs: canonicalize_id_list(xs, alt) if alt else list(map(int, xs))
    labels, gos, seen, zero, rare = align_to_cache(
        cache, labels, canonical(benchmark), canonical(seen), canonical(zero), canonical(rare))

    encoder = BioMedBERTEncoder(
        model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        device=dev, max_length=512,
        attention_pooling_strategy=args.go_encoder_inner_pooling,
        attn_hidden=128, attn_dropout=0.1, gradient_checkpointing=False)
    if not expected_lora:
        for param in encoder.parameters():
            param.requires_grad_(False)
    encoder.eval()
    texts = build_go_text_store(args, encoder)
    texts.materialize_tokens_once(batch_size=512, show_progress=True)
    residues = build_residue_store(args)
    dataset = build_datasets(
        args, residues, texts, labels,
        _build_dag(args.go_basic_json)[0] if args.use_dag_in_ds else None,
        zero, rare)['val']
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=ContrastiveEmbCollator())
    sample = dataset[0]
    context = TrainingContext(
        go_cache=cache, device=dev, go_text_store=texts,
        run_name='e5_test_diagnostic', fp16_enabled=args.fp16,
        protein_pooling_strategy=args.protein_pooling_strategy,
        go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,
        go_segment_representation_mode=args.go_segment_representation_mode,
        eval_id_list=list(gos), eval_seen_go_ids=list(seen),
        eval_unseen_ids=list(zero), eval_rare_go_ids=list(rare),
        logger=None, logging=LoggingConfig(log_every=args.log_every))
    cfg = TrainerConfig(
        d_h=int(sample['prot_emb'].shape[1]), d_g=int(cache.embs.shape[1]),
        d_z=args.align_dim, device=str(dev), lr=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        max_epochs=args.epochs, batch_size=args.batch_size,
        eval_batch_size=batch_size, fp16=args.fp16,
        warmstart_path=None, monitor_metric=args.monitor_metric,
        secondary_monitor_metric=args.secondary_monitor_metric,
        monitor_mode=args.monitor_mode,
        protein_pooling_strategy=args.protein_pooling_strategy,
        use_lora=expected_lora, go_pooling=args.go_pooling,
        go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,
        go_segment_representation_mode=args.go_segment_representation_mode,
        eval_go_bs=args.eval_go_bs, temperature=args.temperature,
        lambda_con=args.lambda_con, pbr_lambda=args.pbr_lambda,
        pbr_margin=args.pbr_margin, pbr_tau=args.pbr_tau,
        retrieval_eval_ks=args.retrieval_eval_ks,
        positive_rank_quantiles=args.positive_rank_quantiles,
        log_gradient_norms=False, log_positive_rank_cdf=False,
        log_cardinality_metrics=False, log_go_segment_weights=False)
    trainer = OppTrainer(cfg=cfg, ctx=context, go_encoder=encoder)
    load_model_state(checkpoint, trainer.model, expected_lora)
    trainer.model.eval()
    all_scores, all_labels, all_ids = [], [], []
    with torch.inference_mode():
        for idx, batch in enumerate(loader):
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                enabled=bool(args.fp16 and dev.type == 'cuda')):
                score, _ = trainer._score_full_go(batch)
            all_scores.append(score.float().cpu().numpy())
            all_labels.append(trainer._positive_mask(batch).cpu().numpy().astype(np.uint8))
            all_ids.extend(map(str, batch['protein_ids']))
            if (idx + 1) % 100 == 0:
                print(f'[{"lora" if expected_lora else "frozen"}] batches {idx+1}/{len(loader)}', flush=True)
    score = np.concatenate(all_scores)
    y = np.concatenate(all_labels)
    if score.shape != (len(all_ids), len(gos)) or y.shape != score.shape:
        raise RuntimeError('Prediction matrix shape mismatch')
    if not np.isfinite(score).all():
        raise RuntimeError('Nonfinite model scores')
    if len(all_ids) != 3416:
        raise RuntimeError(f'Expected 3416 test proteins, got {len(all_ids)}; refusing parity claim')
    return dict(scores=score, labels=y, protein_ids=np.asarray(all_ids),
                go_ids=np.asarray(gos, dtype=np.int64))


def original_metrics(original, d, obo, branch, outfile):
    # Same conversion as the existing src.evaluation.pfresgo_exact module.
    s = d['scores'].astype(np.float64)
    s = (1 / (1 + np.exp(-np.clip(s, -80, 80)))).astype(np.float32)
    payload = {'Y_true': d['labels'], 'Y_pred': s,
               'goterms': np.asarray([go_string(g) for g in d['go_ids']]),
               'proteins': d['protein_ids']}
    with open(outfile, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    method = original.Method('our_model', str(outfile), branch, str(obo))
    ix = np.arange(len(d['protein_ids']))
    return {'fmax_test_oracle_diagnostic': float(method.fmax(ix)),
            'macro_aupr': float(method.macro_aupr(ix)),
            'micro_aupr': float(method.micro_aupr(ix)),
            'flattened_auc': float(method.AUC(ix))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-config', required=True)
    p.add_argument('--lora-config', required=True)
    p.add_argument('--frozen-ckpt', required=True)
    p.add_argument('--lora-ckpt', required=True)
    p.add_argument('--test-ids', default='/workspace/stargo/datasets/pfresgo/test.txt')
    p.add_argument('--original-eval', default='/workspace/stargo/pfresgo_eval.py')
    p.add_argument('--obo', default='/workspace/stargo/datasets/pfresgo/go.obo')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    for path in (a.frozen_config, a.lora_config, a.frozen_ckpt, a.lora_ckpt,
                 a.test_ids, a.original_eval, a.obo):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen = extract(a.frozen_config, a.frozen_ckpt, a.test_ids, False, a.device, a.batch_size)
    np.savez_compressed(out/'frozen_e5_test.npz', **frozen)
    if a.device.startswith('cuda'):
        torch.cuda.empty_cache()
    lora = extract(a.lora_config, a.lora_ckpt, a.test_ids, True, a.device, a.batch_size)
    np.savez_compressed(out/'lora_e5_test.npz', **lora)
    ri = strict_order(lora['protein_ids'], frozen['protein_ids'], 'protein')
    ci = strict_order([go_string(x) for x in lora['go_ids']],
                      [go_string(x) for x in frozen['go_ids']], 'GO')
    lora = {**lora, 'scores': lora['scores'][np.ix_(ri, ci)],
            'labels': lora['labels'][np.ix_(ri, ci)],
            'protein_ids': lora['protein_ids'][ri], 'go_ids': lora['go_ids'][ci]}
    if not np.array_equal(frozen['labels'], lora['labels']):
        raise RuntimeError('Frozen/LoRA ground-truth labels differ')
    original = load_original(Path(a.original_eval))
    results = {'protocol': 'original PFresGO evaluator, matched protein/GO universe, epoch index 5',
               'original_eval_sha256': hashlib.sha256(Path(a.original_eval).read_bytes()).hexdigest(),
               'n_proteins': len(frozen['protein_ids']),
               'n_go_terms': len(frozen['go_ids']),
               'note': 'Test Fmax is oracle diagnostic; not validation-selected threshold performance.',
               'models': {}}
    for name, data in (('frozen', frozen), ('lora', lora)):
        results['models'][name] = original_metrics(original, data, a.obo, 'bp', out/f'{name}_pfresgo_input.pckl')
    results['delta_lora_minus_frozen'] = {
        k: results['models']['lora'][k] - results['models']['frozen'][k]
        for k in results['models']['frozen']}
    (out/'results.json').write_text(json.dumps(results, indent=2))
    with (out/'results.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'frozen_e5', 'lora_e5', 'delta'])
        for key in results['models']['frozen']:
            w.writerow([key, results['models']['frozen'][key],
                        results['models']['lora'][key], results['delta_lora_minus_frozen'][key]])
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
