"""Evaluate exactly the checkpoint supplied with --checkpoint.

Run in the existing repository. Data root, ontology branch, architecture and
batch size are read from the checkpoint. Its run_manifest.json is found beside
it. MZSGO protein-centric unseen AUPR is the primary validation metric.
Results are written beside the checkpoint as <stem>.zero_shot.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .zero_shot_eval_io import load_cohort, validate_checkpoint_meta, sha256_file
from .zero_shot_eval_metrics import evaluate_rankings
from .zero_shot_eval_mzsgo_metrics import evaluate_mzsgo
from .zero_shot_eval_model import resolve_loader, read_checkpoint_meta, score_checkpoint


def default_repo_root():
    """Find src's enclosing repository at any script nesting depth."""
    for directory in Path(__file__).resolve().parents:
        if (directory / 'src').is_dir():
            return directory
    return Path.cwd().resolve()


def cosine_unit_scores(scores, architecture, runtime):
    """One fixed, unfitted cosine adapter for the MZSGO threshold grid.

    Official MZSGO applies sigmoid to classifier logits. Our cosine scorer has
    different output semantics, so we explicitly map its theoretical [-1,1]
    range to [0,1]. This is an adapter for our model, not their logit conversion.
    Raw scores remain untouched for retrieval and term-AP diagnostics.
    """
    if architecture.get('normalize') is not True:
        raise ValueError('The fixed cosine adapter requires architecture.normalize=True')
    if float(architecture.get('parent_strength', float('nan'))) != 0.0:
        raise ValueError('The bounded cosine adapter requires parent adjustment OFF')
    reduced_precision = runtime.get('autocast_dtype') == 'torch.bfloat16'
    tolerance = 0.01 if reduced_precision else 1e-5
    values = np.asarray(scores)
    if values.dtype.kind != 'f' or not np.isfinite(values).all():
        raise ValueError('Expected finite floating-point similarity scores')
    smallest, largest = float(values.min()), float(values.max())
    if smallest < -1.0 - tolerance or largest > 1.0 + tolerance:
        raise ValueError(f'Scores exceed the bounded cosine range: [{smallest}, {largest}], tolerance={tolerance}')
    clipped_count = int(np.count_nonzero((values < -1.0) | (values > 1.0)))
    unit = values.astype(np.float64, copy=True)
    np.clip(unit, -1.0, 1.0, out=unit)
    unit += 1.0
    unit *= 0.5
    return unit, {
        'name': 'fixed_cosine_affine_v1', 'formula': '(clip(s, -1, 1) + 1) / 2',
        'fitted_parameters': False, 'per_protein_normalization': False,
        'raw_min': smallest, 'raw_max': largest, 'bound_tolerance': tolerance,
        'clipped_count': clipped_count,
        'max_clipped_overshoot': max(0.0, -1.0-smallest, largest-1.0),
        'output_dtype': str(unit.dtype),
        'reference_model_conversion': 'sigmoid(classifier_logits)',
        'reference_conversion_reused': False,
        'note': 'Same fixed adapter across all ablations. The reference 101-point grid is sensitive to score transformations.',
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path, help='The exact checkpoint to evaluate')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path, help='Result JSON path; defaults beside the checkpoint')
    p.add_argument('--root', type=Path, help='Data root override; normally read from checkpoint')
    p.add_argument('--cohort-dir', type=Path, help='Prepared cohort override')
    p.add_argument('--repo-root', type=Path, default=default_repo_root())
    p.add_argument('--run-manifest', type=Path, help='Only needed if the checkpoint was moved away from its manifest')
    p.add_argument('--model-module', help='Optional original inference module override')
    p.add_argument('--ks', nargs='+', type=int, default=[50, 100, 200, 500, 1000])
    p.add_argument('--torch-threads', type=int, default=4)
    return p


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def execute(args):
    started = time.perf_counter()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f'Checkpoint not found: {checkpoint}')
    if args.torch_threads < 1 or any(k < 1 for k in args.ks):
        raise ValueError('Thread count and K values must be positive')
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use cpu or cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use the training environment or --device cpu')

    manifest_path = (args.run_manifest or checkpoint.parent / 'run_manifest.json').expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f'Run manifest not found beside checkpoint. Supply --run-manifest /path/run_manifest.json: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    repo_root = args.repo_root.expanduser().resolve()
    loader, source_audit = resolve_loader(manifest, repo_root, args.model_module)
    print(f"[LOADER] {loader} | training source hashes matched", flush=True)
    checkpoint_hash = sha256_file(checkpoint)
    meta = read_checkpoint_meta(loader, checkpoint)
    config = meta['config']
    branch = meta['data_identity']['branch']
    if branch not in ('MF', 'BP', 'CC') or config.get('branch') != branch:
        raise ValueError('Checkpoint has inconsistent ontology branch metadata')
    if meta['architecture'] != manifest.get('architecture'):
        raise ValueError('Checkpoint architecture differs from its run manifest')
    saved_root = args.root or config.get('root')
    if not saved_root and args.cohort_dir is None:
        raise ValueError('Checkpoint has no data root; supply --root or --cohort-dir')
    cohort_path = args.cohort_dir or Path(saved_root) / 'analysis' / f'zero_shot_validation_{branch.lower()}'
    cohort = load_cohort(Path(cohort_path).expanduser().resolve())
    audit = validate_checkpoint_meta(meta, cohort)
    output = (args.output or checkpoint.with_suffix('.zero_shot.json')).expanduser().resolve()
    protected = {checkpoint, manifest_path, Path(cohort['summary_path']).resolve()}
    protected.update(Path(p).expanduser().resolve() for p in cohort['summary']['inputs'].values())
    if output in protected:
        raise ValueError('Choose a result path separate from the checkpoint and input data')

    print(f'[CHECKPOINT] {checkpoint}', flush=True)
    print(f"[MODEL] ablation={config['ablation_id']} branch={branch} epoch={meta['epoch']}", flush=True)
    print(f"[COHORT] proteins={len(cohort['protein_ids'])} active_go={len(cohort['go_ids'])} unseen_go={len(cohort['unseen_go_ids'])}", flush=True)
    scores, loaded_meta, runtime = score_checkpoint(
        checkpoint, loader, cohort, device, repo_root, expected_meta=meta, log_every=10)
    if loaded_meta != meta or sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed during evaluation')
    index = {go: j for j, go in enumerate(cohort['go_ids'])}
    truth = np.zeros(scores.shape, dtype=bool)
    for i, pid in enumerate(cohort['protein_ids']):
        for go in cohort['labels_full'][pid]:
            truth[i, index[go]] = True
    metrics, term_rows, protein_rows = evaluate_rankings(
        scores, truth, cohort['go_ids'], cohort['seen_go_ids'], cohort['unseen_go_ids'],
        cohort['protein_ids'], ks=sorted(set(args.ks)))
    probabilities, score_adapter = cosine_unit_scores(scores, meta['architecture'], runtime)
    print('[METRIC] MZSGO protein-centric AUPR, fixed cosine adapter, 101 thresholds', flush=True)
    mzsgo = evaluate_mzsgo(probabilities, truth, cohort['go_ids'], cohort['seen_go_ids'], cohort['unseen_go_ids'])
    for group in ('all', 'seen', 'unseen'):
        evaluation = mzsgo[group]
        metrics[group]['aupr'] = None if evaluation is None else evaluation['aupr']
        metrics[group]['mzsgo_fmax'] = None if evaluation is None else evaluation['fmax']
        metrics[group]['mzsgo_aupr_n_proteins'] = 0 if evaluation is None else evaluation['n_proteins']
        metrics[group]['mzsgo_aupr_n_terms'] = 0 if evaluation is None else evaluation['n_terms']
    recorded = meta.get('validation_metrics', {}).get('protein_wise_map')
    actual = metrics['seen']['protein_wise_map']
    delta = None if recorded is None or actual is None else float(actual - recorded)
    if delta is not None and not math.isfinite(delta):
        raise ValueError('Nonfinite seen mAP comparison')
    parity = 'not_available' if delta is None else ('matched_within_1e-5' if abs(delta) <= 1e-5 else 'review_score_or_metric_parity')
    result = {
        'format': 'single_checkpoint_natural_zero_positive_validation_v2_mzsgo_aupr',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
        'ablation': config['ablation_id'], 'branch': branch, 'epoch': meta['epoch'],
        'completed_epochs': meta['epoch'] + 1, 'architecture': meta['architecture'],
        'cohort_counts': cohort['summary']['counts'], 'cohort_signature': cohort['signature'],
        'metrics': metrics, 'primary_metric': 'unseen/aupr',
        'mzsgo_evaluation': mzsgo, 'score_adapter': score_adapter,
        'seen_map_parity': {'recorded': recorded, 'actual': actual, 'difference': delta, 'status': parity},
        'term_metrics': term_rows, 'protein_metrics': protein_rows,
        'source_audit': source_audit, 'checkpoint_identity_audit': audit, 'runtime': runtime,
        'elapsed_seconds': time.perf_counter() - started,
        'metric_note': 'AUPR uses original MZSGO protein-centric subset masks and 101-threshold integration with our fixed cosine adapter. Retrieval ranks remain over all active GO; macro term AP retains all validation proteins. These are different metrics.',
        'unseen_note': 'Zero training-positive count, not a certificate of absence from the training loss. Reserved temporal GO excluded.',
    }
    atomic_json(output, result)
    for group in ('unseen', 'seen', 'all'):
        values = metrics[group]
        details = [f"AUPR={values['aupr']}", f"Fmax={values['mzsgo_fmax']}", f"macro_AP={values['macro_term_ap']}", f"mAP={values['protein_wise_map']}"]
        details.extend(f"R@{k}={values[f'protein_recall@{k}']}" for k in sorted(set(args.ks)))
        print(f"[{group.upper()}] " + ' | '.join(details), flush=True)
    print(f"[PRIMARY] unseen/AUPR={metrics['unseen']['aupr']} | H={mzsgo['harmonic_mean']}", flush=True)
    print(f'[SEEN PARITY] {parity} difference={delta}', flush=True)
    print(f'[SAVED] {output}', flush=True)
    return 0


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        logging.error('%s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
