"""Experiment G: one-pass validation-only residual-scale ablation.

Run as: python -m src.diagnostics.experiment_g_residual_ablation --config ... --checkpoint ... --out ...
Uses the project's normal eval_only loader, one full evaluation pass, no training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from src import main as retriever_main
from src.training.trainer import OppTrainer
from src.metrics.cafa import compute_protein_centric_fmax, compute_term_aupr

SCALES = (0.0, 0.02, 1.0)


def summaries(scores, labels):
    y = labels.astype(bool)
    counts = y.sum(axis=1)
    output = {}
    for k in (50, 100, 200, 500):
        kk = min(k, scores.shape[1])
        top = np.argpartition(-scores, kth=kk - 1, axis=1)[:, :kk]
        hits = np.take_along_axis(y, top, axis=1).sum(axis=1)
        output[f'protein_mean_R@{k}'] = float(np.mean(hits / np.maximum(counts, 1)))
    fmax, threshold = compute_protein_centric_fmax(labels.astype(np.int32), scores.astype(np.float32), num_thresholds=101)
    output['obs_protein_fmax'] = float(fmax)
    output['obs_protein_fmax_threshold'] = float(threshold)
    output['obs_aupr'] = float(compute_term_aupr(labels.astype(np.int32), scores.astype(np.float32)))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = retriever_main.load_structured_cfg(args.config)
    if str(cfg.evaluation_split).lower() != 'valid':
        raise ValueError('Ablation is validation-only. Set pfresgo.evaluation_split: valid')
    cfg.eval_only = True
    cfg.resume = None
    cfg.warmstart_path = str(Path(args.checkpoint).resolve())
    cfg.output_dir = str(out)
    cfg.wandb = False
    retriever_main.setup_logging(out)
    retriever_main.set_seed(cfg.seed)

    original_eval = OppTrainer.eval_epoch
    results = {}

    def wrapped_eval(self, loader, epoch_idx):
        scorer = self.model.interaction_scorer
        if not hasattr(scorer, 'residual_scale'):
            raise RuntimeError('Experiment G interaction_scorer.residual_scale not found')
        original_scale = scorer.residual_scale
        original_score = self._score_full_go
        collected = {s: [] for s in SCALES}
        labels = []
        pids = []
        correction_sum = 0.0
        correction_max = 0.0
        correction_count = 0
        try:
            def intercept(batch):
                nonlocal correction_sum, correction_max, correction_count
                zp = self._encode_protein(batch)
                zg, weights = self._project_go_bank()
                cosine = (zp.float() @ zg.float().T).detach()
                scorer.residual_scale = 1.0
                full = self.model.score_projected(zp, zg).detach()
                delta = full - cosine
                correction_sum += float(delta.abs().sum().item())
                correction_max = max(correction_max, float(delta.abs().max().item()))
                correction_count += delta.numel()
                for scale in SCALES:
                    collected[scale].append((cosine + scale * delta).float().cpu().numpy())
                labels.append(self._positive_mask(batch).cpu().numpy().astype(np.uint8))
                pids.extend(map(str, batch['protein_ids']))
                # Return the original trained scale to the native eval.
                scorer.residual_scale = original_scale
                return cosine + original_scale * delta, weights

            self._score_full_go = intercept
            native_metrics = original_eval(self, loader, epoch_idx)
        finally:
            scorer.residual_scale = original_scale
            self._score_full_go = original_score

        y = np.concatenate(labels)
        if len(pids) != y.shape[0] or len(set(pids)) != len(pids):
            raise RuntimeError('Protein ID count mismatch or duplicates')
        if y.shape[1] != len(self.eval_id_list):
            raise RuntimeError('GO column count mismatch')
        scores_by_scale = {scale: np.concatenate(chunks) for scale, chunks in collected.items()}
        for scale, scores in scores_by_scale.items():
            if scores.shape != y.shape or not np.isfinite(scores).all():
                raise RuntimeError(f'Invalid scores for scale {scale}')
        # Compare to the native metric at the trained scale as a parity check.
        trained = scores_by_scale[0.02]
        native_f = float(native_metrics['obs_protein_fmax'])
        check_f = summaries(trained, y)['obs_protein_fmax']
        if abs(native_f - check_f) > 2e-4:
            raise RuntimeError(f'Native metric parity failed: {native_f} vs {check_f}')

        for scale, scores in scores_by_scale.items():
            results[str(scale)] = summaries(scores, y)
        base = scores_by_scale[0.0]
        base_top = np.argsort(-base, axis=1)[:, :50]
        for scale in (0.02, 1.0):
            other_top = np.argsort(-scores_by_scale[scale], axis=1)[:, :50]
            overlap = np.mean([len(set(a) & set(b)) / 50 for a, b in zip(base_top, other_top)])
            results[str(scale)]['top50_overlap_with_cosine'] = float(overlap)
        results['correction'] = {
            'mean_abs_raw_delta': correction_sum / max(correction_count, 1),
            'max_abs_raw_delta': correction_max,
            'mean_abs_trained_correction': 0.02 * correction_sum / max(correction_count, 1),
            'trained_scale': original_scale,
        }
        results['metadata'] = {'split': 'valid', 'checkpoint': str(cfg.warmstart_path),
                               'proteins': len(pids), 'go_terms': y.shape[1],
                               'note': 'scale=1.0 is sensitivity-only, not a trained configuration.'}
        (out / 'residual_ablation.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        print('SAVED', out / 'residual_ablation.json')
        return native_metrics

    OppTrainer.eval_epoch = wrapped_eval
    try:
        retriever_main.run_training(cfg)
    finally:
        OppTrainer.eval_epoch = original_eval


if __name__ == '__main__':
    main()
