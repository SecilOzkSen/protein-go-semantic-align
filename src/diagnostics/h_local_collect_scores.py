"""Collect exact validation/test score matrices for Experiment H Local-Only.
Uses the project's existing evaluation path. No training or optimizer updates.
Requires the corrected src/utils/checkpoint.py when loading via resume.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from src import main as retriever_main
from src.training.trainer import OppTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', choices=['valid', 'test'], required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = retriever_main.load_structured_cfg(args.config)
    cfg.eval_only = True
    cfg.evaluation_split = args.split
    cfg.resume = args.checkpoint
    cfg.warmstart_path = None
    cfg.wandb = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out)
    retriever_main.setup_logging(out)
    retriever_main.set_seed(cfg.seed)

    original_eval = OppTrainer.eval_epoch

    def capture(self, loader, epoch_idx):
        original_score = self._score_full_go
        scores_all, labels_all, pids = [], [], []

        def intercepted(batch):
            scores, weights = original_score(batch)
            scores_all.append(scores.detach().float().cpu().numpy())
            labels_all.append(self._positive_mask(batch).detach().cpu().numpy().astype(np.uint8))
            pids.extend(map(str, batch['protein_ids']))
            return scores, weights

        self._score_full_go = intercepted
        try:
            metrics = original_eval(self, loader, epoch_idx)
        finally:
            self._score_full_go = original_score
        if not scores_all:
            raise RuntimeError('No evaluation batches captured')
        scores = np.concatenate(scores_all)
        labels = np.concatenate(labels_all)
        go_ids = np.asarray(self.eval_id_list, dtype=np.int64)
        if scores.shape != labels.shape or scores.shape != (len(pids), len(go_ids)):
            raise RuntimeError('Scores/labels/protein IDs/GO IDs shape mismatch')
        if not np.isfinite(scores).all():
            raise RuntimeError('Nonfinite scores')
        np.savez_compressed(out / 'scores.npz', scores=scores, labels=labels,
                            protein_ids=np.asarray(pids, dtype=str), go_ids=go_ids)
        (out / 'metadata.json').write_text(json.dumps({
            'split': args.split, 'checkpoint': args.checkpoint,
            'n_proteins': len(pids), 'n_go': len(go_ids),
            'metrics': {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        }, indent=2))
        print('SAVED', out / 'scores.npz')
        return metrics

    OppTrainer.eval_epoch = capture
    try:
        retriever_main.run_training(cfg)
    finally:
        OppTrainer.eval_epoch = original_eval


if __name__ == '__main__':
    main()
