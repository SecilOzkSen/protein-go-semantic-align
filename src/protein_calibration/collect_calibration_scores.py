"""Standalone raw-score exporter. Does not edit main.py or trainer.py.

Usage:
 python -m src.protein_calibration.collect_scores --split valid --out /path/scores.npz -- python -m src.main --config /path/eval.yaml

The command after -- must be your already-working eval_only command configured
for the requested split and the frozen checkpoint. The script intercepts the
trainer's existing evaluation scoring and writes exactly those scores/labels.
"""
import argparse
import runpy
import sys
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['valid', 'test'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expected-go', type=int, default=1943)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if len(command) < 3 or command[:3] != ['python', '-m', 'src.main']:
        parser.error('Expected: -- python -m src.main [your existing eval_only arguments]')

    from src.training.trainer import OppTrainer
    original_eval = OppTrainer.eval_epoch
    captured = {'calls': 0}

    def intercepted_eval(self, loader, epoch_idx):
        scores_list, labels_list = [], []
        original_score = self._score_full_go
        original_mask = self._positive_mask
        pending = []

        def capture_scores(batch):
            result = original_score(batch)
            pending.append(result[0].detach().cpu().float().numpy().copy())
            return result

        def capture_mask(batch):
            mask = original_mask(batch)
            if not pending:
                raise RuntimeError('Unexpected evaluation call order: labels before scores')
            scores_list.append(pending.pop(0))
            labels_list.append(mask.detach().cpu().numpy().astype(np.uint8, copy=True))
            return mask

        self._score_full_go = capture_scores
        self._positive_mask = capture_mask
        try:
            metrics = original_eval(self, loader, epoch_idx)
        finally:
            self._score_full_go = original_score
            self._positive_mask = original_mask
        if pending or not scores_list:
            raise RuntimeError('Score/label capture failed')
        scores = np.concatenate(scores_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        go_ids = np.asarray(self.eval_id_list, dtype=np.int64)
        if scores.shape != labels.shape or scores.shape[1] != args.expected_go:
            raise RuntimeError(f'Unexpected matrices: scores={scores.shape} labels={labels.shape}')
        if not np.isfinite(scores).all():
            raise RuntimeError('Nonfinite scores')
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, scores=scores, labels=labels, go_ids=go_ids)
        captured['calls'] += 1
        print(f'[CALIBRATION EXPORT] split={args.split} saved={out} scores={scores.shape} labels={labels.shape} GO={len(go_ids)}', flush=True)
        return metrics

    OppTrainer.eval_epoch = intercepted_eval
    try:
        sys.argv = ['src.main', *command[3:]]
        runpy.run_module('src.main', run_name='__main__', alter_sys=True)
    finally:
        OppTrainer.eval_epoch = original_eval
    if captured['calls'] != 1:
        raise RuntimeError(f'Expected exactly one eval_epoch call, got {captured["calls"]}')


if __name__ == '__main__':
    main()
