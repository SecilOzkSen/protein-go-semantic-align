"""Real-data, one-batch Experiment H smoke test, with NO optimizer update.

Place at src/diagnostics/experiment_h_one_batch_smoke.py and run:
python -m src.diagnostics.experiment_h_one_batch_smoke --config src/pfresgo_experiment_h_bp.yaml

Uses the project's normal run_training setup to construct the real dataset,
GO segment bank and parent bank. Intercepts the first training batch before
optimizer.backward/step. No checkpoint is saved by this test.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src import main as retriever_main
from src.training.trainer import OppTrainer, multi_positive_full_go_infonce, positive_block_rank_loss


class _SmokeCompleted(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default=None, help='Optional JSON report path')
    args = ap.parse_args()

    cfg = retriever_main.load_structured_cfg(args.config)
    if str(cfg.evaluation_split).lower() != 'valid':
        raise ValueError('Safety: pfresgo.evaluation_split must be valid, not test.')
    if cfg.warmstart_path or cfg.resume:
        raise ValueError('Safety: clear warmstart_path and resume for this smoke test.')
    cfg.eval_only = False
    cfg.wandb = False
    cfg.epochs = 1
    # Normal initialization only. The intercepted first batch aborts before optimizer.step.
    original = OppTrainer.step_losses
    seen = {'done': False}
    report = {}

    def intercept(self, batch, epoch_idx, debug=False):
        if seen['done']:
            raise RuntimeError('Unexpected second training batch')
        seen['done'] = True
        self.model.train()
        self.model.go_encoder.eval()
        self.opt.zero_grad(set_to_none=True)

        H = batch['prot_emb_pad']
        residue_mask = batch['prot_attn_mask'].bool()
        B = int(H.shape[0])
        lengths = residue_mask.sum(1).tolist()
        assert B > 0 and min(lengths) > 0, 'Empty protein or residue mask'
        assert not residue_mask[:, 0].logical_not().any(), 'Unexpected left padding'
        assert len(self.eval_id_list) == 1943, 'Expected PFresGO BP benchmark universe of 1943'

        parent_mask = self._parent_present_cpu
        parent_bank = self._parent_bank_cpu
        assert parent_mask.shape == parent_bank.shape[:2]
        assert parent_mask.shape[0] == 1943
        assert torch.isfinite(parent_bank).all()
        assert not (parent_mask.sum(1) > parent_mask.shape[1]).any()
        assert 'name' in self._segment_names and 'definition' in self._segment_names

        scores, weights = self._score_full_go(batch)
        y = self._positive_mask(batch)
        assert scores.shape == y.shape == (B, 1943), (scores.shape, y.shape)
        assert torch.isfinite(scores).all(), 'Nonfinite scores'
        assert (y.sum(1) > 0).all(), 'Protein missing positives'
        assert torch.isfinite(weights).all()

        l_con = multi_positive_full_go_infonce(scores, y, self.cfg.temperature)
        l_pbr, _ = positive_block_rank_loss(scores, y, self.cfg.pbr_margin, self.cfg.pbr_tau)
        total = self.cfg.lambda_con * l_con + self.cfg.pbr_lambda * l_pbr
        assert torch.isfinite(total), 'Nonfinite loss'
        # Check gradients for the actual training objective, without optimizer step.
        trainables = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        grads = torch.autograd.grad(total, [p for _, p in trainables], allow_unused=True)
        norms = {n: (float(g.detach().float().norm().item()) if g is not None else 0.0)
                 for (n, _), g in zip(trainables, grads)}
        required = ('proj_p', 'proj_local', 'proj_g', 'fusion_gate')
        failures = [prefix for prefix in required if not any(n.startswith(prefix + '.') and v > 0 and math.isfinite(v) for n, v in norms.items())]
        if failures:
            raise AssertionError('Missing/zero gradients for: ' + ', '.join(failures))
        if any(not math.isfinite(v) for v in norms.values()):
            raise AssertionError('Nonfinite gradient norm')

        hdiag = dict(getattr(self.model, 'last_h_diagnostics', {}))
        for k, v in hdiag.items():
            if isinstance(v, (int, float)) and not math.isfinite(float(v)):
                raise AssertionError(f'Nonfinite diagnostic: {k}={v}')
        if 'h/window_count_mean' not in hdiag or hdiag['h/window_count_mean'] <= 0:
            raise AssertionError('Window extraction diagnostic missing/invalid')
        report.update({
            'status': 'PASS', 'optimizer_steps': 0, 'proteins': B,
            'residue_lengths': list(map(int, lengths)), 'go_terms': 1943,
            'go_segments': list(self._segment_names),
            'parent_bank_shape': list(parent_bank.shape),
            'go_with_parents': int(parent_mask.any(1).sum().item()),
            'mean_parents_per_go': float(parent_mask.sum(1).float().mean().item()),
            'positive_counts': y.sum(1).tolist(),
            'score_shape': list(scores.shape),
            'score_range': [float(scores.min().item()), float(scores.max().item())],
            'contrastive_loss': float(l_con.item()), 'pbr_loss': float(l_pbr.item()),
            'total_loss': float(total.item()), 'diagnostics': hdiag,
            'gradient_norms_by_module': {prefix: sum(v for n, v in norms.items() if n.startswith(prefix + '.')) for prefix in required},
        })
        raise _SmokeCompleted()

    OppTrainer.step_losses = intercept
    try:
        try:
            retriever_main.run_training(cfg)
        except _SmokeCompleted:
            pass
        else:
            raise RuntimeError('No training batch intercepted. Smoke test did not run.')
    finally:
        OppTrainer.step_losses = original

    print('EXPERIMENT H REAL-DATA ONE-BATCH SMOKE PASSED (no optimizer step)')
    print(json.dumps(report, indent=2, default=str))
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
        print('SAVED', dest)


if __name__ == '__main__':
    main()
