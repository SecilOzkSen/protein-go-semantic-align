"""F1 checkpoint pooling intervention diagnostic, no training.

Run with the SAME YAML used for F1 evaluation (eval_only=true,
 warmstart_path=.../best_primary.pt). Run separately on valid and test.
"""
import argparse
import json
from pathlib import Path
from types import MethodType

import torch

from src import main as retriever_main
from src.training.trainer import OppTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    cfg = retriever_main.load_structured_cfg(args.config)
    if args.device:
        cfg.general_device = args.device
    if not cfg.eval_only or not cfg.warmstart_path:
        raise ValueError('Set training.eval_only=true and warmstart_path=F1 best_primary.pt')
    if cfg.resume:
        raise ValueError('Set training.resume=null')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out)
    cfg.wandb = False
    retriever_main.setup_logging(out)
    retriever_main.set_seed(cfg.seed)

    original_eval = OppTrainer.eval_epoch

    def compare(self, loader, epoch_idx):
        pool = self.model.protein_mean_attn_gate_pool
        original_forward = pool.forward
        results = {}
        gates = []

        def make_forward(mode):
            def replacement(this, x, mask=None):
                h_mean = this.masked_mean(x, mask)
                h_attn, alpha = this.attn_pool(x, mask)
                gate = torch.sigmoid(this.gate(torch.cat([h_mean, h_attn], dim=-1)))
                if mode == 'gated':
                    h = (1 - gate) * h_mean + gate * h_attn
                    gates.append(gate.detach().float().cpu())
                elif mode == 'mean_only':
                    h = h_mean
                else:
                    h = h_attn
                return h, {'protein_attn_alpha': alpha, 'protein_attn_gate': gate}

            return replacement

        try:
            for mode in ('gated', 'mean_only', 'attention_only'):
                pool.forward = MethodType(make_forward(mode), pool)
                metrics = original_eval(self, loader, epoch_idx)
                results[mode] = {k: float(v) for k, v in metrics.items()
                                 if isinstance(v, (float, int))}
                print(f'[{mode}] ' + ' | '.join(
                    f'{k}={metrics[k]:.5f}' for k in
                    ('align_R@50', 'align_R@100', 'align_R@200',
                     'align_R@500', 'positive_block_coverage',
                     'obs_protein_fmax', 'obs_aupr')))
        finally:
            pool.forward = original_forward

        if gates:
            v = torch.cat(gates).flatten()
            results['gate_distribution'] = {
                'mean': float(v.mean()), 'median': float(v.median()),
                'p10': float(torch.quantile(v, .1)),
                'p90': float(torch.quantile(v, .9)),
                'min': float(v.min()), 'max': float(v.max()),
            }
        results['metadata'] = {
            'checkpoint': str(cfg.warmstart_path),
            'split': str(cfg.evaluation_split),
            'note': 'Mean/attention modes are inference interventions using a projection trained on gated pooling. Not retrained ablations.',
        }
        (out / 'pooling_ablation.json').write_text(json.dumps(results, indent=2))
        print('SAVED', out / 'pooling_ablation.json')
        # Main expects a single metric dictionary for its ordinary eval-only logging.
        return results['gated']

    OppTrainer.eval_epoch = compare
    try:
        retriever_main.run_training(cfg)
    finally:
        OppTrainer.eval_epoch = original_eval


if __name__ == '__main__':
    main()
