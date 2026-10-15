"""No-training diagnostic: global F1 cosine versus window-local MaxSim.

Run once with F1 eval_only config for valid, once for test.
Projection was trained on pooled vectors, NOT windows: exploratory evidence only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src import main as retriever_main
from src.training.trainer import OppTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max_proteins', type=int, default=300)
    parser.add_argument('--window', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    args = parser.parse_args()
    cfg = retriever_main.load_structured_cfg(args.config)
    if not cfg.eval_only or not cfg.warmstart_path or cfg.resume:
        raise ValueError('Use eval_only=true, F1 best_primary.pt warmstart, resume=null')
    out = Path(args.out);
    out.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out)
    cfg.wandb = False
    retriever_main.setup_logging(out)
    retriever_main.set_seed(cfg.seed)
    original_eval = OppTrainer.eval_epoch

    def capture(self, loader, epoch_idx):
        original_score = self._score_full_go
        rows = {'global': [], 'local': [], 'labels': [], 'ids': []}
        go_bank, _ = self._project_go_bank()
        go_bank = go_bank.detach().float()
        n = 0

        def wrapped(batch):
            nonlocal n
            scores, weights = original_score(batch)
            B = scores.shape[0]
            if n >= args.max_proteins:
                return scores, weights
            H = batch['prot_emb_pad'].to(self.device).float()
            mask = batch['prot_attn_mask'].to(self.device).bool()
            labels = self._positive_mask(batch).detach().cpu().numpy().astype(np.uint8)
            for i in range(min(B, args.max_proteins - n)):
                L = int(mask[i].sum().item())
                if L == 0:
                    raise RuntimeError('Empty protein sequence')
                windows = []
                starts = list(range(0, max(1, L - args.window + 1), args.stride))
                last = max(0, L - args.window)
                if not starts or starts[-1] != last:
                    starts.append(last)
                for start in starts:
                    segment = H[i, start:min(start + args.window, L)]
                    windows.append(segment.mean(dim=0))
                x = torch.stack(windows)
                # Same learned projection, but on window means: OOD relative to training.
                z = F.normalize(self.model.proj_p(x).float(), dim=-1)
                local = (z @ go_bank.T).max(dim=0).values
                rows['global'].append(scores[i].detach().float().cpu().numpy())
                rows['local'].append(local.detach().cpu().numpy())
                rows['labels'].append(labels[i])
                rows['ids'].append(str(batch['protein_ids'][i]))
                n += 1
            return scores, weights

        self._score_full_go = wrapped
        try:
            metrics = original_eval(self, loader, epoch_idx)
        finally:
            self._score_full_go = original_score
        if not rows['ids']:
            raise RuntimeError('No proteins captured')
        dest = out / 'global_local.npz'
        np.savez_compressed(dest, global_scores=np.stack(rows['global']),
                            local_scores=np.stack(rows['local']), labels=np.stack(rows['labels']),
                            protein_ids=np.asarray(rows['ids']), go_ids=np.asarray(self.eval_id_list))
        print('SAVED', dest, 'proteins=', n)
        return metrics

    OppTrainer.eval_epoch = capture
    try:
        retriever_main.run_training(cfg)
    finally:
        OppTrainer.eval_epoch = original_eval


if __name__ == '__main__':
    with torch.no_grad():
        main()
