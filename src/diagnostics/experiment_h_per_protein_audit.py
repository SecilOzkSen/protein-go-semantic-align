"""Experiment H per-protein global versus fused retrieval audit, validation only.

Place at src/diagnostics/experiment_h_per_protein_audit.py.
Run: python -m src.diagnostics.experiment_h_per_protein_audit --config src/pfresgo_experiment_h_bp.yaml --checkpoint /path/to/best_primary.pt --out /path/to/per_protein_audit.json

No optimizer step. Does not select thresholds using test data.
"""
from __future__ import annotations
import argparse
import json
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src import main as retriever_main
from src.training.trainer import OppTrainer
from src.metrics.cafa import compute_protein_centric_fmax, compute_term_aupr


class _Finished(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    cfg = retriever_main.load_structured_cfg(args.config)
    if str(cfg.evaluation_split).lower() != 'valid':
        raise ValueError('Only validation is permitted for architecture ablation')
    cfg.wandb = False
    cfg.resume = None
    cfg.warmstart_path = str(ckpt_path)
    cfg.eval_only = True
    cfg.epochs = 1

    def evaluate(self, loader, epoch_idx):
        self.model.eval()
        self.model.go_encoder.eval()
        results = {k: {'scores': [], 'true': []} for k in ('global_only', 'local_only', 'fused', 'fused_parents_off')}
        diagnostics = {'alpha_sum': 0., 'alpha_n': 0, 'parent_abs_sum': 0., 'parent_n': 0}
        protein_rows = []
        with torch.no_grad():
            raw, _ = self._current_go_raw_bank()
            seg = self._segment_embs_cpu.to(self.device)
            present = self._segment_present_cpu.to(self.device)
            name_idx = self._segment_names.index('name')
            def_idx = self._segment_names.index('definition')
            gn = self.model.project_go(seg[:, name_idx]).float()
            gd = self.model.project_go(seg[:, def_idx]).float()
            zg = self.model.project_go(raw).float()
            parent = self._parent_bank_cpu.to(self.device)
            parent_mask = self._parent_present_cpu.to(self.device).bool()
            # Parent projections are computed once, not once per protein batch.
            gp = self.model.project_go(parent.reshape(-1, parent.shape[-1])).float().reshape(parent.shape[0], parent.shape[1], -1)
            chunk_size = int(self.model.go_chunk_size)
            for batch in loader:
                H = batch['prot_emb_pad'].to(self.device).float()
                mask = batch['prot_attn_mask'].to(self.device).bool()
                y = self._positive_mask(batch).cpu().numpy().astype(np.int8)
                zp = self.model.encode_protein_for_scoring(H, mask).float()
                zw, valid = self.model.encode_local_windows(H, mask)
                base = zp @ zg.T
                local_parts, no_parent_parts, alpha_parts, adj_parts = [], [], [], []
                B, G = base.shape
                for st in range(0, G, chunk_size):
                    ed = min(G, st + chunk_size)
                    sn = self.model.mil(torch.einsum('bkd,gd->bkg', zw, gn[st:ed]), valid)
                    sd = self.model.mil(torch.einsum('bkd,gd->bkg', zw, gd[st:ed]), valid)
                    child = torch.where(present[st:ed, def_idx][None, :], .5 * (sn + sd), sn)
                    pm = parent_mask[st:ed]
                    if pm.any():
                        ps = torch.einsum('bkd,cpd->bkcp', zw, gp[st:ed])
                        C, P = pm.shape
                        flat = ps.permute(0, 3, 1, 2).reshape(B * P, zw.shape[1], C)
                        vm = valid[:, None, :].expand(B, P, valid.shape[1]).reshape(B * P, -1)
                        pmil = self.model.mil(flat, vm).reshape(B, P, C).permute(0, 2, 1)
                        pmil = pmil.masked_fill(~pm[None, :, :], 0.)
                        avg = pmil.sum(-1) / pm.sum(-1).clamp_min(1)[None, :]
                        adj = -self.model.parent_strength * pm.any(-1)[None, :] * F.relu(child - avg)
                    else:
                        adj = torch.zeros_like(child)
                    local = child + adj
                    features = torch.stack((base[:, st:ed], local, local - base[:, st:ed]), -1)
                    alpha = torch.sigmoid(self.model.fusion_gate(features)).squeeze(-1)
                    # parents-off recomputes the gate from the parent-free local score.
                    feat_off = torch.stack((base[:, st:ed], child, child - base[:, st:ed]), -1)
                    alpha_off = torch.sigmoid(self.model.fusion_gate(feat_off)).squeeze(-1)
                    local_parts.append(local)
                    no_parent_parts.append((1 - alpha_off) * base[:, st:ed] + alpha_off * child)
                    alpha_parts.append(alpha)
                    adj_parts.append(adj)
                local = torch.cat(local_parts, dim=1)
                alpha = torch.cat(alpha_parts, dim=1)
                fused = (1 - alpha) * base + alpha * local
                off = torch.cat(no_parent_parts, dim=1)
                diagnostics['alpha_sum'] += alpha.sum().item()
                diagnostics['alpha_n'] += alpha.numel()
                diagnostics['parent_abs_sum'] += torch.cat(adj_parts, dim=1).abs().sum().item()
                diagnostics['parent_n'] += alpha.numel()
                # Per-protein, per-GO retrieval diagnostics, computed without changing scores.
                for b in range(B):
                    yt = y[b].astype(bool)
                    global_s = base[b].detach().float().cpu().numpy()
                    fused_s = fused[b].detach().float().cpu().numpy()
                    global_order = np.argsort(-global_s, kind='stable')
                    fused_order = np.argsort(-fused_s, kind='stable')
                    row = {
                        'protein_id': str(batch['protein_ids'][b]),
                        'residue_length': int(mask[b].sum().item()),
                        'cardinality': int(yt.sum()),
                        'mean_local_weight': float(alpha[b].mean().item()),
                        'mean_abs_local_global_gap': float((local[b] - base[b]).abs().mean().item()),
                        'mean_abs_parent_adjustment': float(torch.cat(adj_parts, dim=1)[b].abs().mean().item()),
                    }
                    for k in (50, 100, 200, 500):
                        g_hits = set(int(i) for i in global_order[:k] if yt[i])
                        f_hits = set(int(i) for i in fused_order[:k] if yt[i])
                        denom = max(1, int(yt.sum()))
                        row[f'global_R@{k}'] = len(g_hits) / denom
                        row[f'fused_R@{k}'] = len(f_hits) / denom
                        row[f'delta_R@{k}'] = (len(f_hits) - len(g_hits)) / denom
                        row[f'gain_count@{k}'] = len(f_hits - g_hits)
                        row[f'loss_count@{k}'] = len(g_hits - f_hits)
                        row[f'gain_go_ids@{k}'] = ';'.join(str(self.eval_id_list[i]) for i in sorted(f_hits - g_hits))
                        row[f'loss_go_ids@{k}'] = ';'.join(str(self.eval_id_list[i]) for i in sorted(g_hits - f_hits))
                    protein_rows.append(row)
                for key, scores in [('global_only', base), ('local_only', local), ('fused', fused), ('fused_parents_off', off)]:
                    results[key]['scores'].append(scores.float().cpu().numpy())
                    results[key]['true'].append(y)
        output = {'metadata': {'split': 'valid', 'checkpoint': str(ckpt_path), 'go_terms': len(self.eval_id_list)}, 'diagnostics': {
            'mean_local_fusion_weight': diagnostics['alpha_sum'] / max(1, diagnostics['alpha_n']),
            'mean_abs_parent_adjustment': diagnostics['parent_abs_sum'] / max(1, diagnostics['parent_n'])}}
        for key, values in results.items():
            s = np.concatenate(values['scores'], axis=0).astype(np.float32)
            y = np.concatenate(values['true'], axis=0).astype(np.int32)
            if not np.isfinite(s).all(): raise RuntimeError(f'Nonfinite {key}')
            order = np.argsort(-s, axis=1)
            counts = np.maximum(y.sum(1), 1)
            entry = {}
            for k in (50, 100, 200, 500):
                hit = np.take_along_axis(y, order[:, :k], axis=1).sum(1)
                entry[f'protein_mean_R@{k}'] = float(np.mean(hit / counts))
            entry['obs_protein_fmax'], entry['obs_protein_fmax_threshold'] = map(float, compute_protein_centric_fmax(y, s, num_thresholds=101))
            entry['obs_aupr'] = float(compute_term_aupr(y, s))
            output[key] = entry
        output['metadata']['proteins'] = int(y.shape[0])
        output['metadata']['audit_version'] = 'per_protein_v2'
        if len(protein_rows) != int(y.shape[0]):
            raise RuntimeError('Protein rows and score matrix length disagree')

        def summarize(rows):
            if not rows:
                return {'n': 0}
            summary = {'n': len(rows),
                       'mean_length': float(np.mean([r['residue_length'] for r in rows])),
                       'mean_cardinality': float(np.mean([r['cardinality'] for r in rows])),
                       'mean_local_weight': float(np.mean([r['mean_local_weight'] for r in rows]))}
            for k in (50, 100, 200, 500):
                ds = np.array([r[f'delta_R@{k}'] for r in rows], dtype=float)
                summary[f'delta_R@{k}_mean'] = float(ds.mean())
                summary[f'proteins_improved@{k}'] = int((ds > 1e-12).sum())
                summary[f'proteins_worsened@{k}'] = int((ds < -1e-12).sum())
                summary[f'proteins_unchanged@{k}'] = int((np.abs(ds) <= 1e-12).sum())
                summary[f'GO_gains@{k}'] = int(sum(r[f'gain_count@{k}'] for r in rows))
                summary[f'GO_losses@{k}'] = int(sum(r[f'loss_count@{k}'] for r in rows))
            return summary

        output['per_protein_audit'] = {'overall': summarize(protein_rows)}
        card_groups = [('1_5', 1, 5), ('6_10', 6, 10), ('11_20', 11, 20), ('21_40', 21, 40), ('41_80', 41, 80), ('81_160', 81, 160), ('161plus', 161, 10 ** 9)]
        output['per_protein_audit']['by_cardinality'] = {name: summarize([r for r in protein_rows if lo <= r['cardinality'] <= hi]) for name, lo, hi in
                                                         card_groups}
        length_bins = [('1_128', 1, 128), ('129_256', 129, 256), ('257_512', 257, 512), ('513_1024', 513, 1024), ('1025plus', 1025, 10 ** 9)]
        output['per_protein_audit']['by_length'] = {name: summarize([r for r in protein_rows if lo <= r['residue_length'] <= hi]) for name, lo, hi in
                                                    length_bins}
        output['per_protein_audit']['top_improved_R50'] = [
            {'protein_id': r['protein_id'], 'cardinality': r['cardinality'], 'residue_length': r['residue_length'], 'delta_R@50': r['delta_R@50'],
             'mean_local_weight': r['mean_local_weight']} for r in sorted(protein_rows, key=lambda r: r['delta_R@50'], reverse=True)[:15]]
        output['per_protein_audit']['top_worsened_R50'] = [
            {'protein_id': r['protein_id'], 'cardinality': r['cardinality'], 'residue_length': r['residue_length'], 'delta_R@50': r['delta_R@50'],
             'mean_local_weight': r['mean_local_weight']} for r in sorted(protein_rows, key=lambda r: r['delta_R@50'])[:15]]
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = path.with_name(path.stem + '_per_protein.csv')
        with csv_path.open('w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(protein_rows[0].keys()))
            writer.writeheader()
            writer.writerows(protein_rows)
        output['metadata']['per_protein_csv'] = str(csv_path)
        path.write_text(json.dumps(output, indent=2))
        compact = {k: v for k, v in output.items() if k not in ('global_only', 'local_only', 'fused', 'fused_parents_off')}
        print(json.dumps(compact, indent=2))
        print('SAVED', path, csv_path)
        raise _Finished()

    original = OppTrainer.eval_epoch
    OppTrainer.eval_epoch = evaluate
    try:
        retriever_main.run_training(cfg)
    except _Finished:
        pass
    finally:
        OppTrainer.eval_epoch = original


if __name__ == '__main__':
    main()
