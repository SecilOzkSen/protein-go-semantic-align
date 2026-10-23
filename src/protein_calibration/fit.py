"""Fit cardinality-guided decision calibration on unseen proteins only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .io import load_matrix, verify_compatible, hash_go_ids
from .model import CardinalityGuidedCalibrator, soft_f1_loss
from src.metrics.cafa import compute_protein_centric_fmax, protein_centric_f_at_threshold


def predict(model, scores, device, batch_size=256):
    model.eval()
    chunks = {'scores': [], 'cardinality': [], 'boundary': [], 'scale': []}
    with torch.no_grad():
        for start in range(0, len(scores), batch_size):
            out = model(torch.from_numpy(scores[start:start+batch_size]).to(device))
            for k in chunks: chunks[k].append(out[k].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def cardinality_bin_weights(train_labels, max_ratio=5.0):
    """Train-only sqrt inverse-frequency weights, anchored at most common bin."""
    counts = train_labels.sum(dim=1).cpu().numpy()
    edges = np.array([5, 10, 20, 40, 80, 160])
    bins = np.searchsorted(edges, counts, side='left')
    frequencies = np.bincount(bins, minlength=7)
    if (frequencies == 0).any():
        raise ValueError(f'Empty cardinality bin in calibration training: {frequencies.tolist()}')
    weights = np.sqrt(frequencies.max() / frequencies.astype(np.float64))
    weights = np.minimum(weights, max_ratio)
    # Normalize so the average training-protein weight is one.
    weights /= weights[bins].mean()
    return torch.tensor(weights[bins], dtype=torch.float32), torch.tensor(weights, dtype=torch.float32), frequencies


def weighted_card_loss(pred_log, true_log, weights):
    per_protein = F.smooth_l1_loss(pred_log, true_log, reduction='none').flatten()
    return (per_protein * weights.flatten()).sum() / weights.sum().clamp_min(1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration', required=True)
    p.add_argument('--valid', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--card-weight', type=float, default=.2)
    p.add_argument('--card-bin-max-ratio', type=float, default=5.0)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tr, va = load_matrix(a.calibration), load_matrix(a.valid)
    verify_compatible(tr, va)
    if (tr['labels'].sum(1) == 0).any(): raise ValueError('Calibration train contains zero-positive proteins')
    device = torch.device(a.device)
    model = CardinalityGuidedCalibrator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    x = torch.from_numpy(tr['scores'].astype('float32'))
    y = torch.from_numpy(tr['labels'].astype('float32'))
    xv = torch.from_numpy(va['scores'].astype('float32'))
    yv = torch.from_numpy(va['labels'].astype('float32'))
    train_weights, bin_weights, bin_counts = cardinality_bin_weights(y, a.card_bin_max_ratio)
    print('Cardinality bins:', bin_counts.tolist(), 'normalized weights:', bin_weights.tolist())
    outdir = Path(a.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    best = float('inf'); wait = 0; best_epoch = -1
    history = []
    for epoch in range(a.epochs):
        model.train()
        order = torch.randperm(len(x))
        for idx in order.split(a.batch_size):
            result = model(x[idx].to(device))
            card_target = torch.log1p(y[idx].sum(1, keepdim=True).to(device))
            loss_card = weighted_card_loss(result['log_card'], card_target, train_weights[idx].to(device))
            loss_f = soft_f1_loss(result['scores'], y[idx].to(device), a.temperature)
            # Mild boundary regularizer; keep adaptive corrections modest.
            reg = 1e-3 * (result['boundary'] - model.global_boundary).square().mean()
            loss = loss_f + a.card_weight * loss_card + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            # Selection is on fixed-threshold (zero) differentiable surrogate, not swept Fmax.
            totals = []
            for idx in torch.arange(len(xv)).split(a.batch_size):
                res = model(xv[idx].to(device))
                card = torch.log1p(yv[idx].sum(1, keepdim=True).to(device))
                l = soft_f1_loss(res['scores'], yv[idx].to(device), a.temperature)
                # Validation selection remains unweighted: no validation-distribution tuning.
                l += a.card_weight * F.smooth_l1_loss(res['log_card'], card)
                totals.append((float(l.item()), len(idx)))
            metric = sum(v*n for v,n in totals)/sum(n for _,n in totals)
        history.append({'epoch': epoch, 'valid_surrogate_loss': metric})
        if metric < best - 1e-6:
            best, best_epoch, wait = metric, epoch, 0
            torch.save({'model': model.state_dict(), 'architecture': 'cardinality_guided_v1',
                        'go_hash': hash_go_ids(tr['go_ids']), 'epoch': epoch,
                        'card_bin_counts': bin_counts.tolist(), 'card_bin_weights': bin_weights.tolist()}, outdir/'head.pt')
        else:
            wait += 1
            if wait >= a.patience: break
    ckpt = torch.load(outdir/'head.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    val = predict(model, va['scores'].astype('float32'), device)
    raw_f, raw_t = compute_protein_centric_fmax(va['labels'], va['scores'])
    cal_f, cal_t = compute_protein_centric_fmax(va['labels'], val['scores'])
    result = {'best_epoch': best_epoch, 'raw_valid_oracle_fmax': raw_f, 'raw_valid_threshold': raw_t,
              'calibrated_valid_oracle_fmax': cal_f, 'calibrated_valid_threshold': cal_t,
              'valid_F_at_calibrated_threshold': protein_centric_f_at_threshold(va['labels'], val['scores'], cal_t),
              'cardinality_mae': float(np.mean(abs(val['cardinality'].ravel() - va['labels'].sum(1)))),
              'go_id_hash': ckpt['go_hash'], 'n_calibration': len(x), 'n_valid': len(xv),
              'card_bin_counts': bin_counts.tolist(), 'card_bin_weights': bin_weights.tolist()}
    (outdir/'validation_results.json').write_text(json.dumps(result, indent=2))
    (outdir/'history.json').write_text(json.dumps(history, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
