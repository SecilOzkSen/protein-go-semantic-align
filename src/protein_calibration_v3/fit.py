"""Fit cardinality-guided decision calibration on unseen proteins only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .io import load_matrix, verify_compatible, hash_go_ids
from .model import ProteinCalibrationV3, cafa_soft_f_loss
from src.metrics.cafa import compute_protein_centric_fmax, protein_centric_f_at_threshold


def predict(model, scores, protein_embeddings, device, batch_size=256):
    model.eval()
    chunks = {'scores': [], 'boundary': [], 'scale': []}
    with torch.no_grad():
        for start in range(0, len(scores), batch_size):
            out = model(torch.from_numpy(scores[start:start+batch_size]).to(device),
                        torch.from_numpy(protein_embeddings[start:start+batch_size]).to(device))
            for k in chunks: chunks[k].append(out[k].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration', required=True)
    p.add_argument('--valid', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
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
    model = ProteinCalibrationV3(protein_dim=tr['protein_embeddings'].shape[1]).to(device)
    model.set_protein_normalization(tr['protein_embeddings'])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    x = torch.from_numpy(tr['scores'].astype('float32'))
    xp = torch.from_numpy(tr['protein_embeddings'].astype('float32'))
    y = torch.from_numpy(tr['labels'].astype('float32'))
    xv = torch.from_numpy(va['scores'].astype('float32'))
    xvp = torch.from_numpy(va['protein_embeddings'].astype('float32'))
    yv = torch.from_numpy(va['labels'].astype('float32'))
    outdir = Path(a.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    best = float('inf'); wait = 0; best_epoch = -1
    history = []
    for epoch in range(a.epochs):
        model.train()
        order = torch.randperm(len(x))
        for idx in order.split(a.batch_size):
            result = model(x[idx].to(device), xp[idx].to(device))
            loss_f = cafa_soft_f_loss(result['scores'], y[idx].to(device), a.temperature)
            # Mild boundary regularizer; keep adaptive corrections modest.
            reg = 1e-3 * (result['boundary'] - model.global_boundary).square().mean()
            loss = loss_f + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pieces = []
            for idx in torch.arange(len(xv)).split(a.batch_size):
                pieces.append(model(xv[idx].to(device), xvp[idx].to(device))['scores'])
            metric = float(cafa_soft_f_loss(torch.cat(pieces), yv.to(device), a.temperature).item())
        history.append({'epoch': epoch, 'valid_surrogate_loss': metric})
        if metric < best - 1e-6:
            best, best_epoch, wait = metric, epoch, 0
            torch.save({'model': model.state_dict(), 'architecture': 'protein_calibration_v3_raw_esm',
                        'go_hash': hash_go_ids(tr['go_ids']), 'epoch': epoch}, outdir/'head.pt')
        else:
            wait += 1
            if wait >= a.patience: break
    ckpt = torch.load(outdir/'head.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    val = predict(model, va['scores'].astype('float32'), va['protein_embeddings'].astype('float32'), device)
    raw_f, raw_t = compute_protein_centric_fmax(va['labels'], va['scores'])
    cal_f, cal_t = compute_protein_centric_fmax(va['labels'], val['scores'])
    result = {'best_epoch': best_epoch, 'raw_valid_oracle_fmax': raw_f, 'raw_valid_threshold': raw_t,
              'calibrated_valid_oracle_fmax': cal_f, 'calibrated_valid_threshold': cal_t,
              'valid_F_at_calibrated_threshold': protein_centric_f_at_threshold(va['labels'], val['scores'], cal_t),
              'go_id_hash': ckpt['go_hash'], 'n_calibration': len(x), 'n_valid': len(xv)}
    (outdir/'validation_results.json').write_text(json.dumps(result, indent=2))
    (outdir/'history.json').write_text(json.dumps(history, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
