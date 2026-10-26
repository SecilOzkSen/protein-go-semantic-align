"""V1 balanced calibration, checkpoint + lambda selection using exact PFresGO validation.

Training objective and V1 model unchanged. Validation chooses checkpoint, blend
lambda, and global threshold jointly. Never uses test labels.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import obonet

from .io import load_matrix, verify_compatible, hash_go_ids
from .model import CardinalityGuidedCalibrator, soft_f1_loss
from src.evaluation.pfresgo_comparison import sigmoid, go_string
from src.evaluation.pfresgo_fast_exact import (
    FullAncestors, go_ancestors, metrics, parity, load_original, ROOTS
)


def predict(model, scores, device, batch_size=256):
    model.eval()
    chunks = {'scores': [], 'cardinality': [], 'boundary': [], 'scale': []}
    with torch.no_grad():
        for start in range(0, len(scores), batch_size):
            out = model(torch.from_numpy(scores[start:start + batch_size]).to(device))
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


def annotated_labels(data, path, branch):
    """Read original annot.tsv, aligning both protein and GO IDs; fail if missing."""
    if not path:
        return data['labels'].astype(np.uint8)
    cols = {'mf': 1, 'bp': 2, 'cc': 3}
    annotations = {}
    in_table = False
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if fields[0].startswith('### PDB-chain'):
                in_table = True
                continue
            if not in_table or len(fields) < 4 or not fields[0].strip():
                continue
            pid = fields[0].strip()
            if pid in annotations:
                raise ValueError(f'Duplicate annotation protein: {pid}')
            annotations[pid] = set(x.strip() for x in fields[cols[branch]].split(',') if x.strip())
    if not annotations:
        raise ValueError('Could not parse annotation table')
    gos = [go_string(x) for x in data['go_ids']]
    idx = {g: j for j, g in enumerate(gos)}
    if len(idx) != len(gos):
        raise ValueError('Duplicate GO IDs')
    y = np.zeros(data['scores'].shape, dtype=np.uint8)
    for i, protein in enumerate(map(str, data['protein_ids'])):
        if protein not in annotations:
            raise ValueError(f'Protein absent from annot.tsv: {protein}')
        for go in annotations[protein]:
            j = idx.get(go)
            if j is not None:
                y[i, j] = 1
    return y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', required=True)
    p.add_argument('--valid', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--obo', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--original-eval', required=True)
    p.add_argument('--branch', choices=['bp', 'mf', 'cc'], default='bp')
    p.add_argument('--lambdas', default='0,0.25,0.5,0.75,1')
    p.add_argument('--min-retention', type=float, default=.98)
    p.add_argument('--parity-n', type=int, default=20)
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

    lambdas = [float(t.strip()) for t in a.lambdas.split(',')]
    if len(lambdas) != len(set(lambdas)) or 0.0 not in lambdas or 1.0 not in lambdas or any(
            not (0 <= t <= 1) for t in lambdas
    ):
        raise ValueError('Lambdas must be unique, in [0,1], and include 0 and 1')
    if not 0 < a.min_retention <= 1:
        raise ValueError('--min-retention must be in (0,1]')

    tr, va = load_matrix(a.calibration), load_matrix(a.valid)
    verify_compatible(tr, va)
    if (tr['labels'].sum(1) == 0).any():
        raise ValueError('Calibration training contains zero-positive proteins')
    device = torch.device(a.device)
    model = CardinalityGuidedCalibrator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    x = torch.from_numpy(tr['scores'].astype('float32'))
    y = torch.from_numpy(tr['labels'].astype('float32'))
    weights, bin_weights, bin_counts = cardinality_bin_weights(y, a.card_bin_max_ratio)

    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    gos = [go_string(v) for v in va['go_ids']]
    graph = obonet.read_obo(a.obo)
    anc = go_ancestors(graph, gos)
    full = FullAncestors(graph, gos)
    root = ROOTS[a.branch]
    original = load_original(a.original_eval)
    vy = annotated_labels(va, a.annotations, a.branch)
    raw = sigmoid(va['scores'])

    n = min(a.parity_n, len(vy))
    if n > 0:
        parity(original, vy[:n], raw[:n], va['protein_ids'][:n], gos,
               a.obo, a.branch, anc, None, full, outdir / 'parity_raw.pckl')

    def evaluate(prob):
        m, (curve, _, _, grid) = metrics(vy, prob, anc, None, full, root)
        i = int(np.argmax(curve))
        return {
            'fmax': float(curve[i]),
            'threshold': float(grid[i]),
            'macro_aupr': float(m['macro_aupr']),
            'micro_aupr': float(m['micro_aupr']),
            'auroc': float(m['flattened_auc'])
        }

    baseline = evaluate(raw)
    print('RAW PFRESGO EXACT', json.dumps(baseline), flush=True)
    best = dict(baseline, epoch=-1, lambda_blend=0.0, feasible=True)
    best_epoch = -1
    wait = 0
    history = []
    best_path = outdir / 'head.pt'
    if best_path.exists():
        raise FileExistsError(
            f'{best_path} already exists. Use a fresh --out-dir to avoid stale checkpoints.'
        )
    (outdir / 'baseline.json').write_text(json.dumps(baseline, indent=2))

    for epoch in range(a.epochs):
        model.train()
        order = torch.randperm(len(x))
        losses = []
        for idx in order.split(a.batch_size):
            result = model(x[idx].to(device))
            labels = y[idx].to(device)
            card_target = torch.log1p(labels.sum(1, keepdim=True))
            loss_card = weighted_card_loss(
                result['log_card'], card_target, weights[idx].to(device)
            )
            loss_f = soft_f1_loss(result['scores'], labels, a.temperature)
            reg = 1e-3 * (result['boundary'] - model.global_boundary).square().mean()
            loss = loss_f + a.card_weight * loss_card + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        calibrated_logits = predict(
            model, va['scores'].astype('float32'), device
        )['scores']
        calibrated = sigmoid(calibrated_logits)
        candidates = []
        for lam in lambdas:
            prob = (1.0 - lam) * raw + lam * calibrated
            result = evaluate(prob)
            result['lambda_blend'] = lam
            result['feasible'] = all(
                result[k] >= a.min_retention * baseline[k]
                for k in ('macro_aupr', 'micro_aupr', 'auroc')
            )
            candidates.append(result)

        feasible = [r for r in candidates if r['feasible']]
        chosen = max(feasible, key=lambda r: (r['fmax'], -r['lambda_blend']))
        rec = {
            'epoch': epoch,
            'train_loss': float(np.mean(losses)),
            'best_lambda_this_epoch': chosen['lambda_blend'],
            'valid_fmax': chosen['fmax'],
            'valid_threshold': chosen['threshold'],
            'valid_macro_aupr': chosen['macro_aupr'],
            'valid_micro_aupr': chosen['micro_aupr'],
            'valid_auroc': chosen['auroc'],
            'improves_raw': chosen['fmax'] > baseline['fmax'] + 1e-8,
            'candidates': candidates,
        }
        history.append(rec)
        (outdir / 'history.json').write_text(json.dumps(history, indent=2))
        print('EPOCH', json.dumps({k: v for k, v in rec.items() if k != 'candidates'}), flush=True)

        if chosen['fmax'] > best['fmax'] + 1e-8:
            best = dict(chosen, epoch=epoch)
            best_epoch = epoch
            wait = 0
            torch.save({
                'model': model.state_dict(),
                'architecture': 'cardinality_guided_v1',
                'go_hash': hash_go_ids(tr['go_ids']),
                'epoch': epoch,
                'selected_lambda': chosen['lambda_blend'],
                'valid_threshold': chosen['threshold'],
                'card_bin_counts': bin_counts.tolist(),
                'card_bin_weights': bin_weights.tolist(),
            }, best_path)
        elif best_epoch >= 0:
            wait += 1
            if wait >= a.patience:
                print('EARLY STOP: no improvement over selected checkpoint', flush=True)
                break

    summary = {
        'protocol': 'PFresGO exact validation, annotation aligned, parity checked',
        'selection': 'joint epoch + lambda + global threshold; >= retention on AP/AUROC',
        'raw_validation': baseline,
        'best_validation': best,
        'best_epoch': best_epoch,
        'raw_fallback': best_epoch < 0,
        'n_calibration': len(tr['scores']),
        'n_validation': len(va['scores']),
        'n_go_terms': len(gos),
        'min_retention': a.min_retention,
    }
    (outdir / 'validation_results.json').write_text(json.dumps(summary, indent=2))
    print('VALIDATION SELECTED', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
