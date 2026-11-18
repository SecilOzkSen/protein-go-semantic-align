"""Run a trained EmbSetCal reranker on a generic prepared NPZ split.

If labels are present, evaluate with the original MZSGO evaluate_annotations.
No labels are required for inference. Never trains or selects a checkpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .trainer import load, predict
from .model import EmbSetCal


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--mzsgo-repo', default='/workspace/MZSGO')
    ap.add_argument('--no-eval', action='store_true', help='Skip evaluation even when NPZ has labels')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    data = load(args.npz, labels=False)  # labels optional, not discarded
    if 'labels' in data:
        if data['labels'].shape != data['scores'].shape:
            raise ValueError('NPZ labels shape differs from score matrix')
        if not np.isin(data['labels'], (0, 1)).all():
            raise ValueError('NPZ labels must be binary')

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    p_dim, g_dim, k = int(cfg['protein_dim']), int(cfg['go_dim']), int(cfg['k'])
    if data['protein_z'].shape[1] != p_dim or data['go_z'].shape[1] != g_dim:
        raise ValueError('Checkpoint and NPZ embedding dimensions differ')
    if not 1 <= k <= data['scores'].shape[1]:
        raise ValueError(f'Checkpoint k={k} incompatible with {data["scores"].shape[1]} candidates')
    if len(set(data['go_ids'].tolist())) != len(data['go_ids']):
        raise ValueError('Duplicate candidate GO IDs')

    # Check all outputs before expensive inference, avoid silently replacing files.
    out = args.out_dir
    destinations = [out / 'scores_01.npy', out / 'protein_ids.json', out / 'go_ids.json']
    if 'labels' in data and not args.no_eval:
        destinations.append(out / 'metrics.json')
    existing = [str(p) for p in destinations if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f'Output exists (use --overwrite to replace): {existing}')

    device = torch.device(args.device)
    model = EmbSetCal(p_dim, g_dim).to(device)
    model.load_state_dict(ck['model'], strict=True)
    scores = predict(model, data, k, device, args.batch_size)
    if scores.shape != data['scores'].shape or not np.isfinite(scores).all():
        raise RuntimeError('Invalid prediction scores')
    if scores.min() < 0 or scores.max() > 1:
        raise RuntimeError('Prediction scores outside [0,1]')

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'scores_01.npy', scores)
    (out / 'protein_ids.json').write_text(json.dumps(data['protein_ids'].tolist()))
    (out / 'go_ids.json').write_text(json.dumps(data['go_ids'].tolist()))

    if 'labels' in data and not args.no_eval:
        from .metrics import evaluate
        metrics = evaluate(data['labels'], scores, args.mzsgo_repo, 'embsetcal_test')
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print('[METRICS]', json.dumps(metrics), flush=True)
    else:
        print('[INFO] Evaluation skipped: no labels or --no-eval', flush=True)
    print(f'[SAVED] {out} scores={scores.shape} checkpoint_epoch={ck.get("epoch", "unknown")}', flush=True)


if __name__ == '__main__':
    main()
