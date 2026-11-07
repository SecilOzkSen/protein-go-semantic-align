"""50-epoch alignment relevance head with protein-disjoint early stopping and held-out test.
Run on frozen retriever validation extraction. No test labels read.
"""
import argparse, json, copy, sys, importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight);
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # Retain retriever score as a fixed anchor; learn bounded residual.
        return 4.0 * x[:, 0] + 0.5 * torch.tanh(self.net(x).squeeze(-1))


def predict(model, feat, score, idx, mu, sd, columns, device, chunk=8192):
    n_go = score.shape[1]
    out = np.empty(n_go, dtype=np.float32)
    with torch.inference_mode():
        for st in range(0, n_go, chunk):
            ed = min(n_go, st + chunk)
            x = np.array(feat[idx, st:ed], dtype=np.float32, copy=True)
            x[:, 0] = np.asarray(score[idx, st:ed], dtype=np.float32)
            x[:, 1:] = (x[:, 1:] - mu) / sd
            z = model(torch.from_numpy(x[:, columns]).to(device))
            out[st:ed] = torch.sigmoid(z).cpu().numpy()
    return out


def labels_for(i, pids, lab, gindex, g):
    y = np.zeros(g, dtype=np.bool_)
    for term in lab.get(pids[i], []):
        if term in gindex: y[gindex[term]] = True
    return y


def exact_map(model, indices, feat, score, pids, lab, gindex, mu, sd, columns, device):
    aps = []
    for i in indices:
        y = labels_for(i, pids, lab, gindex, score.shape[1])
        if not y.any(): continue
        p = predict(model, feat, score, int(i), mu, sd, columns, device)
        aps.append(average_precision_score(y, p))
    return float(np.mean(aps)) if aps else float('nan')


def evaluate(model, indices, feat, score, pids, lab, gindex, mu, sd, columns, device):
    thresholds = np.linspace(0, 1, 201)
    P = np.zeros(201);
    R = np.zeros(201)
    ap = [];
    raw_ap = []
    for i in indices:
        i = int(i);
        y = labels_for(i, pids, lab, gindex, score.shape[1])
        if not y.any(): continue
        pred = predict(model, feat, score, i, mu, sd, columns, device)
        raw = np.asarray(score[i], dtype=np.float32)
        ap.append(average_precision_score(y, pred))
        raw_ap.append(average_precision_score(y, raw))
        for arr, threshold_scores in ((pred, False),):
            bins = np.clip((arr * 200).astype(np.int32), 0, 200)
            hp = np.bincount(bins[y], minlength=201)[::-1].cumsum()[::-1]
            hn = np.bincount(bins[~y], minlength=201)[::-1].cumsum()[::-1]
            P += np.divide(hp, hp + hn, out=np.zeros(201), where=(hp + hn) > 0)
            R += hp / y.sum()
    count = len(ap)
    P /= count;
    R /= count
    f = np.divide(2 * P * R, P + R, out=np.zeros(201), where=(P + R) > 0)
    j = int(np.argmax(f))
    return {'n_proteins': count, 'exact_protein_map': float(np.mean(ap)),
            'raw_exact_protein_map': float(np.mean(raw_ap)),
            'delta_exact_map': float(np.mean(np.array(ap) - np.array(raw_ap))),
            'protein_macro_fmax_201grid': float(f[j]),
            'threshold': float(thresholds[j])}


def mzsgo_fmax_diagnostic(model, indices, feat, score, pids, lab, gindex, mu, sd, columns, device, repo):
    """Call original MZSGO evaluate_annotations, not a surrogate Fmax implementation."""
    # Load by absolute file path: avoids collisions with another installed `utils` package.
    roots = [Path(repo), Path('/workspace/MZSGO_temporal'), Path('/workspace/protein-go-semantic-align')]
    candidates = []
    for root in roots:
        if root.is_file() and root.name == 'util.py':
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(root.glob('**/utils/util.py'))
    util = next((f for f in candidates if f.is_file() and 'evaluate_annotations' in f.read_text(errors='replace')), None)
    if util is None:
        raise FileNotFoundError('Cannot locate original MZSGO evaluate_annotations. '
                                'Pass --mzsgo-repo PATH_TO_MZSGO_REPOSITORY; searched: ' + ', '.join(map(str, roots)))
    # Some MZSGO util.py implementations import sibling modules.
    sys.path.insert(0, str(util.parent.parent.resolve()))
    spec = importlib.util.spec_from_file_location('_mzsgo_original_util', str(util))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate_annotations = module.evaluate_annotations
    print(f'[MZSGO evaluator] {util}', flush=True)
    n = len(indices);
    g = score.shape[1]
    truth = np.zeros((n, g), dtype=np.int32)
    raw = np.empty((n, g), dtype=np.float32)
    head = np.empty((n, g), dtype=np.float32)
    for j, i in enumerate(indices):
        i = int(i)
        truth[j] = labels_for(i, pids, lab, gindex, g)
        raw[j] = np.clip((np.asarray(score[i], dtype=np.float32) + 1) / 2, 0, 1)
        head[j] = predict(model, feat, score, i, mu, sd, columns, device)
        if (j + 1) % 100 == 0:
            print(f'[MZSGO diagnostic] {j + 1}/{n}', flush=True)
    results = {}
    for name, pred in [('raw', raw), ('alignment', head)]:
        f, precision, recall, aupr, threshold = evaluate_annotations(
            torch.from_numpy(truth), torch.from_numpy(pred), 'general')
        results[name] = {'Fmax': float(f), 'precision': float(precision),
                         'recall': float(recall), 'AUPR': float(aupr),
                         'threshold': float(threshold)}
    results['delta_Fmax'] = results['alignment']['Fmax'] - results['raw']['Fmax']
    results['delta_AUPR'] = results['alignment']['AUPR'] - results['raw']['AUPR']
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True);
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--min-delta', type=float, default=0.00001)
    ap.add_argument('--negatives-per-positive', type=int, default=5)
    ap.add_argument('--max-train-pairs', type=int, default=500000)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--mzsgo-repo', default='/workspace/MZSGO')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = Path(args.data);
    out = Path(args.out);
    out.mkdir(parents=True, exist_ok=True)
    feat = np.load(data / 'features.npy', mmap_mode='r')
    score = np.load(data / 'scores.npy', mmap_mode='r')
    pids = json.loads((data / 'protein_ids.json').read_text())
    gids = json.loads((data / 'go_ids.json').read_text())
    lab = json.loads((data / 'labels.json').read_text())
    gindex = {term: i for i, term in enumerate(gids)}
    n, g, d = feat.shape
    if len(pids) != n or score.shape != (n, g): raise ValueError('Input dimensions mismatch')
    # Preserve original 20% heldout set from the 8-epoch run (seed 42).
    perm = rng.permutation(n);
    outer = int(.8 * n)
    train_pool = perm[:outer];
    heldout = perm[outer:]
    # Split original train pool 87.5/12.5 = 70/10/20 overall.
    inner_rng = np.random.default_rng(args.seed + 1)
    inner = inner_rng.permutation(train_pool)
    cut = int(.875 * len(inner));
    tr = inner[:cut];
    stop = inner[cut:]
    print(f'[SPLIT] train={len(tr)} early_stop={len(stop)} heldout={len(heldout)}', flush=True)
    rows = [];
    cols = [];
    ys = []
    for i in tr:
        pos = np.array([gindex[x] for x in lab.get(pids[i], []) if x in gindex], dtype=np.int64)
        if not len(pos): continue
        s = np.asarray(score[i])
        hard_n = min(len(pos) * args.negatives_per_positive // 2, g)
        k = min(g, hard_n + len(pos) + 64)
        pool = np.argpartition(s, -k)[-k:]
        pool = pool[~np.isin(pool, pos)]
        hard = pool[np.argsort(s[pool])[-hard_n:]] if hard_n else np.array([], dtype=np.int64)
        rand = rng.integers(0, g, size=max(0, len(pos) * args.negatives_per_positive - hard_n))
        rand = rand[~np.isin(rand, pos)]
        neg = np.unique(np.concatenate([hard, rand]))
        c = np.concatenate([pos, neg])
        rows.extend([int(i)] * len(c));
        cols.extend(c.tolist())
        ys.extend([1] * len(pos) + [0] * len(neg))
    pick = rng.permutation(len(rows))[:args.max_train_pairs]
    rows = np.asarray(rows, dtype=np.int64)[pick]
    cols = np.asarray(cols, dtype=np.int64)[pick]
    ys = np.asarray(ys, dtype=np.float32)[pick]
    X = np.asarray(feat[rows, cols], dtype=np.float32)
    X[:, 0] = np.asarray(score[rows, cols], dtype=np.float32)
    mu = X[:, 1:].mean(0);
    sd = X[:, 1:].std(0).clip(1e-4)
    X[:, 1:] = (X[:, 1:] - mu) / sd
    device = torch.device(args.device)
    xx = torch.from_numpy(X).to(device);
    yy = torch.from_numpy(ys).to(device)
    columns = list(range(d))
    model = Head(d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best = -float('inf');
    best_epoch = 0;
    last_significant = -float('inf');
    bad = 0;
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train();
        order = torch.randperm(len(xx), device=device)
        losses = [];
        total = 0
        for sub in order.split(8192):
            logit = model(xx[sub])
            bce = nn.functional.binary_cross_entropy_with_logits(logit, yy[sub])
            loss = bce + 1e-4 * sum(p.square().sum() for p in model.parameters())
            opt.zero_grad();
            loss.backward();
            opt.step()
            losses.append(float(bce.item()) * len(sub));
            total += len(sub)
        model.eval()
        stop_map = exact_map(model, stop, feat, score, pids, lab, gindex, mu, sd, columns, device)
        train_bce = sum(losses) / total
        # Always save the true highest exact mAP, regardless of min_delta.
        improved = stop_map > best
        if improved:
            best = stop_map;
            best_epoch = epoch
            torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        'columns': columns, 'mean': mu, 'std': sd,
                        'best_epoch': best_epoch, 'early_stop_exact_map': best,
                        'feature_names': json.loads((data / 'metadata.json').read_text()).get('feature_names', []) if (
                                    data / 'metadata.json').exists() else []},
                       out / 'alignment_best.pt')
        # Patience resets only after an improvement of at least min_delta
        # relative to the last significant reference, not the latest tiny best.
        significant = stop_map > last_significant + args.min_delta
        if significant:
            last_significant = stop_map
            bad = 0
        else:
            bad += 1
        history.append({'epoch': epoch, 'train_bce': train_bce, 'stop_exact_map': stop_map,
                        'best': best, 'checkpoint_updated': improved,
                        'significant_improvement': significant})
        print(
            f'[alignment] epoch={epoch} train_bce={train_bce:.5f} stop_exact_mAP={stop_map:.6f} best={best:.6f} best_epoch={best_epoch} patience={bad}/{args.patience}',
            flush=True)
        if bad >= args.patience:
            print('[EARLY STOP]', flush=True);
            break
    ckpt = torch.load(out / 'alignment_best.pt', map_location='cpu', weights_only=False)
    model = Head(d).to(device);
    model.load_state_dict(ckpt['state_dict']);
    model.eval()
    metrics = evaluate(model, heldout, feat, score, pids, lab, gindex, mu, sd, columns, device)
    mzsgo = mzsgo_fmax_diagnostic(model, heldout, feat, score, pids, lab, gindex, mu, sd, columns, device, args.mzsgo_repo)
    result = {'mzsgo_original_heldout_diagnostic': mzsgo, 'checkpoint': str(data), 'seed': args.seed, 'train_proteins': len(tr),
              'early_stop_proteins': len(stop), 'heldout_proteins': len(heldout),
              'best_epoch': best_epoch, 'best_stop_exact_map': best,
              'heldout': metrics,
              'note': 'Original MZSGO evaluate_annotations on held-out validation, NOT general test benchmark. Raw uses linear score mapping; head uses sigmoid output. Threshold selected on heldout for diagnostic only. BCE uses sampled negatives, so sigmoid is not calibrated.'}
    (out / 'history.json').write_text(json.dumps(history, indent=2))
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    print('[DONE]', json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
