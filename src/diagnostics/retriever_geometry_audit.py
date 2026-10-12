"""ESM mean vs Retriever-v2 protein geometry audit. No model training.

Compares matched protein representations and whether geometric distortion correlates
with raw retriever per-protein average precision. This cannot isolate projection
from pooling or establish causality.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def read_ids(path):
    ids = [str(x) for x in json.loads(path.read_text())]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate IDs in {path}')
    return ids


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite embeddings')
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_split(split, dump, neighbours, seed, max_proteins, pairs, k):
    ids = read_ids(dump / 'protein_ids.json')
    src_ids = read_ids(neighbours / f'{split}_protein_ids.json')
    index = {p: i for i, p in enumerate(src_ids)}
    missing = set(ids) - set(index)
    if missing:
        raise ValueError(f'{split}: {len(missing)} IDs absent from ESM embeddings: {list(missing)[:5]}')
    if len(ids) < k + 2:
        raise ValueError('Too few proteins')
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(ids), min(max_proteins, len(ids)), replace=False))
    ids_chosen = [ids[i] for i in chosen]
    esm_file = neighbours / f'{split}_esm_mean_normalized.npy'
    esm = np.load(esm_file, mmap_mode='r')
    z = np.load(dump / 'protein_z.float16.npy', mmap_mode='r')
    labels = np.load(dump / 'labels.int8.npy', mmap_mode='r')
    scores = np.load(dump / 'retriever_scores.float16.npy', mmap_mode='r')
    if z.shape[0] != len(ids) or labels.shape != scores.shape or labels.shape[0] != len(ids):
        raise ValueError(f'{split}: dump shape mismatch')
    if esm.shape[0] != len(src_ids):
        raise ValueError(f'{split}: ESM shape mismatch')
    e = unit(esm[[index[x] for x in ids_chosen]])
    r = unit(z[chosen])
    a, b = e @ e.T, r @ r.T
    np.fill_diagonal(a, -np.inf)
    np.fill_diagonal(b, -np.inf)
    ka = np.argpartition(-a, k, axis=1)[:, :k]
    kb = np.argpartition(-b, k, axis=1)[:, :k]
    overlap = np.array([len(set(x) & set(y)) / k for x, y in zip(ka, kb)])
    n = len(chosen)
    i = rng.integers(0, n, pairs * 2)
    j = rng.integers(0, n, pairs * 2)
    valid = i != j
    i, j = i[valid][:pairs], j[valid][:pairs]
    ea = np.sum(e[i] * e[j], axis=1)
    ra = np.sum(r[i] * r[j], axis=1)
    aps = []
    for ix in chosen:
        y = np.asarray(labels[ix], dtype=np.int8)
        s = np.asarray(scores[ix], dtype=np.float32)
        aps.append(float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan)
    aps = np.asarray(aps)
    valid_ap = np.isfinite(aps)
    ap_corr = spearmanr(overlap[valid_ap], aps[valid_ap]).statistic if valid_ap.sum() > 10 else np.nan
    return {'split': split, 'n_dump': len(ids), 'n_sample': n,
            'esm_mean_norm': float(np.mean(np.linalg.norm(e, axis=1))),
            'retriever_mean_norm': float(np.mean(np.linalg.norm(r, axis=1))),
            'random_pair_esm_cos_mean': float(ea.mean()),
            'random_pair_retriever_cos_mean': float(ra.mean()),
            'pairwise_cos_spearman': float(spearmanr(ea, ra).statistic),
            f'top{k}_neighbour_overlap_mean': float(overlap.mean()),
            'raw_retriever_per_protein_AP_mean': float(np.nanmean(aps)),
            'overlap_vs_per_protein_AP_spearman': float(ap_corr),
            'note': 'ESM vectors are masked-mean pooled, Retriever vectors include learned pooling and projection. Geometry changes cannot be attributed solely to projection.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--val_dump', required=True, type=Path)
    ap.add_argument('--test_dump', required=True, type=Path)
    ap.add_argument('--neighbour_dir', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--max_proteins', type=int, default=1500)
    ap.add_argument('--pairs', type=int, default=10000)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    results = []
    for split, dump in [('val', a.val_dump), ('test', a.test_dump)]:
        result = load_split(split, dump, a.neighbour_dir, a.seed, a.max_proteins, a.pairs, a.topk)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'results': results}, indent=2))
    print('SAVED', a.out)


if __name__ == '__main__':
    main()
