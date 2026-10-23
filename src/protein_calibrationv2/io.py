"""Strict score matrix loading, metadata alignment and disjointness checks."""
from pathlib import Path
import numpy as np
import hashlib


def load_matrix(path):
    with np.load(path, allow_pickle=False) as d:
        for key in ('scores', 'labels', 'protein_ids', 'go_ids', 'protein_embeddings'):
            if key not in d: raise ValueError(f'{path}: missing {key}')
        obj = {k: d[k].copy() for k in ('scores', 'labels', 'protein_ids', 'go_ids', 'protein_embeddings')}
    s, y = obj['scores'], obj['labels']
    if s.ndim != 2 or s.shape != y.shape or s.shape[1] != len(obj['go_ids']) or s.shape[0] != len(obj['protein_ids']):
        raise ValueError(f'{path}: incompatible matrix dimensions')
    if obj['protein_embeddings'].ndim != 2 or obj['protein_embeddings'].shape[0] != s.shape[0]:
        raise ValueError(f'{path}: invalid protein embedding shape')
    if not np.isfinite(obj['protein_embeddings']).all():
        raise ValueError(f'{path}: non-finite protein embeddings')
    if not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f'{path}: invalid scores or labels')
    if len(set(obj['protein_ids'].tolist())) != len(obj['protein_ids']):
        raise ValueError(f'{path}: duplicate proteins')
    if len(set(obj['go_ids'].tolist())) != len(obj['go_ids']):
        raise ValueError(f'{path}: duplicate GO IDs')
    return obj


def verify_compatible(*items):
    for obj in items[1:]:
        if items[0]['protein_embeddings'].shape[1] != obj['protein_embeddings'].shape[1]:
            raise ValueError('Protein representation dimension mismatch')
        if not np.array_equal(items[0]['go_ids'], obj['go_ids']):
            raise ValueError('GO universe or ordering differs between matrices')
    sets = [set(x['protein_ids'].tolist()) for x in items]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]: raise ValueError('Protein overlap across splits')


def hash_go_ids(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
