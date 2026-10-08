"""Strict ID-aligned Retriever-v2 + ESM neighbour evidence dataset."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def read_ids(p):
    x = json.loads(Path(p).read_text())
    if not isinstance(x, list): raise ValueError(f'Expected ID list: {p}')
    x = [str(v) for v in x]
    if len(x) != len(set(x)): raise ValueError(f'Duplicate IDs in {p}')
    return x


class MoEDataset(Dataset):
    def __init__(self, dump_dir, evidence_dir, neighbour_dir, split):
        self.dump_dir = Path(dump_dir);
        self.evidence_dir = Path(evidence_dir);
        self.neighbour_dir = Path(neighbour_dir)
        if not (self.dump_dir / 'DONE').exists(): raise ValueError(f'Incomplete dump: {dump_dir}')
        self.ids = read_ids(self.dump_dir / 'protein_ids.json')
        self.go_ids = np.load(self.dump_dir / 'eval_go_ids.int64.npy', allow_pickle=False).astype(np.int64)
        e_go = np.asarray([int(str(x).split(':')[-1]) for x in read_ids(self.evidence_dir / 'go_ids.json')], dtype=np.int64)
        if not np.array_equal(self.go_ids, e_go): raise ValueError('GO order mismatch between dump and evidence')
        source_ids = read_ids(self.neighbour_dir / f'{split}_protein_ids.json')
        ix = {p: i for i, p in enumerate(source_ids)}
        missing = [p for p in self.ids if p not in ix]
        if missing: raise ValueError(f'{len(missing)} missing evidence protein IDs, examples={missing[:5]}')
        self.indices = np.asarray([ix[p] for p in self.ids], dtype=np.int64)
        self.scores = np.load(self.dump_dir / 'retriever_scores.float16.npy', mmap_mode='r') if (
                    self.dump_dir / 'retriever_scores.float16.npy').exists() else np.load(self.dump_dir / 'retriever_scores.float32.npy', mmap_mode='r')
        self.labels = np.load(self.dump_dir / 'labels.int8.npy', mmap_mode='r')
        self.features = [np.load(self.evidence_dir / f'{split}_{k}.npy', mmap_mode='r') for k in ('direct_support', 'semantic_support', 'neighbour_similarity')]
        N = len(self.ids);
        G = len(self.go_ids)
        if self.scores.shape != (N, G) or self.labels.shape != (N, G): raise ValueError('Retriever score/label shape mismatch')
        if any(x.shape != (len(source_ids), G) for x in self.features): raise ValueError('Evidence shape mismatch')
        if not np.isin(self.labels, [0, 1]).all(): raise ValueError('Labels must be binary')
        if not np.isfinite(self.scores).all() or any(not np.isfinite(x).all() for x in self.features): raise ValueError('NaN/Inf in input matrices')
        if len(set(self.ids)) != N: raise ValueError('Duplicate dump IDs')
        self.split = split

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return tuple(torch.from_numpy(np.array(x, dtype=np.float32, copy=True)) for x in (self.scores[i], *(a[j] for a in self.features))) + (
            torch.from_numpy(np.array(self.labels[i], dtype=np.float32, copy=True)),)


def verify_neighbours(neighbour_dir, split, query_ids, training_bank_ids):
    root = Path(neighbour_dir)
    source = read_ids(root / f'{split}_protein_ids.json')
    neighbours = json.loads((root / f'{split}_neighbour_ids.json').read_text())
    if len(source) != len(neighbours): raise ValueError('Neighbour rows mismatch')
    bank = set(training_bank_ids)
    lookup = {p: i for i, p in enumerate(source)}
    for pid in query_ids:
        row = neighbours[lookup[pid]]
        if len(row) != 5 or len(set(row)) != 5: raise ValueError(f'Bad k=5 for {pid}')
        if pid in row: raise ValueError(f'Self-neighbour leakage: {pid}')
        if not set(row) <= bank: raise ValueError(f'Non-train neighbour: {pid}')
