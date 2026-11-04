#!/usr/bin/env python3
"""Extract full-length ESM-2 residue embeddings using overlapping windows.

Output: res_esm2_XXXXX.data.npy, .index.tsv (protein_id, start, end), .meta.json.
The NPY stores concatenated FP16 residue vectors, including padding capacity;
index end offsets delimit actual data. Never writes into an existing shard.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm

MODEL_NAME = 'esm2_t33_650M_UR50D'
DIM = 1280
LAYER = 33
PREFIX = 'res_esm2'
VALID = set('ACDEFGHIKLMNPQRSTVWYBXZOU')


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta', type=Path, default=Path('/workspace/MZSGO_temporal/sequences/all_train_temporal.fasta'))
    p.add_argument('--out-dir', type=Path, default=Path('/workspace/MZSGO_temporal/protein_embeddings/esm2_residue_overlap'))
    p.add_argument('--window-size', type=int, default=1000)
    p.add_argument('--overlap', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-tokens-per-batch', type=int, default=8000)
    p.add_argument('--shard-max-rows', type=int, default=2_000_000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--model-cache', type=Path, default=None, help='Optional local ESM-2 checkpoint path; otherwise esm.pretrained downloads/caches')
    p.add_argument('--limit-proteins', type=int, default=0, help='Smoke test only; 0 means all')
    p.add_argument('--resume', action='store_true', help='Skip only IDs in completed, indexed shards')
    return p.parse_args()


def clean_sequence(seq):
    seq = re.sub(r'\s+|\d+', '', seq).upper().replace('-', '').replace('.', '').replace('_', '')
    if seq.endswith('*'):
        seq = seq[:-1]
    return ''.join(c if c in VALID else 'X' for c in seq)


def read_fasta(path):
    seqs = {}
    pid, parts = None, []
    def flush():
        if pid is not None:
            if pid in seqs:
                raise ValueError(f'Duplicate FASTA ID: {pid}')
            s = clean_sequence(''.join(parts))
            if not s:
                raise ValueError(f'Empty sequence: {pid}')
            seqs[pid] = s
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                flush()
                pid, parts = line[1:].split()[0], []
            else:
                if pid is None:
                    raise ValueError('Sequence before FASTA header')
                parts.append(line)
    flush()
    return seqs


def window_starts(n, w, overlap):
    if n <= w:
        return [0]
    stride = w - overlap
    starts = list(range(0, n - w + 1, stride))
    last = n - w
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_window_jobs(pid, sequence, w, overlap):
    return [(pid, st, sequence[st:st+w]) for st in window_starts(len(sequence), w, overlap)]


@torch.inference_mode()
def encode_batch(model, converter, jobs, device):
    _, _, tokens = converter([(f'{pid}:{st}', s) for pid, st, s in jobs])
    tokens = tokens.to(device, non_blocking=True)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
        out = model(tokens, repr_layers=[LAYER], return_contacts=False)['representations'][LAYER]
    result = []
    for i, (_, _, s) in enumerate(jobs):
        v = out[i, 1:len(s)+1].float().cpu().numpy()
        if v.shape != (len(s), DIM) or not np.isfinite(v).all():
            raise RuntimeError('Invalid ESM-2 window embedding')
        result.append(v)
    return result


def encode_jobs(model, converter, jobs, device):
    try:
        return encode_batch(model, converter, jobs, device)
    except torch.cuda.OutOfMemoryError:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        if len(jobs) == 1:
            raise RuntimeError(f'OOM on a single window: {jobs[0][0]} offset {jobs[0][1]}')
        mid = len(jobs) // 2
        return encode_jobs(model, converter, jobs[:mid], device) + encode_jobs(model, converter, jobs[mid:], device)


def extract_protein(model, converter, pid, sequence, w, overlap, batch_size, token_budget, device):
    n = len(sequence)
    sums = np.zeros((n, DIM), dtype=np.float32)
    counts = np.zeros(n, dtype=np.uint16)
    jobs = make_window_jobs(pid, sequence, w, overlap)
    i = 0
    while i < len(jobs):
        batch = []
        token_count = 0
        while i < len(jobs) and len(batch) < batch_size:
            job = jobs[i]
            cost = len(job[2]) + 2
            if batch and token_count + cost > token_budget:
                break
            batch.append(job)
            token_count += cost
            i += 1
        for (_, st, s), vec in zip(batch, encode_jobs(model, converter, batch, device)):
            sums[st:st+len(s)] += vec
            counts[st:st+len(s)] += 1
    if np.any(counts == 0):
        raise RuntimeError(f'Uncovered residues: {pid}')
    merged = sums / counts[:, None]
    if not np.isfinite(merged).all():
        raise RuntimeError(f'Nonfinite merged protein: {pid}')
    return merged.astype(np.float16), len(jobs), int(counts.max())


class ShardWriter:
    def __init__(self, out_dir, shard_id, capacity, config):
        self.dir, self.id, self.capacity, self.config = out_dir, shard_id, capacity, config
        self.stem = f'{PREFIX}_{shard_id:05d}'
        self.data = out_dir / (self.stem + '.data.npy')
        self.index = out_dir / (self.stem + '.index.tsv')
        self.meta = out_dir / (self.stem + '.meta.json')
        if any(p.exists() for p in (self.data, self.index, self.meta)):
            raise FileExistsError(f'Shard already exists: {self.stem}')
        self.array = open_memmap(str(self.data), mode='w+', dtype=np.float16, shape=(capacity, DIM))
        self.cursor = 0
        self.rows = []

    def append(self, pid, vec):
        n = len(vec)
        if n > self.capacity - self.cursor:
            return False
        start = self.cursor
        self.array[start:start+n] = vec
        self.cursor += n
        self.rows.append((pid, start, self.cursor))
        return True

    def close(self):
        if not self.rows:
            del self.array
            self.data.unlink(missing_ok=True)
            return
        self.array.flush()
        del self.array
        # Publish index last so --resume never trusts a partially written shard.
        self.meta.write_text(json.dumps({**self.config, 'shape': [self.capacity, DIM], 'used_rows': self.cursor,
                                         'proteins': len(self.rows), 'format': 'residue_concat_with_index',
                                         'dtype': 'float16', 'embedding_dim': DIM}, indent=2))
        tmp = self.index.with_suffix('.index.tmp')
        with tmp.open('w') as f:
            for pid, start, end in self.rows:
                f.write(f'{pid}\t{start}\t{end}\n')
        os.replace(tmp, self.index)
        print(f'[SHARD] {self.stem}: {len(self.rows)} proteins, {self.cursor} residues', flush=True)


def completed_ids(out):
    done = set()
    shard_ids = []
    for index in sorted(out.glob(f'{PREFIX}_*.index.tsv')):
        stem = index.name[:-len('.index.tsv')]
        data = out / (stem + '.data.npy')
        meta = out / (stem + '.meta.json')
        if not data.exists() or not meta.exists():
            raise RuntimeError(f'Incomplete shard: {stem}')
        shard_ids.append(int(stem.rsplit('_', 1)[1]))
        info = json.loads(meta.read_text())
        if info.get('model') != MODEL_NAME or info.get('window_size') != out_config['window_size'] or info.get('overlap') != out_config['overlap']:
            raise RuntimeError(f'Shard extraction configuration mismatch: {stem}')
        for line in index.read_text().splitlines():
            pid, start, end = line.split('\t')
            if pid in done:
                raise RuntimeError(f'Duplicate indexed protein: {pid}')
            if not (0 <= int(start) < int(end) <= info['used_rows']):
                raise RuntimeError(f'Invalid index offsets: {pid}')
            done.add(pid)
    return done, max(shard_ids, default=-1) + 1


def main():
    global out_config
    a = args_parser()
    if not 1 <= a.window_size <= 1022 or not 0 <= a.overlap < a.window_size:
        raise ValueError('ESM-2 requires 1 <= window-size <= 1022 and overlap < window-size')
    if a.shard_max_rows < a.window_size or a.batch_size < 1 or a.max_tokens_per_batch < a.window_size + 2:
        raise ValueError('Invalid shard capacity or batching parameters')
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_config = {'model': MODEL_NAME, 'representation_layer': LAYER, 'window_size': a.window_size,
                  'overlap': a.overlap, 'aggregation': 'mean_per_residue', 'long_sequence_policy': 'overlap_no_truncation',
                  'source_fasta': str(a.fasta.resolve())}
    seqs = read_fasta(a.fasta)
    print(f'[FASTA] proteins={len(seqs)} max_length={max(map(len, seqs.values()))}', flush=True)
    if a.resume:
        done, shard_id = completed_ids(a.out_dir)
    else:
        existing = list(a.out_dir.glob(f'{PREFIX}_*'))
        if existing:
            raise RuntimeError('Output contains ESM-2 shards. Use --resume or a new output directory.')
        done, shard_id = set(), 0
    remaining = sorted((p for p in seqs if p not in done), key=lambda p: (len(seqs[p]), p))
    if a.limit_proteins:
        remaining = remaining[:a.limit_proteins]
    print(f'[RUN] completed={len(done)} remaining={len(remaining)} output={a.out_dir}', flush=True)
    if not remaining:
        return
    from esm import pretrained
    print('[MODEL] loading ESM-2', flush=True)
    if a.model_cache:
        model, alphabet = pretrained.load_model_and_alphabet_local(str(a.model_cache))
    else:
        model, alphabet = pretrained.esm2_t33_650M_UR50D()
    model.eval().to(device)
    converter = alphabet.get_batch_converter()
    writer = None
    total_windows, total_residues, max_overlap = 0, 0, 0
    for pid in tqdm(remaining, desc='ESM-2 proteins'):
        seq = seqs[pid]
        if len(seq) > a.shard_max_rows:
            raise ValueError(f'Protein {pid} longer than shard capacity; increase --shard-max-rows')
        vec, windows, depth = extract_protein(model, converter, pid, seq, a.window_size, a.overlap,
                                              a.batch_size, a.max_tokens_per_batch, device)
        if writer is None:
            writer = ShardWriter(a.out_dir, shard_id, a.shard_max_rows, out_config)
        if not writer.append(pid, vec):
            writer.close()
            shard_id += 1
            writer = ShardWriter(a.out_dir, shard_id, a.shard_max_rows, out_config)
            if not writer.append(pid, vec):
                raise RuntimeError(f'Cannot append {pid}')
        total_windows += windows
        total_residues += len(seq)
        max_overlap = max(max_overlap, depth)
    if writer:
        writer.close()
    summary = {**out_config, 'fasta_proteins': len(seqs), 'previously_completed': len(done),
               'proteins_processed_this_run': len(remaining), 'residues_processed_this_run': total_residues,
               'windows_processed_this_run': total_windows, 'maximum_windows_per_residue': max_overlap,
               'truncated_proteins': 0, 'dtype': 'float16', 'embedding_dim': DIM}
    (a.out_dir / 'extraction_run_summary.json').write_text(json.dumps(summary, indent=2))
    print('[DONE]', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
