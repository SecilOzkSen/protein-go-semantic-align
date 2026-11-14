#!/usr/bin/env python3
"""Create an evaluation-only NPZ with BP labels exactly from PFresGO annot.tsv.
Never overwrites input; preserves protein and GO column order and all score arrays.
"""
import argparse
import csv
from pathlib import Path
import numpy as np

def go_id(x):
    s = str(x).strip()
    return f"GO:{int(s.split(':')[-1]):07d}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    src, dest = Path(a.input).resolve(), Path(a.out).resolve()
    if src == dest:
        raise ValueError('Output must differ from input: original NPZ must not be overwritten')
    with np.load(src, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
    required = {'scores','labels','protein_ids','go_ids'}
    if not required.issubset(arrays):
        raise ValueError(f'Missing NPZ keys: {sorted(required-set(arrays))}')
    proteins = [str(x) for x in arrays['protein_ids']]
    gos = [go_id(x) for x in arrays['go_ids']]
    if len(set(proteins)) != len(proteins) or len(set(gos)) != len(gos):
        raise ValueError('Duplicate protein or GO IDs')
    if arrays['labels'].shape != arrays['scores'].shape or arrays['labels'].shape != (len(proteins),len(gos)):
        raise ValueError('Score/label/ID dimensions inconsistent')
    annotations = {}
    with open(a.annotations, newline='') as f:
        lines = f.readlines()
    header_line = next((i for i,line in enumerate(lines) if line.startswith('### PDB-chain')), None)
    if header_line is None:
        raise ValueError('PFresGO annotation table header not found')
    reader = csv.reader(lines[header_line+1:], delimiter='\t')
    for row in reader:
        if not row or not row[0].strip():
            continue
        if len(row) < 3:
            raise ValueError(f'Malformed annotation row: {row[:2]}')
        pid = row[0].strip()
        if pid in annotations:
            raise ValueError(f'Duplicate annotation protein: {pid}')
        annotations[pid] = set(x.strip() for x in row[2].split(',') if x.strip())
    missing = sorted(set(proteins)-set(annotations))
    if missing:
        raise ValueError(f'{len(missing)} missing proteins, e.g. {missing[:5]}')
    go_index = {go:i for i,go in enumerate(gos)}
    new = np.zeros_like(arrays['labels'])
    for i,pid in enumerate(proteins):
        for go in annotations[pid]:
            j = go_index.get(go)
            if j is not None:
                new[i,j] = 1
    old = arrays['labels'] > 0
    fresh = new > 0
    mismatch = np.any(old != fresh,axis=1)
    print('Proteins:',len(proteins),'GO terms:',len(gos))
    print('Mismatched proteins:',int(mismatch.sum()))
    print('Added positive labels:',int((fresh & ~old).sum()))
    print('Removed positive labels:',int((old & ~fresh).sum()))
    # For this known BP case, fail rather than silently accept an unexpected discrepancy.
    if (len(proteins),len(gos),int(mismatch.sum()),int((fresh & ~old).sum()),int((old & ~fresh).sum())) != (3416,1943,70,271,0):
        raise ValueError('Unexpected label differences: inspect before writing')
    arrays['labels'] = new
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest, **arrays)
    with np.load(dest,allow_pickle=True) as verify:
        assert np.array_equal(verify['labels'],new)
        assert np.array_equal(verify['scores'],arrays['scores'])
        assert np.array_equal(verify['go_ids'],arrays['go_ids'])
        assert np.array_equal(verify['protein_ids'],arrays['protein_ids'])
    print('PASS: saved evaluation-only corrected labels:',dest)

if __name__ == '__main__':
    main()
