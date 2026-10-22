"""Create deterministic protein-level cardinality-stratified retriever/calibration split.

Usage:
 python create_calibration_split.py --train-ids TRAIN.txt --pid2pos PID2POS.json \
   --go-ids ACTIVE_GO_IDS.txt --out-dir OUT --fraction 0.2 --seed 42

Does not read validation/test data. GO IDs may be GO:0001234 or integers.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np

BINS = ((1, 5, '1_5'), (6, 10, '6_10'), (11, 20, '11_20'), (21, 40, '21_40'), (41, 80, '41_80'), (81, 160, '81_160'), (161, 10 ** 12, '161plus'))


def go_int(x): return int(str(x).split(':')[-1])


def bin_of(n):
    if n == 0: return 'zero'
    return next(name for lo, hi, name in BINS if lo <= n <= hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-ids', required=True)
    ap.add_argument('--pid2pos', required=True)
    ap.add_argument('--go-ids', required=True, help='Fixed active GO universe, one ID per line')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--fraction', type=float, default=.2)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    if not 0 < a.fraction < 1: raise ValueError('fraction must be between 0 and 1')
    ids = [x.strip() for x in Path(a.train_ids).read_text().splitlines() if x.strip()]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate protein IDs in train list')
    labels = json.loads(Path(a.pid2pos).read_text())
    universe = {go_int(x.strip()) for x in Path(a.go_ids).read_text().splitlines() if x.strip()}
    if not universe: raise ValueError('Empty GO universe')
    groups = {name: [] for _, _, name in BINS};
    groups['zero'] = []
    for pid in ids:
        if pid not in labels: raise ValueError(f'Missing labels for {pid}')
        n = len({go_int(g) for g in labels[pid]} & universe)
        groups[bin_of(n)].append(pid)
    rng = np.random.default_rng(a.seed)
    ret = []
    cal = []
    counts = {}
    for name, group in groups.items():
        perm = np.array(sorted(group), dtype=object)
        rng.shuffle(perm)
        nc = round(len(perm) * a.fraction)
        if len(perm) > 1:
            nc = max(1, min(len(perm) - 1, nc))
        else:
            nc = 0
        cal.extend(perm[:nc].tolist());
        ret.extend(perm[nc:].tolist())
        counts[name] = {'total': len(perm), 'retriever': len(perm) - nc, 'calibration': nc}
    assert set(ret).isdisjoint(cal) and set(ret) | set(cal) == set(ids)
    out = Path(a.out_dir);
    out.mkdir(parents=True, exist_ok=True)
    (out / 'retriever_train_ids.txt').write_text('\n'.join(sorted(ret)) + '\n')
    (out / 'calibration_train_ids.txt').write_text('\n'.join(sorted(cal)) + '\n')
    manifest = {'seed': a.seed, 'fraction_requested': a.fraction, 'n_retriever': len(ret), 'n_calibration': len(cal),
                'n_go_terms': len(universe), 'bins': counts,
                'train_ids_sha256': hashlib.sha256(Path(a.train_ids).read_bytes()).hexdigest(),
                'pid2pos_sha256': hashlib.sha256(Path(a.pid2pos).read_bytes()).hexdigest(),
                'go_ids_sha256': hashlib.sha256(Path(a.go_ids).read_bytes()).hexdigest()}
    (out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
