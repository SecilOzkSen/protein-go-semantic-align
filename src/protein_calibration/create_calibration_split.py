"""Deterministic, protein-level cardinality-stratified calibration split.

The original validation/test splits are never read or modified.
GO universe is the union of annotated GO IDs in pid2pos, optionally
intersected with an explicitly supplied branch GO list.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

BINS = ((1, 5, '1_5'), (6, 10, '6_10'), (11, 20, '11_20'),
        (21, 40, '21_40'), (41, 80, '41_80'), (81, 160, '81_160'),
        (161, 10**12, '161plus'))


def go_int(value):
    return int(str(value).strip().split(':')[-1])


def read_go_ids(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        obj = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(obj, dict):
            obj = obj.get('ids', obj.get('go_ids', obj.get('terms', obj)))
        if isinstance(obj, dict):
            obj = list(obj.keys())
        if not isinstance(obj, list):
            raise ValueError('GO JSON must contain a list or ID-keyed object')
        return {go_int(x) for x in obj}
    return {go_int(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()}


def card_bin(n):
    return next(name for lo, hi, name in BINS if lo <= n <= hi)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_ids(path, ids):
    path.write_text(''.join(f'{x}\n' for x in sorted(ids)), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-ids', required=True)
    parser.add_argument('--pid2pos', required=True)
    parser.add_argument('--go-ids', help='Optional BP branch filter, JSON or TXT; annotated intersection used')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--expected-go-count', type=int, default=1943)
    parser.add_argument('--branch', default='BP')
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        parser.error('--fraction must be between 0 and 1')
    train_ids = [s.strip() for s in Path(args.train_ids).read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError('Duplicate train protein IDs')
    raw_labels = json.loads(Path(args.pid2pos).read_text(encoding='utf-8'))
    if not isinstance(raw_labels, dict):
        raise ValueError('pid2pos must be a protein-ID keyed JSON object')
    missing = set(train_ids) - set(raw_labels)
    if missing:
        raise ValueError(f'{len(missing)} train proteins absent from pid2pos, examples: {sorted(missing)[:5]}')
    labels = {pid: {go_int(x) for x in (gos or [])} for pid, gos in raw_labels.items()}
    universe = set().union(*labels.values()) if labels else set()
    branch_ids = read_go_ids(args.go_ids) if args.go_ids else None
    if branch_ids is not None:
        universe &= branch_ids
    if not universe:
        raise ValueError('Empty annotated GO universe')
    if args.expected_go_count and len(universe) != args.expected_go_count:
        raise ValueError(f'Expected {args.expected_go_count} GO IDs but found {len(universe)}. Check BP annotations/branch filter.')

    groups = {name: [] for _, _, name in BINS}
    excluded_zero = []
    protein_go = {}
    for pid in train_ids:
        gos = labels[pid] & universe
        if not gos:
            excluded_zero.append(pid)
            continue
        protein_go[pid] = gos
        groups[card_bin(len(gos))].append(pid)

    rng = np.random.default_rng(args.seed)
    retriever, calibration, bin_counts = [], [], {}
    for name, members in groups.items():
        shuffled = np.array(sorted(members), dtype=object)
        rng.shuffle(shuffled)
        n_cal = round(len(shuffled) * args.fraction)
        if len(shuffled) > 1:
            n_cal = max(1, min(len(shuffled) - 1, n_cal))
        else:
            n_cal = 0
        calibration.extend(shuffled[:n_cal].tolist())
        retriever.extend(shuffled[n_cal:].tolist())
        bin_counts[name] = {'total': len(members), 'retriever': len(members)-n_cal, 'calibration': n_cal}

    ret, cal, zero = set(retriever), set(calibration), set(excluded_zero)
    if (ret & cal) or (ret & zero) or (cal & zero) or ret | cal | zero != set(train_ids):
        raise RuntimeError('Split overlap or incomplete partition')
    if any(not protein_go[pid] for pid in ret | cal):
        raise RuntimeError('Zero-positive protein entered train split')

    ret_counts = Counter(g for pid in ret for g in protein_go[pid])
    cal_counts = Counter(g for pid in cal for g in protein_go[pid])
    train_terms = set(ret_counts) | set(cal_counts)
    only_cal = sorted(set(cal_counts) - set(ret_counts))
    only_ret = sorted(set(ret_counts) - set(cal_counts))
    never_train = sorted(universe - train_terms)
    singleton_terms = sorted(g for g in train_terms if ret_counts[g] + cal_counts[g] == 1)
    coverage = {
        'go_universe_size': len(universe),
        'go_observed_in_original_train': len(train_terms),
        'go_observed_in_retriever_train': len(ret_counts),
        'go_observed_in_calibration_train': len(cal_counts),
        'go_only_in_calibration': only_cal,
        'go_only_in_retriever': only_ret,
        'go_not_observed_in_original_train': never_train,
        'singleton_go_in_original_train': singleton_terms,
        'go_only_in_calibration_count': len(only_cal),
        'go_not_observed_in_original_train_count': len(never_train),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_ids(out / 'retriever_train_ids.txt', ret)
    write_ids(out / 'calibration_train_ids.txt', cal)
    write_ids(out / 'excluded_zero_positive_ids.txt', zero)
    write_ids(out / 'active_go_ids.txt', universe)
    (out / 'go_coverage.json').write_text(json.dumps(coverage, indent=2) + '\n', encoding='utf-8')
    manifest = {
        'branch_label': args.branch,
        'seed': args.seed, 'fraction_requested': args.fraction,
        'n_original_train': len(train_ids), 'n_annotated_train': len(protein_go),
        'n_excluded_zero_positive': len(zero),
        'n_retriever': len(ret), 'n_calibration': len(cal),
        'n_go_terms': len(universe), 'bins': bin_counts,
        'coverage_summary': {k: v for k, v in coverage.items() if not isinstance(v, list)},
        'inputs_sha256': {
            'train_ids': digest(args.train_ids),
            'pid2pos': digest(args.pid2pos),
            'go_ids': digest(args.go_ids) if args.go_ids else None,
        },
        'active_go_ids_sha256': digest(out / 'active_go_ids.txt'),
    }
    (out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    print(f'GO-only-calibration IDs: {only_cal[:20]} (total {len(only_cal)})')
    print('Original validation and test files were not modified.')


if __name__ == '__main__':
    main()
