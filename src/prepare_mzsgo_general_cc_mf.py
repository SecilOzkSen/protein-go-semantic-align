#!/usr/bin/env python3
"""Prepare CC/MF 2023 general-test ground truth from MZSGO CAFA5 SwissProt.

CPU-only. No ontology propagation (matches MZSGO preprocess_dataset).
Does not overwrite existing files. Reports BP parity as a diagnostic.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

NS = {'bp': 'biological_process', 'cc': 'cellular_component', 'mf': 'molecular_function'}


def goid(value):
    return f"GO:{int(str(value).split(':')[-1]):07d}"


def read_obo(path):
    result = {}
    row = None
    def flush():
        if row and row.get('id') and not row.get('obsolete') and row.get('namespace'):
            result[row['id']] = row['namespace']
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line == '[Term]':
                flush(); row = {}
            elif line.startswith('['):
                flush(); row = None
            elif row is not None:
                if line.startswith('id: '): row['id'] = line[4:].strip()
                elif line.startswith('namespace: '): row['namespace'] = line[11:].strip()
                elif line == 'is_obsolete: true': row['obsolete'] = True
    flush()
    return result


def load_raw(path, namespaces):
    out = {b: {} for b in NS}
    all_pids = set()
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            if not line.startswith('>'): continue
            parts = line[1:].strip().split()
            if len(parts) != 2:
                raise ValueError(f'Unexpected FASTA header: {line[:120]}')
            pid, labels = parts
            if pid in all_pids: raise ValueError(f'Duplicate protein: {pid}')
            all_pids.add(pid)
            branch_labels = defaultdict(set)
            for raw in labels.split(';'):
                go = goid(raw)
                ns = namespaces.get(go)
                if ns: branch_labels[ns].add(go)
            for branch, ns in NS.items():
                if branch_labels[ns]: out[branch][pid] = sorted(branch_labels[ns])
    return out, len(all_pids)


def load_candidate(path):
    data = json.loads(path.read_text())
    if not isinstance(data, list): raise ValueError(f'Expected list: {path}')
    ids = [goid(x) for x in data]
    if len(ids) != len(set(ids)): raise ValueError(f'Duplicate candidate GO: {path}')
    return set(ids)


def normalize_bp_labels(path):
    raw = json.loads(path.read_text())
    return {str(pid): sorted(set(goid(x) for x in vals)) for pid, vals in raw.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/workspace/MZSGO_temporal'))
    p.add_argument('--test-source', type=Path, default=Path('/workspace/MZSGO/data/sequence/cafa5_test_in_swissprot.txt'))
    p.add_argument('--obo', type=Path, default=Path('/workspace/MZSGO/data/go_2023_01_01.obo'))
    p.add_argument('--branches', nargs='+', choices=['CC','MF'], default=['CC','MF'])
    p.add_argument('--allow-bp-parity-mismatch', action='store_true', help='Proceed despite BP parity discrepancy')
    p.add_argument('--write', action='store_true', help='Write files after checks (default: dry run)')
    a = p.parse_args()
    for x in [a.test_source, a.obo]:
        if not x.is_file(): raise FileNotFoundError(x)
    namespaces = read_obo(a.obo)
    annotations, total = load_raw(a.test_source, namespaces)
    print(f'[SOURCE] proteins={total}, 2023 active GO terms={len(namespaces)}')
    bp_path = a.root / 'processed/bp/pid_to_go_general_test_bp.json'
    if bp_path.is_file():
        reference = normalize_bp_labels(bp_path)
        observed = annotations['bp']
        shared = set(reference) & set(observed)
        mismatch = sum(reference[p] != observed[p] for p in shared)
        print(f'[BP PARITY] reference={len(reference)}, source={len(observed)}, shared={len(shared)}, '
              f'labels_differ={mismatch}, ref_only={len(set(reference)-set(observed))}, '
              f'source_only={len(set(observed)-set(reference))}')
        if (mismatch or set(reference) != set(observed)) and not a.allow_bp_parity_mismatch:
            raise RuntimeError('BP reference differs from direct 2023 namespace filtering. '
                               'Do not write CC/MF until BP selection/filtering is understood. '
                               'If discrepancy is expected and verified, use --allow-bp-parity-mismatch.')
    else:
        print('[WARNING] BP reference not found; parity cannot be checked')
    outputs = []
    for branch_upper in a.branches:
        branch = branch_upper.lower()
        proc = a.root / 'processed' / branch
        candidate_file = proc / f'candidate_go_ids_{branch}.json'
        temporal_file = proc / f'temporal_go_ids_{branch}.json'
        if not candidate_file.is_file(): raise FileNotFoundError(candidate_file)
        if not temporal_file.is_file(): raise FileNotFoundError(temporal_file)
        candidate = load_candidate(candidate_file)
        temporal = load_candidate(temporal_file)
        if not temporal <= candidate: raise RuntimeError(f'{branch}: temporal IDs not subset of candidates')
        labels = annotations[branch]
        positive = set().union(*(set(v) for v in labels.values())) if labels else set()
        missing = sorted(positive - candidate)
        if not all(namespaces.get(x) == NS[branch] for x in missing):
            raise RuntimeError(f'{branch}: missing GO has wrong 2023 namespace')
        files = [
            (proc / f'general_test_ids_{branch}.txt', ''.join(p + '\n' for p in sorted(labels))),
            (proc / f'pid_to_go_general_test_{branch}.json', json.dumps(dict(sorted(labels.items())), indent=2) + '\n'),
            (proc / f'general_test_missing_go_{branch}.json', json.dumps({
                'branch': branch_upper, 'missing_go_ids': missing,
                'missing_count': len(missing), 'general_test_proteins': len(labels),
                'unique_positive_go': len(positive), 'candidate_count': len(candidate),
                'source': str(a.test_source), 'ontology': str(a.obo)
            }, indent=2) + '\n'),
        ]
        print(f'[{branch_upper}] proteins={len(labels)} positive_terms={len(positive)} '
              f'candidate={len(candidate)} temporal={len(temporal)} missing={len(missing)} '
              f'general_candidate_size={len((candidate | set(missing)) - temporal)}')
        outputs.extend(files)
    existing = [str(path) for path, _ in outputs if path.exists()]
    if existing: raise FileExistsError('Refusing overwrite: ' + ', '.join(existing))
    if not a.write:
        print('[DRY RUN] No files written. Add --write to save all six files.')
        return
    for path, contents in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        print('[SAVED]', path)


if __name__ == '__main__':
    main()
