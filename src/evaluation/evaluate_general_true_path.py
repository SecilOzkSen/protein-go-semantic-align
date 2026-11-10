"""BP general-test True Path Rule on SAVED V2 RAW scores, without model inference.

Propagate child predictions to ancestors in the 20,639-term candidate graph.
Uses MZSGO's original evaluate_annotations for RAW and propagated metrics.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def go_id(value):
    return 'GO:' + str(int(str(value).split(':')[-1])).zfill(7)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_mzsgo_util(repo):
    path = Path(repo) / 'utils' / 'util.py'
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('mzsgo_util_general_tpr', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(values):
    f, p, r, a, t = values[:5]
    return {'Fmax': float(f), 'precision': float(p), 'recall': float(r),
            'AUPR': float(a), 'threshold': float(t)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--root', default='/workspace/MZSGO_temporal')
    parser.add_argument('--mzsgo-repo', default='/workspace/MZSGO')
    args = parser.parse_args()
    raw_dir, out = Path(args.raw_dir), Path(args.out_dir)
    if raw_dir.resolve() == out.resolve():
        raise ValueError('RAW and TPR directories must differ')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {out}')
    proc = Path(args.root) / 'processed' / 'bp'
    raw_path = raw_dir / 'scores_01.npy'
    pids_path = raw_dir / 'protein_ids.json'
    gids_path = raw_dir / 'go_ids.json'
    raw_metrics_path = raw_dir / 'metrics.json'
    labels_path = proc / 'pid_to_go_general_test_bp.json'
    ontology_path = proc / 'go_texts_canonical_segmented_dual_2023_2025.jsonl'
    for p in (raw_path, pids_path, gids_path, raw_metrics_path, labels_path, ontology_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    pids = json.loads(pids_path.read_text())
    gids = [go_id(g) for g in json.loads(gids_path.read_text())]
    if len(pids) != 5712 or len(gids) != 20639:
        raise RuntimeError(f'Unexpected general BP dimensions: {len(pids)}, {len(gids)}')
    if len(set(pids)) != len(pids) or len(set(gids)) != len(gids):
        raise RuntimeError('Duplicate IDs')
    raw = np.load(raw_path, mmap_mode='r')
    if raw.shape != (5712, 20639):
        raise RuntimeError(f'Unexpected score shape: {raw.shape}')
    if not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise RuntimeError('Expected finite scores in [0,1]')
    existing = json.loads(raw_metrics_path.read_text())
    if existing.get('mode') != 'general' or existing.get('branch') != 'BP':
        raise RuntimeError('Not BP general evaluation')
    if existing.get('score_transform') != 'linear':
        raise RuntimeError('Unexpected score transform')

    labels = json.loads(labels_path.read_text())
    if set(pids) != set(labels):
        raise RuntimeError('Protein IDs do not match general labels')
    col = {g: i for i, g in enumerate(gids)}
    gold = np.zeros(raw.shape, dtype=np.uint8)
    for i, p in enumerate(pids):
        for term in labels[p]:
            g = go_id(term)
            if g not in col:
                raise RuntimeError(f'Ground-truth GO not in candidate universe: {g}')
            gold[i, col[g]] = 1
    if np.any(gold.sum(axis=1) == 0):
        raise RuntimeError('Test protein without positive annotations')

    parents = {}
    with ontology_path.open(encoding='utf-8') as handle:
        for line in handle:
            item = json.loads(line)
            parents[go_id(item['go_id'])] = {go_id(x) for x in item.get('is_a_parent_ids', [])}
    missing = set(gids) - set(parents)
    if missing:
        raise RuntimeError(f'{len(missing)} candidate GO terms absent from ontology; examples: {sorted(missing)[:5]}')

    # Kahn topological order for child -> parent edges. Leaves first.
    from collections import deque
    direct = {g: tuple(sorted(p for p in parents[g] if p in col)) for g in gids}
    indegree = {g: 0 for g in gids}
    for child in gids:
        for parent in direct[child]:
            indegree[parent] += 1
    queue = deque(g for g in gids if indegree[g] == 0)
    order = []
    while queue:
        child = queue.popleft()
        order.append(child)
        for parent in direct[child]:
            indegree[parent] -= 1
            if indegree[parent] == 0:
                queue.append(parent)
    if len(order) != len(gids):
        raise RuntimeError('GO ontology contains a cycle')
    edges = [(col[child], col[parent]) for child in order for parent in direct[child]]
    print(f'[GO] candidates={len(gids)} direct_is_a_edges={len(edges)}', flush=True)

    out.mkdir(parents=True, exist_ok=True)
    prop_path = out / 'scores_01_tpr.npy'
    prop = np.lib.format.open_memmap(prop_path, mode='w+', dtype=np.float32, shape=raw.shape)
    changed = 0
    for start in range(0, len(pids), 32):
        end = min(start + 32, len(pids))
        chunk = np.array(raw[start:end], dtype=np.float32, copy=True)
        for child, parent in edges:
            np.maximum(chunk[:, parent], chunk[:, child], out=chunk[:, parent])
        changed += int(np.count_nonzero(chunk > raw[start:end] + 1e-7))
        if np.any(chunk + 1e-7 < raw[start:end]):
            raise RuntimeError('TPR unexpectedly lowered a score')
        prop[start:end] = chunk
        if end % 512 == 0 or end == len(pids):
            print(f'[TPR] {end}/{len(pids)} proteins', flush=True)
    prop.flush()

    util = load_mzsgo_util(args.mzsgo_repo)
    y = torch.from_numpy(gold)
    raw_metric = metrics(util.evaluate_annotations(y, torch.from_numpy(np.array(raw)), 'general_raw_recheck'))
    expected = existing.get('general_original_mzsgo', {})
    for key in ('Fmax', 'AUPR'):
        if abs(raw_metric[key] - float(expected[key])) > 1e-6:
            raise RuntimeError(f'RAW {key} mismatch: {raw_metric[key]} vs {expected[key]}')
    print('[CHECK] RAW metrics match saved general test', flush=True)
    propagated_metric = metrics(util.evaluate_annotations(y, torch.from_numpy(np.array(prop)), 'general_tpr'))
    report = {
        'scope': 'BP general 20,639 candidate GO terms',
        'prediction_propagation': 'child-to-parent max over is_a graph',
        'ground_truth_propagated': False,
        'n_proteins': len(pids), 'n_candidates': len(gids),
        'n_direct_is_a_edges': len(edges), 'n_changed_scores': changed,
        'raw': raw_metric, 'propagated': propagated_metric,
        'delta_Fmax': propagated_metric['Fmax'] - raw_metric['Fmax'],
        'delta_AUPR': propagated_metric['AUPR'] - raw_metric['AUPR'],
        'source_raw_dir': str(raw_dir),
        'source_scores_sha256': digest(raw_path),
        'ontology_sha256': digest(ontology_path),
        'note': 'Ontology source is canonical JSONL is_a parents; compare with official MZSGO OBO adjacency before claiming exact official propagation equivalence.'
    }
    (out / 'metrics_tpr.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
