"""Apply 2023 GO is_a True Path Rule to saved V2 BP general-test scores.

No inference. Intermediate non-candidate GO nodes are used for propagation.
Uses original MZSGO evaluate_annotations for RAW and TPR metrics.
"""
import argparse
import hashlib
import importlib.util
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch


def gid(value):
    return 'GO:' + str(int(str(value).split(':')[-1])).zfill(7)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_obo(path):
    terms = {}
    current = None

    def save():
        if current and current.get('id'):
            key = current['id']
            if key in terms:
                raise RuntimeError(f'Duplicate ontology ID: {key}')
            terms[key] = current.copy()

    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line == '[Term]':
                save()
                current = {'parents': set(), 'alt': set(), 'obsolete': False, 'replaced_by': set()}
            elif line.startswith('['):
                save()
                current = None
            elif current is not None:
                if line.startswith('id: GO:'):
                    current['id'] = gid(line.split(': ', 1)[1])
                elif line.startswith('alt_id: GO:'):
                    current['alt'].add(gid(line.split(': ', 1)[1]))
                elif line.startswith('is_a: GO:'):
                    current['parents'].add(gid(line.split(': ', 1)[1].split()[0]))
                elif line == 'is_obsolete: true':
                    current['obsolete'] = True
                elif line.startswith('replaced_by: GO:'):
                    current['replaced_by'].add(gid(line.split(': ', 1)[1].split()[0]))
    save()
    aliases = {}
    for key, term in terms.items():
        for alt in term['alt']:
            if alt in terms or (alt in aliases and aliases[alt] != key):
                raise RuntimeError(f'Ambiguous alternate ID: {alt}')
            aliases[alt] = key
    return terms, aliases


def load_util(repo):
    path = Path(repo) / 'utils' / 'util.py'
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('mzsgo_general_tpr_util', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_metrics(values):
    f, p, r, a, t = values[:5]
    return dict(Fmax=float(f), precision=float(p), recall=float(r), AUPR=float(a), threshold=float(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--root', default='/workspace/MZSGO_temporal')
    ap.add_argument('--mzsgo-repo', default='/workspace/MZSGO')
    ap.add_argument('--obo-path', default='/workspace/MZSGO/data/go_2023_01_01.obo')
    ap.add_argument('--chunk-size', type=int, default=32)
    args = ap.parse_args()
    raw_dir, out = Path(args.raw_dir), Path(args.out_dir)
    if raw_dir.resolve() == out.resolve():
        raise ValueError('Output must be separate from RAW directory')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {out}')
    if args.chunk_size < 1:
        raise ValueError('chunk-size must be positive')

    files = {
        'raw': raw_dir / 'scores_01.npy',
        'pids': raw_dir / 'protein_ids.json',
        'gids': raw_dir / 'go_ids.json',
        'metrics': raw_dir / 'metrics.json',
        'labels': Path(args.root) / 'processed/bp/pid_to_go_general_test_bp.json',
        'obo': Path(args.obo_path),
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    pids = json.loads(files['pids'].read_text())
    gids = [gid(g) for g in json.loads(files['gids'].read_text())]
    if len(pids) != 5712 or len(gids) != 20639:
        raise RuntimeError(f'Unexpected general BP dimensions: {len(pids)}, {len(gids)}')
    if len(set(pids)) != len(pids) or len(set(gids)) != len(gids):
        raise RuntimeError('Duplicate IDs')
    raw = np.load(files['raw'], mmap_mode='r')
    if raw.shape != (5712, 20639) or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise RuntimeError('Invalid RAW score matrix')
    previous = json.loads(files['metrics'].read_text())
    if previous.get('mode') != 'general' or previous.get('branch') != 'BP' or previous.get('score_transform') != 'linear':
        raise RuntimeError('Unexpected RAW evaluation protocol')

    terms, aliases = read_obo(files['obo'])

    def resolve(x):
        return aliases.get(x, x)

    unresolved = sorted(g for g in gids if resolve(g) not in terms)
    obsolete = sorted(g for g in gids if resolve(g) in terms and terms[resolve(g)]['obsolete'])
    if unresolved or obsolete:
        raise RuntimeError(f'2023 OBO candidate coverage failed: missing={len(unresolved)} {unresolved[:12]}, '
                           f'obsolete={len(obsolete)} {[(g, sorted(terms[resolve(g)]["replaced_by"])) for g in obsolete[:12]]}')
    canonical = [resolve(g) for g in gids]
    if len(set(canonical)) != len(canonical):
        raise RuntimeError('Multiple scored candidate IDs resolve to the same GO term')
    print(f'[GO] 2023 ontology terms={len(terms)} candidate_coverage={len(canonical)}/{len(gids)} '
          f'alt_id_resolved={sum(g != c for g, c in zip(gids, canonical))}', flush=True)

    # Traverse through intermediate GO terms, even if they have no scored column.
    # A candidate's score travels to every scored ancestor through the full is_a DAG.
    reachable = set(canonical)
    stack = list(canonical)
    direct = {}
    while stack:
        node = stack.pop()
        if node in direct:
            continue
        if node not in terms:
            raise RuntimeError(f'Unresolved intermediate GO ancestor: {node}')
        term = terms[node]
        if term['obsolete']:
            raise RuntimeError(f'Obsolete intermediate GO ancestor: {node}')
        parents = set()
        for parent in term['parents']:
            p = resolve(parent)
            if p not in terms or terms[p]['obsolete']:
                raise RuntimeError(f'Invalid parent {parent} of {node}')
            parents.add(p)
            if p not in reachable:
                reachable.add(p)
                stack.append(p)
        direct[node] = parents
    indegree = {node: 0 for node in reachable}
    for child, parents in direct.items():
        for parent in parents:
            indegree[parent] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for parent in direct[node]:
            indegree[parent] -= 1
            if indegree[parent] == 0:
                queue.append(parent)
    if len(order) != len(reachable):
        raise RuntimeError('Cycle in GO is_a graph')
    nodes = sorted(reachable)
    node_col = {node: i for i, node in enumerate(nodes)}
    candidate_cols = np.asarray([node_col[g] for g in canonical], dtype=np.int32)
    edges = [(node_col[child], node_col[parent]) for child in order for parent in direct[child]]
    print(f'[GO] reachable_nodes={len(nodes)} intermediate_nodes={len(nodes) - len(gids)} '
          f'direct_is_a_edges={len(edges)}', flush=True)

    labels = json.loads(files['labels'].read_text())
    if set(labels) != set(pids):
        raise RuntimeError('Protein IDs do not match labels')
    candidate_index = {g: i for i, g in enumerate(gids)}
    gold = np.zeros(raw.shape, dtype=np.uint8)
    for i, protein in enumerate(pids):
        for term in labels[protein]:
            g = gid(term)
            if g not in candidate_index:
                raise RuntimeError(f'Ground truth missing from scored GO IDs: {g}')
            gold[i, candidate_index[g]] = 1
    if np.any(gold.sum(axis=1) == 0):
        raise RuntimeError('Protein without positive annotations')

    util = load_util(args.mzsgo_repo)
    y = torch.from_numpy(gold)
    raw_metrics = format_metrics(util.evaluate_annotations(y, torch.from_numpy(np.asarray(raw).copy()), 'general_raw_recheck'))
    expected = previous.get('general_original_mzsgo', {})
    for key in ('Fmax', 'AUPR'):
        if key not in expected or abs(raw_metrics[key] - float(expected[key])) > 1e-6:
            raise RuntimeError(f'RAW {key} mismatch: {raw_metrics[key]} vs {expected.get(key)}')
    print('[CHECK] RAW metrics match original general evaluation', flush=True)

    out.mkdir(parents=True, exist_ok=True)
    prop_path = out / 'scores_01_tpr.npy'
    prop = np.lib.format.open_memmap(prop_path, mode='w+', dtype=np.float32, shape=raw.shape)
    changed = 0
    for start in range(0, len(pids), args.chunk_size):
        end = min(start + args.chunk_size, len(pids))
        chunk = np.zeros((end - start, len(nodes)), dtype=np.float32)
        chunk[:, candidate_cols] = raw[start:end]
        for child, parent in edges:
            np.maximum(chunk[:, parent], chunk[:, child], out=chunk[:, parent])
        result = chunk[:, candidate_cols]
        if np.any(result + 1e-7 < raw[start:end]):
            raise RuntimeError('TPR lowered a score')
        changed += int(np.count_nonzero(result > raw[start:end] + 1e-7))
        prop[start:end] = result
        if end % 512 == 0 or end == len(pids):
            print(f'[TPR] {end}/{len(pids)} proteins', flush=True)
    prop.flush()
    propagated_metrics = format_metrics(util.evaluate_annotations(y, torch.from_numpy(np.asarray(prop)), 'general_tpr'))
    report = {
        'scope': 'BP general test, 2023 GO is_a, all 20639 scored candidates',
        'ground_truth_propagated': False,
        'prediction_propagation': 'child-to-ancestor max, including intermediate unscored ontology nodes',
        'n_proteins': len(pids), 'n_candidates': len(gids),
        'n_ontology_nodes_reachable': len(nodes), 'n_direct_is_a_edges': len(edges),
        'n_candidate_alt_ids_resolved': sum(g != c for g, c in zip(gids, canonical)),
        'n_changed_scores': changed,
        'raw': raw_metrics, 'propagated': propagated_metrics,
        'delta_Fmax': propagated_metrics['Fmax'] - raw_metrics['Fmax'],
        'delta_AUPR': propagated_metrics['AUPR'] - raw_metrics['AUPR'],
        'source_raw_dir': str(raw_dir),
        'source_scores_sha256': digest(files['raw']),
        'ontology_path': str(files['obo']), 'ontology_sha256': digest(files['obo']),
        'note': 'Full 2023 is_a ancestor propagation; official MZSGO adjacency equivalence not independently verified.'
    }
    (out / 'metrics_tpr.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
