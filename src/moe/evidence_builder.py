"""Experiment D: turn ESM 5-NN outputs into GO evidence for MoE.

All evidence is derived from TRAIN annotations only. For train queries,
query itself must not occur among neighbours. Exact-sequence exclusion
must be handled by neighbour_builder using --exclusion_groups.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import numpy as np

LOG = logging.getLogger('moe.evidence_builder')


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def parse_go_id(value):
    s = str(value).strip()
    return int(s.split(':', 1)[1]) if s.upper().startswith('GO:') else int(s)


def load_annotation_map(path):
    """JSON {protein_id: [GO:..., ...]} or {protein_id: {GO:...: ...}}."""
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError('Annotation file must be a JSON mapping protein ID -> GO IDs')
    result = {}
    for pid, annotations in data.items():
        if isinstance(annotations, dict):
            annotations = annotations.keys()
        if not isinstance(annotations, (list, tuple, set, dict_keys_type())):
            raise ValueError(f'Unexpected annotations for {pid}: {type(annotations)}')
        result[str(pid)] = {parse_go_id(g) for g in annotations}
    return result


def dict_keys_type():
    return type({}.keys())


def load_go_bank(go_ids_path, go_embeddings_path):
    ids_raw = read_json(go_ids_path) if go_ids_path.suffix == '.json' else np.load(go_ids_path, allow_pickle=False).tolist()
    if isinstance(ids_raw, dict):
        raise ValueError('GO IDs must be ordered list, not dictionary')
    ids = [parse_go_id(x) for x in ids_raw]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate GO IDs')
    emb = np.load(go_embeddings_path, allow_pickle=False)
    if emb.ndim != 2 or emb.shape[0] != len(ids):
        raise ValueError(f'GO bank shape mismatch: {emb.shape}, ids={len(ids)}')
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if not np.isfinite(emb).all() or (norms < 1e-10).any():
        raise ValueError('Nonfinite/zero GO embeddings')
    return ids, emb / norms


def build_one_split(split, neighbour_dir, out_dir, train_ids, annotations,
                    go_ids, go_similarity, k, temperature, chunk_size):
    query_ids = [str(x) for x in read_json(neighbour_dir / f'{split}_protein_ids.json')]
    neighbours = read_json(neighbour_dir / f'{split}_neighbour_ids.json')
    scores = np.load(neighbour_dir / f'{split}_neighbour_scores.npy', allow_pickle=False)
    if len(query_ids) != len(neighbours) or scores.shape != (len(query_ids), k):
        raise ValueError(f'{split}: neighbour shape/order mismatch')
    index = {g: j for j, g in enumerate(go_ids)}
    n, G = len(query_ids), len(go_ids)
    direct = np.lib.format.open_memmap(out_dir / f'{split}_direct_support.npy', mode='w+', dtype='float32', shape=(n, G))
    semantic = np.lib.format.open_memmap(out_dir / f'{split}_semantic_support.npy', mode='w+', dtype='float32', shape=(n, G))
    similarity = np.lib.format.open_memmap(out_dir / f'{split}_neighbour_similarity.npy', mode='w+', dtype='float32', shape=(n, G))
    coverage = np.zeros(n, dtype=np.float32)
    train_set = set(train_ids)
    missing_ann = set()
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        block = np.zeros((end - start, G), dtype=np.float32)
        sim_block = np.zeros((end - start, G), dtype=np.float32)
        for row, i in enumerate(range(start, end)):
            ns = [str(x) for x in neighbours[i]]
            if len(ns) != k or len(set(ns)) != k:
                raise ValueError(f'{split}: bad neighbour count/duplicates for {query_ids[i]}')
            if split == 'train' and query_ids[i] in ns:
                raise ValueError(f'LEAKAGE: query {query_ids[i]} appears as own neighbour')
            if any(pid not in train_set for pid in ns):
                raise ValueError(f'LEAKAGE: non-training neighbour for {query_ids[i]}')
            sims = scores[i].astype(np.float64)
            if not np.isfinite(sims).all():
                raise ValueError(f'Nonfinite cosine scores for {query_ids[i]}')
            weights = np.exp((sims - sims.max()) / temperature)
            weights /= weights.sum()
            sim_block[row] = np.float32(np.dot(weights, sims))
            for pid, weight in zip(ns, weights):
                if pid not in annotations:
                    missing_ann.add(pid)
                    continue
                cols = [index[g] for g in annotations[pid] if g in index]
                if cols:
                    block[row, cols] += float(weight)
            coverage[i] = float(np.count_nonzero(block[row]))
        direct[start:end] = block
        # Weighted semantic expansion, bounded to [0,1], preserving direct evidence separately.
        # Max similarity to a supported annotated GO avoids high-frequency sum amplification.
        for row in range(end - start):
            cols = np.flatnonzero(block[row] > 0)
            if len(cols):
                sim = np.maximum(go_similarity[cols], 0.0)
                semantic[start + row] = np.max(sim * block[row, cols, None], axis=0)
            else:
                semantic[start + row] = 0
        similarity[start:end] = sim_block
        LOG.info('[%s] %d/%d', split, end, n)
    if missing_ann:
        raise ValueError(f'Missing TRAIN annotations for {len(missing_ann)} neighbours, e.g. {sorted(missing_ann)[:5]}')
    del direct, semantic, similarity
    return {'queries': n, 'mean_direct_GO_count': float(coverage.mean()), 'k': k}


def main():
    p = argparse.ArgumentParser(description='Experiment D GO evidence builder')
    p.add_argument('--neighbour_dir', type=Path, required=True)
    p.add_argument('--train_annotations', type=Path, required=True, help='TRAIN-ONLY pid_to_positives_bp.json')
    p.add_argument('--go_ids', type=Path, required=True, help='Ordered active GO IDs, exactly matching embedding rows')
    p.add_argument('--go_embeddings', type=Path, required=True, help='Retriever-v2 GO embeddings [G,D], same GO order')
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--chunk_size', type=int, default=64)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if a.temperature <= 0 or a.chunk_size < 1:
        raise ValueError('temperature and chunk_size must be positive')
    meta = read_json(a.neighbour_dir / 'metadata.json')
    k = int(meta['k'])
    train_ids = [str(x) for x in read_json(a.neighbour_dir / 'bank_protein_ids.json')]
    if len(train_ids) != meta['train_count']:
        raise ValueError('Neighbour metadata train count mismatch')
    ann = load_annotation_map(a.train_annotations)
    missing = set(train_ids) - set(ann)
    if missing:
        raise ValueError(f'Train bank IDs missing from annotations: {len(missing)}, e.g. {sorted(missing)[:5]}')
    go_ids, go_emb = load_go_bank(a.go_ids, a.go_embeddings)
    S = go_emb @ go_emb.T
    a.out_dir.mkdir(parents=True, exist_ok=True)
    marker = a.out_dir / 'metadata.json'
    if marker.exists() and not a.overwrite:
        raise FileExistsError(f'{marker} exists, use --overwrite')
    outputs = {}
    for split in ('train', 'val'):
        outputs[split] = build_one_split(split, a.neighbour_dir, a.out_dir,
                                         train_ids, ann, go_ids, S, k, a.temperature, a.chunk_size)
    (a.out_dir / 'go_ids.json').write_text(json.dumps(go_ids), encoding='utf-8')
    marker.write_text(json.dumps({
        'method': 'ESM5NN-weighted direct GO support + Retriever-v2 GO cosine max semantic transfer',
        'neighbour_dir': str(a.neighbour_dir),
        'train_annotations': str(a.train_annotations),
        'go_ids_source': str(a.go_ids), 'go_embeddings_source': str(a.go_embeddings),
        'G': len(go_ids), 'temperature': a.temperature,
        'training_annotations_only': True, 'splits': outputs,
        'semantic_rule': 'max_over_annotated_go(weighted_support * max(cosine,0))',
    }, indent=2), encoding='utf-8')
    LOG.info('DONE %s', a.out_dir)


if __name__ == '__main__':
    main()