"""Audit ESM 5-NN homology and direct-transfer Fmax, without retraining.
Dependencies: numpy, biopython, parasail; existing StarGO evaluator.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from Bio import SeqIO
import parasail
from src.metrics.stargo_pfresgo_metrics import StarGOPFresGOEvaluator


def read_json(path):
    return json.loads(Path(path).read_text())


def global_identity(a, b):
    # Protein global alignment; identity = identical aligned positions / alignment columns.
    result = parasail.nw_trace_scan_16(a, b, 10, 1, parasail.blosum62)
    qa, qb = result.traceback.query, result.traceback.ref
    if not qa or len(qa) != len(qb):
        raise RuntimeError('Invalid alignment traceback')
    matches = sum(x == y and x != '-' for x, y in zip(qa, qb))
    return matches / len(qa), matches, len(qa)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fasta', required=True, help='PFresGO sequence FASTA containing train and validation proteins')
    ap.add_argument('--neighbour_dir', required=True)
    ap.add_argument('--val_dump', required=True)
    ap.add_argument('--evidence_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--go_graph_path', required=True)
    ap.add_argument('--ontology', default='bp')
    ap.add_argument('--topk', type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nd, vd, ed = map(Path, (args.neighbour_dir, args.val_dump, args.evidence_dir))
    query_ids = list(map(str, read_json(vd / 'protein_ids.json')))
    neighbour_query_ids = list(map(str, read_json(nd / 'val_protein_ids.json')))
    neighbour_ids = read_json(nd / 'val_neighbour_ids.json')
    scores = np.load(nd / 'val_neighbour_scores.npy', mmap_mode='r')
    direct = np.load(ed / 'val_direct_support.npy', mmap_mode='r')
    labels = np.load(vd / 'labels.int8.npy', mmap_mode='r')
    go_ids = np.load(vd / 'eval_go_ids.int64.npy')
    evidence_go_ids = list(map(int, read_json(ed / 'go_ids.json')))
    if go_ids.tolist() != evidence_go_ids:
        raise RuntimeError('GO ID order mismatch')
    if len(set(query_ids)) != len(query_ids) or len(set(neighbour_query_ids)) != len(neighbour_query_ids):
        raise RuntimeError('Duplicate protein IDs')
    if not (len(neighbour_query_ids) == len(neighbour_ids) == len(scores) == len(direct)):
        raise RuntimeError('Evidence/neighbour row mismatch')
    idx = {p: i for i, p in enumerate(neighbour_query_ids)}
    if set(query_ids) - set(idx):
        raise RuntimeError('Missing validation evidence')
    if labels.shape != (len(query_ids), len(go_ids)):
        raise RuntimeError('Validation label shape mismatch')
    if direct.shape[1] != len(go_ids):
        raise RuntimeError('Evidence GO dimension mismatch')
    print('Loading FASTA...', flush=True)
    seq = {}
    for rec in SeqIO.parse(args.fasta, 'fasta'):
        pid = rec.id
        if pid in seq:
            raise RuntimeError(f'Duplicate FASTA ID {pid}')
        seq[pid] = str(rec.seq).upper()
    required = set(query_ids)
    for pid in query_ids:
        required.update(map(str, neighbour_ids[idx[pid]][:args.topk]))
    missing = sorted(required - set(seq))
    if missing:
        raise RuntimeError(f'{len(missing)} protein IDs missing in FASTA, e.g. {missing[:12]}. Check FASTA ID format; do not silently skip.')
    bank = set(map(str, read_json(nd / 'bank_protein_ids.json')))
    if any(pid not in bank for pid in required - set(query_ids)):
        raise RuntimeError('Non-training neighbour detected')
    rows = []
    bins = [('0-30%', 0, .30), ('30-50%', .30, .50), ('50-70%', .50, .70), ('70-90%', .70, .90), ('90-99%', .90, .99), ('99-100%', .99, 1.000001)]
    print(f'Aligning {len(query_ids)} query proteins to {args.topk} neighbours...', flush=True)
    for t, pid in enumerate(query_ids):
        j = idx[pid]
        if pid in set(map(str, neighbour_ids[j][:args.topk])):
            raise RuntimeError(f'Self neighbour {pid}')
        pair_identities = []
        for rank, npid in enumerate(neighbour_ids[j][:args.topk]):
            npid = str(npid)
            identity, matches, aln_len = global_identity(seq[pid], seq[npid])
            pair_identities.append(identity)
            rows.append({'query_id': pid, 'neighbour_id': npid, 'rank': rank + 1,
                         'esm_cosine': float(scores[j, rank]), 'global_identity': identity,
                         'alignment_matches': matches, 'alignment_columns': aln_len,
                         'query_length': len(seq[pid]), 'neighbour_length': len(seq[npid])})
        if (t + 1) % 250 == 0:
            print(f'Aligned {t + 1}/{len(query_ids)} queries', flush=True)
    with (out / 'pairs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));
        w.writeheader();
        w.writerows(rows)
    top1 = np.array([r['global_identity'] for r in rows if r['rank'] == 1])
    max5 = np.array([max(r['global_identity'] for r in rows[i * args.topk:(i + 1) * args.topk]) for i in range(len(query_ids))])
    val_rows = np.array([idx[p] for p in query_ids])
    evidence = np.asarray(direct[val_rows], dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    evaluator = StarGOPFresGOEvaluator(goterms=go_ids, ontology=args.ontology, go_graph_path=Path(args.go_graph_path))

    def calc(mask):
        if not np.any(mask): return {'n': 0}
        metrics = evaluator.evaluate(y[mask], evidence[mask])
        return {'n': int(mask.sum()), **{k: float(v) for k, v in metrics.items()}}

    results = {'n_proteins': len(query_ids), 'topk': args.topk,
               'identity_definition': 'global identical alignment columns / all alignment columns, including gaps',
               'top1_identity': {'mean': float(top1.mean()), 'median': float(np.median(top1)),
                                 'p05': float(np.quantile(top1, .05)), 'p95': float(np.quantile(top1, .95))},
               'max5_identity': {'mean': float(max5.mean()), 'median': float(np.median(max5))},
               'direct_5nn_overall': calc(np.ones(len(query_ids), dtype=bool)),
               'by_top1_identity': {}, 'by_max5_identity': {}}
    for name, arr in [('by_top1_identity', top1), ('by_max5_identity', max5)]:
        for label, lo, hi in bins:
            results[name][label] = calc((arr >= lo) & (arr < hi))
    (out / 'summary.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    print('DONE:', out, flush=True)


if __name__ == '__main__': main()
