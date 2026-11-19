"""Unified MZSGO temporal/general evaluation, including full-train seen/unseen/H and retrieval, for BP/MF/CC.
General evaluation uses a branch-specific expanded bank, excluding temporal-only GO terms.
No test-set checkpoint selection.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.datasets.residue_store import ESMResidueStore
from src.training.collate import ContrastiveEmbCollator
from src.encoders import BioMedBERTEncoder
from src.models.alignment_model_v2_original import ProteinGoAligner

MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'


def gid(x): return int(str(x).split(':')[-1])

def normalize_go(x):
    s = str(x).strip()
    if s.startswith('GO:'):
        return 'GO:' + str(int(s[3:])).zfill(7)
    return 'GO:' + str(int(s)).zfill(7)


def parse_annotation_json(path):
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, dict):
        raise ValueError('Expected training annotations as {protein_id: [GO IDs]} JSON')
    terms = set()
    for protein, value in obj.items():
        if isinstance(value, dict):
            for key in ('go_ids', 'gos', 'labels', 'annotations', 'pos_go_ids'):
                if key in value:
                    value = value[key]
                    break
            else:
                raise ValueError(f'No GO list found for protein {protein}')
        if not isinstance(value, (list, tuple)):
            raise ValueError(f'GO annotations for {protein} are not a list')
        for item in value:
            terms.add(normalize_go(item))
    return terms


def mzsgo_metrics(y, scores):
    """Exact evaluation algorithm in MZSGO utils/util.py evaluate_annotations.

    Drop proteins with no positive annotations in the selected GO columns;
    threshold grid 0..1 step .01; protein-averaged precision over proteins
    with >=1 prediction, recall over all retained proteins; trapezoidal AUPR
    after sorting PR points by recall.
    """
    keep = np.asarray(y.sum(axis=1) > 0).reshape(-1)
    y, scores = y[keep], scores[keep]
    if not len(y):
        return None
    y = y.astype(bool)
    counts = y.sum(axis=1)
    precisions, recalls, fs = [], [], []
    for t in np.linspace(0, 1, 101):
        pred = scores >= t
        tp = (y & pred).sum(axis=1)
        pred_count = pred.sum(axis=1)
        has_pred = pred_count > 0
        precision = float((tp[has_pred] / pred_count[has_pred]).mean()) if has_pred.any() else 0.
        recall = float((tp / counts).mean())
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        precisions.append(precision); recalls.append(recall); fs.append(f)
    best = int(np.argmax(fs))
    order = np.argsort(recalls)
    return dict(Fmax=float(fs[best]), precision=float(precisions[best]),
                recall=float(recalls[best]), AUPR=float(np.trapz(np.asarray(precisions)[order], np.asarray(recalls)[order])),
                threshold=float(np.linspace(0, 1, 101)[best]), proteins_with_positive=int(len(y)),
                positive_pairs=int(y.sum()))





def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


class TestDataset(Dataset):
    def __init__(self, ids, labels, store): self.ids, self.labels, self.store = ids, labels, store

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        p = self.ids[i]
        return {'protein_id': p, 'prot_emb': self.store.get(p),
                'pos_go_ids': torch.tensor(self.labels[p], dtype=torch.long)}


def checkpoint_state(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    for _ in range(10):
        if isinstance(d, dict) and sum(torch.is_tensor(v) for v in d.values()) >= 5: break
        d = next((d[k] for k in ('model', 'model_state_dict', 'state_dict', 'net', 'module')
                  if isinstance(d, dict) and isinstance(d.get(k), dict)), None)
        if d is None: raise RuntimeError('Checkpoint model state not found')
    out = {}
    for k, v in d.items():
        if not torch.is_tensor(v): continue
        while True:
            prefix = next((p for p in ('trainer.model.', 'module.', 'model.') if k.startswith(p)), None)
            if prefix is None: break
            k = k[len(prefix):]
        out[k] = v
    return out


def original_metrics(y, s, repo, metric_tag="temporal_unseen"):
    if not (Path(repo) / 'utils' / 'util.py').is_file():
        raise FileNotFoundError('Original MZSGO utils/util.py not found')
    sys.path.insert(0, str(Path(repo).resolve()))
    import importlib.util

    util_path = Path(repo) / "utils" / "util.py"

    spec = importlib.util.spec_from_file_location(
        "mzsgo_original_util", util_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    evaluate_annotations = module.evaluate_annotations
    f, p, r, a, t = evaluate_annotations(torch.from_numpy(y), torch.from_numpy(s), metric_tag)
    return dict(Fmax=float(f), precision=float(p), recall=float(r), AUPR=float(a), threshold=float(t))


def retrieval_metrics(s, y, cols, ks):
    n, g = s.shape
    order = np.argsort(-s, axis=1, kind='stable')
    inv = np.empty_like(order)
    inv[np.arange(n)[:, None], order] = np.arange(g)[None, :]
    rank_list = [];
    aps = [];
    mrr = [];
    recall = {k: [] for k in ks}
    term_hits = {k: np.zeros(y.shape[1], dtype=np.int64) for k in ks}
    term_count = y.sum(0)
    for i in range(n):
        positives = np.flatnonzero(y[i]);
        ranks = np.sort(inv[i, cols[positives]] + 1)
        if not len(ranks): raise RuntimeError('Temporal test protein has no positives')
        rank_list.extend(ranks.tolist());
        mrr.append(1 / float(ranks[0]))
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
        for k in ks:
            hits = inv[i, cols[positives]] < min(k, g)
            recall[k].append(float(hits.mean()))
            term_hits[k][positives] += hits.astype(np.int64)
    result = {'temporal_positive_mAP': float(np.mean(aps)),
              'temporal_positive_MRR': float(np.mean(mrr)),
              'temporal_positive_rank_median': float(np.median(rank_list)),
              'temporal_positive_rank_p90': float(np.quantile(rank_list, .9))}
    present = term_count > 0
    for k in ks:
        result[f'temporal_recall@{k}'] = float(np.mean(recall[k]))
        result[f'macro_temporal_term_recall@{k}'] = float(np.mean(term_hits[k][present] / term_count[present]))
    return result


def retrieval_metrics_general(scores, labels, ks):
    """Protein-wise AP, recall and macro term recall for all general-test labels."""
    n, g = scores.shape
    ranks_sum = 0.0
    rank_count = 0
    aps, recalls = [], {k: [] for k in ks}
    hits = {k: np.zeros(g, dtype=np.int64) for k in ks}
    term_count = labels.sum(axis=0)
    for i in range(n):
        pos = np.flatnonzero(labels[i])
        if len(pos) == 0:
            raise RuntimeError('General test protein without positive labels')
        order = np.argsort(-scores[i], kind='stable')
        inv = np.empty(g, dtype=np.int32)
        inv[order] = np.arange(1, g+1)
        r = np.sort(inv[pos])
        ranks_sum += float(r.sum())
        rank_count += len(r)
        aps.append(float(np.mean(np.arange(1,len(r)+1) / r)))
        for k in ks:
            found = inv[pos] <= k
            recalls[k].append(float(found.mean()))
            hits[k][pos] += found.astype(np.int64)
    present = term_count > 0
    out = {'protein_wise_map': float(np.mean(aps)), 'positive_rank_mean': ranks_sum/rank_count}
    for k in ks:
        out[f'align_R@{k}'] = float(np.mean(recalls[k]))
        out[f'macro_term_recall@{k}'] = float(np.mean(hits[k][present]/term_count[present]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--train-labels-json', help='Full TRAIN+VAL protein->GO annotations JSON, required for general seen/unseen metrics')
    ap.add_argument('--mode', choices=['temporal', 'general'], default='temporal')
    ap.add_argument('--branch', choices=['BP', 'MF', 'CC'], default='BP')
    ap.add_argument('--root', default='/workspace/MZSGO_temporal')
    ap.add_argument('--mzsgo-repo', default='/workspace/MZSGO')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--go-chunk-size', type=int, default=128)
    ap.add_argument('--out-dir', required=True, help='New, empty output directory')
    ap.add_argument('--smoke-proteins', type=int, default=0)
    ap.add_argument('--ks', type=int, nargs='+', default=[50, 100, 200, 500, 1000])
    ap.add_argument('--score-transform', choices=['linear', 'sigmoid'], default='linear',
                    help='Fixed score mapping for MZSGO 0..1 threshold grid; NOT learned calibration')
    args = ap.parse_args()
    if args.mode == 'general' and not args.train_labels_json:
        ap.error('--train-labels-json is required for general mode; must contain full TRAIN+VAL annotations')
    if args.batch_size < 1 or args.go_chunk_size < 1: raise ValueError('Invalid batch/chunk size')
    root = Path(args.root);
    branch = args.branch.lower();
    proc = root / 'processed' / branch
    candidate = proc / f'candidate_go_ids_{branch}.json'
    temporal = proc / f'temporal_go_ids_{branch}.json'
    annot = proc / (f'pid_to_go_general_test_{branch}.json' if args.mode == 'general' else f'pid_to_go_temporal_{branch}.json')
    canonical = proc / 'go_texts_canonical_segmented_dual_2023_2025.jsonl'
    cache = root / 'go_cache' / branch / ('frozen_segment_parent_bank_general_test.pt' if args.mode == 'general' else 'frozen_segment_parent_bank.pt')
    for p in (candidate, temporal, annot, canonical, cache, Path(args.checkpoint)) + ((Path(args.train_labels_json),) if args.train_labels_json else ()):
        if not p.is_file(): raise FileNotFoundError(p)
    ids = sorted(map(gid, json.loads(candidate.read_text())))
    unseen = sorted(map(gid, json.loads(temporal.read_text())))
    if len(ids) != len(set(ids)) or len(unseen) != len(set(unseen)) or not set(unseen) <= set(ids):
        raise RuntimeError('Candidate/temporal GO universe invalid')
    if branch == 'bp' and (len(ids), len(unseen)) != (20345, 34):
        raise RuntimeError('BP protocol GO count mismatch')
    labels = {p: sorted(set(map(gid, v))) for p, v in json.loads(annot.read_text()).items()}
    if any(not v for v in labels.values()):
        raise RuntimeError('Test protein with no positive GO labels')
    if args.mode == 'temporal':
        if any(not set(v) <= set(unseen) for v in labels.values()):
            raise RuntimeError('Temporal ground truth contains non-temporal GO labels')
        if branch == 'bp' and len(labels) != 3687: raise RuntimeError('BP temporal protein count mismatch')
    elif branch == 'bp' and len(labels) != 5712:
        raise RuntimeError('BP general protein count mismatch')
    bank = torch.load(cache, map_location='cpu', weights_only=False)
    if bank.get('format') != 'mzsgo_frozen_segment_parent_v1': raise RuntimeError('Bank format mismatch')
    metadata = bank.get('metadata', {})
    for key, p in [('canonical_jsonl_sha256', canonical), ('candidate_ids_sha256', candidate),
                   ('temporal_ids_sha256', temporal)]:
        if metadata.get(key) != digest(p): raise RuntimeError(f'Bank hash mismatch: {key}')
    if metadata.get('model') != MODEL or metadata.get('normalized') is not False:
        raise RuntimeError('Encoder/normalization mismatch')
    if list(bank['segment_names']) != ['name', 'definition', 'is_a']:
        raise RuntimeError('Segment order mismatch')
    bank_ids = list(map(int, bank['go_ids']))
    if len(bank_ids) != len(set(bank_ids)):
        raise RuntimeError('Duplicate bank GO IDs')
    if args.mode == 'general':
        # General evaluation requires the branch-specific expanded frozen bank.
        # The ordinary temporal bank is NOT a valid substitute: it can omit
        # 2023 general-test GO terms that were not present in the train universe.
        base = root / 'go_cache' / branch / 'frozen_segment_parent_bank.pt'
        if not base.is_file():
            raise FileNotFoundError(f'Base frozen GO bank missing: {base}')
        if not set(ids) <= set(bank_ids):
            raise RuntimeError('General GO bank does not contain all training candidates')
        if metadata.get('base_bank_sha256') != digest(base):
            raise RuntimeError('General bank does not match original bank')
        if branch == 'bp':
            # Preserve the existing published BP general-test protocol exactly.
            if len(bank_ids) != 20673:
                raise RuntimeError('General BP bank universe mismatch')
            if metadata.get('purpose') != 'general_BP_test_only' or metadata.get('added_unseen_go') != 328:
                raise RuntimeError('General BP bank provenance mismatch')
        else:
            # CC/MF must use an independently constructed expanded bank.
            # Reject an unprovenanced or accidentally copied temporal bank.
            purpose = str(metadata.get('purpose', '')).lower()
            if 'general' not in purpose or branch not in purpose:
                raise RuntimeError(
                    f'General {args.branch} bank provenance mismatch: purpose={metadata.get("purpose")!r}. '
                    'Build a branch-specific general-test frozen GO bank first.'
                )
            if len(bank_ids) <= len(set(ids)):
                raise RuntimeError('General bank is not expanded beyond training candidates')
        # Exclude 2025 temporal-only terms, as in the existing BP protocol.
        ids = sorted(set(bank_ids) - set(unseen))
        if branch == 'bp' and len(ids) != 20639:
            raise RuntimeError(f'Expected 20639 general BP GO IDs, got {len(ids)}')
        if not ids:
            raise RuntimeError('Empty general candidate universe')
        print(f'[general] {args.branch}: {len(ids)} GO candidates, '
              f'{len(labels)} proteins, {len(bank_ids)} bank entries', flush=True)
    elif set(bank_ids) != set(ids):
        raise RuntimeError('Temporal bank GO IDs mismatch')
    if any(not set(v) <= set(ids) for v in labels.values()):
        missing = sorted(set().union(*(set(v) for v in labels.values())) - set(ids))
        raise RuntimeError(f'Some ground-truth GO IDs are missing from the evaluation universe: {missing[:20]}')
    index = {g: i for i, g in enumerate(bank_ids)}
    selection = torch.tensor([index[g] for g in ids], dtype=torch.long)
    seg = bank['segment_embs'].index_select(0, selection).float()
    present = bank['segment_present'].index_select(0, selection).bool()
    parent = bank['parent_embs'].index_select(0, selection).float()
    parent_present = bank['parent_present'].index_select(0, selection).bool()
    if seg.shape != (len(ids), 3, 768) or parent.shape[0] != len(ids):
        raise RuntimeError('Bank tensor shape mismatch')
    store = ESMResidueStore(embed_dir=str(root / 'protein_embeddings' / 'esm2_residue_overlap'),
                            max_len=1000, overlap=256)
    pids = sorted(labels)
    if not all(store.has(p) for p in pids): raise RuntimeError('Missing test ESM-2 embeddings')
    if args.smoke_proteins: pids = pids[:args.smoke_proteins]
    loader = DataLoader(TestDataset(pids, labels, store), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ContrastiveEmbCollator())
    device = torch.device(args.device)
    encoder = BioMedBERTEncoder(model_name=MODEL, device=device, max_length=512,
                                attention_pooling_strategy='mean', attn_hidden=128,
                                attn_dropout=.1, gradient_checkpointing=False)
    model = ProteinGoAligner(d_h=1280, d_g=768, d_z=768, go_encoder=encoder,
                             normalize=True, protein_pool_type='mean_attn_gate',
                             go_pool_type='mean', go_segment_representation_mode='segments_only').to(device)
    state = checkpoint_state(args.checkpoint)
    model.load_state_dict(state, strict=True)
    print(f'[CHECK] V2 original architecture strict load: {len(state)} tensors', flush=True)
    model.go_chunk_size = args.go_chunk_size
    model.eval();
    model.go_encoder.eval()
    name = seg[:, 0].to(device);
    definition = seg[:, 1].to(device)
    dp = present[:, 1].to(device);
    parent = parent.to(device)
    pp = parent_present.to(device);
    raw = seg.mean(1).to(device)
    chunks = [];
    ordered = []
    with torch.inference_mode():
        for i, batch in enumerate(loader, 1):
            h = batch['prot_emb_pad'].to(device).float()
            mask = batch['prot_attn_mask'].to(device).bool()
            scores = model.score_fine_grained(h, mask, name, definition, dp, parent, pp, raw)
            if not torch.isfinite(scores).all(): raise RuntimeError('Nonfinite scores')
            chunks.append(scores.cpu().numpy().astype(np.float32))
            ordered.extend(batch['protein_ids'])
            if i % 20 == 0: print(f'[{args.mode}] batches={i} proteins={len(ordered)}', flush=True)
    raw_scores = np.concatenate(chunks)
    if raw_scores.shape != (len(pids), len(ids)): raise RuntimeError('Score shape mismatch')
    # Fixed monotonic mapping only, not a learned calibration model.
    score01 = (1 / (1 + np.exp(-np.clip(raw_scores, -80, 80)))) if args.score_transform == 'sigmoid' else np.clip((raw_scores + 1) / 2, 0, 1)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Output directory not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'scores_raw.npy', raw_scores)
    np.save(out / 'scores_01.npy', score01.astype(np.float32))
    (out / 'protein_ids.json').write_text(json.dumps(ordered, indent=2))
    (out / 'go_ids.json').write_text(json.dumps([f'GO:{g:07d}' for g in ids], indent=2))
    result = {'mode': args.mode, 'branch': args.branch, 'n_proteins': len(ordered),
              'n_candidates': len(ids), 'n_temporal_go': len(unseen),
              'checkpoint_sha256': digest(args.checkpoint), 'bank_sha256': digest(cache),
              'score_transform': args.score_transform,
              'annotation_sha256': digest(annot),
              'candidate_universe_sha256': hashlib.sha256(
                  json.dumps(ids).encode('utf-8')).hexdigest(),
              'note': 'Test-selected Fmax is an oracle benchmark metric, not a validation-selected operating threshold. Fixed score mapping is not probability calibration.'}
    if args.smoke_proteins:
        result['smoke_test_only'] = True
    else:
        columns = {g: i for i, g in enumerate(ids)}
        if args.mode == 'general':
            # Full candidate universe: negatives outside observed test labels are retained.
            y = np.zeros((len(ordered), len(ids)), dtype=np.int32)
            for i, p in enumerate(ordered):
                for g in labels[p]: y[i, columns[g]] = 1
            result['general_original_mzsgo'] = original_metrics(
                y, score01.astype(np.float32), args.mzsgo_repo, 'general')
            # Separate diagnostic, not comparable to the paper's full-universe Fmax.
            # Do not calculate full-universe argsort metrics: high memory cost.
            result['general_positive_pairs'] = int(y.sum())
            result['general_unique_positive_terms'] = int((y.sum(0) > 0).sum())
            result['general_unseen_positive_pairs'] = int(y[:, [columns[g] for g in unseen if g in columns]].sum())
            # Seen/unseen membership is based on ALL GO annotations used in full training,
            # not the old train-only split and not temporal-only GO IDs.
            train_json = json.loads(Path(args.train_labels_json).read_text())
            if not isinstance(train_json, dict):
                raise ValueError('Full-training annotation JSON must be {protein_id: [GO IDs]}')
            full_train_ids_path = Path(args.checkpoint).parent / 'full_train_ids.txt'
            if full_train_ids_path.is_file():
                full_ids = set(full_train_ids_path.read_text().splitlines())
                if set(train_json) != full_ids:
                    raise RuntimeError(f'Train annotation protein IDs do not match full_train_ids.txt: annotations={len(train_json)}, full_ids={len(full_ids)}, missing={len(full_ids-set(train_json))}, extra={len(set(train_json)-full_ids)}')
            if set(ordered) & set(train_json):
                raise RuntimeError('DATA LEAKAGE: general test proteins overlap full training annotation IDs')
            train_go = parse_annotation_json(args.train_labels_json)
            go_names = [normalize_go(g) for g in ids]
            pos_by_go = y.sum(axis=0) > 0
            seen_mask = np.asarray([g in train_go and bool(pos_by_go[j]) for j,g in enumerate(go_names)], dtype=bool)
            unseen_mask = np.asarray([g not in train_go and bool(pos_by_go[j]) for j,g in enumerate(go_names)], dtype=bool)
            if not np.array_equal(seen_mask | unseen_mask, pos_by_go):
                raise RuntimeError('Seen/unseen partition failed')
            seen_metrics = mzsgo_metrics(y[:,seen_mask], score01[:,seen_mask])
            unseen_metrics = mzsgo_metrics(y[:,unseen_mask], score01[:,unseen_mask])
            if seen_metrics is None or unseen_metrics is None:
                raise RuntimeError('Seen or unseen test-positive subset empty; cannot compute H')
            a_seen, a_unseen = seen_metrics['AUPR'], unseen_metrics['AUPR']
            h = 2*a_seen*a_unseen/(a_seen+a_unseen) if a_seen+a_unseen else 0.0
            result['seen_unseen'] = {
                'definition': 'GO term observed in full TRAIN+VAL annotations; only terms with test positives included in per-group score matrices',
                'train_labels_sha256': digest(args.train_labels_json),
                'n_train_go': len(train_go),
                'n_seen_test_go': int(seen_mask.sum()),
                'n_unseen_test_go': int(unseen_mask.sum()),
                'seen': seen_metrics, 'unseen': unseen_metrics, 'H': float(h),
                'unseen_go_ids': [go_names[j] for j in np.flatnonzero(unseen_mask)],
            }
            # Recompute overall with generic implementation as a diagnostic.
            # Original MZSGO util remains the authoritative overall metric.
            result['overall_generic_crosscheck'] = mzsgo_metrics(y, score01)
            result['overall_crosscheck_abs_difference'] = {
                k: abs(result['general_original_mzsgo'][k] - result['overall_generic_crosscheck'][k])
                for k in ('Fmax', 'AUPR')
            }
            if max(result['overall_crosscheck_abs_difference'].values()) > 1e-5:
                print('[WARNING] Generic and original MZSGO overall metric implementations differ. Do not claim exact parity for seen/unseen without checking original util semantics.', flush=True)
            # Retrieval over all general-test GO labels, not temporal-only labels.
            result['general_retrieval'] = retrieval_metrics_general(raw_scores, y, args.ks)

        else:
            ucols = np.array([columns[g] for g in unseen])
            y = np.zeros((len(ordered), len(unseen)), dtype=np.int32)
            uindex = {g: i for i, g in enumerate(unseen)}
            for i, p in enumerate(ordered):
                for g in labels[p]: y[i, uindex[g]] = 1
            result['unseen_original_mzsgo'] = original_metrics(
                y, score01[:, ucols].astype(np.float32), args.mzsgo_repo)
            result['full_universe_retrieval'] = retrieval_metrics(raw_scores, y, ucols, args.ks)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print('Saved:', out)


if __name__ == '__main__': main()
