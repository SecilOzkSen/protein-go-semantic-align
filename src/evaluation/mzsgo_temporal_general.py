"""MZSGO temporal or general BP evaluation using the frozen GO bank.
Run: python -m src.evaluation.mzsgo_temporal --mode general --checkpoint PATH --out-dir PATH
General test scores 20,639 2023 GO candidates; no test-set checkpoint selection.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.datasets.residue_store import ESMResidueStore
from src.training.collate import ContrastiveEmbCollator
from src.encoders import BioMedBERTEncoder
from src.models.alignment_model import ProteinGoAligner

MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'


def gid(x): return int(str(x).split(':')[-1])


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
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
    if args.batch_size < 1 or args.go_chunk_size < 1: raise ValueError('Invalid batch/chunk size')
    root = Path(args.root);
    branch = args.branch.lower();
    proc = root / 'processed' / branch
    candidate = proc / f'candidate_go_ids_{branch}.json'
    temporal = proc / f'temporal_go_ids_{branch}.json'
    if args.mode == 'general' and branch != 'bp':
        raise ValueError('General mode currently supports BP only')
    annot = proc / (f'pid_to_go_general_test_{branch}.json' if args.mode == 'general' else f'pid_to_go_temporal_{branch}.json')
    canonical = proc / 'go_texts_canonical_segmented_dual_2023_2025.jsonl'
    cache = root / 'go_cache' / branch / ('frozen_segment_parent_bank_general_test.pt' if args.mode == 'general' else 'frozen_segment_parent_bank.pt')
    for p in (candidate, temporal, annot, canonical, cache, Path(args.checkpoint)):
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
    elif len(labels) != 5712:
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
        if len(bank_ids) != 20673 or not set(ids) <= set(bank_ids):
            raise RuntimeError('General GO bank universe mismatch')
        if metadata.get('purpose') != 'general_BP_test_only' or metadata.get('added_unseen_go') != 328:
            raise RuntimeError('General bank provenance mismatch')
        base = root / 'go_cache' / branch / 'frozen_segment_parent_bank.pt'
        if metadata.get('base_bank_sha256') != digest(base):
            raise RuntimeError('General bank does not match original bank')
        # The expanded bank also includes 34 2025 temporal-only terms.
        # Exclude them to match the paper's 2023 general BP universe (20,639).
        ids = sorted(set(bank_ids) - set(unseen))
        if len(ids) != 20639:
            raise RuntimeError(f'Expected 20639 general BP GO IDs, got {len(ids)}')
    elif set(bank_ids) != set(ids):
        raise RuntimeError('Temporal bank GO IDs mismatch')
    if any(not set(v) <= set(ids) for v in labels.values()):
        raise RuntimeError('Some ground-truth GO IDs are missing from bank')
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
