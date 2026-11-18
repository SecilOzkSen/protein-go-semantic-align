#!/usr/bin/env python3
"""Fast, checkpoint-faithful NPZ extraction for local-only MZSGO V2.

Caches frozen GO projections once instead of repeating them per protein.
Uses the original model's local-window extraction and SmoothMIL to preserve
its scoring semantics. Optional --verify-reference checks scores against
score_fine_grained on the first batch before writing output.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.residue_store import ESMResidueStore
from src.encoders import BioMedBERTEncoder
from src.models.alignment_model_v2_original import ProteinGoAligner

MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'


def gid(x):
    return int(str(x).split(':')[-1])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    for _ in range(10):
        if isinstance(d, dict) and sum(torch.is_tensor(v) for v in d.values()) >= 5:
            break
        d = next((d[k] for k in ('model', 'model_state_dict', 'state_dict', 'net', 'module')
                  if isinstance(d, dict) and isinstance(d.get(k), dict)), None)
        if d is None:
            raise RuntimeError('No model state in checkpoint')
    result = {}
    for k, v in d.items():
        if not torch.is_tensor(v):
            continue
        for prefix in ('trainer.model.', 'module.', 'model.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
        result[k] = v
    return result


class Proteins(Dataset):
    def __init__(self, pids, store):
        self.pids, self.store = pids, store

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        return pid, self.store.get(pid)


def collate(batch):
    ids, tensors = zip(*batch)
    tensors = [torch.as_tensor(x) for x in tensors]
    lengths = [x.shape[0] for x in tensors]
    h = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    mask = torch.arange(h.shape[1])[None, :] < torch.tensor(lengths)[:, None]
    return list(ids), h, mask


class FastLocalScorer:
    def __init__(self, model, name, definition, definition_present, parent, parent_present, chunk_size):
        self.model = model
        self.chunk_size = chunk_size
        self.dp = definition_present.bool()
        self.pm = parent_present.bool()
        self.parent_strength = model.parent_strength
        self.go_chunk_size = chunk_size
        # Project GO vectors exactly once. The frozen model uses the same
        # project_go operation for every protein.
        with torch.inference_mode():
            self.gn = model.project_go(name).float()
            self.gd = model.project_go(definition).float()
            self.gp = model.project_go(parent.reshape(-1, parent.shape[-1])).float()
            self.gp = self.gp.reshape(parent.shape[0], parent.shape[1], -1)

    @torch.inference_mode()
    def __call__(self, h, mask):
        model = self.model
        zw, valid = model.encode_local_windows(h, mask)
        B, W, _ = zw.shape
        chunks = []
        for st in range(0, self.gn.shape[0], self.chunk_size):
            ed = min(st + self.chunk_size, self.gn.shape[0])
            sn = model.mil(torch.einsum('bkd,gd->bkg', zw, self.gn[st:ed]), valid)
            sd = model.mil(torch.einsum('bkd,gd->bkg', zw, self.gd[st:ed]), valid)
            child = torch.where(self.dp[st:ed][None, :], 0.5 * (sn + sd), sn)
            pm = self.pm[st:ed]
            if pm.any():
                pr = self.gp[st:ed]
                parent_scores = torch.einsum('bkd,cpd->bkcp', zw, pr)
                C, P = pm.shape
                parent_flat = parent_scores.permute(0, 3, 1, 2).reshape(B * P, W, C)
                parent_valid = valid[:, None, :].expand(B, P, W).reshape(B * P, W)
                parent_mil = model.mil(parent_flat, parent_valid).reshape(B, P, C).permute(0, 2, 1)
                parent_mil = parent_mil.masked_fill(~pm[None, :, :], 0)
                parent_avg = parent_mil.sum(-1) / pm.sum(-1).clamp_min(1)[None, :]
                adjust = -self.parent_strength * pm.any(-1)[None, :] * torch.relu(child - parent_avg)
                child = child + adjust
            chunks.append(child)
        return torch.cat(chunks, dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('split', 'checkpoint', 'protein-ids', 'candidate-ids', 'go-bank', 'embedding-dir', 'out-file'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--labels')
    ap.add_argument('--branch', choices=['BP', 'CC', 'MF'], required=True)
    ap.add_argument('--mode', choices=['train', 'val', 'general', 'temporal', 'custom'], default='custom')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--go-chunk-size', type=int, default=256)
    ap.add_argument('--cpu-threads', type=int, default=4)
    ap.add_argument('--max-len', type=int, default=1000)
    ap.add_argument('--overlap', type=int, default=256)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--verify-reference', action='store_true',
                    help='Check cached scorer vs original on first batch; abort on mismatch')
    ap.add_argument('--verify-tol', type=float, default=2e-5)
    args = ap.parse_args()
    if args.batch_size < 1 or args.go_chunk_size < 1 or args.cpu_threads < 1:
        raise ValueError('batch-size, go-chunk-size and cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out_file)
    if out.exists() and not args.overwrite:
        raise FileExistsError(out)
    ids_file, candidate_file, bank_file = map(Path, (args.protein_ids, args.candidate_ids, args.go_bank))

    def read_ids(path):
        if path.suffix.lower() == '.json':
            return json.loads(path.read_text())
        return path.read_text().split()

    pids = read_ids(ids_file)
    candidates = [gid(x) for x in read_ids(candidate_file)]
    if not isinstance(pids, list) or not pids:
        raise ValueError('Protein IDs must be a nonempty JSON or TXT list')
    if len(set(pids)) != len(pids) or len(set(candidates)) != len(candidates):
        raise ValueError('Duplicate protein or GO IDs')
    if args.limit:
        pids = pids[:args.limit]
    labels = None
    if args.labels:
        labels = {pid: {gid(x) for x in terms} for pid, terms in json.loads(Path(args.labels).read_text()).items()}
        if not set(pids).issubset(labels):
            raise ValueError('Protein IDs missing from annotation JSON')
        if args.mode in ('train', 'val') and any(not labels[pid] for pid in pids):
            raise ValueError('Unlabelled training/validation protein')
        absent = {g for pid in pids for g in labels[pid]} - set(candidates)
        if absent:
            raise ValueError(f'Positive GO terms outside candidate universe: {sorted(absent)[:10]}')
    elif args.mode in ('train', 'val'):
        raise ValueError('Training/validation requires --labels')

    bank = torch.load(bank_file, map_location='cpu', weights_only=False)
    if bank.get('format') != 'mzsgo_frozen_segment_parent_v1':
        raise ValueError('Unsupported bank format')
    bank_ids = [gid(x) for x in bank['go_ids']]
    lookup = {g: i for i, g in enumerate(bank_ids)}
    if len(lookup) != len(bank_ids) or not set(candidates).issubset(lookup):
        raise ValueError('Candidate GO missing or duplicate in bank')
    ix = torch.tensor([lookup[g] for g in candidates], dtype=torch.long)
    seg = bank['segment_embs'].index_select(0, ix).float()
    present = bank['segment_present'].index_select(0, ix).bool()
    parent = bank['parent_embs'].index_select(0, ix).float()
    pp = bank['parent_present'].index_select(0, ix).bool()
    weights = present.float().unsqueeze(-1)
    go_z = ((seg * weights).sum(1) / weights.sum(1).clamp_min(1)).numpy().astype(np.float32)
    encoder = BioMedBERTEncoder(model_name=MODEL, device=device, max_length=512,
                                attention_pooling_strategy='mean', attn_hidden=128,
                                attn_dropout=.1, gradient_checkpointing=False)
    model = ProteinGoAligner(d_h=1280, d_g=768, d_z=768, go_encoder=encoder,
                             normalize=True, protein_pool_type='mean_attn_gate',
                             go_pool_type='mean', go_segment_representation_mode='segments_only').to(device)
    model.load_state_dict(state(args.checkpoint), strict=True)
    model.go_chunk_size = args.go_chunk_size
    model.eval()
    model.go_encoder.eval()
    name = seg[:, 0].to(device)
    definition = seg[:, 1].to(device)
    dp = present[:, 1].to(device)
    parent = parent.to(device)
    pp = pp.to(device)
    raw = seg.mean(1).to(device)
    scorer = FastLocalScorer(model, name, definition, dp, parent, pp, args.go_chunk_size)
    store = ESMResidueStore(embed_dir=args.embedding_dir, max_len=args.max_len, overlap=args.overlap)
    missing = [pid for pid in pids if not store.has(pid)]
    if missing:
        raise RuntimeError(f'Missing {len(missing)} ESM-2 proteins, examples={missing[:5]}')
    loader = DataLoader(Proteins(pids, store), batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)
    scores, proteins, ordered = [], [], []
    started = time.monotonic()
    with torch.inference_mode():
        for j, (ids, h, mask) in enumerate(loader, 1):
            h = h.to(device).float()
            mask = mask.to(device).bool()
            s = scorer(h, mask)
            if args.verify_reference and j == 1:
                ref = model.score_fine_grained(h, mask, name, definition, dp, parent, pp, raw)
                err = (s - ref).abs().max().item()
                print(f'[VERIFY] cached-vs-original max_abs_diff={err:.8g}', flush=True)
                if err > args.verify_tol:
                    raise RuntimeError(f'Cached scorer mismatch {err} > {args.verify_tol}')
            if not torch.isfinite(s).all():
                raise RuntimeError('Nonfinite retriever scores')
            scores.append(((s.cpu().numpy().astype(np.float32) + 1) / 2).clip(0, 1))
            proteins.append(((h * mask.unsqueeze(-1)).sum(1) /
                             mask.sum(1, keepdim=True).clamp_min(1)).cpu().numpy().astype(np.float32))
            ordered.extend(ids)
            if j % 100 == 0 or len(ordered) == len(pids):
                elapsed = time.monotonic() - started
                print(f'[{args.split}] {len(ordered)}/{len(pids)} proteins, '
                      f'{len(ordered) / max(elapsed, 1):.1f} proteins/s', flush=True)
    s = np.concatenate(scores)
    protein_z = np.concatenate(proteins)
    payload = {'scores': s, 'protein_z': protein_z, 'go_z': go_z,
               'protein_ids': np.asarray(ordered),
               'go_ids': np.asarray([f'GO:{g:07d}' for g in candidates])}
    if labels is not None:
        gold = np.zeros(s.shape, dtype=np.uint8)
        cols = {g: i for i, g in enumerate(candidates)}
        for i, pid in enumerate(ordered):
            for g in labels[pid]:
                gold[i, cols[g]] = 1
        payload['labels'] = gold
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    provenance = {'split': args.split, 'mode': args.mode, 'branch': args.branch,
                  'checkpoint': str(Path(args.checkpoint).resolve()),
                  'checkpoint_sha256': sha(args.checkpoint),
                  'protein_ids_file': str(ids_file), 'candidate_ids_file': str(candidate_file),
                  'bank': str(bank_file), 'bank_sha256': sha(bank_file),
                  'labels_file': args.labels, 'score_transform': 'linear',
                  'protein_feature': 'masked mean ESM-2 residue vectors, NOT retriever projected',
                  'go_feature': 'mean frozen GO text segments, NOT retriever projected',
                  'scorer': 'cached frozen GO projections; original local SmoothMIL and parent penalty',
                  'n_proteins': len(ordered), 'n_candidates': len(candidates),
                  'limit': args.limit}
    out.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'[SAVED] {out} scores={s.shape} labels={labels is not None}', flush=True)


if __name__ == '__main__':
    main()
