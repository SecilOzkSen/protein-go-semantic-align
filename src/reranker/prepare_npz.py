"""Prepare a single split NPZ using a frozen MZSGO V2 retriever.

All paths and GO universes are explicit. Train, validation, general and temporal
splits share identical extraction code. Test labels are optional and never used
by the scorer. Original frozen ESM-2 / BioMedBERT pooled features are used,
not learned projected representations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.datasets.residue_store import ESMResidueStore
from src.encoders import BioMedBERTEncoder
from src.models.alignment_model_v2_original import ProteinGoAligner

MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'


def gid(x): return int(str(x).split(':')[-1])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def state(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    for _ in range(10):
        if isinstance(d, dict) and sum(torch.is_tensor(v) for v in d.values()) >= 5: break
        d = next((d[k] for k in ('model', 'model_state_dict', 'state_dict', 'net', 'module') if isinstance(d, dict) and isinstance(d.get(k), dict)), None)
        if d is None: raise RuntimeError('No model state in checkpoint')
    result = {}
    for k, v in d.items():
        if not torch.is_tensor(v): continue
        for prefix in ('trainer.model.', 'module.', 'model.'):
            if k.startswith(prefix): k = k[len(prefix):]
        result[k] = v
    return result


class InferenceCollator:
    """Pad residue embeddings without requiring training GO positives."""

    def __call__(self, batch):
        vectors = [item['prot_emb'] for item in batch]
        vectors = [v if torch.is_tensor(v) else torch.as_tensor(v) for v in vectors]
        max_len = max(v.shape[0] for v in vectors)
        dim = vectors[0].shape[1]
        padded = torch.zeros((len(vectors), max_len, dim), dtype=vectors[0].dtype)
        mask = torch.zeros((len(vectors), max_len), dtype=torch.bool)
        for i, v in enumerate(vectors):
            padded[i, :v.shape[0]] = v
            mask[i, :v.shape[0]] = True
        return {'protein_ids': [item['protein_id'] for item in batch],
                'prot_emb_pad': padded, 'prot_attn_mask': mask}


class Proteins(Dataset):
    def __init__(self, pids, store): self.pids, self.store = pids, store

    def __len__(self): return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        return {'protein_id': pid, 'prot_emb': self.store.get(pid), 'pos_go_ids': torch.empty(0, dtype=torch.long)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('split', 'checkpoint', 'protein-ids', 'candidate-ids', 'go-bank', 'embedding-dir', 'out-file'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--labels', help='Optional JSON protein ID -> list of GO IDs. Required for train/val, optional for test')
    p.add_argument('--branch', choices=['BP', 'CC', 'MF'], required=True)
    p.add_argument('--mode', choices=['train', 'val', 'general', 'temporal', 'custom'], default='custom')
    p.add_argument('--device', default='cuda');
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--go-chunk-size', type=int, default=128)
    p.add_argument('--max-len', type=int, default=1000);
    p.add_argument('--overlap', type=int, default=256)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    out = Path(a.out_file)
    if out.exists() and not a.overwrite: raise FileExistsError(out)
    ids_file = Path(a.protein_ids);
    candidate_file = Path(a.candidate_ids);
    bank_file = Path(a.go_bank)
    pids = json.loads(ids_file.read_text());
    candidates = [gid(x) for x in json.loads(candidate_file.read_text())]
    if not isinstance(pids, list) or not pids: raise ValueError('Protein IDs must be a nonempty JSON list')
    if len(set(pids)) != len(pids) or len(set(candidates)) != len(candidates): raise ValueError('Duplicate protein/GO IDs')
    if a.limit: pids = pids[:a.limit]
    labels = None
    if a.labels:
        labels = {pid: {gid(x) for x in terms} for pid, terms in json.loads(Path(a.labels).read_text()).items()}
        if not set(pids).issubset(labels): raise ValueError('Protein IDs missing from annotation JSON')
        if a.mode in ('train', 'val') and any(not labels[pid] for pid in pids): raise ValueError('Unlabelled training/validation protein')
        absent = {g for pid in pids for g in labels[pid]} - set(candidates)
        if absent: raise ValueError(f'Positive GO terms outside candidate universe: {sorted(absent)[:10]}')
    elif a.mode in ('train', 'val'):
        raise ValueError('Training/validation requires --labels')
    bank = torch.load(bank_file, map_location='cpu', weights_only=False)
    if bank.get('format') != 'mzsgo_frozen_segment_parent_v1': raise ValueError('Unsupported bank format')
    bank_ids = [gid(x) for x in bank['go_ids']]
    lookup = {g: i for i, g in enumerate(bank_ids)}
    if len(lookup) != len(bank_ids) or not set(candidates).issubset(lookup): raise ValueError('Candidate GO missing or duplicate in bank')
    ix = torch.tensor([lookup[g] for g in candidates], dtype=torch.long)
    seg = bank['segment_embs'].index_select(0, ix).float()
    present = bank['segment_present'].index_select(0, ix).bool()
    parent = bank['parent_embs'].index_select(0, ix).float()
    pp = bank['parent_present'].index_select(0, ix).bool()
    weights = present.float().unsqueeze(-1)
    go_z = ((seg * weights).sum(1) / weights.sum(1).clamp_min(1)).numpy().astype(np.float32)
    device = torch.device(a.device)
    encoder = BioMedBERTEncoder(model_name=MODEL, device=device, max_length=512, attention_pooling_strategy='mean', attn_hidden=128, attn_dropout=.1,
                                gradient_checkpointing=False)
    model = ProteinGoAligner(d_h=1280, d_g=768, d_z=768, go_encoder=encoder, normalize=True, protein_pool_type='mean_attn_gate', go_pool_type='mean',
                             go_segment_representation_mode='segments_only').to(device)
    model.load_state_dict(state(a.checkpoint), strict=True)
    model.go_chunk_size = a.go_chunk_size;
    model.eval();
    model.go_encoder.eval()
    name = seg[:, 0].to(device);
    definition = seg[:, 1].to(device);
    dp = present[:, 1].to(device)
    parent = parent.to(device);
    pp = pp.to(device);
    raw = seg.mean(1).to(device)
    store = ESMResidueStore(embed_dir=a.embedding_dir, max_len=a.max_len, overlap=a.overlap)
    missing = [pid for pid in pids if not store.has(pid)]
    if missing: raise RuntimeError(f'Missing {len(missing)} ESM-2 proteins, examples={missing[:5]}')
    loader = DataLoader(Proteins(pids, store), batch_size=a.batch_size, shuffle=False, num_workers=0, collate_fn=InferenceCollator())
    scores = [];
    proteins = [];
    ordered = []
    with torch.inference_mode():
        for j, batch in enumerate(loader, 1):
            h = batch['prot_emb_pad'].to(device).float();
            mask = batch['prot_attn_mask'].to(device).bool()
            s = model.score_fine_grained(h, mask, name, definition, dp, parent, pp, raw)
            if not torch.isfinite(s).all(): raise RuntimeError('Nonfinite retriever scores')
            scores.append(((s.cpu().numpy().astype(np.float32) + 1) / 2).clip(0, 1))
            proteins.append(((h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)).cpu().numpy().astype(np.float32))
            ordered.extend(batch['protein_ids'])
            if j % 100 == 0: print(f'[{a.split}] batches={j} proteins={len(ordered)}', flush=True)
    s = np.concatenate(scores);
    protein_z = np.concatenate(proteins)
    payload = {'scores': s, 'protein_z': protein_z, 'go_z': go_z, 'protein_ids': np.asarray(ordered), 'go_ids': np.asarray([f'GO:{g:07d}' for g in candidates])}
    if labels is not None:
        gold = np.zeros(s.shape, dtype=np.uint8);
        cols = {g: i for i, g in enumerate(candidates)}
        for i, pid in enumerate(ordered):
            for g in labels[pid]: gold[i, cols[g]] = 1
        payload['labels'] = gold
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    provenance = {'split': a.split, 'mode': a.mode, 'branch': a.branch, 'checkpoint': str(Path(a.checkpoint).resolve()), 'checkpoint_sha256': sha(a.checkpoint),
                  'protein_ids_file': str(ids_file), 'candidate_ids_file': str(candidate_file), 'bank': str(bank_file), 'bank_sha256': sha(bank_file),
                  'labels_file': a.labels, 'score_transform': 'linear', 'protein_feature': 'masked mean ESM-2 residue vectors, NOT retriever projected',
                  'go_feature': 'mean frozen GO text segments, NOT retriever projected', 'n_proteins': len(ordered), 'n_candidates': len(candidates),
                  'limit': a.limit}
    out.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'[SAVED] {out} scores={s.shape} labels={labels is not None}', flush=True)


if __name__ == '__main__': main()
