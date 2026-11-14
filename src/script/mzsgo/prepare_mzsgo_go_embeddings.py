"""Precompute frozen GO segment and parent banks matching OppTrainer (Nov 2026).

Run from repository root: python -m src.script.mzsgo.prepare_mzsgo_go_embeddings
(or python /path/to/prepare_mzsgo_go_embeddings.py with repo root on PYTHONPATH).

The resulting .pt is a NEW cache format. OppTrainer needs a separate loader patch
before it can consume this file; do not pass it as stores.go_cache_path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch

from src.encoders.go_encoder import BioMedBERTEncoder
from src.datasets import GoTextStore
from src.utils.helpers import load_go_texts_canonical

MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
SEGMENTS = ("name", "definition", "is_a")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gid_str(g):
    return f"GO:{int(str(g).split(':')[-1]):07d}"


@torch.inference_mode()
def encode(encoder, input_ids, attention_mask, device, batch_size):
    output = []
    for start in range(0, len(input_ids), batch_size):
        end = start + batch_size
        z = encoder(input_ids=input_ids[start:end].to(device),
                    attention_mask=attention_mask[start:end].to(device),
                    output_mode="pooled")
        if isinstance(z, tuple):
            z = z[0]
        elif isinstance(z, dict):
            z = z.get("pooled", z.get("pooler_output"))
        if not isinstance(z, torch.Tensor) or z.ndim != 2:
            raise RuntimeError("Unexpected BioMedBERT output")
        output.append(z.float().cpu())
        print(f"Encoded {min(end, len(input_ids))}/{len(input_ids)}", flush=True)
    return torch.cat(output) if output else torch.empty(0, encoder.model.config.hidden_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=["BP", "MF", "CC"], default="BP")
    ap.add_argument("--root", default="/workspace/MZSGO_temporal")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--segment-max-len", type=int, default=64)
    ap.add_argument("--text-max-len", type=int, default=128)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    branch = args.branch.lower()
    source = root / "processed" / branch / "go_texts_canonical_segmented_dual_2023_2025.jsonl"
    candidate = root / "processed" / branch / f"candidate_go_ids_{branch}.json"
    temporal = root / "processed" / branch / f"temporal_go_ids_{branch}.json"
    dest = Path(args.output) if args.output else root / "go_cache" / branch / "frozen_segment_parent_bank.pt"
    if dest.exists():
        raise FileExistsError(f"Cache exists, refusing overwrite: {dest}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    records = {}
    with source.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                key = gid_str(record["go_id"])
                if key in records:
                    raise ValueError(f"Duplicate GO: {key}")
                records[key] = record
    ids = [int(str(x).split(':')[-1]) for x in json.loads(candidate.read_text())]
    if len(ids) != len(set(ids)) or set(map(gid_str, ids)) != set(records):
        raise RuntimeError("Candidate ID list and canonical JSONL must match exactly")
    temporal_ids = [int(str(x).split(':')[-1]) for x in json.loads(temporal.read_text())]
    if not set(temporal_ids).issubset(ids):
        raise RuntimeError("Temporal GO not contained in candidate universe")
    ids = sorted(ids)
    device = torch.device(args.device)
    encoder = BioMedBERTEncoder(model_name=args.model, device=device, max_length=512,
                                attention_pooling_strategy="mean", attn_hidden=128,
                                attn_dropout=0.1, gradient_checkpointing=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Identical tokenization path to main.py -> GoTextStore -> OppTrainer.
    id2text, id2segments, id2present = load_go_texts_canonical(
        str(source), phase=-2, return_segments=True, enabled_segments=list(SEGMENTS))
    store = GoTextStore({-2: id2text}, encoder.tokenizer, phase=-2, lazy=True,
                        max_len=args.text_max_len, is_segmented=True,
                        segment_max_len=args.segment_max_len,
                        full_id2segments={-2: id2segments},
                        full_id2seg_present={-2: id2present},
                        segment_names=list(SEGMENTS))
    store.materialize_tokens_once(batch_size=512, show_progress=True)
    toks = store.batch(ids)
    present = toks["seg_present"].bool().cpu()
    G, S, L = toks["seg_input_ids"].shape
    assert (G, S) == (len(ids), 3)
    flat_present = present.reshape(-1)
    indexes = flat_present.nonzero(as_tuple=False).flatten()
    segment_bank = torch.zeros(G * S, encoder.model.config.hidden_size, dtype=torch.float32)
    if indexes.numel():
        vectors = encode(encoder,
                         toks["seg_input_ids"].reshape(-1, L).index_select(0, indexes),
                         toks["seg_attention_mask"].reshape(-1, L).index_select(0, indexes),
                         device, args.batch_size)
        segment_bank.index_copy_(0, indexes, vectors)
    segment_bank = segment_bank.reshape(G, S, -1)

    # Match current OppTrainer parent behavior: only parent IDs also present
    # in the canonical candidate JSONL are used. Record omitted links explicitly.
    parent_lists, omitted = [], set()
    for gid in ids:
        row = records[gid_str(gid)]
        parents = []
        for raw in row.get("is_a_parent_ids", []):
            p = gid_str(raw)
            if p in records and records[p].get("name"):
                parents.append(p)
            else:
                omitted.add(p)
        parent_lists.append(parents)
    unique = sorted({p for ps in parent_lists for p in ps})
    parent_vectors = {}
    if unique:
        enc = encoder.tokenizer([records[p]["name"] for p in unique],
                                truncation=True, max_length=args.segment_max_len,
                                padding="max_length", return_tensors="pt")
        values = encode(encoder, enc["input_ids"], enc["attention_mask"], device, args.batch_size)
        parent_vectors = dict(zip(unique, values))
    P = max(1, max(map(len, parent_lists)))
    parent_bank = torch.zeros(G, P, encoder.model.config.hidden_size, dtype=torch.float32)
    parent_present = torch.zeros(G, P, dtype=torch.bool)
    for i, parents in enumerate(parent_lists):
        for j, p in enumerate(parents):
            parent_bank[i, j] = parent_vectors[p]
            parent_present[i, j] = True
    for t in (segment_bank, parent_bank):
        if not torch.isfinite(t).all():
            raise RuntimeError("Non-finite embeddings")

    payload = {
        "format": "mzsgo_frozen_segment_parent_v1",
        "go_ids": ids,
        "segment_names": list(toks.get("segment_names", SEGMENTS)),
        "segment_embs": segment_bank,
        "segment_present": present,
        "parent_embs": parent_bank,
        "parent_present": parent_present,
        "metadata": {
            "branch": args.branch, "model": args.model, "pooling": "mean",
            "normalized": False, "segment_max_len": args.segment_max_len,
            "text_max_len": args.text_max_len,
            "canonical_jsonl_sha256": sha256(source),
            "candidate_ids_sha256": sha256(candidate),
            "temporal_ids_sha256": sha256(temporal),
            "n_seen": len(ids) - len(temporal_ids), "n_temporal": len(temporal_ids),
            "n_parent_names_encoded": len(unique),
            "parent_ids_outside_candidate_count": len(omitted),
            "parent_ids_outside_candidate_examples": sorted(omitted)[:20],
        },
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(dest)
    print(f"[DONE] {dest} | segments={tuple(segment_bank.shape)} parents={tuple(parent_bank.shape)}")
    print(json.dumps(payload["metadata"], indent=2))
    print("NOTE: OppTrainer needs a cache-loading patch; this file is not a GoLookupCache .npy.")


if __name__ == "__main__":
    main()
