#!/usr/bin/env python3
"""Extend the frozen BP GO bank with 328 2023 unseen labels for general test.

Does not modify training bank, checkpoint, or canonical source JSONL.
Run: python -m src.script.mzsgo.prepare_mzsgo_go_test_embeddings
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import torch

from src.encoders.go_encoder import BioMedBERTEncoder
from src.datasets import GoTextStore
from src.utils.helpers import load_go_texts_canonical

MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
SEGMENTS = ["name", "definition", "is_a"]


def gid(x):
    return f"GO:{int(str(x).split(':')[-1]):07d}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parse_obo(path):
    """Read primary 2023 GO term names, definitions, is_a IDs, namespace."""
    terms = {}
    row = None

    def flush():
        if row is not None and row.get("id"):
            k = row["id"]
            if k in terms:
                raise RuntimeError(f"Duplicate GO ID in OBO: {k}")
            terms[k] = dict(row)

    for raw in path.open(encoding="utf-8"):
        line = raw.strip()
        if line == "[Term]":
            flush()
            row = {"is_a": []}
        elif line.startswith("["):
            flush()
            row = None
        elif row is not None and ": " in line:
            key, val = line.split(": ", 1)
            if key in {"id", "name", "namespace"}:
                row[key] = val.strip()
            elif key == "def":
                match = re.search(r'"((?:\\.|[^"\\])*)"', val)
                row["definition"] = match.group(1).replace('\\"', '"') if match else val.strip()
            elif key == "is_a":
                row["is_a"].append(gid(val.split()[0]))
            elif key == "is_obsolete":
                row["is_obsolete"] = val.strip() == "true"
    flush()
    return terms


def sentence(s):
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    if not s:
        return "none"
    return s if s[-1] in ".!?" else s + "."


def make_record(go, terms):
    t = terms[go]
    if t.get("is_obsolete") or t.get("namespace") != "biological_process":
        raise RuntimeError(f"Invalid 2023 BP GO: {go}")
    if not t.get("name"):
        raise RuntimeError(f"GO term without name: {go}")
    parent_ids = sorted(set(p for p in t["is_a"] if p in terms
                            and not terms[p].get("is_obsolete")
                            and terms[p].get("name")))
    parent_names = [terms[p]["name"] for p in parent_ids]
    segments = {
        "name": sentence(t["name"]),
        "namespace": "biological process.",
        "definition": sentence(t.get("definition", "")),
        "is_a": sentence("; ".join(parent_names[:3]) if parent_names else "none"),
        "part_of": "none.",
    }
    return {
        "go_id": go,
        "domain": "Biological Process",
        "namespace": "biological process",
        "name": t["name"],
        "definition": t.get("definition", ""),
        "is_a_parent_ids": parent_ids,
        "part_of_parent_ids": [],
        "is_a_parents": parent_names,
        "part_of_parents": [],
        "segments": segments,
        "segment_order": SEGMENTS,
        "segment_format": "markerless_v1",
        "text": "\n".join(segments[k] for k in SEGMENTS),
        "ontology_source": "2023-01-01",
        "label_exposure": "general_test_unseen",
    }


@torch.inference_mode()
def encode(encoder, tokens, masks, device, bs):
    outputs = []
    for st in range(0, len(tokens), bs):
        z = encoder(input_ids=tokens[st:st+bs].to(device),
                    attention_mask=masks[st:st+bs].to(device), output_mode="pooled")
        if isinstance(z, tuple):
            z = z[0]
        elif isinstance(z, dict):
            z = z.get("pooled", z.get("pooler_output"))
        if not torch.is_tensor(z) or z.ndim != 2:
            raise RuntimeError("Unexpected BioMedBERT output")
        outputs.append(z.float().cpu())
    return torch.cat(outputs, 0) if outputs else torch.empty((0, 768))


def make_store(jsonl, encoder, segment_max_len, text_max_len):
    texts, segs, present = load_go_texts_canonical(
        str(jsonl), phase=-2, return_segments=True, enabled_segments=SEGMENTS)
    store = GoTextStore({-2: texts}, encoder.tokenizer, phase=-2, lazy=True,
                        max_len=text_max_len, is_segmented=True,
                        segment_max_len=segment_max_len,
                        full_id2segments={-2: segs},
                        full_id2seg_present={-2: present},
                        segment_names=SEGMENTS)
    store.materialize_tokens_once(batch_size=512, show_progress=False)
    return store


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/workspace/MZSGO_temporal"))
    ap.add_argument("--obo", type=Path, default=Path("/workspace/MZSGO/data/go_2023_01_01.obo"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.batch_size < 1:
        raise ValueError("batch-size must be positive")
    proc = a.root / "processed/bp"
    original = a.root / "go_cache/bp/frozen_segment_parent_bank.pt"
    source = proc / "go_texts_canonical_segmented_dual_2023_2025.jsonl"
    missing_path = proc / "general_test_missing_go_bp.json"
    candidate_path = proc / "candidate_go_ids_bp.json"
    dest = a.output or a.root / "go_cache/bp/frozen_segment_parent_bank_general_test.pt"
    if dest.exists():
        raise FileExistsError(f"Refusing overwrite: {dest}")
    for p in (original, source, missing_path, candidate_path, a.obo):
        if not p.is_file():
            raise FileNotFoundError(p)

    base = torch.load(original, map_location="cpu", weights_only=False)
    if base.get("format") != "mzsgo_frozen_segment_parent_v1":
        raise RuntimeError("Unexpected base bank format")
    meta = base["metadata"]
    if meta.get("model") != MODEL or meta.get("normalized") is not False:
        raise RuntimeError("Encoder or normalization mismatch")
    if meta.get("canonical_jsonl_sha256") != sha256(source):
        raise RuntimeError("Canonical JSONL hash mismatch")
    if meta.get("candidate_ids_sha256") != sha256(candidate_path):
        raise RuntimeError("Candidate JSON hash mismatch")
    if list(base["segment_names"]) != SEGMENTS:
        raise RuntimeError("Segment order mismatch")
    seg_max = int(meta["segment_max_len"])
    text_max = int(meta["text_max_len"])
    old_ids = [int(x) for x in base["go_ids"]]
    if len(old_ids) != 20345 or len(set(old_ids)) != len(old_ids):
        raise RuntimeError("Expected 20345 unique original GO IDs")
    requested = json.loads(missing_path.read_text())
    new_ids = sorted({int(g.split(":")[-1]) for g in requested["missing_go_ids"]})
    if len(new_ids) != 328 or set(new_ids) & set(old_ids):
        raise RuntimeError("Expected 328 distinct, new GO IDs")
    terms = parse_obo(a.obo)
    records = [make_record(gid(x), terms) for x in new_ids]

    # Use the identical canonical loader + GoTextStore tokenization as the base-bank creator.
    with tempfile.TemporaryDirectory(prefix="mzsgo_general_go_") as td:
        temp = Path(td) / "new_go.jsonl"
        temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        device = torch.device(a.device)
        encoder = BioMedBERTEncoder(model_name=MODEL, device=device, max_length=512,
                                    attention_pooling_strategy="mean", attn_hidden=128,
                                    attn_dropout=0.1, gradient_checkpointing=False)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        store = make_store(temp, encoder, seg_max, text_max)
        toks = store.batch(new_ids)
        present = toks["seg_present"].bool().cpu()
        G, S, L = toks["seg_input_ids"].shape
        if (G, S) != (328, 3):
            raise RuntimeError(f"Unexpected new GO token shape: {(G, S, L)}")
        flat = present.reshape(-1)
        indices = flat.nonzero(as_tuple=False).flatten()
        new_seg = torch.zeros(G * S, 768, dtype=torch.float32)
        if len(indices):
            vectors = encode(encoder,
                             toks["seg_input_ids"].reshape(-1, L).index_select(0, indices),
                             toks["seg_attention_mask"].reshape(-1, L).index_select(0, indices),
                             device, a.batch_size)
            new_seg.index_copy_(0, indices, vectors)
        new_seg = new_seg.reshape(G, S, 768)

        # Regression test: re-encode existing canonical GO rows and compare to bank.
        existing = {}
        with source.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[gid(r["go_id"])] = r
        sample_ids = [old_ids[i] for i in sorted(set([0, len(old_ids)//4,
                       len(old_ids)//2, 3*len(old_ids)//4, len(old_ids)-1]))]
        sample_path = Path(td) / "sample.jsonl"
        sample_path.write_text("".join(json.dumps(existing[gid(x)]) + "\n" for x in sample_ids))
        sample_toks = make_store(sample_path, encoder, seg_max, text_max).batch(sample_ids)
        sample_present = sample_toks["seg_present"].bool().cpu()
        sample_flat = sample_present.reshape(-1)
        sample_idx = sample_flat.nonzero(as_tuple=False).flatten()
        sample_vec = torch.zeros(len(sample_ids)*3, 768)
        sample_enc = encode(encoder,
                            sample_toks["seg_input_ids"].reshape(-1, sample_toks["seg_input_ids"].shape[-1]).index_select(0, sample_idx),
                            sample_toks["seg_attention_mask"].reshape(-1, sample_toks["seg_attention_mask"].shape[-1]).index_select(0, sample_idx),
                            device, a.batch_size)
        sample_vec.index_copy_(0, sample_idx, sample_enc)
        sample_vec = sample_vec.reshape(len(sample_ids), 3, 768)
        base_index = {g: i for i, g in enumerate(old_ids)}
        ref = base["segment_embs"][torch.tensor([base_index[x] for x in sample_ids])].float()
        delta = (sample_vec - ref).abs().max().item()
        print(f"[PARITY] existing GO segment max_abs_diff={delta:.7g}", flush=True)
        if not torch.allclose(sample_vec, ref, atol=1e-4, rtol=1e-4):
            raise RuntimeError("Existing-bank parity failed. Refusing to write output")

        # Match the base builder: only parent IDs in the candidate record universe.
        # For new GO terms, the expanded universe includes the 328 new IDs.
        available = set(existing) | {gid(x) for x in new_ids}
        parent_lists = []
        for rec in records:
            parent_lists.append([p for p in rec["is_a_parent_ids"]
                                 if p in available and p in terms and terms[p].get("name")])
        parent_names = sorted({p for ps in parent_lists for p in ps})
        parent_vec = {}
        if parent_names:
            batch = encoder.tokenizer([terms[p]["name"] for p in parent_names],
                                      truncation=True, max_length=seg_max,
                                      padding="max_length", return_tensors="pt")
            vectors = encode(encoder, batch["input_ids"], batch["attention_mask"], device, a.batch_size)
            parent_vec = dict(zip(parent_names, vectors))
        old_parent = base["parent_embs"].float()
        old_mask = base["parent_present"].bool()
        P = max(old_parent.shape[1], max((len(ps) for ps in parent_lists), default=0), 1)
        new_parent = torch.zeros((328, P, 768), dtype=torch.float32)
        new_mask = torch.zeros((328, P), dtype=torch.bool)
        for i, ps in enumerate(parent_lists):
            for j, p in enumerate(ps):
                new_parent[i, j] = parent_vec[p]
                new_mask[i, j] = True
        if old_parent.shape[1] < P:
            pad = P - old_parent.shape[1]
            old_parent = torch.nn.functional.pad(old_parent, (0, 0, 0, pad))
            old_mask = torch.nn.functional.pad(old_mask, (0, pad), value=False)

    all_ids = sorted(set(old_ids) | set(new_ids))
    old_idx = {g: i for i, g in enumerate(old_ids)}
    new_idx = {g: i for i, g in enumerate(new_ids)}
    ordered_seg = torch.stack([base["segment_embs"][old_idx[g]] if g in old_idx else new_seg[new_idx[g]] for g in all_ids])
    ordered_present = torch.stack([base["segment_present"][old_idx[g]] if g in old_idx else present[new_idx[g]] for g in all_ids])
    ordered_parent = torch.stack([old_parent[old_idx[g]] if g in old_idx else new_parent[new_idx[g]] for g in all_ids])
    ordered_mask = torch.stack([old_mask[old_idx[g]] if g in old_idx else new_mask[new_idx[g]] for g in all_ids])
    assert len(all_ids) == 20673
    if not all(torch.isfinite(x).all() for x in (ordered_seg, ordered_parent)):
        raise RuntimeError("Nonfinite vectors")
    result = {
        "format": "mzsgo_frozen_segment_parent_v1",
        "go_ids": all_ids,
        "segment_names": SEGMENTS,
        "segment_embs": ordered_seg,
        "segment_present": ordered_present,
        "parent_embs": ordered_parent,
        "parent_present": ordered_mask,
        "metadata": {**meta,
            "purpose": "general_BP_test_only",
            "base_bank_sha256": sha256(original),
            "2023_obo_sha256": sha256(a.obo),
            "missing_ids_sha256": sha256(missing_path),
            "added_unseen_go": len(new_ids),
            "base_go": len(old_ids),
            "total_go": len(all_ids),
            "existing_segment_parity_max_abs_diff": delta,
            "note": "Original bank entries unchanged; new 2023 unseen labels added. Not a training bank.",
        },
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_out = dest.with_suffix(dest.suffix + ".tmp")
    if temp_out.exists():
        raise FileExistsError(f"Temporary output already exists: {temp_out}")
    torch.save(result, temp_out)
    temp_out.replace(dest)
    print(f"[DONE] {dest} | original={len(old_ids)} added={len(new_ids)} total={len(all_ids)}", flush=True)
    print(f"[SHAPES] segments={tuple(ordered_seg.shape)} parents={tuple(ordered_parent.shape)}", flush=True)


if __name__ == "__main__":
    main()
