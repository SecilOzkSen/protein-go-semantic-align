"""
Created by Secil Sen

Build a phase-locked GO text embedding cache.

This script reads GO term texts, encodes them
with the BioMedBERTEncoder, L2-normalizes, and saves a compact cache:
    processed/go_text_embeddings.pt

Saved fields
------------
- embs   : FloatTensor [N_go, D]      (L2-normalized)
- id2row : Dict[int, int]             (global GO id -> row index)
- row2id : List[int]                  (inverse mapping)
- texts  : Dict[int, str]             (optional; if --store-texts)
- tokens : Any                        (optional; if encoder exposes .tokenize_texts)
- meta   : Dict[str, Any]             (phase, model_name, timestamp, etc.)

Usage
-----
python scripts/build_go_embeddings.py \
  --go-texts s3://bucket/go_texts_phase1.json \
  --output processed/go_text_embeddings.pt \
  --phase phase1 --model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --batch-size 256 --device cuda --store-texts

You can also compare to an existing cache to estimate drift:
  --compare-to processed/go_text_embeddings_prev.pt
"""

from __future__ import annotations
import os,  datetime
from typing import List, Dict, Any
import torch
from src.encoders.go_encoder import BioMedBERTEncoder
from src.utils import load_go_texts
import json
import torch.nn.functional as F



@torch.no_grad()
def cosine_drift(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute mean cosine distance between two embedding matrices of identical shape.
    Returns a scalar drift in [0, 2]; small (~0.0-0.05) = close, large = drifted.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for drift: {a.shape} vs {b.shape}")
    sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # [N]
    return float((1.0 - sim).mean().item())

def main(go_path: str,
         output_path: str,
         phase:str,
         *,
         model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
         device: str = "cpu",
         batch_size: int = 256,
         max_length: int = 512,
         store_texts: bool = True,
         store_tokens: bool = False,
         compare_to: str | None = None
         ):

    go_texts = load_go_texts(go_path)
    go_texts = dict(sorted(go_texts.items()))
    print(f"[INFO] Loaded {len(go_texts.keys())} GO texts.")
    encoder = BioMedBERTEncoder(model_name=model_name, device=device, max_length=max_length)
    # 3) Encode in batches
    encoder.eval()
    all_embs = encoder.encode_texts(
        go_texts=list(go_texts.values()),
        batch_size=batch_size
    )
    all_embs = all_embs.detach().to(torch.float32).contiguous()
    all_embs = F.normalize(all_embs, p=2, dim=1) # FAISS expects L2 normalized vectors!

    # Son kontroller (opsiyonel ama faydalı)
    if not torch.isfinite(all_embs).all():
        raise ValueError("Non-finite values after normalization!")

    print(f"[INFO] Embeddings ready: {all_embs.shape}, L2-normalized")
    N, D = all_embs.shape
    print(f"[INFO] Encoded GO embeddings: {N} terms, dim={D} (L2-normalized).")
    go_ids = list(go_texts.keys())
    # 4) Build mappings
    id2row = {int(g): i for i, g in enumerate(go_ids)}
    row2id = go_ids[:]  # already sorted

    # 5) Optional: store tokens (if encoder provides a tokenizer)
    cache_tokens: Any = None
    if store_tokens and hasattr(encoder, "tokenize_texts"):
        print("[INFO] Storing tokenized inputs.")
        toks_all: List[Any] = []
        with torch.no_grad():
            for i in range(0, len(go_ids), batch_size):
                chunk_ids = go_ids[i:i + batch_size]
                texts = [go_texts[g] for g in chunk_ids]
                toks = encoder.tokenize_texts(texts, max_length=max_length, device="cpu")
                toks_all.append(toks)
        # Caution: this can be large; keep CPU tensors
        cache_tokens = toks_all

    # 6) Compare drift (optional)
    drift_val = None
    if compare_to and os.path.isfile(compare_to):
        prev = torch.load(compare_to, map_location="cpu")
        if "embs" in prev and prev["embs"].shape == all_embs.shape:
            drift_val = cosine_drift(all_embs, prev["embs"])
            print(f"[INFO] Mean cosine drift vs {compare_to}: {drift_val:.5f}")
        else:
            print("[WARN] Cannot compare drift (shape mismatch or missing embs).")

    # 7) Save cache
    meta = {
        "phase": phase,
        "model_name": model_name,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "normalized": True,
        "N": int(N),
        "D": int(D),
        "compare_to": compare_to,
        "drift": float(drift_val) if drift_val is not None else None,
    }

    blob = {
        "embs": all_embs,  # [N, D] float32
        "id2row": id2row,  # dict[int,int]
        "row2id": row2id,  # list[int]
        "meta": meta,
    }
    if store_texts:
        blob["texts"] = {int(k): str(v) for k, v in go_texts.items()}
    if cache_tokens is not None:
        blob["tokens"] = cache_tokens

    torch.save(blob, output_path)
    print(f"[OK] Saved GO embedding cache → {output_path}")

    # 8) Quick self-check
    check = torch.load(output_path, map_location="cpu")
    assert check["embs"].shape == (N, D)
    assert isinstance(check["id2row"], dict) and len(check["id2row"]) == N
    assert len(check["row2id"]) == N
    print("[OK] Self-check passed. Cache is consistent.")


if __name__ == "__main__":
    main(go_path="/Users/secilsen/PhD/protein-go-semantic-align/src/data/processed/go_terms/go_phases/go_texts_phase_4.jsonl",
         output_path="/src/data/processed/go_terms/go_phases/go_embeddings/go_text_embeddings.pt",
         phase="phase_4")