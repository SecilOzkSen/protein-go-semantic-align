"""
Build canonical GO text embeddings in the *old* .npy + sidecar format
expected by GoLookupCache / GoMemoryBank.

Üretilen dosyalar:
- <prefix>.npy          : [N, D] float32, L2-normalized GO embeddings
- <prefix>.npy.meta.pt  : {"shape": (N, D), "dtype": "float32"}
- ids.json              : {"ids": [0, 1, 2, ..., N-1]}
Opsiyonel:
- <prefix>.pt           : debugging için full blob (embs, id2row, row2id, meta, texts)

Global GO id'ler integer 0..N-1 olarak kabul ediliyor.
Gerçek GO accession stringleri meta["go_ids"] içinde tutuluyor.
"""

from __future__ import annotations
import os
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import json
import numpy as np
import torch
import torch.nn.functional as F

from src.encoders.go_encoder import BioMedBERTEncoder


@torch.no_grad()
def cosine_drift(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for drift: {a.shape} vs {b.shape}")
    sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return float((1.0 - sim).mean().item())


def load_canonical_go_texts(path: str) -> Dict[str, str]:
    """
    JSONL:
      {"go_id": "...", "domain": "...", "name": "...", "definition": "..."}

    Prompt:
      "[Domain: {domain}] {name}. Definition: {definition}"
    """
    go_texts: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            go_id = j["go_id"]  # "GO:0003677" gibi
            domain = j.get("domain", "").strip()
            name = j.get("name", "").strip()
            definition = j.get("definition", "").strip()

            if not name:
                name = go_id

            domain_str = domain if domain else "Unknown"

            prompt = f"[Domain: {domain_str}] {name}."
            if definition:
                prompt += f" Definition: {definition}"

            go_texts[go_id] = prompt

    # GO accession stringine göre deterministik sıralama
    go_texts = dict(sorted(go_texts.items(), key=lambda kv: kv[0]))
    return go_texts


def main(
    go_path: str,
    output_prefix: str,
    phase: str = "canonical",
    *,
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    device: str = "cpu",
    batch_size: int = 256,
    max_length: int = 512,
    store_texts: bool = True,
    store_pt_blob: bool = True,
    store_tokens: bool = False,
    compare_to: Optional[str] = None,
) -> None:

    # 1) GO textlerini yükle: {go_accession: prompt_text}
    go_texts_str_keyed = load_canonical_go_texts(go_path)
    print(f"[INFO] Loaded {len(go_texts_str_keyed)} canonical GO texts from {go_path}")

    go_ids_str: List[str] = list(go_texts_str_keyed.keys())      # ["GO:0003677", ...]
    N = len(go_ids_str)
    int_ids: List[int] = list(range(N))                          # 0..N-1

    all_texts: List[str] = [go_texts_str_keyed[g] for g in go_ids_str]

    # 2) Encoder
    encoder = BioMedBERTEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
    )
    encoder.eval()

    # 3) Encode in batches
    print(f"[INFO] Encoding {N} GO texts with {model_name}")
    all_embs = encoder.encode_texts(
        go_texts=all_texts,
        batch_size=batch_size,
    )  # [N, D]

    all_embs = all_embs.detach().to(torch.float32).contiguous()
    all_embs = F.normalize(all_embs, p=2, dim=1)

    if not torch.isfinite(all_embs).all():
        raise ValueError("Non-finite values after normalization!")

    N_check, D = all_embs.shape
    assert N_check == N, f"Embedding row count mismatch: {N_check} vs {N}"
    print(f"[INFO] Encoded GO embeddings: {N} terms, dim={D} (L2-normalized).")

    # 4) Optional drift check
    drift_val: Optional[float] = None
    if compare_to and os.path.isfile(compare_to):
        prev = torch.load(compare_to, map_location="cpu")
        if "embs" in prev and prev["embs"].shape == all_embs.shape:
            drift_val = cosine_drift(all_embs, prev["embs"])
            print(f"[INFO] Mean cosine drift vs {compare_to}: {drift_val:.5f}")
        else:
            print("[WARN] Cannot compare drift (shape mismatch or missing embs).")

    # 5) Çıktı yollarını ayarla
    out_prefix = Path(output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    npy_path = out_prefix.with_suffix(".npy")                    # memmap path
    meta_path = Path(str(npy_path) + ".meta.pt")                # memmap meta
    ids_json_path = npy_path.with_name("ids.json")              # id listesi
    pt_blob_path = out_prefix.with_suffix(".pt")                # opsiyonel .pt blob

    # 6) .npy memmap formatına yaz
    embs_np = all_embs.cpu().numpy().astype("float32")
    np.save(npy_path, embs_np)
    print(f"[OK] Saved embeddings memmap → {npy_path}")

    # 7) memmap meta (GoLookupCache bunu bekliyor)
    meta_for_memmap: Dict[str, Any] = {
        "shape": embs_np.shape,
        "dtype": str(embs_np.dtype),
    }
    torch.save(meta_for_memmap, meta_path)
    print(f"[OK] Saved memmap meta → {meta_path}")

    # 8) ids.json (build_go_cache memmap branch bunu okuyabiliyor)
    with open(ids_json_path, "w", encoding="utf-8") as f:
        json.dump({"ids": int_ids}, f)
    print(f"[OK] Saved ids → {ids_json_path}")

    # 9) İstersen .pt blob da üret (debug / analiz için)
    if store_pt_blob:
        id2row = {int(i): int(i) for i in int_ids}  # identity mapping
        row2id = int_ids[:]

        meta_blob: Dict[str, Any] = {
            "phase": phase,
            "model_name": model_name,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "normalized": True,
            "N": int(N),
            "D": int(D),
            "compare_to": compare_to,
            "drift": float(drift_val) if drift_val is not None else None,
            "input_path": go_path,
            "prompt_template": "[Domain: {domain}] {name}. Definition: {definition}",
            "go_ids": go_ids_str,   # row sırasına göre gerçek GO accession listesi
        }

        texts_int_keyed: Dict[int, str] = {int(i): txt for i, txt in zip(int_ids, all_texts)}

        blob: Dict[str, Any] = {
            "embs": all_embs,   # [N, D] float32
            "id2row": id2row,   # dict[int,int]
            "row2id": row2id,   # list[int]
            "meta": meta_blob,
        }
        if store_texts:
            blob["texts"] = texts_int_keyed

        if store_tokens and hasattr(encoder, "tokenize_texts"):
            print("[INFO] Storing tokenized inputs into .pt blob.")
            toks_all: List[Any] = []
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    texts_chunk = all_texts[i : i + batch_size]
                    toks = encoder.tokenize_texts(
                        texts_chunk, max_length=max_length, device="cpu"
                    )
                    toks_all.append(toks)
            blob["tokens"] = toks_all

        torch.save(blob, pt_blob_path)
        print(f"[OK] Saved debug PT blob → {pt_blob_path}")

    print("[OK] GO embedding cache generation finished.")


if __name__ == "__main__":
    main(
        go_path="/Users/secilsen/PhD/protein-go-semantic-align/src/data/processed/go_terms/canonical/go_texts_canonical.jsonl",
        output_prefix="/Users/secilsen/PhD/protein-go-semantic-align/src/data/training_ready/go_indexes/go_text_embeddings_canonical",
        phase="canonical",
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device="cpu",
        batch_size=256,
        max_length=512,
        store_texts=True,
        store_pt_blob=True,
        store_tokens=False,
        compare_to=None,
    )
