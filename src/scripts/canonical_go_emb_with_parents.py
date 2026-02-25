"""
Build GO text embeddings in the old .npy + sidecar format expected by GoLookupCache / GoMemoryBank.

Input JSONL supports either:
1) New format:
   {"go_id":"GO:0001234", "domain":"...", "name":"...", "definition":"...", "text":"..."}   # text preferred
   or minimal:
   {"go_id":"GO:0001234", "text":"Name: ...\\n\\nDefinition: ...\\n\\nParents: ..."}
2) Old canonical format:
   {"go_id":"GO:0001234", "domain":"...", "name":"...", "definition":"..."}

Outputs:
- <prefix>.npy          : [N, D] float32 (optionally L2-normalized)
- <prefix>.npy.meta.pt  : {"shape": (N, D), "dtype": "float32"}
- ids.json              : {"ids": [0..N-1]}
Optional:
- <prefix>.pt           : debug blob (embs, id2row, row2id, meta, texts)
"""

from __future__ import annotations

import os
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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


def _prompt_from_canonical(domain: str, name: str, definition: str) -> str:
    domain_str = (domain or "").strip() or "Unknown"
    name = (name or "").strip()
    definition = (definition or "").strip()

    if not name:
        name = "Unknown"

    prompt = f"[Domain: {domain_str}] {name}."
    if definition:
        prompt += f" Definition: {definition}"
    return prompt


def load_go_texts_jsonl(path: str) -> Tuple[List[str], List[str]]:
    """
    Returns:
      go_ids_str: list of accession strings in deterministic order
      texts: list of text prompts aligned with go_ids_str

    Priority:
      - if "text" exists and non-empty: use it (new format)
      - else build prompt from domain/name/definition (old format)
    """
    rows: List[Tuple[str, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)

            go_id = (j.get("go_id") or "").strip()
            if not go_id:
                continue

            txt = (j.get("text") or "").strip()
            if txt:
                rows.append((go_id, txt))
                continue

            # fallback: canonical fields
            domain = (j.get("domain") or "").strip()
            name = (j.get("name") or "").strip()
            definition = (j.get("definition") or "").strip()
            prompt = _prompt_from_canonical(domain, name or go_id, definition)
            rows.append((go_id, prompt))

    # deterministic order by GO accession string
    rows.sort(key=lambda x: x[0])
    go_ids_str = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return go_ids_str, texts


def save_checkpoint_blob(
    pt_blob_path: Path,
    *,
    embs: torch.Tensor,
    go_ids_str: List[str],
    texts: List[str],
    phase: str,
    model_name: str,
    max_length: int,
    normalized: bool,
    compare_to: Optional[str],
    drift: Optional[float],
    input_path: str,
) -> None:
    N, D = embs.shape
    int_ids = list(range(N))
    id2row = {int(i): int(i) for i in int_ids}
    row2id = int_ids[:]

    meta_blob: Dict[str, Any] = {
        "phase": phase,
        "model_name": model_name,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "normalized": bool(normalized),
        "N": int(N),
        "D": int(D),
        "compare_to": compare_to,
        "drift": float(drift) if drift is not None else None,
        "input_path": input_path,
        "max_length": int(max_length),
        "go_ids": go_ids_str,
    }

    texts_int_keyed: Dict[int, str] = {int(i): txt for i, txt in enumerate(texts)}

    blob: Dict[str, Any] = {
        "embs": embs,     # [N,D] float32 on CPU
        "id2row": id2row,
        "row2id": row2id,
        "meta": meta_blob,
        "texts": texts_int_keyed,
    }

    torch.save(blob, pt_blob_path)


def main(
    go_path: str,
    output_prefix: str,
    phase: str = "canonical",
    *,
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    device: str = "cpu",
    batch_size: int = 256,
    max_length: int = 512,
    normalize: bool = True,
    store_pt_blob: bool = True,
    compare_to: Optional[str] = None,
) -> None:
    # 1) Load texts
    go_ids_str, texts = load_go_texts_jsonl(go_path)
    N = len(go_ids_str)
    if N == 0:
        raise RuntimeError(f"No GO texts loaded from {go_path}")

    print(f"[INFO] Loaded {N} GO texts from {go_path}")

    # 2) Encoder
    encoder = BioMedBERTEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
    )
    encoder.eval()

    # 3) Encode
    print(f"[INFO] Encoding {N} GO texts with {model_name} | batch_size={batch_size} | max_length={max_length}")
    embs = encoder.encode_texts(go_texts=texts, batch_size=batch_size)  # [N,D]
    embs = embs.detach().to(torch.float32).contiguous()

    if normalize:
        embs = F.normalize(embs, p=2, dim=1)

    if not torch.isfinite(embs).all():
        raise ValueError("Non-finite values in embeddings!")

    N_check, D = embs.shape
    assert N_check == N

    print(f"[INFO] Encoded embeddings: N={N} D={D} normalize={normalize}")

    # 4) Optional drift check against a previous .pt blob
    drift_val: Optional[float] = None
    if compare_to and os.path.isfile(compare_to):
        prev = torch.load(compare_to, map_location="cpu")
        prev_embs = prev.get("embs", None)
        if isinstance(prev_embs, torch.Tensor) and prev_embs.shape == embs.shape:
            drift_val = cosine_drift(embs, prev_embs.to(torch.float32))
            print(f"[INFO] Mean cosine drift vs {compare_to}: {drift_val:.6f}")
        else:
            print("[WARN] Drift compare skipped (missing embs or shape mismatch).")

    # 5) Output paths
    out_prefix = Path(output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    npy_path = out_prefix.with_suffix(".npy")
    meta_path = Path(str(npy_path) + ".meta.pt")
    ids_json_path = npy_path.with_name("ids.json")
    pt_blob_path = out_prefix.with_suffix(".pt")

    # 6) Save .npy
    embs_np = embs.cpu().numpy().astype("float32")
    np.save(npy_path, embs_np)
    print(f"[OK] Saved embeddings → {npy_path}")

    # 7) Save meta
    meta_for_memmap: Dict[str, Any] = {"shape": embs_np.shape, "dtype": str(embs_np.dtype)}
    torch.save(meta_for_memmap, meta_path)
    print(f"[OK] Saved meta → {meta_path}")

    # 8) ids.json, global ids 0..N-1
    int_ids = list(range(N))
    with open(ids_json_path, "w", encoding="utf-8") as f:
        json.dump({"ids": int_ids}, f)
    print(f"[OK] Saved ids → {ids_json_path}")

    # 9) Optional debug blob
    if store_pt_blob:
        save_checkpoint_blob(
            pt_blob_path,
            embs=embs.cpu(),
            go_ids_str=go_ids_str,
            texts=texts,
            phase=phase,
            model_name=model_name,
            max_length=max_length,
            normalized=normalize,
            compare_to=compare_to,
            drift=drift_val,
            input_path=go_path,
        )
        print(f"[OK] Saved debug blob → {pt_blob_path}")

    print("[OK] Done.")


if __name__ == "__main__":
    main(
        go_path="/workspace/data/processed/go_terms/canonical2/go_texts_canonical_2.jsonl",
        output_prefix="/workspace/data/training_ready/go_indexes/go_text_embeddings_canonical_2",
        phase="-2",
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device="cpu",
        batch_size=256,
        max_length=512,
        normalize=True,
        store_pt_blob=True,
        compare_to=None,
    )