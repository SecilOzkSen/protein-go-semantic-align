"""
Build GO text embeddings in .npy + sidecar format expected by GoLookupCache / GoMemoryBank.

Fixes:
- ids.json now stores REAL GO ids (ints like 8150), not 0..N-1
- optional debug .pt blob uses the same ids
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


def go_str_to_int(go_id: str) -> int:
    go_id = (go_id or "").strip()
    if not go_id:
        raise ValueError("Empty go_id")
    if go_id.upper().startswith("GO:"):
        return int(go_id.split(":")[1])
    return int(go_id)


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
    rows: List[Tuple[str, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            j = json.loads(line)

            go_id = (j.get("go_id") or "").strip()
            if not go_id:
                raise ValueError(f"Missing go_id at line {line_no}")

            txt = (j.get("text") or "").strip()
            if not txt:
                raise ValueError(f"Missing text field for {go_id} at line {line_no}")

            rows.append((go_id, txt))

    rows.sort(key=lambda x: go_str_to_int(x[0]))
    go_ids_str = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return go_ids_str, texts


def save_checkpoint_blob(
    pt_blob_path: Path,
    *,
    embs: torch.Tensor,
    go_ids_str: List[str],
    go_ids_int: List[int],
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
    assert len(go_ids_int) == N

    # correct mappings
    row2id = go_ids_int[:]  # row -> gid
    id2row = {int(g): int(i) for i, g in enumerate(go_ids_int)}  # gid -> row

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
        "go_ids_str": go_ids_str[:],
        "go_ids_int": go_ids_int[:],
    }

    # Useful for debugging: map by GLOBAL GO id (not row)
    texts_keyed: Dict[int, str] = {int(g): txt for g, txt in zip(go_ids_int, texts)}

    blob: Dict[str, Any] = {
        "embs": embs,        # [N,D] float32 CPU
        "id2row": id2row,    # gid -> row
        "row2id": row2id,    # row -> gid
        "meta": meta_blob,
        "texts": texts_keyed,
    }
    torch.save(blob, pt_blob_path)


def write_sidecars(npy_path: Path, go_ids_int: List[int]) -> None:
    """
    Writes ids.json in the format expected by your build_go_cache():
      ids.json: {"ids": [gid0, gid1, ...]} where index is row.
    """
    ids_json_path = npy_path.with_name("ids.json")
    with open(ids_json_path, "w", encoding="utf-8") as f:
        json.dump({"ids": [int(x) for x in go_ids_int]}, f)
    print(f"[OK] Saved ids → {ids_json_path}")


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
    pid_positives_path: Optional[str] = None,
) -> None:
    # 1) Load texts
    go_ids_str, texts = load_go_texts_jsonl(go_path)
    N = len(go_ids_str)
    if N == 0:
        raise RuntimeError(f"No GO texts loaded from {go_path}")

    go_ids_int = [go_str_to_int(x) for x in go_ids_str]
    # sanity: no duplicates
    if len(set(go_ids_int)) != len(go_ids_int):
        raise RuntimeError("Duplicate GO ids after parsing. Check input JSONL.")

    print(f"[INFO] Loaded {N} GO texts from {go_path}")

    if pid_positives_path and os.path.isfile(pid_positives_path):
        with open(pid_positives_path, "r", encoding="utf-8") as f:
            pid2pos = json.load(f)

        pos_ids = set()
        for gids in pid2pos.values():
            for g in gids:
                pos_ids.add(int(g))

        cache_ids = set(go_ids_int)
        missing = sorted(pos_ids - cache_ids)

        print(f"[CHECK] unique positive GO ids: {len(pos_ids)}")
        print(f"[CHECK] cache GO ids: {len(cache_ids)}")
        print(f"[CHECK] positives missing from cache: {len(missing)}")

        if missing:
            print("[WARN] example missing positive GO ids:", missing[:20])

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

    # 4) Optional drift check against previous .pt
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
    pt_blob_path = out_prefix.with_suffix(".pt")

    # 6) Save .npy
    embs_np = embs.cpu().numpy().astype("float32")
    np.save(npy_path, embs_np)
    print(f"[OK] Saved embeddings → {npy_path}")

    # 7) Save meta
    meta_for_memmap: Dict[str, Any] = {"shape": embs_np.shape, "dtype": str(embs_np.dtype)}
    torch.save(meta_for_memmap, meta_path)
    print(f"[OK] Saved meta → {meta_path}")

    # 8) ids.json with REAL GO ids
    write_sidecars(npy_path, go_ids_int)

    # 9) Optional debug blob
    if store_pt_blob:
        save_checkpoint_blob(
            pt_blob_path,
            embs=embs.cpu(),
            go_ids_str=go_ids_str,
            go_ids_int=go_ids_int,
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
        go_path=(
            "/workspace/data_pfresgo/processed/"
            "go_texts_canonical_segmented.jsonl"
        ),
        output_prefix=(
            "/workspace/data_pfresgo/go_cache/all/"
            "go_text_embeddings_canonical"
        ),
        phase="gor2023_2023_01_01",
        model_name=(
            "microsoft/"
            "BiomedNLP-PubMedBERT-base-uncased-"
            "abstract-fulltext"
        ),
        device="cuda",
        batch_size=256,
        max_length=128,
        normalize=False,
        store_pt_blob=True,
        compare_to=None,
        pid_positives_path=None,
    )