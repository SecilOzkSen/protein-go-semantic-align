
"""
Created by @Secil Sen
ESM embedding extraction CLI

- Main pipeline: esm_embed_and_save2 (resumable, safe to re-run)
- Legacy pipeline: esm_embed_and_save (one-shot)

Example usage:
    python esm_embed_extractor_cli.py \
        --sequence-full /path/to/sequences_full.pkl \
        --save-dir /path/to/esm_output \
        --max-len 1024 --overlap 256 --shard-size 2000 --max-batch 10

Requires: fair-esm  (pip install fair-esm)
"""

"""
ESM Embedding Extraction Pipeline (CLI)

This script extracts residue-level proteins embeddings using Meta AI's ESM models
and saves them in shards for downstream tasks such as proteins function prediction.

Pipeline Overview:
------------------
1. **Input**: 
   - `sequences_full.pkl`: A dictionary mapping proteins IDs → amino acid sequences.
   - These sequences typically come from curated datasets (e.g., Swiss-Prot, TrEMBL, CAFA5)
     with GO annotations filtered by strong experimental evidence.

2. **Segmentation**:
   - Long sequences are split into overlapping segments (`max_len`, `overlap`).
   - This ensures compatibility with ESM’s maximum token length.

3. **Embedding Extraction**:
   - Segments are tokenized and passed through a pretrained ESM model 
     (default: `esm2_t33_650M_UR50D`).
   - The model is kept frozen for efficiency; only embeddings are extracted.
   - CLS/EOS tokens are removed, retaining only residue embeddings.

4. **Sharding & Storage**:
   - Segment embeddings are concatenated per proteins.
   - Proteins are grouped into shards (`shard_size`) and stored as `.pt` files 
     (`esm_embed_00000.pt`, `esm_embed_00001.pt`, …).
   - This keeps memory usage low and makes processing resumable.

5. **Resumable Processing**:
   - The main function `esm_embed_and_save2` skips already-processed proteins 
     by checking existing shards.
   - This allows safe re-runs without duplicating work.

6. **Batch Sizing**:
   - The pipeline automatically probes the maximum feasible batch size 
     for the current GPU/CPU, preventing OOM errors.

7. **Output**:
   - A directory of `.pt` files with proteins embeddings.
   - Auxiliary pickle files (`all_proteins.pkl`, `remaining_proteins.pkl`) 
     tracking dataset progress.

Usage:
------
Example CLI command:
    python esm_embed_extractor_cli.py \
        --sequence-full /path/to/sequences_full.pkl \
        --save-dir /path/to/esm_output \
        --pipeline resumable

By default, the script runs in resumable mode (`esm_embed_and_save2`). 
The legacy one-shot pipeline can be invoked with `--pipeline legacy`.
"""

import os
import re
import gc
import argparse
import pickle
import numpy as np
from collections import defaultdict

import torch
from tqdm import tqdm
from esm import pretrained

# -----------------------------
# Helpers & utilities
# -----------------------------
def validate_sequence(seq: str, allowed_regex=re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO\-]+$", re.IGNORECASE)):
    """
    Returns True if the sequence contains only valid tokens for ESM-style tokenizers.
    Allows 20 canonical AA + B, X, Z, J, U (selenocysteine), O (pyrrolysine), and '-' (gap).
    Rejects digits, underscores, spaces, or other symbols that cause tokenizer KeyErrors.
    """
    if not isinstance(seq, str):
        return False
    s = seq.strip().upper()
    return bool(allowed_regex.match(s))

def segment_sequence(seq, max_len, overlap):
    if len(seq) <= max_len:
        return [seq]
    segments, start = [], 0
    while start < len(seq):
        end = min(start + max_len, len(seq))
        segments.append(seq[start:end])
        if end == len(seq):
            break
        start += (max_len - overlap)
    return segments

def auto_batch_size(model, batch_converter, device, seqs, max_try=16):
    for batch_size in range(1, max_try + 1):
        try:
            batch_labels = [(f"proteins{i}", seq) for i, seq in enumerate(seqs[:batch_size])]
            labels, sequences, tokens = batch_converter(batch_labels)
            tokens = tokens.to(device)
            with torch.no_grad():
                _ = model(tokens, repr_layers=[33])
            print(f"Batch size {batch_size} successful.")
        except RuntimeError as e:
            print(f"Batch size {batch_size} failed: {str(e).splitlines()[0]}")
            torch.cuda.empty_cache()
            return batch_size - 1 if batch_size > 1 else 1
    return max_try

def load_model(model_name: str, device: torch.device):
    """
    Load an ESM model by short name.
    Common options:
      - esm2_t33_650M_UR50D (default)
      - esm2_t30_150M_UR50D
      - esm2_t12_35M_UR50D
    """
    loader = getattr(pretrained, model_name, None)
    if loader is None or not callable(loader):
        raise ValueError(f"Unknown model '{model_name}'. Expected a function under esm.pretrained (e.g., esm2_t33_650M_UR50D).")
    model, alphabet = loader()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter

def get_already_processed_ids(embed_dir: str):
    processed_ids = set()
    if not os.path.isdir(embed_dir):
        return processed_ids
    for fname in sorted(os.listdir(embed_dir)):
        if fname.endswith(".pt") and fname.startswith("esm_embed_"):
            fpath = os.path.join(embed_dir, fname)
            try:
                shard = torch.load(fpath, map_location="cpu", weights_only=False)
                if isinstance(shard, dict):
                    processed_ids.update(shard.keys())
            except Exception as e:
                print(f"[warn] Failed to load '{fname}': {e}. Deleting file...")
                try:
                    os.remove(fpath)
                    print(f"[info] {fname} deleted.")
                except Exception as delete_err:
                    print(f"[warn] Could not delete {fname}: {delete_err}")
    return processed_ids

def flush_buffer(model, alphabet, batch_converter, device, segment_buffer, shard_embed):
    if not segment_buffer:
        return 0
    try:
        batch_labels = [(f"{pid}_seg{i}", seg) for i, (pid, seg) in enumerate(segment_buffer)]
        labels, sequences, tokens = batch_converter(batch_labels)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[33])
            reps = out["representations"][33]
        # tokens shape: [B, L]; CLS at index 0, EOS at index length-1
        added = 0
        for idx, (pid, _) in enumerate(segment_buffer):
            length = (tokens[idx] != alphabet.padding_idx).sum().item()
            # exclude CLS and EOS: 1 .. length-2 inclusive
            if length >= 2:
                emb = reps[idx, 1:length - 1].detach().cpu().numpy()
                shard_embed[pid].append(emb)
                added += 1
        segment_buffer.clear()
        return added
    except Exception as e:
        print(f"[warn] Failed to process batch: {e}")
        segment_buffer.clear()
        return 0

def save_shard(embed_dir: str, shard_id: int, shard_embed: dict):
    try:
        final_shard = {pid: np.concatenate(emb_list, axis=0) for pid, emb_list in shard_embed.items() if emb_list}
        if not final_shard:
            return False
        save_path = os.path.join(embed_dir, f"esm_embed_{shard_id:05d}.pt")
        torch.save(final_shard, save_path)
        print(f"[info] Saved shard {shard_id:05d} with {len(final_shard)} proteins")
        shard_embed.clear()
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"[warn] Failed to save shard {shard_id:05d}: {e}")
        return False

# -----------------------------
# Pipelines
# -----------------------------
def esm_embed_and_save(
    sequence_full: str,
    save_dir: str,
    max_len: int = 1024,
    overlap: int = 256,
    shard_size: int = 2000,
    max_batch_probe: int = 10,
    model_name: str = "esm2_t33_650M_UR50D",
    skip_invalid: bool = True,
):
    """
    Legacy one-shot pipeline. Processes all proteins and writes shards sequentially.
    """
    embed_dir = os.path.join(save_dir, "embeddings")
    os.makedirs(embed_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet, batch_converter = load_model(model_name, device)

    with open(sequence_full, "rb") as f:
        all_proteins = pickle.load(f)

    protein_ids = list(all_proteins.keys())

    with open(os.path.join(save_dir, "all_proteins.pkl"), "wb") as f:
        pickle.dump(protein_ids, f)

    # Probe an automatic batch size using a small subset of segments
    test_segs = []
    probe_count = 0
    for pid in protein_ids:
        seq = all_proteins[pid]
        if skip_invalid and not validate_sequence(seq):
            continue
        test_segs.extend(segment_sequence(seq, max_len, overlap))
        probe_count += 1
        if probe_count >= max_batch_probe:
            break
    BATCH_SIZE = auto_batch_size(model, batch_converter, device, test_segs) or 1
    print(f"[info] Batch size determined: {BATCH_SIZE}")

    shard_embed = defaultdict(list)
    segment_buffer = []
    shard_id = 0

    processed_in_shard = 0

    for pid in tqdm(protein_ids, desc="Processing (legacy)"):
        seq = all_proteins[pid]
        if skip_invalid and not validate_sequence(seq):
            print(f"[skip] Invalid tokens in '{pid}'.")
            continue

        for seg in segment_sequence(seq, max_len, overlap):
            segment_buffer.append((pid, seg))
            if len(segment_buffer) == BATCH_SIZE:
                processed_in_shard += flush_buffer(model, alphabet, batch_converter, device, segment_buffer, shard_embed)

        # Save shard when enough distinct proteins collected
        if len(shard_embed) >= shard_size:
            if processed_in_shard > 0:
                saved = save_shard(embed_dir, shard_id, shard_embed)
                if saved:
                    shard_id += 1
                processed_in_shard = 0

    # Flush leftovers
    if segment_buffer:
        processed_in_shard += flush_buffer(model, alphabet, batch_converter, device, segment_buffer, shard_embed)
    if shard_embed:
        if processed_in_shard > 0:
            saved = save_shard(embed_dir, shard_id, shard_embed)
            if saved:
                shard_id += 1

    print("[done] All residue-level embeddings saved (legacy).")

def esm_embed_and_save2(
    sequence_full: str,
    save_dir: str,
    max_len: int = 1024,
    overlap: int = 256,
    shard_size: int = 2000,
    max_batch_probe: int = 10,
    model_name: str = "esm2_t33_650M_UR50D",
    skip_invalid: bool = True,
):
    """
    Resumable pipeline.
    - Skips proteins already present in existing shards (based on IDs).
    - Safe to re-run; will only process remaining proteins.
    """
    embed_dir = os.path.join(save_dir, "embeddings")
    os.makedirs(embed_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet, batch_converter = load_model(model_name, device)

    with open(sequence_full, "rb") as f:
        all_proteins = pickle.load(f)

    all_ids = set(all_proteins.keys())
    done_ids = get_already_processed_ids(embed_dir)
    remaining_ids = [pid for pid in all_ids - done_ids]

    print(f"[info] Total proteins: {len(all_ids)}")
    print(f"[info] Already embedded: {len(done_ids)}")
    print(f"[info] Remaining: {len(remaining_ids)}")

    with open(os.path.join(save_dir, "remaining_proteins.pkl"), "wb") as f:
        pickle.dump(remaining_ids, f)

    # Probe an automatic batch size using a subset of segments from remaining proteins
    test_segs = []
    probe_count = 0
    for pid in remaining_ids:
        seq = all_proteins[pid]
        if skip_invalid and not validate_sequence(seq):
            continue
        test_segs.extend(segment_sequence(seq, max_len, overlap))
        probe_count += 1
        if probe_count >= max_batch_probe:
            break
    BATCH_SIZE = auto_batch_size(model, batch_converter, device, test_segs) or 1
    print(f"[info] Batch size determined: {BATCH_SIZE}")

    shard_embed = defaultdict(list)
    segment_buffer = []
    # Continue shard numbering after existing files
    existing = [f for f in os.listdir(embed_dir) if f.startswith("esm_embed_") and f.endswith(".pt")]
    shard_id = len(existing)

    processed_in_shard = 0

    for pid in tqdm(remaining_ids, desc="Processing (resumable)"):
        seq = all_proteins[pid]
        if skip_invalid and not validate_sequence(seq):
            print(f"[skip] Invalid tokens in '{pid}'.")
            continue

        for seg in segment_sequence(seq, max_len, overlap):
            segment_buffer.append((pid, seg))
            if len(segment_buffer) == BATCH_SIZE:
                processed_in_shard += flush_buffer(model, alphabet, batch_converter, device, segment_buffer, shard_embed)

        # Save shard when enough distinct proteins collected
        if len(shard_embed) >= shard_size:
            if processed_in_shard > 0:
                saved = save_shard(embed_dir, shard_id, shard_embed)
                if saved:
                    shard_id += 1
                processed_in_shard = 0

    # Flush leftovers
    if segment_buffer:
        processed_in_shard += flush_buffer(model, alphabet, batch_converter, device, segment_buffer, shard_embed)
    if shard_embed:
        if processed_in_shard > 0:
            saved = save_shard(embed_dir, shard_id, shard_embed)
            if saved:
                shard_id += 1

    print("[done] All residue-level embeddings saved (resumable).")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ESM embedding extractor (CLI)")
    p.add_argument("--sequence-full", required=True, help="Path to sequences_full.pkl (dict: {protein_id: sequence})")
    p.add_argument("--save-dir", required=True, help="Output directory (will create 'embeddings' subfolder)")
    p.add_argument("--max-len", type=int, default=1024, help="Max tokens per segment")
    p.add_argument("--overlap", type=int, default=256, help="Sliding window overlap between segments")
    p.add_argument("--shard-size", type=int, default=2000, help="Number of proteins per shard")
    p.add_argument("--max-batch", type=int, default=10, help="How many proteins to probe for auto batch sizing")
    p.add_argument("--model-name", type=str, default="esm2_t33_650M_UR50D", help="esm.pretrained loader name")
    p.add_argument("--skip-invalid", action="store_true", default=True, help="Skip sequences with invalid tokens")
    p.add_argument("--no-skip-invalid", dest="skip_invalid", action="store_false", help="Do not skip invalid sequences")
    p.add_argument("--pipeline", choices=["resumable", "legacy"], default="resumable",
                   help="'resumable' runs esm_embed_and_save2; 'legacy' runs esm_embed_and_save.")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.pipeline == "resumable":
        esm_embed_and_save2(
            sequence_full=args.sequence_full,
            save_dir=args.save_dir,
            max_len=args.max_len,
            overlap=args.overlap,
            shard_size=args.shard_size,
            max_batch_probe=args.max_batch,
            model_name=args.model_name,
            skip_invalid=args.skip_invalid,
        )
    else:
        esm_embed_and_save(
            sequence_full=args.sequence_full,
            save_dir=args.save_dir,
            max_len=args.max_len,
            overlap=args.overlap,
            shard_size=args.shard_size,
            max_batch_probe=args.max_batch,
            model_name=args.model_name,
            skip_invalid=args.skip_invalid,
        )

if __name__ == "__main__":
    main()
