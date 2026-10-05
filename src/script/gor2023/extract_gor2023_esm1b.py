'''
cd /workspace/protein-go-semantic-align

python -m src.script.extract_pfresgo_esm1b \
  --fasta /workspace/stargo/datasets/pfresgo/nrPDB-GO_2019.06.18_sequences.fasta \
  --res-out-dir /workspace/data_pfresgo/protein_embeddings/esm1b_residue \
  --no-protein-bank \
  --batch-size 32 \
  --max-tokens-per-batch 24000 \
  --res-shard-max-rows 2000000
'''

import argparse
import gc
import json
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from esm import pretrained
from numpy.lib.format import open_memmap
from tqdm import tqdm

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

EMB_DIM = 1280
REPRESENTATION_LAYER = 33
NPY_DTYPE = np.float16

VALID_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWYBXZOU"
)

GAPLIKE_CHARACTERS = {"-", ".", "_"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract ESM-1b residue and pooled protein "
            "embeddings from the GOR2023 FASTA."
        )
    )

    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path(
            "/workspace/GOR2023/sequences/final/"
            "all_proteins_temporal_safe.fasta"
        ),
    )

    parser.add_argument(
        "--res-out-dir",
        type=Path,
        default=Path(
            "/workspace/GOR2023/"
            "protein_embeddings/esm1b_residue"
        ),
    )

    parser.add_argument(
        "--prot-out-dir",
        type=Path,
        default=Path(
            "/workspace/GOR2023/"
            "protein_embeddings/esm1b_protein"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=6000,
        help=(
            "Maximum sum of sequence lengths in a batch. "
            "Actual batch size is also limited by --batch-size."
        ),
    )

    parser.add_argument(
        "--res-shard-max-rows",
        type=int,
        default=2_000_000,
    )

    parser.add_argument(
        "--protein-shard-size",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--long-sequence-policy",
        choices=("error", "truncate"),
        default="truncate",
        help=(
            "How to handle proteins longer than --max-seq-length. "
            "GOR2023 uses deterministic N-terminal truncation."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--no-protein-bank",
        action="store_true",
        help="Only generate residue embeddings.",
    )

    return parser.parse_args()


def clean_sequence(sequence: str) -> str:
    sequence = re.sub(
        r"\s+|\d+",
        "",
        sequence,
    ).upper()

    sequence = "".join(
        character
        for character in sequence
        if character not in GAPLIKE_CHARACTERS
    )

    if sequence.endswith("*"):
        sequence = sequence[:-1]

    sequence = "".join(
        character
        if character in VALID_AMINO_ACIDS
        else "X"
        for character in sequence
    )

    return sequence


def load_fasta(path: Path) -> Dict[str, str]:
    sequences = {}

    current_id = None
    current_parts = []

    def flush():
        if current_id is None:
            return

        if current_id in sequences:
            raise ValueError(
                f"Duplicate FASTA protein ID: {current_id}"
            )

        sequence = clean_sequence(
            "".join(current_parts)
        )

        if not sequence:
            raise ValueError(
                f"Empty sequence for {current_id}"
            )

        sequences[current_id] = sequence

    with path.open(
            "r",
            encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                flush()

                header = line[1:].strip()
                current_id = header.split()[0]
                current_parts = []
            else:
                if current_id is None:
                    raise ValueError(
                        "Sequence encountered before FASTA header."
                    )

                current_parts.append(line)

    flush()

    return sequences


def list_shard_ids(
        output_dir: Path,
        prefix: str,
        suffix: str,
) -> List[int]:
    shard_ids = []

    for path in output_dir.glob(
            f"{prefix}_*{suffix}"
    ):
        filename = path.name

        try:
            shard_text = (
                filename
                .replace(f"{prefix}_", "")
                .replace(suffix, "")
            )

            shard_ids.append(int(shard_text))
        except ValueError:
            continue

    return sorted(shard_ids)


def read_completed_residue_ids(
        output_dir: Path,
) -> Set[str]:
    completed = set()

    for index_path in sorted(
            output_dir.glob(
                "res_esm1b_*.index.tsv"
            )
    ):
        with index_path.open(
                "r",
                encoding="utf-8",
        ) as file:
            for line in file:
                columns = line.rstrip("\n").split("\t")

                if columns and columns[0]:
                    completed.add(columns[0])

    return completed


def read_completed_protein_ids(
        output_dir: Path,
) -> Set[str]:
    completed = set()

    for ids_path in sorted(
            output_dir.glob(
                "fused_esm1b_*.ids.txt"
            )
    ):
        with ids_path.open(
                "r",
                encoding="utf-8",
        ) as file:
            for line in file:
                protein_id = line.strip()

                if protein_id:
                    completed.add(protein_id)

    return completed


def l2_normalize(
        vector: np.ndarray,
        epsilon: float = 1e-12,
):
    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm < epsilon:
        return None

    return (
            vector / norm
    ).astype(
        np.float32,
        copy=False,
    )


class ResidueShardWriter:
    def __init__(
            self,
            output_dir: Path,
            shard_id: int,
            capacity_rows: int,
            embedding_dim: int,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.shard_id = shard_id
        self.capacity = capacity_rows
        self.embedding_dim = embedding_dim

        self.data_path = (
                self.output_dir
                / f"res_esm1b_{shard_id:05d}.data.npy"
        )

        self.index_path = (
                self.output_dir
                / f"res_esm1b_{shard_id:05d}.index.tsv"
        )

        self.meta_path = (
                self.output_dir
                / f"res_esm1b_{shard_id:05d}.meta.json"
        )

        self.memmap = open_memmap(
            str(self.data_path),
            mode="w+",
            dtype=NPY_DTYPE,
            shape=(
                capacity_rows,
                embedding_dim,
            ),
        )

        self.cursor = 0
        self.index_rows = []

    def remaining(self):
        return self.capacity - self.cursor

    def append(
            self,
            protein_id: str,
            embedding: np.ndarray,
    ):
        sequence_length = embedding.shape[0]

        if sequence_length <= 0:
            return False

        if sequence_length > self.remaining():
            return False

        start = self.cursor
        end = start + sequence_length

        self.memmap[start:end] = embedding
        self.index_rows.append(
            (
                protein_id,
                start,
                end,
            )
        )

        self.cursor = end
        return True

    def close(self):
        self.memmap.flush()
        del self.memmap

        with self.index_path.open(
                "w",
                encoding="utf-8",
        ) as file:
            for protein_id, start, end in self.index_rows:
                file.write(
                    f"{protein_id}\t{start}\t{end}\n"
                )

        metadata = {
            "shape": [
                self.capacity,
                self.embedding_dim,
            ],
            "used_rows": self.cursor,
            "dtype": "float16",
            "embed_dim": self.embedding_dim,
            "model": "esm1b_t33_650M_UR50S",
            "representation_layer": (
                REPRESENTATION_LAYER
            ),
            "format": (
                "residue_concat_with_index"
            ),
        }

        with self.meta_path.open(
                "w",
                encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
            )

        print(
            f"[res-save] shard={self.shard_id:05d} "
            f"proteins={len(self.index_rows)} "
            f"used_rows={self.cursor}"
        )


def save_protein_shard(
        output_dir: Path,
        shard_id: int,
        vectors: List[np.ndarray],
        protein_ids: List[str],
):
    if not vectors:
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = np.ascontiguousarray(
        np.stack(
            vectors,
            axis=0,
        ).astype(
            np.float16,
            copy=False,
        )
    )

    data_path = (
            output_dir
            / f"fused_esm1b_{shard_id:05d}.npy"
    )

    ids_path = (
            output_dir
            / f"fused_esm1b_{shard_id:05d}.ids.txt"
    )

    meta_path = (
            output_dir
            / f"fused_esm1b_{shard_id:05d}.meta.json"
    )

    np.save(
        data_path,
        data,
        allow_pickle=False,
    )

    with ids_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(protein_ids) + "\n"
        )

    metadata = {
        "shape": list(data.shape),
        "dtype": "float16",
        "l2_normalized": True,
        "embed_dim": EMB_DIM,
        "model": "esm1b_t33_650M_UR50S",
        "representation_layer": (
            REPRESENTATION_LAYER
        ),
        "pooling": "mean(residue)",
        "format": "protein_bank_shard",
    }

    with meta_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"[protein-save] shard={shard_id:05d} "
        f"proteins={len(protein_ids)}"
    )


def build_batches(
        protein_ids: List[str],
        sequences: Dict[str, str],
        max_batch_size: int,
        max_tokens_per_batch: int,
):
    batch = []
    token_count = 0

    for protein_id in protein_ids:
        sequence_length = len(
            sequences[protein_id]
        )

        would_exceed_size = (
                len(batch) >= max_batch_size
        )

        would_exceed_tokens = (
                batch
                and token_count + sequence_length
                > max_tokens_per_batch
        )

        if would_exceed_size or would_exceed_tokens:
            yield batch
            batch = []
            token_count = 0

        batch.append(protein_id)
        token_count += sequence_length

    if batch:
        yield batch


@torch.inference_mode()
def run_model_batch(
        model,
        batch_converter,
        protein_ids: List[str],
        sequences: Dict[str, str],
        device: torch.device,
):
    labels = [
        (
            protein_id,
            sequences[protein_id],
        )
        for protein_id in protein_ids
    ]

    _, _, tokens = batch_converter(labels)

    tokens = tokens.to(
        device,
        non_blocking=True,
    )

    autocast_enabled = (
            device.type == "cuda"
    )

    with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
    ):
        output = model(
            tokens,
            repr_layers=[
                REPRESENTATION_LAYER
            ],
            return_contacts=False,
        )

    representations = output[
        "representations"
    ][REPRESENTATION_LAYER]

    results = {}

    for row_index, protein_id in enumerate(
            protein_ids
    ):
        sequence_length = len(
            sequences[protein_id]
        )

        residue_embedding = (
            representations[
                row_index,
                1:sequence_length + 1,
            ]
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        if residue_embedding.shape != (
                sequence_length,
                EMB_DIM,
        ):
            raise ValueError(
                f"{protein_id}: expected "
                f"{(sequence_length, EMB_DIM)}, "
                f"got {residue_embedding.shape}"
            )

        results[protein_id] = (
            residue_embedding
        )

    del output
    del representations
    del tokens

    return results


def next_shard_id(
        output_dir: Path,
        prefix: str,
        suffix: str,
):
    existing = list_shard_ids(
        output_dir,
        prefix,
        suffix,
    )

    return (
        max(existing) + 1
        if existing
        else 0
    )


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available."
        )

    device = torch.device(args.device)

    args.res_out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.prot_out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading FASTA...")
    sequences = load_fasta(args.fasta)

    original_lengths = {
        protein_id: len(sequence)
        for protein_id, sequence in sequences.items()
    }

    too_long = {
        protein_id: len(sequence)
        for protein_id, sequence
        in sequences.items()
        if len(sequence) > args.max_seq_length
    }

    if too_long and args.long_sequence_policy == "error":
        raise ValueError(
            f"{len(too_long)} sequences exceed "
            f"--max-seq-length={args.max_seq_length}. "
            f"Examples: {list(too_long.items())[:10]}"
        )

    if too_long and args.long_sequence_policy == "truncate":
        sequences = {
            protein_id: sequence[:args.max_seq_length]
            for protein_id, sequence in sequences.items()
        }

    embedded_lengths = {
        protein_id: len(sequence)
        for protein_id, sequence in sequences.items()
    }

    length_manifest_path = (
            args.res_out_dir / "sequence_length_manifest.tsv"
    )
    with length_manifest_path.open("w", encoding="utf-8") as file:
        file.write(
            "protein_id\toriginal_length\tembedded_length\ttruncated\n"
        )
        for protein_id in sorted(sequences):
            original_length = original_lengths[protein_id]
            embedded_length = embedded_lengths[protein_id]
            file.write(
                f"{protein_id}\t{original_length}\t{embedded_length}\t"
                f"{int(original_length != embedded_length)}\n"
            )

    with (args.res_out_dir / "seq_len_lookup.pkl").open("wb") as file:
        pickle.dump(
            embedded_lengths,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"FASTA proteins: {len(sequences)}")
    print(f"Truncated proteins: {len(too_long)}")
    print(f"Maximum embedded length: {max(embedded_lengths.values())}")

    print("Loading ESM-1b...")
    model, alphabet = (
        pretrained.esm1b_t33_650M_UR50S()
    )

    model.eval()
    model.to(device)

    batch_converter = (
        alphabet.get_batch_converter()
    )

    completed_residue = (
        read_completed_residue_ids(
            args.res_out_dir
        )
    )

    if args.no_protein_bank:
        completed = completed_residue
    else:
        completed_protein = (
            read_completed_protein_ids(
                args.prot_out_dir
            )
        )

        completed = (
                completed_residue
                & completed_protein
        )

    remaining_ids = [
        protein_id
        for protein_id in sequences
        if protein_id not in completed
    ]

    # Sorting by length reduces padding waste.
    remaining_ids.sort(
        key=lambda protein_id: (
            len(sequences[protein_id]),
            protein_id,
        )
    )

    print(f"Already completed: {len(completed)}")
    print(f"Remaining proteins: {len(remaining_ids)}")
    print(f"Device: {device}")

    remaining_rows = sum(
        embedded_lengths[protein_id]
        for protein_id in remaining_ids
    )
    estimated_bytes = remaining_rows * EMB_DIM * np.dtype(NPY_DTYPE).itemsize
    disk_free = shutil.disk_usage(args.res_out_dir).free
    print(
        "Estimated new residue storage: "
        f"{estimated_bytes / (1024 ** 3):.2f} GiB"
    )
    print(f"Available disk: {disk_free / (1024 ** 3):.2f} GiB")
    if estimated_bytes > disk_free * 0.95:
        raise RuntimeError(
            "Insufficient free disk for residue embeddings: "
            f"need approximately {estimated_bytes / (1024 ** 3):.2f} GiB, "
            f"available {disk_free / (1024 ** 3):.2f} GiB."
        )

    residue_shard_id = next_shard_id(
        args.res_out_dir,
        "res_esm1b",
        ".index.tsv",
    )

    protein_shard_id = next_shard_id(
        args.prot_out_dir,
        "fused_esm1b",
        ".ids.txt",
    )

    residue_writer = ResidueShardWriter(
        output_dir=args.res_out_dir,
        shard_id=residue_shard_id,
        capacity_rows=args.res_shard_max_rows,
        embedding_dim=EMB_DIM,
    )

    protein_vectors = []
    protein_vector_ids = []

    skipped_path = (
            args.res_out_dir
            / "skipped_proteins.txt"
    )

    batches = list(
        build_batches(
            protein_ids=remaining_ids,
            sequences=sequences,
            max_batch_size=args.batch_size,
            max_tokens_per_batch=(
                args.max_tokens_per_batch
            ),
        )
    )

    progress = tqdm(
        total=len(remaining_ids),
        desc="ESM-1b",
    )

    with skipped_path.open(
            "a",
            encoding="utf-8",
    ) as skipped_file:
        for batch_ids in batches:
            pending_batches = [batch_ids]

            while pending_batches:
                current_batch = (
                    pending_batches.pop(0)
                )

                try:
                    results = run_model_batch(
                        model=model,
                        batch_converter=(
                            batch_converter
                        ),
                        protein_ids=current_batch,
                        sequences=sequences,
                        device=device,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()

                    if len(current_batch) == 1:
                        protein_id = current_batch[0]
                        raise RuntimeError(
                            f"Single-protein CUDA OOM for {protein_id}; "
                            "refusing to create an incomplete embedding store."
                        )

                    midpoint = (
                            len(current_batch) // 2
                    )

                    pending_batches.insert(
                        0,
                        current_batch[midpoint:],
                    )

                    pending_batches.insert(
                        0,
                        current_batch[:midpoint],
                    )

                    continue

                for protein_id in current_batch:
                    residue_float32 = results[
                        protein_id
                    ]

                    residue_float16 = (
                        residue_float32.astype(
                            np.float16,
                            copy=False,
                        )
                    )

                    if (
                            residue_float16.shape[0]
                            > residue_writer.remaining()
                    ):
                        residue_writer.close()
                        residue_shard_id += 1

                        residue_writer = (
                            ResidueShardWriter(
                                output_dir=(
                                    args.res_out_dir
                                ),
                                shard_id=(
                                    residue_shard_id
                                ),
                                capacity_rows=(
                                    args.res_shard_max_rows
                                ),
                                embedding_dim=EMB_DIM,
                            )
                        )

                    if not residue_writer.append(
                            protein_id,
                            residue_float16,
                    ):
                        raise RuntimeError(
                            f"Could not append "
                            f"{protein_id} to residue shard."
                        )

                    if not args.no_protein_bank:
                        pooled = (
                            residue_float32.mean(
                                axis=0
                            )
                        )

                        pooled = l2_normalize(
                            pooled
                        )

                        if (
                                pooled is None
                                or not np.isfinite(
                            pooled
                        ).all()
                        ):
                            skipped_file.write(
                                protein_id + "\n"
                            )
                            skipped_file.flush()
                            continue

                        protein_vectors.append(
                            pooled.astype(
                                np.float16,
                                copy=False,
                            )
                        )

                        protein_vector_ids.append(
                            protein_id
                        )

                        if (
                                len(protein_vector_ids)
                                >= args.protein_shard_size
                        ):
                            save_protein_shard(
                                output_dir=(
                                    args.prot_out_dir
                                ),
                                shard_id=(
                                    protein_shard_id
                                ),
                                vectors=(
                                    protein_vectors
                                ),
                                protein_ids=(
                                    protein_vector_ids
                                ),
                            )

                            protein_vectors.clear()
                            protein_vector_ids.clear()
                            protein_shard_id += 1

                    progress.update(1)

                del results
                gc.collect()

                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if protein_vector_ids:
        save_protein_shard(
            output_dir=args.prot_out_dir,
            shard_id=protein_shard_id,
            vectors=protein_vectors,
            protein_ids=protein_vector_ids,
        )

    residue_writer.close()
    progress.close()

    completed_after = read_completed_residue_ids(args.res_out_dir)
    expected_ids = set(sequences)
    missing_after = sorted(expected_ids - completed_after)
    unexpected_after = sorted(completed_after - expected_ids)

    if missing_after or unexpected_after:
        raise RuntimeError(
            "Embedding coverage audit failed: "
            f"expected={len(expected_ids)}, completed={len(completed_after)}, "
            f"missing={len(missing_after)}, unexpected={len(unexpected_after)}, "
            f"missing_examples={missing_after[:20]}, "
            f"unexpected_examples={unexpected_after[:20]}"
        )

    extraction_summary = {
        "model": "esm1b_t33_650M_UR50S",
        "representation_layer": REPRESENTATION_LAYER,
        "embedding_dim": EMB_DIM,
        "dtype": "float16",
        "fasta": str(args.fasta),
        "proteins": len(expected_ids),
        "coverage_fraction": 1.0,
        "max_original_length": max(original_lengths.values()),
        "max_embedded_length": max(embedded_lengths.values()),
        "truncated_proteins": len(too_long),
        "long_sequence_policy": args.long_sequence_policy,
        "sequence_length_manifest": str(length_manifest_path),
        "seq_len_lookup": str(args.res_out_dir / "seq_len_lookup.pkl"),
    }
    with (args.res_out_dir / "extraction_summary.json").open(
            "w", encoding="utf-8"
    ) as file:
        json.dump(extraction_summary, file, indent=2, sort_keys=True)
        file.write("\n")

    print("\nExtraction completed with 100% protein coverage.")


if __name__ == "__main__":
    main()
