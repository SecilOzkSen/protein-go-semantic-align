import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.main import (
    load_structured_cfg,
    set_seed,
    build_go_cache,
    build_stores,
    build_datasets,
    enforce_cache_alignment,
)
from src.go import load_go_parents, load_go_children
from src.utils import load_raw_json, load_raw_pickle, load_go_set
from src.utils.helpers import (
    build_altid_map_from_go_terms,
    canonicalize_id_list,
    canonicalize_pid2pos,
    go_str_to_int_any,
)

# Reuse helper functions from the retriever dump script.
# This keeps dataset/collator behavior identical to the P3a dump pipeline.
from src.script.dump_retriever_candidates import (  # type: ignore
    build_go_encoder_and_text_store,
    make_dump_loader,
)


def setup_logging_simple() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def canonicalize_and_align_inputs(args, go_cache, logger):
    if args.eval_space == "seen":
        eval_id_list = load_raw_pickle(args.go_path_seen)
    else:
        eval_id_list = load_raw_pickle(args.go_path_observed)

    eval_seen_go_ids = load_raw_pickle(args.go_path_seen)
    eval_unseen_ids = load_raw_pickle(args.zero_shot_path)
    eval_rare_go_ids = load_raw_pickle(args.few_shot_path)

    go_terms = load_raw_json(args.go_basic_json)
    alt_map = build_altid_map_from_go_terms(go_terms) if go_terms else {}
    logger.info("[canon] alt_id map size = %d", len(alt_map))

    pid2pos_raw = load_raw_json(args.pid2pos)
    pid2pos = canonicalize_pid2pos(pid2pos_raw, alt_map) if alt_map else pid2pos_raw

    zs = load_go_set(args.zero_shot_path)
    fs = load_go_set(args.few_shot_path)

    zs = canonicalize_id_list(list(zs), alt_map) if alt_map else zs
    fs = canonicalize_id_list(list(fs), alt_map) if alt_map else fs

    eval_id_list = (
        canonicalize_id_list(eval_id_list, alt_map)
        if alt_map
        else [go_str_to_int_any(x) for x in eval_id_list]
    )
    eval_seen_go_ids = (
        canonicalize_id_list(eval_seen_go_ids, alt_map)
        if alt_map
        else [go_str_to_int_any(x) for x in eval_seen_go_ids]
    )
    eval_unseen_ids = (
        canonicalize_id_list(eval_unseen_ids, alt_map)
        if alt_map
        else [go_str_to_int_any(x) for x in eval_unseen_ids]
    )
    eval_rare_go_ids = (
        canonicalize_id_list(eval_rare_go_ids, alt_map)
        if alt_map
        else [go_str_to_int_any(x) for x in eval_rare_go_ids]
    )

    pid2pos, eval_id_list, eval_seen_go_ids, eval_unseen_ids, eval_rare_go_ids = enforce_cache_alignment(
        go_cache=go_cache,
        pid2pos=pid2pos,
        eval_id_list=eval_id_list,
        eval_seen_go_ids=eval_seen_go_ids,
        eval_unseen_ids=eval_unseen_ids,
        eval_rare_go_ids=eval_rare_go_ids,
        logger=logger,
        drop_empty_proteins=False,
    )

    return {
        "pid2pos": pid2pos,
        "zs": zs,
        "fs": fs,
        "eval_id_list": eval_id_list,
        "eval_seen_go_ids": eval_seen_go_ids,
        "eval_unseen_ids": eval_unseen_ids,
        "eval_rare_go_ids": eval_rare_go_ids,
    }


def _valid_mask_from_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    m = batch["prot_attn_mask"].to(device, non_blocking=True)
    if m.dim() == 3 and m.size(-1) == 1:
        m = m.squeeze(-1)
    if m.dtype != torch.bool:
        m = m != 0
    return m


@torch.no_grad()
def collect_mean_esm_embeddings(
    loader: DataLoader,
    *,
    device: torch.device,
    eval_id_to_col: Dict[int, int],
    desc: str,
) -> Tuple[List[str], np.ndarray, List[np.ndarray]]:
    """
    Returns:
      protein_ids: list[str]
      Z: [N,D] float32, L2-normalized raw ESM masked-mean embeddings
      label_cols: list[np.ndarray], true labels restricted to eval_id_to_col
    """
    protein_ids: List[str] = []
    z_chunks: List[np.ndarray] = []
    label_cols: List[np.ndarray] = []

    for batch in tqdm(loader, desc=desc):
        H = batch["prot_emb_pad"].to(device, non_blocking=True).float()  # [B,T,D]
        valid = _valid_mask_from_batch(batch, device)

        w = valid.to(H.dtype).unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        z = (H * w).sum(dim=1) / denom
        z = torch.nn.functional.normalize(z.float(), dim=-1)

        z_chunks.append(z.detach().cpu().numpy().astype(np.float32))

        pids = batch.get("protein_ids", None)
        if pids is None:
            pids = [f"row_{len(protein_ids) + i}" for i in range(int(H.size(0)))]
        protein_ids.extend([str(x) for x in pids])

        for gids in batch["pos_go_global"]:
            cols = []
            for g in gids.detach().cpu().tolist():
                j = eval_id_to_col.get(int(g), None)
                if j is not None:
                    cols.append(int(j))
            if cols:
                label_cols.append(np.asarray(sorted(set(cols)), dtype=np.int32))
            else:
                label_cols.append(np.empty(0, dtype=np.int32))

    Z = np.concatenate(z_chunks, axis=0)
    if len(protein_ids) != Z.shape[0] or len(label_cols) != Z.shape[0]:
        raise RuntimeError(
            f"collection size mismatch: ids={len(protein_ids)} labels={len(label_cols)} Z={Z.shape}"
        )
    return protein_ids, Z, label_cols


@torch.no_grad()
def compute_knn(
    train_z: np.ndarray,
    query_z: np.ndarray,
    *,
    train_ids: List[str],
    query_ids: List[str],
    n_neighbors: int,
    device: torch.device,
    query_batch_size: int = 64,
    use_fp16: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine kNN over already-normalized embeddings.

    Returns:
      nn_idx: [Nq, M] int64 indices into train_z
      nn_sim: [Nq, M] float32 cosine similarities
    """
    n_train = int(train_z.shape[0])
    n_query = int(query_z.shape[0])
    M = min(int(n_neighbors), n_train)

    dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32
    train_t = torch.from_numpy(train_z).to(device=device, dtype=dtype)

    id2train = {pid: i for i, pid in enumerate(train_ids)}

    nn_idx = np.empty((n_query, M), dtype=np.int64)
    nn_sim = np.empty((n_query, M), dtype=np.float32)

    for s in tqdm(range(0, n_query, query_batch_size), desc="ESM kNN search"):
        e = min(n_query, s + query_batch_size)
        q = torch.from_numpy(query_z[s:e]).to(device=device, dtype=dtype)
        sims = q @ train_t.T  # [B,Ntrain]

        # Exclude exact self for train split or overlapping query/train ids.
        for bi, pid in enumerate(query_ids[s:e]):
            j = id2train.get(pid, None)
            if j is not None:
                sims[bi, int(j)] = -float("inf")

        vals, idx = torch.topk(sims.float(), k=M, dim=1)
        nn_idx[s:e] = idx.detach().cpu().numpy().astype(np.int64)
        nn_sim[s:e] = vals.detach().cpu().numpy().astype(np.float32)

    return nn_idx, nn_sim


def _softmax_np(x: np.ndarray, tau: float) -> np.ndarray:
    x = x.astype(np.float64)
    tau = max(float(tau), 1e-8)
    x = x / tau
    x = x - np.max(x)
    w = np.exp(x)
    denom = np.sum(w)
    if not np.isfinite(denom) or denom <= 0:
        return np.ones_like(x, dtype=np.float64) / max(1, x.size)
    return w / denom


def build_knn_candidate_arrays(
    *,
    nn_idx: np.ndarray,
    nn_sim: np.ndarray,
    train_label_cols: List[np.ndarray],
    true_label_cols: List[np.ndarray],
    n_go: int,
    topk: int,
    tau: float,
    ref_top_cols: Optional[np.ndarray] = None,
    fill_from_ref: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      top_cols: [N,K] int32
      top_scores: [N,K] float32
      top_labels: [N,K] int8
      top_valid: [N,K] int8, 1 if from actual kNN transfer, 0 if filler
    """
    N, M = nn_idx.shape
    K = int(topk)

    top_cols = np.zeros((N, K), dtype=np.int32)
    top_scores = np.full((N, K), -1e6, dtype=np.float32)
    top_labels = np.zeros((N, K), dtype=np.int8)
    top_valid = np.zeros((N, K), dtype=np.int8)

    fill_rows = 0

    for i in tqdm(range(N), desc="build ESM-kNN GO candidates"):
        weights = _softmax_np(nn_sim[i], tau=tau)
        scores: Dict[int, float] = {}

        for r, tr_i in enumerate(nn_idx[i]):
            cols = train_label_cols[int(tr_i)]
            if cols.size == 0:
                continue
            w = float(weights[r])
            for c in cols.tolist():
                scores[int(c)] = scores.get(int(c), 0.0) + w

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected_cols: List[int] = []
        selected_scores: List[float] = []
        selected_valid: List[int] = []
        seen = set()

        for c, sc in ranked[:K]:
            c = int(c)
            selected_cols.append(c)
            selected_scores.append(float(sc))
            selected_valid.append(1)
            seen.add(c)

        if len(selected_cols) < K:
            fill_rows += 1
            need = K - len(selected_cols)

            filler: List[int] = []
            if fill_from_ref and ref_top_cols is not None:
                for c in ref_top_cols[i].tolist():
                    c = int(c)
                    if c not in seen:
                        filler.append(c)
                        seen.add(c)
                    if len(filler) >= need:
                        break

            # Last-resort deterministic filler from eval columns.
            if len(filler) < need:
                for c in range(n_go):
                    if c not in seen:
                        filler.append(c)
                        seen.add(c)
                    if len(filler) >= need:
                        break

            for c in filler[:need]:
                selected_cols.append(int(c))
                selected_scores.append(-1e6)
                selected_valid.append(0)

        cols_arr = np.asarray(selected_cols[:K], dtype=np.int32)
        scores_arr = np.asarray(selected_scores[:K], dtype=np.float32)
        valid_arr = np.asarray(selected_valid[:K], dtype=np.int8)

        true_set = set(int(x) for x in true_label_cols[i].tolist())
        labels_arr = np.asarray([1 if int(c) in true_set else 0 for c in cols_arr], dtype=np.int8)

        top_cols[i] = cols_arr
        top_scores[i] = scores_arr
        top_labels[i] = labels_arr
        top_valid[i] = valid_arr

    logging.info("[esm-knn] rows requiring filler: %d / %d", fill_rows, N)
    return top_cols, top_scores, top_labels, top_valid


def pad_true_ids_from_cols(label_cols: List[np.ndarray], eval_go_ids: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    true_ids: List[List[int]] = []
    max_len = max((int(x.size) for x in label_cols), default=0)
    padded = np.full((len(label_cols), max_len), -1, dtype=np.int64)

    for i, cols in enumerate(label_cols):
        gids = [int(eval_go_ids[int(c)]) for c in cols.tolist()]
        true_ids.append(gids)
        if gids:
            padded[i, : len(gids)] = np.asarray(gids, dtype=np.int64)

    return padded, true_ids


def copy_reference_embedding_files(ref_dump: Path, out_dir: Path) -> None:
    for name in ["eval_go_ids.npy", "go_z.float16.npy", "protein_z.float16.npy"]:
        src = ref_dump / name
        if not src.exists():
            raise FileNotFoundError(f"Missing reference file: {src}")
        shutil.copy2(src, out_dir / name)


def load_ref_ids(ref_dump: Path) -> List[str]:
    with (ref_dump / "protein_ids.json").open("r", encoding="utf-8") as f:
        return [str(x) for x in json.load(f)]


def validate_ref_alignment(ref_dump: Path, query_ids: List[str]) -> None:
    ref_ids = load_ref_ids(ref_dump)
    if len(ref_ids) != len(query_ids):
        raise RuntimeError(f"ref ids length {len(ref_ids)} != query ids length {len(query_ids)}")
    bad = [(i, ref_ids[i], query_ids[i]) for i in range(len(query_ids)) if ref_ids[i] != query_ids[i]]
    if bad:
        raise RuntimeError(f"Reference dump row order mismatch. First examples: {bad[:5]}")


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if overwrite:
            logging.warning("[esm-knn] removing existing output dir: %s", out_dir)
            shutil.rmtree(out_dir)
        elif any(out_dir.iterdir()):
            raise RuntimeError(f"Output dir exists and is not empty: {out_dir}. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser("Dump ESM-kNN GO candidates in reranker-compatible schema.")

    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ref_dump", type=str, required=True, help="P3a dump for the same split. Used for protein_z/go_z/eval ids and row order.")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--ids_path", type=str, default=None, help="Optional override for val_ids_path. Use this for test/custom split.")
    p.add_argument("--pid2pos_path", type=str, default=None)

    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--n_neighbors", type=int, default=500)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--query_batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_fp16_knn", action="store_true")
    p.add_argument("--no_ref_fill", action="store_true", help="Do not fill short kNN candidate lists from ref_dump top_go_cols.")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def main():
    cli = parse_args()
    setup_logging_simple()

    args = load_structured_cfg(cli.config)
    if cli.device is not None:
        args.general_device = cli.device
    if cli.ids_path is not None:
        args.val_ids_path = Path(cli.ids_path)
    if cli.pid2pos_path is not None:
        args.pid2pos = Path(cli.pid2pos_path)

    args.resume = None
    args.warmstart_path = None
    args.eval_only = True
    args.wandb = False

    set_seed(args.seed)

    device = torch.device(args.general_device if args.general_device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    logging.info("[esm-knn] device=%s", device)

    ref_dump = Path(cli.ref_dump)
    out_dir = Path(cli.out_dir)
    prepare_output_dir(out_dir, overwrite=bool(cli.overwrite))

    go_cache = build_go_cache(str(args.go_cache_path))
    dag_parents = load_go_parents() if args.use_dag_in_ds else None
    dag_children = load_go_children() if args.use_dag_in_ds else None

    go_encoder, go_text_store = build_go_encoder_and_text_store(args, device)
    logging.info("[esm-knn] materializing GO text tokens")
    go_text_store.materialize_tokens_once(batch_size=512, show_progress=True)

    aligned = canonicalize_and_align_inputs(args=args, go_cache=go_cache, logger=logging.getLogger("esm-knn"))
    eval_go_ids = np.asarray(aligned["eval_id_list"], dtype=np.int64)
    eval_id_to_col = {int(g): i for i, g in enumerate(eval_go_ids.tolist())}

    res_store = build_stores(args)
    datasets = build_datasets(
        args,
        res_store,
        go_text_store,
        dag_parents=dag_parents,
        pid2pos=aligned["pid2pos"],
        zs=aligned["zs"],
        fs=aligned["fs"],
    )

    batch_size = int(cli.batch_size or args.eval_batch_size or args.batch_size)

    train_loader = make_dump_loader(
        dataset=datasets["train"],
        args=args,
        go_text_store=go_text_store,
        batch_size=batch_size,
        num_workers=int(cli.num_workers),
    )

    if cli.split == "train":
        query_dataset = datasets["train"]
        split_name = "train"
    else:
        query_dataset = datasets["val"]
        split_name = "val" if cli.ids_path is None else "custom"

    query_loader = make_dump_loader(
        dataset=query_dataset,
        args=args,
        go_text_store=go_text_store,
        batch_size=batch_size,
        num_workers=int(cli.num_workers),
    )

    train_ids, train_z, train_label_cols = collect_mean_esm_embeddings(
        train_loader,
        device=device,
        eval_id_to_col=eval_id_to_col,
        desc="collect train ESM means",
    )

    query_ids, query_z, query_label_cols = collect_mean_esm_embeddings(
        query_loader,
        device=device,
        eval_id_to_col=eval_id_to_col,
        desc=f"collect {split_name} ESM means",
    )

    validate_ref_alignment(ref_dump, query_ids)

    nn_idx, nn_sim = compute_knn(
        train_z=train_z,
        query_z=query_z,
        train_ids=train_ids,
        query_ids=query_ids,
        n_neighbors=int(cli.n_neighbors),
        device=device,
        query_batch_size=int(cli.query_batch_size),
        use_fp16=not bool(cli.no_fp16_knn),
    )

    ref_top_cols = None
    if not bool(cli.no_ref_fill):
        ref_top_cols_path = ref_dump / "top_go_cols.int32.npy"
        if ref_top_cols_path.exists():
            ref_top_cols = np.load(ref_top_cols_path, mmap_mode="r")[:, : int(cli.topk)]
        else:
            logging.warning("[esm-knn] no ref top_go_cols found; disabling ref fill")

    top_cols, top_scores, top_labels, top_valid = build_knn_candidate_arrays(
        nn_idx=nn_idx,
        nn_sim=nn_sim,
        train_label_cols=train_label_cols,
        true_label_cols=query_label_cols,
        n_go=int(eval_go_ids.shape[0]),
        topk=int(cli.topk),
        tau=float(cli.tau),
        ref_top_cols=ref_top_cols,
        fill_from_ref=not bool(cli.no_ref_fill),
    )

    # Copy P3a embedding files so the existing reranker can consume this dump.
    copy_reference_embedding_files(ref_dump, out_dir)

    # Write candidate arrays.
    np.save(out_dir / "top_go_cols.int32.npy", top_cols.astype(np.int32))
    np.save(out_dir / "top_scores.float32.npy", top_scores.astype(np.float32))
    np.save(out_dir / "top_labels.int8.npy", top_labels.astype(np.int8))
    np.save(out_dir / "top_valid.int8.npy", top_valid.astype(np.int8))

    true_padded, true_json = pad_true_ids_from_cols(query_label_cols, eval_go_ids)
    np.save(out_dir / "true_go_ids.npy", true_padded)

    with (out_dir / "protein_ids.json").open("w", encoding="utf-8") as f:
        json.dump(query_ids, f)
    with (out_dir / "true_go_ids.json").open("w", encoding="utf-8") as f:
        json.dump(true_json, f)

    metadata = {
        "status": "complete",
        "source": "esm_knn",
        "split": split_name,
        "config": str(cli.config),
        "ref_dump": str(ref_dump),
        "n_samples": int(len(query_ids)),
        "n_train": int(len(train_ids)),
        "n_go": int(eval_go_ids.shape[0]),
        "topk": int(cli.topk),
        "n_neighbors": int(cli.n_neighbors),
        "tau": float(cli.tau),
        "fill_from_ref": not bool(cli.no_ref_fill),
        "files": {
            "protein_ids": "protein_ids.json",
            "true_go_ids_json": "true_go_ids.json",
            "true_go_ids_padded": "true_go_ids.npy",
            "eval_go_ids": "eval_go_ids.npy",
            "go_z": "go_z.float16.npy",
            "protein_z": "protein_z.float16.npy",
            "top_go_cols": "top_go_cols.int32.npy",
            "top_scores": "top_scores.float32.npy",
            "top_labels": "top_labels.int8.npy",
            "top_valid": "top_valid.int8.npy",
        },
        "schema_note": "top_go_cols are ESM-kNN transferred-label candidates. protein_z/go_z are copied from ref_dump for reranker compatibility.",
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (out_dir / "DONE").open("w", encoding="utf-8") as f:
        f.write("complete\n")

    logging.info("[esm-knn] done: %s", out_dir)


if __name__ == "__main__":
    main()
