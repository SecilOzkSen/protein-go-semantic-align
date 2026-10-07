from __future__ import annotations
import argparse, dataclasses, json, logging, shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import src.main as base
from src.configs.data_classes import TrainerConfig, TrainingContext, LoggingConfig
from src.encoders import BioMedBERTEncoder
from src.training.collate import ContrastiveEmbCollator
from src.training.trainer import OppTrainer
from src.utils import load_raw_json
from src.utils.helpers import build_altid_map_from_go_terms, canonicalize_id_list, canonicalize_pid2pos


def _unwrap(x):
    for _ in range(10):
        if isinstance(x, dict) and sum(torch.is_tensor(v) for v in x.values()) >= 5:
            return x
        for k in ("model", "state_dict", "model_state_dict", "module", "net"):
            if isinstance(x, dict) and isinstance(x.get(k), dict):
                x = x[k];
                break
        else:
            raise RuntimeError("Cannot unwrap checkpoint model state")
    raise RuntimeError("Checkpoint nesting too deep")


def load_exact(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError("Checkpoint must contain 'model'")
    raw, state = _unwrap(ckpt["model"]), {}
    for k, v in raw.items():
        if not torch.is_tensor(v): continue
        changed = True
        while changed:
            changed = False
            for p in ("trainer.model.", "module.", "model."):
                if k.startswith(p):
                    k = k[len(p):];
                    changed = True
        state[k] = v
    cur = model.state_dict()
    missing = [k for k in cur if k not in state]
    unexpected = [k for k in state if k not in cur]
    shape = [(k, tuple(state[k].shape), tuple(cur[k].shape))
             for k in state if k in cur and state[k].shape != cur[k].shape]
    if missing or unexpected or shape:
        raise RuntimeError(f"Checkpoint mismatch\nmissing={missing[:20]}\nunexpected={unexpected[:20]}\nshape={shape[:10]}")
    model.load_state_dict(state, strict=True)
    print(f"[dump] exact checkpoint load OK: {len(state)} tensors")
    return ckpt.get("meta", {})


def make_cfg(args, d_h, d_g, d_z, device):
    vals = dict(
        d_h=d_h, d_g=d_g, d_z=d_z, device=str(device),
        lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        max_epochs=args.epochs, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, fp16=args.fp16,
        warmstart_path=None, monitor_metric=args.monitor_metric,
        secondary_monitor_metric=args.secondary_monitor_metric,
        monitor_mode=args.monitor_mode,
        protein_pooling_strategy=args.protein_pooling_strategy,
        use_lora=False, go_pooling=args.go_pooling, go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,
        go_segment_representation_mode=args.go_segment_representation_mode,
        eval_go_bs=args.eval_go_bs, temperature=args.temperature,
        lambda_con=args.lambda_con, pbr_lambda=args.pbr_lambda,
        pbr_margin=args.pbr_margin, pbr_tau=args.pbr_tau,
        retrieval_eval_ks=args.retrieval_eval_ks,
        positive_rank_quantiles=args.positive_rank_quantiles,
        log_gradient_norms=False, log_positive_rank_cdf=False,
        log_cardinality_metrics=False, log_go_segment_weights=True)
    allowed = {f.name for f in dataclasses.fields(TrainerConfig)}
    return TrainerConfig(**{k: v for k, v in vals.items() if k in allowed})


def build_runtime(args, device):
    branch = base._read_ids(args.branch_go_ids_path)
    seen, rare, zero, benchmark = base._training_buckets(
        args.train_ids_path, args.pid2pos, branch, args.rare_lt)
    active = benchmark if args.evaluation_space == "benchmark" else branch
    cache = base.build_go_cache(str(args.go_cache_path))

    terms = load_raw_json(args.go_basic_json)
    alt = build_altid_map_from_go_terms(terms) if terms else {}
    raw = load_raw_json(args.pid2pos)
    pid2pos = canonicalize_pid2pos(raw, alt) if alt else raw
    active = canonicalize_id_list(active, alt) if alt else list(map(int, active))
    seen = canonicalize_id_list(seen, alt) if alt else seen
    rare = canonicalize_id_list(rare, alt) if alt else rare
    zero = canonicalize_id_list(zero, alt) if alt else zero
    pid2pos, active, seen, zero, rare = base.align_to_cache(
        cache, pid2pos, active, seen, zero, rare)

    dag, _ = base._build_dag(args.go_basic_json) if args.use_dag_in_ds else (None, None)
    enc = BioMedBERTEncoder(
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        device=device, max_length=512,
        attention_pooling_strategy=args.go_encoder_inner_pooling,
        attn_hidden=128, attn_dropout=0.1, special_token_weights=None,
        enable_lora=False, lora_parameters=None, use_special_tokens=False)
    for p in enc.parameters(): p.requires_grad_(False)
    enc.eval()

    gstore = base.build_go_text_store(args, enc)
    gstore.materialize_tokens_once(batch_size=512, show_progress=True)
    rstore = base.build_residue_store(args)
    ds = base.build_datasets(args, rstore, gstore, pid2pos, dag, zero, rare)
    base.validate_active_universe(pid2pos, active, ds)

    sample = ds["train"][0]
    ctx = TrainingContext(
        go_cache=cache, device=device, go_text_store=gstore,
        run_name="retriever-v2-fullgo-dump", fp16_enabled=args.fp16,
        protein_pooling_strategy=args.protein_pooling_strategy,
        go_pool_type=args.go_pool_type,
        go_encoder_output_mode=args.go_encoder_output_mode,
        go_segment_representation_mode=args.go_segment_representation_mode,
        eval_id_list=list(active), eval_seen_go_ids=list(seen),
        eval_unseen_ids=list(zero), eval_rare_go_ids=list(rare),
        logger=logging.getLogger("dump"),
        logging=LoggingConfig(log_every=max(1, args.log_every)))
    cfg = make_cfg(args, int(sample["prot_emb"].shape[1]),
                   int(cache.embs.shape[1]), int(args.align_dim), device)
    return OppTrainer(cfg=cfg, ctx=ctx, go_encoder=enc, wandb_run=None), ds


def make_loader(ds, bs, nw):
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                      persistent_workers=nw > 0, pin_memory=True,
                      collate_fn=ContrastiveEmbCollator(), drop_last=False)


@torch.no_grad()
def dump_fullgo(trainer, dl, out, checkpoint, config, split, overwrite, score_fp32):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite: raise RuntimeError(f"{out} non-empty; use --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    trainer.model.eval();
    trainer.model.go_encoder.eval()
    for p in trainer.model.parameters(): p.requires_grad_(False)

    # Exact Retriever-v2 GO path: segment bank -> learned gate -> go_ln -> proj_g -> L2.
    zg, segw = trainer._project_go_bank()
    zg = zg.detach().float()
    gids = np.asarray(trainer.eval_id_list, dtype=np.int64)
    N, G, D = len(dl.dataset), len(gids), int(zg.shape[1])

    np.save(out / "go_z.float16.npy", zg.cpu().half().numpy())
    np.save(out / "go_segment_weights.float32.npy",
            segw.detach().float().cpu().numpy().astype(np.float32))
    np.save(out / "eval_go_ids.int64.npy", gids)

    zp_mm = np.lib.format.open_memmap(
        out / "protein_z.float16.npy", "w+", dtype=np.float16, shape=(N, D))
    sdtype = np.float32 if score_fp32 else np.float16
    sname = "retriever_scores.float32.npy" if score_fp32 else "retriever_scores.float16.npy"
    sc_mm = np.lib.format.open_memmap(out / sname, "w+", dtype=sdtype, shape=(N, G))
    y_mm = np.lib.format.open_memmap(
        out / "labels.int8.npy", "w+", dtype=np.int8, shape=(N, G))

    pids, true_ids, off = [], [], 0
    zg_dev = zg.to(trainer.device)
    for batch in tqdm(dl, desc=f"dump {split}"):
        # Exact Retriever-v2 protein path, including normalization.
        zp = trainer._encode_protein(batch).detach().float()
        y = trainer._positive_mask(batch).detach()
        scores = zp @ zg_dev.T
        if not torch.isfinite(scores).all(): raise RuntimeError("non-finite scores")

        B = zp.shape[0];
        a, b = off, off + B
        zp_mm[a:b] = zp.cpu().half().numpy()
        sc_mm[a:b] = scores.cpu().numpy().astype(sdtype, copy=False)
        y_mm[a:b] = y.cpu().numpy().astype(np.int8, copy=False)

        bpids = batch.get("protein_ids") or [f"{split}_{i}" for i in range(a, b)]
        pids.extend(map(str, bpids))
        for row in y.cpu():
            cols = torch.nonzero(row, as_tuple=False).flatten().numpy()
            true_ids.append(gids[cols].tolist())
        off = b

    zp_mm.flush();
    sc_mm.flush();
    y_mm.flush()
    if off != N: raise RuntimeError(f"wrote {off}, expected {N}")
    (out / "protein_ids.json").write_text(json.dumps(pids))
    (out / "true_go_ids.json").write_text(json.dumps(true_ids))

    # Invariant for Experiment B: score must be reconstructible from dumped z.
    n = min(32, N)
    recon = torch.from_numpy(np.asarray(zp_mm[:n], np.float32)) @ zg.cpu().T
    saved = torch.from_numpy(np.asarray(sc_mm[:n], np.float32))
    err = float((recon - saved).abs().max())
    tol = 1e-5 if score_fp32 else 5e-3
    if err > tol: raise RuntimeError(f"score reconstruction err={err} > {tol}")

    meta = dict(
        schema_version="retriever_v2_fullgo_experiment_b_v1",
        split=split, checkpoint=str(checkpoint), config=str(config),
        n_samples=N, n_go=G, d_z=D,
        segments=list(getattr(trainer, "_segment_names", [])),
        protein_mode="single_vector", go_mode="segments_only",
        full_go=True, topk=None, candidate_truncation=False, rank_feature=False,
        score_definition="dot product of L2-normalized Retriever-v2 z_p and z_g",
        score_reconstruction_max_abs_err=err,
        files=dict(
            protein_z="protein_z.float16.npy", go_z="go_z.float16.npy",
            scores=sname, labels="labels.int8.npy",
            go_segment_weights="go_segment_weights.float32.npy",
            eval_go_ids="eval_go_ids.int64.npy",
            protein_ids="protein_ids.json", true_go_ids="true_go_ids.json"))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out / "DONE").write_text("complete\n")
    print(f"[dump] DONE {out} | N={N} G={G} D={D} reconstruction_err={err:.3g}")


def main():
    ap = argparse.ArgumentParser("Retriever v2 full-GO dump for Experiment B")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--ids_path", default=None,
                    help="Override val IDs. For test use --split val --ids_path test.txt")
    ap.add_argument("--pid2pos_path", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--score_fp32", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    args = base.load_structured_cfg(cli.config)
    if cli.device: args.general_device = cli.device
    if cli.ids_path: args.val_ids_path = Path(cli.ids_path).expanduser().resolve()
    if cli.pid2pos_path: args.pid2pos = Path(cli.pid2pos_path).expanduser().resolve()
    args.resume = None;
    args.warmstart_path = None;
    args.wandb = False

    base.set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else args.general_device)
    trainer, datasets = build_runtime(args, device)
    print("[dump] checkpoint meta:", load_exact(trainer.model, cli.checkpoint, device))

    ds = datasets["train"] if cli.split == "train" else datasets["val"]
    split_name = "train" if cli.split == "train" else ("custom" if cli.ids_path else "val")
    dl = make_loader(ds, cli.batch_size or args.eval_batch_size, cli.num_workers)
    dump_fullgo(trainer, dl, cli.out_dir, cli.checkpoint, cli.config,
                split_name, cli.overwrite, cli.score_fp32)


if __name__ == "__main__":
    main()
