from __future__ import annotations
import argparse
import logging
import random
import sys
from pathlib import Path
import numpy as np
import torch
from src.experiment_c_runtime import build, load_retriever, load_head, live
from src.models.rerankerv2_model import PredictionHeadConfig, ProteinGOPredictionHead
from src.loss.asymmetric_loss import AsymmetricLoss, AsymmetricLossConfig
from src.evaluation.stargo_pfresgo_metrics import StarGOPFresGOEvaluator

LOG = logging.getLogger("experiment_c")


def parse_args():
    p = argparse.ArgumentParser("Experiment C full training")
    for k in ("retriever_config", "retriever_checkpoint", "head_checkpoint", "go_graph_path", "output_dir"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--ontology", choices=["bp", "mf", "cc"], default="bp")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--adapter_rank", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--gamma_pos", type=float, default=0.)
    p.add_argument("--gamma_neg", type=float, default=4.)
    p.add_argument("--asl_clip", type=float, default=.05)
    p.add_argument("--retriever_lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.)
    p.add_argument("--grad_clip", type=float, default=1.)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--baseline_fmax", type=float, default=.4206411403909932)
    p.add_argument("--parity_tolerance", type=float, default=.015)
    p.add_argument("--resume", type=Path)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="protein-go-align-pfresgo")
    p.add_argument("--wandb_run_name", default="ExperimentC-BP-EndToEnd")
    return p.parse_args()


@torch.no_grad()
def validate(r, head, criterion, loader, evaluator):
    r.model.eval();
    r.model.go_encoder.eval();
    head.eval()
    ys = [];
    ps = [];
    loss_total = 0.;
    n = 0
    hit = {k: 0. for k in (50, 100, 200, 500, 1000)}
    for b in loader:
        zp = r._encode_protein(b)
        zg, _ = r._project_go_bank()
        scores = zp @ zg.T
        logits = head(protein_z=zp, go_z=zg, retriever_scores=scores)
        y = r._positive_mask(b).float()
        bs = y.size(0);
        n += bs
        loss_total += float(criterion(logits, y).item()) * bs
        ys.append(y.cpu().numpy().astype(np.int8))
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        true_count = y.sum(1).clamp_min(1)
        for k in hit:
            idx = scores.topk(min(k, scores.size(1)), dim=1).indices
            hit[k] += float((y.gather(1, idx).sum(1) / true_count).sum().item())
    metrics = evaluator.evaluate(np.concatenate(ys), np.concatenate(ps))
    metrics["asl_loss"] = loss_total / n
    metrics.update({f"retrieval_R@{k}": v / n for k, v in hit.items()})
    r.model.train();
    r.model.go_encoder.eval();
    head.train()
    return metrics


def save(path, r, head, opt, epoch, step, best, best_epoch, bad):
    tmp = path.with_suffix(".tmp")
    torch.save({"retriever": r.model.state_dict(), "head": head.state_dict(),
                "optimizer": opt.state_dict(),
                "meta": {"epoch": epoch, "global_step": step, "best_fmax": best,
                         "best_epoch": best_epoch, "bad_epochs": bad}}, tmp)
    tmp.replace(path)
    LOG.info("[checkpoint] Saved -> %s", path)


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(a.output_dir / "train.log", encoding="utf-8")],
                        force=True)
    random.seed(a.seed);
    np.random.seed(a.seed);
    torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device(a.device)
    LOG.info("Device: %s", device)
    c, r, tr, va = build(a, device)
    if len(tr.dataset) != 23522 or len(va.dataset) != 3416 or len(r.eval_id_list) != 1943:
        raise RuntimeError(f"Unexpected split sizes train={len(tr.dataset)} val={len(va.dataset)} G={len(r.eval_id_list)}")
    load_retriever(r.model, a.retriever_checkpoint)
    for p in r.model.go_encoder.parameters(): p.requires_grad_(False)
    r.model.go_encoder.eval()
    head = ProteinGOPredictionHead(PredictionHeadConfig(
        dim=int(c.align_dim), adapter_rank=a.adapter_rank,
        hidden_dim=a.hidden_dim, dropout=a.dropout)).to(device)
    load_head(head, a.head_checkpoint)
    criterion = AsymmetricLoss(AsymmetricLossConfig(
        gamma_pos=a.gamma_pos, gamma_neg=a.gamma_neg, clip=a.asl_clip)).to(device)
    rp = [p for p in r.model.parameters() if p.requires_grad]
    hp = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": rp, "lr": a.retriever_lr},
        {"params": hp, "lr": a.head_lr}], weight_decay=a.weight_decay)
    evaluator = StarGOPFresGOEvaluator(goterms=r.eval_id_list,
                                       ontology=a.ontology, go_graph_path=a.go_graph_path)
    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project=a.wandb_project, name=a.wandb_run_name,
                         config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
        run.define_metric("trainer_step")
        run.define_metric("*", step_metric="trainer_step")
    try:
        start = 0;
        step = 0;
        best = -1.;
        best_epoch = -1;
        bad = 0
        if a.resume:
            ck = torch.load(a.resume, map_location=device, weights_only=False)
            r.model.load_state_dict(ck["retriever"], strict=True)
            head.load_state_dict(ck["head"], strict=True)
            opt.load_state_dict(ck["optimizer"])
            meta = ck["meta"];
            start = int(meta["epoch"]) + 1
            step = int(meta["global_step"]);
            best = float(meta["best_fmax"])
            best_epoch = int(meta["best_epoch"]);
            bad = int(meta.get("bad_epochs", 0))
            LOG.info("[resume] epoch=%d step=%d best=%.4f", start, step, best)
        else:
            LOG.info("[parity] Evaluating BEFORE any optimizer step")
            m = validate(r, head, criterion, va, evaluator)
            LOG.info("[parity] Fmax=%.6f | B baseline=%.6f | delta=%+.6f",
                     m["protein_fmax"], a.baseline_fmax, m["protein_fmax"] - a.baseline_fmax)
            if run: run.log({"trainer_step": 0, **{f"val_init/{k}": v for k, v in m.items()}}, step=0)
            if abs(m["protein_fmax"] - a.baseline_fmax) > a.parity_tolerance:
                raise RuntimeError(
                    "Initial B/C Fmax parity FAILED. Training blocked. "
                    "Check GO order, score scaling and frozen encoder parity."
                )
            best = m["protein_fmax"];
            best_epoch = -1
            LOG.info("[parity] PASSED; starting joint fine-tuning")
        for epoch in range(start, a.epochs):
            r.model.train();
            r.model.go_encoder.eval();
            head.train()
            loss_sum = 0.;
            nb = 0
            for b in tr:
                step += 1;
                nb += 1
                opt.zero_grad(set_to_none=True)
                logits, y = live(r, head, b)
                loss, diag = criterion(logits, y, return_diagnostics=True)
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite ASL")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(rp + hp, a.grad_clip)
                if not torch.isfinite(norm): raise RuntimeError("Nonfinite gradient")
                opt.step();
                loss_sum += float(loss.item())
                if step % a.log_every == 0:
                    LOG.info("[train] epoch=%d step=%d loss=%.6f gap=%.4f",
                             epoch, step, diag["loss"], diag["probability_gap"])
                    if run: run.log({"trainer_step": step, "train/loss": diag["loss"],
                                     "train/gap": diag["probability_gap"]}, step=step)
            m = validate(r, head, criterion, va, evaluator)
            LOG.info("[val] epoch=%d Fmax=%.4f threshold=%.2f macroAUPR=%.4f microAUPR=%.4f AUC=%.4f R@50=%.4f R@200=%.4f R@500=%.4f",
                     epoch, m["protein_fmax"], m["protein_fmax_threshold"],
                     m["macro_aupr"], m["micro_aupr"], m["auc"],
                     m["retrieval_R@50"], m["retrieval_R@200"], m["retrieval_R@500"])
            if run: run.log({"trainer_step": step, "epoch": epoch, **{f"val/{k}": v for k, v in m.items()}}, step=step)
            if m["protein_fmax"] > best:
                best = m["protein_fmax"];
                best_epoch = epoch;
                bad = 0
                save(a.output_dir / "checkpoint_best.pt", r, head, opt, epoch, step, best, best_epoch, bad)
            else:
                bad += 1
            save(a.output_dir / "checkpoint_last.pt", r, head, opt, epoch, step, best, best_epoch, bad)
            if a.patience > 0 and bad >= a.patience:
                LOG.info("[early-stop] patience reached");
                break
        LOG.info("[done] best_epoch=%d best_fmax=%.6f B=%.6f", best_epoch, best, a.baseline_fmax)
        if run:
            run.summary["best/protein_fmax"] = best
            run.summary["best/epoch"] = best_epoch
    finally:
        if run: run.finish()


if __name__ == "__main__":
    main()
