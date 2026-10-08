"""ASL training with StarGO-compatible checkpoint selection."""
import logging
from pathlib import Path
import numpy as np
import torch
from src.loss.asymmetric_loss import AsymmetricLoss, AsymmetricLossConfig

LOG = logging.getLogger('moe')


@torch.no_grad()
def evaluate(model, loader, evaluator, device):
    model.eval();
    ys = [];
    ps = [];
    weights = []
    for s, d, e, n, y in loader:
        out = model(s.to(device), d.to(device), e.to(device), n.to(device), return_details=True)
        ps.append(out['probability'].cpu().numpy());
        ys.append(y.numpy())
        weights.append(float(out['retriever_weight'].mean().item()))
    metrics = evaluator.evaluate(np.concatenate(ys), np.concatenate(ps))
    metrics['mean_retriever_gate'] = float(np.mean(weights))
    return metrics


def save(path, model, optimizer, epoch, best, best_epoch, bad, step, go_ids):
    tmp = Path(str(path) + '.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'meta': {'epoch': epoch, 'best_fmax': best, 'best_epoch': best_epoch, 'bad_epochs': bad, 'step': step, 'go_ids': go_ids.tolist()}}, tmp)
    tmp.replace(path)


def train(model, train_loader, val_loader, evaluator, args, go_ids):
    device = torch.device(args.device);
    model.to(device)
    criterion = AsymmetricLoss(AsymmetricLossConfig(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg, clip=args.asl_clip))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = 0;
    best = -1.;
    best_epoch = -1;
    bad = 0;
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ck['meta']['go_ids'] != go_ids.tolist(): raise ValueError('Resume GO universe mismatch')
        model.load_state_dict(ck['model'], strict=True);
        optimizer.load_state_dict(ck['optimizer'])
        m = ck['meta'];
        start = m['epoch'] + 1;
        best = m['best_fmax'];
        best_epoch = m['best_epoch'];
        bad = m['bad_epochs'];
        step = m['step']
        LOG.info('Resume epoch=%d step=%d best=%.5f', start, step, best)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        for epoch in range(start, args.epochs):
            model.train();
            losses = []
            for s, d, e, n, y in train_loader:
                step += 1;
                optimizer.zero_grad(set_to_none=True)
                logits = model(s.to(device), d.to(device), e.to(device), n.to(device))
                loss = criterion(logits, y.to(device))
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(gn): raise RuntimeError('Nonfinite gradient')
                optimizer.step();
                losses.append(float(loss.item()))
                if step % args.log_every == 0: LOG.info('[train] epoch=%d step=%d loss=%.6f', epoch, step, float(np.mean(losses[-args.log_every:])))
            metrics = evaluate(model, val_loader, evaluator, device)
            LOG.info('[val] epoch=%d Fmax=%.5f @ %.2f macroAUPR=%.5f microAUPR=%.5f AUC=%.5f gate=%.4f', epoch, metrics['protein_fmax'],
                     metrics['protein_fmax_threshold'], metrics['macro_aupr'], metrics['micro_aupr'], metrics['auc'], metrics['mean_retriever_gate'])
            if run: run.log({'epoch': epoch, 'step': step, 'train/loss': float(np.mean(losses)), **{f'val/{k}': v for k, v in metrics.items()}})
            if metrics['protein_fmax'] > best:
                best = metrics['protein_fmax'];
                best_epoch = epoch;
                bad = 0
                save(args.output_dir / 'checkpoint_best.pt', model, optimizer, epoch, best, best_epoch, bad, step, go_ids)
            else:
                bad += 1
            save(args.output_dir / 'checkpoint_last.pt', model, optimizer, epoch, best, best_epoch, bad, step, go_ids)
            if args.patience and bad >= args.patience:
                LOG.info('Early stop: best epoch=%d Fmax=%.5f', best_epoch, best);
                break
        LOG.info('DONE best_epoch=%d best_fmax=%.6f', best_epoch, best)
    finally:
        if run: run.finish()
