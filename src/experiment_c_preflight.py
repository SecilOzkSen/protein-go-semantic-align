from __future__ import annotations
import argparse, logging, random, sys
from pathlib import Path
import numpy as np
import torch

from src.main import (_read_ids, _training_buckets, align_to_cache, build_dataloaders,
                      build_datasets, build_go_cache, build_go_text_store, build_residue_store,
                      load_structured_cfg, validate_active_universe)
from src.configs.data_classes import LoggingConfig, TrainerConfig, TrainingContext
from src.encoders import BioMedBERTEncoder
from src.loss.asymmetric_loss import AsymmetricLoss, AsymmetricLossConfig
from src.models.rerankerv2_model import PredictionHeadConfig, ProteinGOPredictionHead
from src.training.trainer import OppTrainer
from src.utils import load_raw_json
from src.utils.helpers import build_altid_map_from_go_terms, canonicalize_id_list, canonicalize_pid2pos

LOG = logging.getLogger('experiment_c_preflight')


def args():
    p = argparse.ArgumentParser('Experiment C gradient preflight')
    p.add_argument('--retriever_config', type=Path, required=True)
    p.add_argument('--retriever_checkpoint', type=Path, required=True)
    p.add_argument('--head_checkpoint', type=Path, required=True)
    p.add_argument('--device', default='cuda:0');
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=4);
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--adapter_rank', type=int, default=32);
    p.add_argument('--hidden_dim', type=int, default=256);
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--gamma_pos', type=float, default=0.);
    p.add_argument('--gamma_neg', type=float, default=4.);
    p.add_argument('--asl_clip', type=float, default=.05)
    p.add_argument('--retriever_lr', type=float, default=1e-5);
    p.add_argument('--head_lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.);
    p.add_argument('--grad_clip', type=float, default=1.)
    p.add_argument('--steps', type=int, default=5)
    return p.parse_args()


def setup():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s', handlers=[logging.StreamHandler(sys.stdout)],
                        force=True)


def seed(s):
    random.seed(s);
    np.random.seed(s);
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def clean(k):
    changed = True
    while changed:
        changed = False
        for pref in ('trainer.model.', 'module.', 'model.'):
            if k.startswith(pref): k = k[len(pref):]; changed = True
    return k


def load_retriever(model, path):
    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(ck, dict) or "model" not in ck:
        raise RuntimeError(
            f"Invalid Retriever checkpoint. Keys: "
            f"{list(ck.keys()) if isinstance(ck, dict) else type(ck)}"
        )

    raw = ck["model"]

    state = {
        clean_key(k): v
        for k, v in raw.items()
        if torch.is_tensor(v)
    }

    current = model.state_dict()

    # These are the Experiment-A alignment blocks that MUST
    # come from the trained Retriever checkpoint.
    required_prefixes = (
        "protein_mean_attn_gate_pool.",
        "protein_ln.",
        "proj_p.",
        "go_ln.",
        "proj_g.",
        "go_segment_gate.",
    )

    required_current = {
        k: v
        for k, v in current.items()
        if k.startswith(required_prefixes)
    }

    missing = [
        k
        for k in required_current
        if k not in state
    ]

    shape_bad = [
        (
            k,
            tuple(state[k].shape),
            tuple(required_current[k].shape),
        )
        for k in required_current
        if k in state
           and tuple(state[k].shape)
           != tuple(required_current[k].shape)
    ]

    if missing:
        raise RuntimeError(
            "[Experiment C] Retriever checkpoint is missing "
            f"trained alignment tensors: {missing[:20]}"
        )

    if shape_bad:
        raise RuntimeError(
            "[Experiment C] Retriever checkpoint has incompatible "
            f"alignment tensors: {shape_bad[:10]}"
        )

    loadable = {
        k: state[k]
        for k in required_current
    }

    missing_after, unexpected = model.load_state_dict(
        loadable,
        strict=False,
    )

    # Missing go_encoder.* is EXPECTED.
    bad_missing = [
        k
        for k in missing_after
        if not k.startswith("go_encoder.")
    ]

    if bad_missing:
        raise RuntimeError(
            "[Experiment C] Unexpected missing Retriever tensors "
            f"after load: {bad_missing[:20]}"
        )

    if unexpected:
        raise RuntimeError(
            "[Experiment C] Unexpected Retriever checkpoint "
            f"tensors: {unexpected[:20]}"
        )

    print(
        f"[Experiment C] Retriever alignment load OK: "
        f"{len(loadable)} tensors"
    )

    for pref in required_prefixes:
        n = sum(
            k.startswith(pref)
            for k in loadable
        )
        print(
            f"[Experiment C][CHECK] {pref} loaded={n}"
        )

    meta = ck.get("meta", {})

    print(
        "[Experiment C] Retriever checkpoint meta: "
        f"epoch={meta.get('epoch')} "
        f"step={meta.get('global_step', meta.get('step'))}"
    )

    print(
        "[Experiment C] go_encoder.* intentionally NOT loaded "
        "from Retriever checkpoint; pretrained BioMedBERT remains frozen."
    )


def load_head(head, path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' not in ck: raise RuntimeError("B checkpoint has no 'model' key")
    head.load_state_dict(ck['model'], strict=True);
    m = ck.get('meta', {})
    LOG.info('[init] Head B exact load OK | epoch=%s best_fmax=%s', m.get('epoch'), m.get('best_fmax'))


def build(a, device):
    c = load_structured_cfg(str(a.retriever_config));
    c.general_device = str(device);
    c.cpu = device.type == 'cpu';
    c.batch_size = a.batch_size;
    c.eval_batch_size = a.batch_size;
    c.num_workers = a.num_workers
    branch = _read_ids(c.branch_go_ids_path);
    seen, rare, zero, bench = _training_buckets(c.train_ids_path, c.pid2pos, branch, c.rare_lt)
    active = bench if c.evaluation_space == 'benchmark' else branch
    cache = build_go_cache(str(c.go_cache_path));
    terms = load_raw_json(c.go_basic_json);
    alt = build_altid_map_from_go_terms(terms) if terms else {}
    raw = load_raw_json(c.pid2pos);
    pid2pos = canonicalize_pid2pos(raw, alt) if alt else raw
    if alt:
        active = canonicalize_id_list(active, alt);
        seen = canonicalize_id_list(seen, alt);
        rare = canonicalize_id_list(rare, alt);
        zero = canonicalize_id_list(zero, alt)
    else:
        active = list(map(int, active))
    pid2pos, active, seen, zero, rare = align_to_cache(cache, pid2pos, active, seen, zero, rare)
    LOG.info('[runtime] active GO=%d', len(active))
    enc = BioMedBERTEncoder(model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', device=device, max_length=512,
                            attention_pooling_strategy=c.go_encoder_inner_pooling, attn_hidden=128, attn_dropout=.1, gradient_checkpointing=False)
    for p in enc.parameters(): p.requires_grad_(False)
    enc.eval();
    text = build_go_text_store(c, enc);
    text.materialize_tokens_once(batch_size=512, show_progress=True)
    residues = build_residue_store(c);
    ds = build_datasets(c, residues, text, pid2pos, None, zero, rare);
    validate_active_universe(pid2pos, active, ds);
    tr, _ = build_dataloaders(ds, c)
    sample = ds['train'][0]
    ctx = TrainingContext(go_cache=cache, device=device, go_text_store=text, run_name='ExperimentC-Preflight', fp16_enabled=c.fp16,
                          protein_pooling_strategy=c.protein_pooling_strategy, go_pool_type=c.go_pool_type, go_encoder_output_mode=c.go_encoder_output_mode,
                          go_segment_representation_mode=c.go_segment_representation_mode, eval_id_list=list(active), eval_seen_go_ids=list(seen),
                          eval_unseen_ids=list(zero), eval_rare_go_ids=list(rare), logger=LOG, logging=LoggingConfig(log_every=500))
    tc = TrainerConfig(d_h=int(sample['prot_emb'].shape[1]), d_g=int(cache.embs.shape[1]), d_z=int(c.align_dim), device=str(device), lr=a.retriever_lr,
                       weight_decay=a.weight_decay, grad_clip=a.grad_clip, max_epochs=1, batch_size=a.batch_size, eval_batch_size=a.batch_size, fp16=c.fp16,
                       warmstart_path=None, monitor_metric='protein_fmax', secondary_monitor_metric='macro_aupr', monitor_mode='max',
                       protein_pooling_strategy=c.protein_pooling_strategy, use_lora=False, go_pooling=c.go_pooling, go_pool_type=c.go_pool_type,
                       go_encoder_output_mode=c.go_encoder_output_mode, go_segment_representation_mode=c.go_segment_representation_mode,
                       eval_go_bs=c.eval_go_bs, temperature=c.temperature, lambda_con=0., pbr_lambda=0., pbr_margin=c.pbr_margin, pbr_tau=c.pbr_tau,
                       retrieval_eval_ks=c.retrieval_eval_ks, positive_rank_quantiles=c.positive_rank_quantiles, log_gradient_norms=True,
                       log_positive_rank_cdf=False, log_cardinality_metrics=False, log_go_segment_weights=True)
    r = OppTrainer(tc, ctx, enc, wandb_run=None);
    r.opt = None
    return c, r, tr


def live(r, head, b):
    zp = r._encode_protein(b);
    zg, _ = r._project_go_bank();
    s = zp @ zg.T  # NO DETACH
    logits = head(protein_z=zp, go_z=zg, retriever_scores=s);
    y = r._positive_mask(b).float()
    if logits.shape != y.shape: raise RuntimeError(f'shape mismatch {logits.shape} vs {y.shape}')
    return logits, y


def gnorm(model, prefixes):
    x = 0.
    for n, p in model.named_parameters():
        if n.startswith(prefixes) and p.grad is not None: x += float(p.grad.detach().float().pow(2).sum())
    return x ** .5


def main():
    setup();
    a = args();
    seed(a.seed);
    device = torch.device(a.device);
    LOG.info('Device: %s', device)
    c, r, tr = build(a, device);
    load_retriever(r.model, a.retriever_checkpoint)
    for p in r.model.go_encoder.parameters(): p.requires_grad_(False)
    r.model.go_encoder.eval()
    bad = [n for n, p in r.model.go_encoder.named_parameters() if p.requires_grad]
    if bad: raise RuntimeError(f'BioMedBERT trainable: {bad[:10]}')
    head = ProteinGOPredictionHead(PredictionHeadConfig(dim=int(c.align_dim), adapter_rank=a.adapter_rank, hidden_dim=a.hidden_dim, dropout=a.dropout)).to(
        device);
    load_head(head, a.head_checkpoint)
    lossfn = AsymmetricLoss(AsymmetricLossConfig(gamma_pos=a.gamma_pos, gamma_neg=a.gamma_neg, clip=a.asl_clip)).to(device)
    rp = [p for p in r.model.parameters() if p.requires_grad];
    hp = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': rp, 'lr': a.retriever_lr}, {'params': hp, 'lr': a.head_lr}], weight_decay=a.weight_decay)
    LOG.info('[optimizer] retriever LR=%.2e | head LR=%.2e', a.retriever_lr, a.head_lr)
    r.model.train();
    r.model.go_encoder.eval();
    head.train();
    seen = {k: False for k in ('pool', 'proj_p', 'go_gate', 'proj_g', 'head')}
    for i, b in enumerate(tr, start=1):
        if i > a.steps: break
        opt.zero_grad(set_to_none=True);
        logits, y = live(r, head, b);
        loss = lossfn(logits, y);
        loss.backward()
        gs = {'pool': gnorm(r.model, ('protein_mean_attn_gate_pool.',)), 'proj_p': gnorm(r.model, ('protein_ln.', 'proj_p.')),
              'go_gate': gnorm(r.model, ('go_segment_gate.',)), 'proj_g': gnorm(r.model, ('go_ln.', 'proj_g.')),
              'head': gnorm(head, ('protein_adapter.', 'go_adapter.', 'predictor.'))}
        for k, v in gs.items(): seen[k] |= v > 0
        LOG.info('[preflight] step=%d loss=%.6f | pool=%.3e proj_p=%.3e go_gate=%.3e proj_g=%.3e head=%.3e', i, float(loss.item()), gs['pool'], gs['proj_p'],
                 gs['go_gate'], gs['proj_g'], gs['head'])
        if i == 1 and any(v <= 0 for v in gs.values()): raise RuntimeError(f'Gradient contract failed at step 1: {gs}')
        torch.nn.utils.clip_grad_norm_(rp + hp, a.grad_clip);
        opt.step()
    missing = [k for k, v in seen.items() if not v]
    if missing: raise RuntimeError(f'Never observed gradients for: {missing}')
    LOG.info('[preflight] PASSED | ASL reaches head + protein pooling/projection + GO gate/projection; BioMedBERT frozen.')


if __name__ == '__main__': main()
