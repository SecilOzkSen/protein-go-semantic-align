"""Experiment D full GO neighbour MoE, frozen Retriever-v2 scores."""
import argparse
import json
import logging
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.moe.dataset import MoEDataset, read_ids, verify_neighbours
from src.moe.model import NeighbourGuidedMoE, MoEConfig
from src.moe.trainer import train
from src.metrics.stargo_pfresgo_metrics import StarGOPFresGOEvaluator


def parse_args():
    p = argparse.ArgumentParser('Experiment D MoE')
    for k in ('train_dump', 'val_dump', 'evidence_dir', 'neighbour_dir', 'go_graph_path', 'output_dir'):
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--device', default='cuda:0');
    p.add_argument('--ontology', default='bp', choices=['bp', 'mf', 'cc'])
    p.add_argument('--epochs', type=int, default=12);
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=32);
    p.add_argument('--eval_batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4);
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--grad_clip', type=float, default=1.0);
    p.add_argument('--hidden_dim', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.1);
    p.add_argument('--initial_retriever_weight', type=float, default=0.9)
    p.add_argument('--gamma_pos', type=float, default=0.0);
    p.add_argument('--gamma_neg', type=float, default=4.0)
    p.add_argument('--asl_clip', type=float, default=0.05);
    p.add_argument('--log_every', type=int, default=500)
    p.add_argument('--seed', type=int, default=42);
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--resume', type=Path);
    p.add_argument('--check_only', action='store_true')
    p.add_argument('--wandb', action='store_true');
    p.add_argument('--wandb_project', default='protein-go-align-pfresgo')
    p.add_argument('--wandb_run_name', default='ExperimentD-BP-ESM5NN-MoE')
    return p.parse_args()


def main():
    a = parse_args();
    a.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    random.seed(a.seed);
    np.random.seed(a.seed);
    torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    tr = MoEDataset(a.train_dump, a.evidence_dir, a.neighbour_dir, 'train')
    va = MoEDataset(a.val_dump, a.evidence_dir, a.neighbour_dir, 'val')
    if not np.array_equal(tr.go_ids, va.go_ids): raise ValueError('Train/val GO ID order mismatch')
    if set(tr.ids) & set(va.ids): raise ValueError('Train/validation overlap')
    bank = read_ids(a.neighbour_dir / 'bank_protein_ids.json')
    verify_neighbours(a.neighbour_dir, 'train', tr.ids, bank)
    verify_neighbours(a.neighbour_dir, 'val', va.ids, bank)
    print(f'CONTRACT OK: train={len(tr)} val={len(va)} GO={len(tr.go_ids)}; no ID/GO mismatch; no self/non-train neighbours', flush=True)
    if a.check_only: return
    loader_tr = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, pin_memory=True)
    loader_va = DataLoader(va, batch_size=a.eval_batch_size, shuffle=False, num_workers=a.num_workers, pin_memory=True)
    evaluator = StarGOPFresGOEvaluator(goterms=tr.go_ids, ontology=a.ontology, go_graph_path=a.go_graph_path)
    model = NeighbourGuidedMoE(MoEConfig(hidden_dim=a.hidden_dim, dropout=a.dropout, initial_retriever_weight=a.initial_retriever_weight))
    train(model, loader_tr, loader_va, evaluator, a, tr.go_ids)


if __name__ == '__main__': main()
