'''
python -m src.retrievercal_main \
  --model_kind retrievercal \
  --train_dump /workspace/candidate_dumps/P3aSemExp500_ESMknn500_union_train_top500 \
  --val_dump /workspace/candidate_dumps/P3aSemExp500_ESMknn500_union_val_top500 \
  --topk 500 \
  --batch_size 1024 \
  --epochs 20 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --hidden_dim 64 \
  --dropout 0.05 \
  --lambda_f1 0.25 \
  --lambda_card 0.05 \
  --lambda_dag 0.0 \
  --eval_every_steps 500 \
  --patience 8 \
  --monitor fmax_full \
  --out_dir /workspace/protein-go-align/outputs/retrievercal/retrievercal_union_top500
'''
import argparse
import logging
from pathlib import Path

from src.training.retrievercal_trainer import CalibConfig, CalibTrainer


def parse_args() -> CalibConfig:
    p = argparse.ArgumentParser("Train RetrieverCal / ScoreSetCal over candidate dumps")
    p.add_argument("--train_dump", required=True)
    p.add_argument("--val_dump", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_kind", default="retrievercal", choices=["retrievercal", "scoresetcal", "step1", "step2", "setcal"])
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--pos_weight_max", type=float, default=50.0)
    p.add_argument("--lambda_f1", type=float, default=0.25)
    p.add_argument("--lambda_card", type=float, default=0.05)
    p.add_argument("--lambda_dag", type=float, default=0.0)
    p.add_argument("--dag_margin", type=float, default=0.0)
    p.add_argument("--dag_parents_json", default="")
    p.add_argument("--eval_every_steps", type=int, default=1000)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score_stat_rows", type=int, default=0)
    p.add_argument("--monitor", default="fmax_full")
    a = p.parse_args()
    return CalibConfig(**vars(a))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = parse_args()
    logging.info("[main] config=%s", cfg)
    trainer = CalibTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
