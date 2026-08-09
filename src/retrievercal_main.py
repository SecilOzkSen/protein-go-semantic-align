'''
python -m src.retrievercal_main \
  --model_kind embcal \
  --train_dump /workspace/candidate_dumps/P3aSemExp500_ESMknn500_union_train_top500 \
  --val_dump /workspace/candidate_dumps/P3aSemExp500_ESMknn500_union_val_top500 \
  --train_embedding_dump /workspace/candidate_dumps/P3a_SemExp_train_top1000 \
  --val_embedding_dump /workspace/candidate_dumps/P3a_SemExp_val_top1000 \
  --topk 500 \
  --batch_size 512 \
  --epochs 20 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --hidden_dim 128 \
  --proj_dim 128 \
  --dropout 0.10 \
  --pos_weight_max 20 \
  --lambda_f1 0.10 \
  --lambda_card 0.01 \
  --lambda_dag 0.0 \
  --eval_every_steps 500 \
  --patience 6 \
  --monitor fmax_full \
  --out_dir /workspace/protein-go-align/outputs/retrievercal/embcal_union_top500

  ## embcalset:

  python -m src.retrievercal_main \
  --model_kind embsetcal \
  --train_dump /workspace/candidate_dumps/P3aSemExp500_ESMknn500_union_train_top500 \
  --val_dump /workspace/candidate_dumps/P3aSemExp500_ESMknn500_union_val_top500 \
  --train_embedding_dump /workspace/candidate_dumps/P3a_SemExp_train_top1000 \
  --val_embedding_dump /workspace/candidate_dumps/P3a_SemExp_val_top1000 \
  --topk 500 \
  --batch_size 256 \
  --epochs 20 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --hidden_dim 128 \
  --proj_dim 128 \
  --n_layers 1 \
  --n_heads 4 \
  --dropout 0.10 \
  --pos_weight_max 20 \
  --lambda_f1 0.10 \
  --lambda_card 0.01 \
  --lambda_dag 0.0 \
  --eval_every_steps 500 \
  --patience 6 \
  --monitor fmax_full \
  --out_dir /workspace/protein-go-align/outputs/retrievercal/embsetcal_union_top500

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
    p.add_argument("--model_kind", default="retrievercal", choices=["retrievercal", "scoresetcal", "embcal", "embsetcal", "embeddingcal", "embeddingsetcal", "step1", "step2", "step3", "setcal"])
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
    p.add_argument("--lambda_f1", type=float, default=0.0)
    p.add_argument("--lambda_card", type=float, default=0.0)
    p.add_argument("--lambda_dag", type=float, default=0.0)
    p.add_argument("--dag_margin", type=float, default=0.0)
    p.add_argument("--dag_parents_json", default="")
    p.add_argument("--eval_every_steps", type=int, default=1000)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score_stat_rows", type=int, default=0)
    p.add_argument("--pooling_type", type=str, default="mean")
    p.add_argument("--monitor", default="fmax_full")
    p.add_argument("--train_embedding_dump", default="", help="Optional dump providing protein_z/go_z for train; defaults to train_dump")
    p.add_argument("--val_embedding_dump", default="", help="Optional dump providing protein_z/go_z for val; defaults to val_dump")
    p.add_argument("--proj_dim", type=int, default=0, help="Projection dim for embedding calibrators; 0 uses hidden_dim")
    p.add_argument("--lambda_pair", type=float, default=0.25,)
    p.add_argument("--pairwise_max_negatives", type=int, default=64,)
    p.add_argument("--use_stargo_eval", action="store_true", help="Compute StarGO/PFresGO metrics during validation.",)
    p.add_argument("--stargo_ontology", default="bp", choices=["bp", "mf", "cc"],)
    p.add_argument("--stargo_go_obo", default="", help="Path to the GO OBO file used by StarGO evaluation.",)
#    p.add_argument("--stargo_test_csv", default="", help="Path to nrPDB-GO_2019.06.18_test.csv.",)
    p.add_argument("--stargo_seqid_column", type=int, default=4, help="StarGO sequence-identity subset column. Default: 4.",)
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
