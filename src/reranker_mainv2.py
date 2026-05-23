import argparse
from src.training.reranker_trainerv2 import RerankerTrainConfig, RerankerTrainer
import yaml, types

YAML_FILE = "/workspace/protein-go-semantic-align/src/rerankerv2.yaml"

def load_structured_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    reranker = cfg.get("reranker", {})

    args = types.SimpleNamespace(
        train_dump = reranker.get("train_dump", None),
        val_dump = reranker.get("val_dump", None),
        test_dump = reranker.get("test_dump", None),
        out_dir = reranker.get("out_dir", None),
        topk = int(reranker.get("topk", 500)),
        model_kind = reranker.get("model_kind", "interaction_mlp"),
        hidden_dim = reranker.get("hidden_dim", 512),
        dropout = float(reranker.get("dropout", 0.10)),
        batch_size = int(reranker.get("batch_size", 8)),
        num_workers = int(reranker.get("num_workers", 0)),
        lr = float(reranker.get("lr", 1e-4)),
        weight_decay = float(reranker.get("weight_decay", 1e-4)),
        epochs = int(reranker.get("epochs", 10)),
        patience = int(reranker.get("patience", 2)),
        grad_clip = float(reranker.get("grad_clip", 1.0)),
        pos_weight_max = float(reranker.get("pos_weight_max", 50.0)),
        monitor = reranker.get("monitor", "fmax"),
        device = reranker.get("device", "cpu"),
    )
    return args


def parse_args():
    p = argparse.ArgumentParser("Train simple frozen-retriever candidate reranker.")

    # --- temel ayarlar ---
    p.add_argument("--config", type=str, default=None,
                        help="YAML config file path (örn: src/configs/colab.yaml)")
    p.add_argument("--device", type=str, default="cuda:0",
                        help="cuda device")

    args = p.parse_args()

    if not args.config:
        args.config = YAML_FILE
        print(f"[main] No --config passed, defaulting to {args.config}")

    args = load_structured_cfg(args.config)

    return args

def main():
    a = parse_args()
    cfg = RerankerTrainConfig(
        train_dump=a.train_dump,
        val_dump=a.val_dump,
        test_dump=a.test_dump,
        out_dir=a.out_dir,
        topk=a.topk,
        model_kind=a.model_kind,
        hidden_dim=a.hidden_dim,
        dropout=a.dropout,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        lr=a.lr,
        weight_decay=a.weight_decay,
        epochs=a.epochs,
        patience=a.patience,
        grad_clip=a.grad_clip,
        pos_weight_max=a.pos_weight_max,
        monitor=a.monitor,
        device=a.device,
    )
    trainer = RerankerTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
