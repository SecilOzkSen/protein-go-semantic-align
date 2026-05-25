'''
python -m src.script.check_esm_knn_leakage \
  --dump_dir /workspace/candidate_dumps/ESMknn_val_top500_nofill \
  --split val

python -m src.script.check_esm_knn_leakage \
  --dump_dir /workspace/candidate_dumps/ESMknn_train_top500_nofill \
  --split train

'''

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser("Check ESM-kNN neighbor leakage and similarity statistics.")
    p.add_argument("--dump_dir", type=str, required=True)
    p.add_argument("--split", type=str, required=True, choices=["train", "val", "test", "custom"])
    p.add_argument("--sim_thresholds", type=float, nargs="+", default=[0.999, 0.995, 0.99, 0.98, 0.95])
    args = p.parse_args()

    dump_dir = Path(args.dump_dir)

    nn_idx_path = dump_dir / "knn_neighbor_idx.int64.npy"
    nn_sim_path = dump_dir / "knn_neighbor_sim.float32.npy"
    train_ids_path = dump_dir / "knn_train_ids.json"
    query_ids_path = dump_dir / "knn_query_ids.json"

    for path in [nn_idx_path, nn_sim_path, train_ids_path, query_ids_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Add neighbor debug saving to dump_esm_knn_candidates.py and re-dump."
            )

    nn_idx = np.load(nn_idx_path, mmap_mode="r")
    nn_sim = np.load(nn_sim_path, mmap_mode="r")
    train_ids = [str(x) for x in load_json(train_ids_path)]
    query_ids = [str(x) for x in load_json(query_ids_path)]

    train_id_set = set(train_ids)
    query_id_set = set(query_ids)

    print("\n[ESM-kNN LEAKAGE CHECK]")
    print("dump_dir:", dump_dir)
    print("split:", args.split)
    print("nn_idx:", nn_idx.shape, nn_idx.dtype)
    print("nn_sim:", nn_sim.shape, nn_sim.dtype)
    print("n_train_ids:", len(train_ids))
    print("n_query_ids:", len(query_ids))

    if nn_idx.shape != nn_sim.shape:
        raise RuntimeError(f"nn_idx shape {nn_idx.shape} != nn_sim shape {nn_sim.shape}")

    if nn_idx.shape[0] != len(query_ids):
        raise RuntimeError(f"nn_idx rows {nn_idx.shape[0]} != query_ids {len(query_ids)}")

    # Query IDs that also exist in train IDs.
    query_train_overlap = query_id_set & train_id_set
    print("\n[ID OVERLAP]")
    print("query ∩ train:", len(query_train_overlap))

    # Check if exact same ID appears as a neighbor.
    bad_self = []
    for i, qid in enumerate(query_ids):
        neighbor_ids = [train_ids[int(j)] for j in nn_idx[i]]
        if qid in neighbor_ids:
            bad_self.append((i, qid))

    print("rows with query ID appearing in neighbors:", len(bad_self))
    if bad_self:
        print("examples:", bad_self[:10])

    if args.split in ["val", "test", "custom"]:
        if len(query_train_overlap) > 0:
            print("[WARN] Some query IDs exist in train IDs. This may be expected only if split overlap exists.")
        if len(bad_self) > 0:
            raise RuntimeError("Leakage risk: query protein appears among train neighbors.")

    if args.split == "train":
        if len(bad_self) > 0:
            raise RuntimeError("Self-neighbor leakage: train query appears among its own neighbors.")
        else:
            print("train self-neighbor exclusion: OK")

    top1 = np.asarray(nn_sim[:, 0], dtype=np.float32)
    print("\n[TOP-1 SIMILARITY]")
    print("mean:", float(top1.mean()))
    print("std :", float(top1.std()))
    print("min :", float(top1.min()))
    print("p50 :", float(np.percentile(top1, 50)))
    print("p90 :", float(np.percentile(top1, 90)))
    print("p95 :", float(np.percentile(top1, 95)))
    print("p99 :", float(np.percentile(top1, 99)))
    print("max :", float(top1.max()))

    print("\n[HIGH-SIM FRACTION]")
    for t in args.sim_thresholds:
        frac = float((top1 >= float(t)).mean())
        print(f"top1 >= {t}: {frac:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()