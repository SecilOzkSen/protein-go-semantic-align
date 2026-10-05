#!/usr/bin/env python3
"""Convert eval_hiercross_checkpoint output to the common candidate-dump schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir
    target = args.output_dir
    target.mkdir(parents=True, exist_ok=True)

    eval_ids = np.load(source / "eval_go_ids.npy").astype(np.int64)
    cand_ids = np.load(source / "cand_ids.int64.npy").astype(np.int64)
    scores = np.load(source / "scores.float32.npy").astype(np.float32)
    labels = np.load(source / "labels.int8.npy").astype(np.int8)
    valid = np.load(source / "valid.int8.npy").astype(np.int8)
    true_ids = np.load(source / "true_go_ids.npy").astype(np.int64)
    protein_ids = json.loads((source / "protein_ids.json").read_text(encoding="utf-8"))

    id_to_col = {int(go_id): col for col, go_id in enumerate(eval_ids.tolist())}
    top_cols = np.full(cand_ids.shape, -1, dtype=np.int32)
    for row in range(cand_ids.shape[0]):
        for rank in range(cand_ids.shape[1]):
            if not valid[row, rank]:
                continue
            go_id = int(cand_ids[row, rank])
            if go_id not in id_to_col:
                raise KeyError(f"Candidate GO ID missing from eval universe: {go_id}")
            top_cols[row, rank] = id_to_col[go_id]

    # Invalid positions need an in-range placeholder because consumers index
    # eval_go_ids before applying top_valid.
    top_cols[top_cols < 0] = 0
    np.save(target / "eval_go_ids.npy", eval_ids)
    np.save(target / "top_go_cols.int32.npy", top_cols)
    np.save(target / "top_scores.float32.npy", scores)
    np.save(target / "top_labels.int8.npy", labels)
    np.save(target / "top_valid.int8.npy", valid)
    np.save(target / "true_go_ids.npy", true_ids)
    (target / "protein_ids.json").write_text(
        json.dumps(protein_ids, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Converted {cand_ids.shape[0]} proteins x {cand_ids.shape[1]} candidates -> {target}")


if __name__ == "__main__":
    main()
