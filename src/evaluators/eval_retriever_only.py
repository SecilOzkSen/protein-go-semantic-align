import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src.evaluators.pfresgo_eval import Method


dump = Path(
    "/workspace/candidate_dumps/"
    "name_def_isa_step52929_test_top500"
)

go_obo = (
    "/workspace/stargo/datasets/pfresgo/go.obo"
)

ontology = "bp"

eval_go_ids = np.load(
    dump / "eval_go_ids.npy"
).astype(np.int64)

top_cols = np.load(
    dump / "top_go_cols.int32.npy"
).astype(np.int64)

top_scores = np.load(
    dump / "top_scores.float32.npy"
).astype(np.float32)

true_go_ids = np.load(
    dump / "true_go_ids.npy"
).astype(np.int64)

with open(
    dump / "protein_ids.json",
    encoding="utf-8",
) as f:
    protein_ids = json.load(f)

n_proteins = len(protein_ids)
n_terms = len(eval_go_ids)

# Retriever scores are logits/ranking scores.
probs = torch.sigmoid(
    torch.from_numpy(top_scores)
).numpy()

full_pred = np.zeros(
    (n_proteins, n_terms),
    dtype=np.float32,
)

full_true = np.zeros(
    (n_proteins, n_terms),
    dtype=np.int8,
)

for i in range(n_proteins):
    full_pred[i, top_cols[i]] = probs[i]

go_to_col = {
    int(go_id): col
    for col, go_id in enumerate(eval_go_ids)
}

for i in range(n_proteins):
    for go_id in true_go_ids[i]:
        go_id = int(go_id)

        if go_id < 0:
            continue

        col = go_to_col.get(go_id)

        if col is not None:
            full_true[i, col] = 1

goterms = [
    f"GO:{int(go_id):07d}"
    for go_id in eval_go_ids
]

eval_file = dump / "retriever_only_eval.pckl"

with eval_file.open("wb") as f:
    pickle.dump(
        {
            "Y_true": full_true,
            "Y_pred": full_pred,
            "goterms": goterms,
            "proteins": protein_ids,
        },
        f,
    )

method = Method(
    "retriever_only",
    str(eval_file),
    ontology,
    go_obo,
)

keep_pidx = np.arange(
    n_proteins,
    dtype=np.int64,
)

fscores, recalls, precisions, thresholds = (
    method._protein_centric_fmax(
        keep_pidx=keep_pidx
    )
)

best = int(np.argmax(fscores))

print("retriever_only_fmax:", fscores[best])
print("threshold:", thresholds[best])
print("precision:", precisions[best])
print("recall:", recalls[best])