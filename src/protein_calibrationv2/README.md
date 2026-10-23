# Protein calibration v2, frozen ESM representation + score statistics

V2 differs from V1 **only in its input representation**: the shared calibration MLP receives the ten score-distribution statistics **and** a small learned projection of frozen ESM-1b residue mean embeddings. It still predicts cardinality, uses predicted cardinality to set a protein-conditioned boundary, learns a positive scale, and preserves each protein's GO ranking.

## Training and evaluation protocol

- Retriever: unchanged frozen InfoNCE checkpoint, trained on the 18,818-protein split.
- Calibration train: 4,704 disjoint BP proteins.
- Validation: original PFresGO validation proteins, for head checkpoint selection and fixed threshold.
- Test: original PFresGO test, once, using validation-selected thresholds.
- Training objective: bin-balanced log-cardinality Huber plus protein-wise soft-F1, same as the V1 bin-balanced experiment.
- Protein input: masked mean of **precomputed ESM-1b residue embeddings**, not the learned retriever's projected protein vector. This choice keeps extraction compatible with existing `prot_emb_pad` and `prot_attn_mask` batch fields and avoids altering the retriever. ESM encoder is not fine-tuned.
- Normalization mean/std for protein embeddings are fitted **only on calibration train** and saved in the head checkpoint.
- No true cardinality or test labels enter inference.

## Installation

Unzip at the repository root, creating `src/protein_calibrationv2/`. V1 files remain unchanged.

## Smoke test

```bash
python -m src.protein_calibrationv2.test_smoke
```

## Extract all three splits (V1 NPZ files cannot be reused, because they lack `protein_embeddings`)

```bash
BASE=/workspace/data_pfresgo/outputs/bp/calibration_study/infonce
SPLIT=/workspace/data_pfresgo/splits/bp_calibration_seed42
CONFIG=src/pfresgo_bp_calib_infonce.yaml
CKPT="$BASE/best_primary.pt"

python -m src.protein_calibrationv2.extract_scores --config "$CONFIG" --checkpoint "$CKPT" --ids "$SPLIT/calibration_train_ids.txt" --out "$BASE/calibrationv2/calibration.npz"
python -m src.protein_calibrationv2.extract_scores --config "$CONFIG" --checkpoint "$CKPT" --ids /workspace/stargo/datasets/pfresgo/valid.txt --out "$BASE/calibrationv2/valid.npz"
python -m src.protein_calibrationv2.extract_scores --config "$CONFIG" --checkpoint "$CKPT" --ids /workspace/stargo/datasets/pfresgo/test.txt --out "$BASE/calibrationv2/test.npz"
```

## Fit

```bash
python -m src.protein_calibrationv2.fit \
  --calibration "$BASE/calibrationv2/calibration.npz" \
  --valid "$BASE/calibrationv2/valid.npz" \
  --out-dir "$BASE/calibrationv2/cardinality_binbalanced" \
  --card-bin-max-ratio 5.0 --seed 42
```

## Evaluate, with the threshold saved from validation

```bash
python -m src.protein_calibrationv2.evaluate \
  --test "$BASE/calibrationv2/test.npz" \
  --head-dir "$BASE/calibrationv2/cardinality_binbalanced" \
  --out "$BASE/calibrationv2/cardinality_binbalanced/test_results.json"
```

## Important limitations

- This is **not** end-to-end retriever fine-tuning; only the small calibration network is trained.
- Results from validation and test should be compared with V1 bin-balanced on exactly the same IDs and GO ordering.
- Test data should not be used for hyperparameter or architecture selection.
- The exact `extract_scores` integration must be smoke-tested on RunPod. Its checkpoint reader supports both `checkpoint['model']` and `checkpoint['model']['model']`.
