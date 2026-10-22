# Cardinality-guided protein-conditioned affine decision calibration, v1

This is a **single shared MLP** with a cardinality output and an explicitly cardinality-conditioned decision boundary. It uses the existing retriever's **raw final [protein, GO] scores**, not probabilities. The retriever is frozen. Within-protein GO ordering is unchanged because `z = positive_scale * (raw_score - protein_boundary)`.

## Data protocol

- Retriever trained on 18,818 annotated BP proteins, from the stratified split.
- Calibration head trained on 4,704 **unseen by retriever** BP proteins.
- Original PFresGO validation selects head checkpoint (validation soft objective) and exact Fmax threshold.
- Original PFresGO test is used once for fixed-threshold reporting.
- GO universe and ordering must match exactly; script checks this.
- The test oracle Fmax is diagnostic **only**.
- The package does not compute wFmax. Match the PFresGO IC weights and propagation protocol before making wFmax benchmark claims.

## Model

`score_features` produces 10 label-free score statistics. A shared MLP produces log-cardinality, a residual boundary adjustment, and positive scale. The decision boundary contains an explicit monotonic cardinality contribution: `boundary = global_boundary + residual - positive_gain * (predicted_log_card - reference_log_card)`.

Training objective: per-protein soft-F1 at zero *training decision threshold* plus `0.2 * Huber(predicted_log_card, log1p(true_count))` and a small boundary regularizer. This is a surrogate, **not** exact CAFA Fmax. A validation sweep selects the final **single global threshold** in transformed score space. Never use true cardinality as model input at inference.

## Install

Extract ZIP at repository root. It contains `src/protein_calibration/` and should replace the old `model.py`, `fit.py`, `evaluate.py`. Keep the existing `create_calibration_split.py` if already present. No change to `src.main`, `src.training.trainer`, or `src.metrics.cafa`.

## Run after the 80% retriever finishes

```bash
BASE=/workspace/data_pfresgo/outputs/bp/calibration_study/infonce
SPLIT=/workspace/data_pfresgo/splits/bp_calibration_seed42
CONFIG=src/pfresgo_bp_calib_infonce.yaml

python -m src.protein_calibration.extract_scores --config "$CONFIG" --checkpoint "$BASE/best_primary.pt" --ids "$SPLIT/calibration_train_ids.txt" --out "$BASE/calibration/calibration.npz"
python -m src.protein_calibration.extract_scores --config "$CONFIG" --checkpoint "$BASE/best_primary.pt" --ids /workspace/stargo/datasets/pfresgo/valid.txt --out "$BASE/calibration/valid.npz"
python -m src.protein_calibration.fit --calibration "$BASE/calibration/calibration.npz" --valid "$BASE/calibration/valid.npz" --out-dir "$BASE/calibration/cardinality_guided" --seed 42
```

**Only after selecting the pipeline on validation**, run:

```bash
python -m src.protein_calibration.extract_scores --config "$CONFIG" --checkpoint "$BASE/best_primary.pt" --ids /workspace/stargo/datasets/pfresgo/test.txt --out "$BASE/calibration/test.npz"
python -m src.protein_calibration.evaluate --test "$BASE/calibration/test.npz" --head-dir "$BASE/calibration/cardinality_guided" --out "$BASE/calibration/cardinality_guided/test_results.json"
```

Repeat with PBR-only and combined models using their own 80% retriever checkpoints, configs and output folders. Do not reuse a head between retrievers.

## Caveats

1. The existing score extractor reconstructs `OppTrainer` and expects the project's checkpoint to contain `state['model']`; run an extraction smoke test on your actual RunPod checkpoint first.
2. The validation selection metric is the soft surrogate, whereas the benchmark metric is exact CAFA Fmax. This is deliberate, but the mismatch can affect selection.
3. If cardinality has no predictive signal in score statistics, this head can overfit. Inspect cardinality MAE and test F at the **validation** threshold.
4. The architecture guarantees a cardinality pathway but does **not** guarantee improved generalization or statistically calibrated probabilities.
5. `src.main` currently builds its active benchmark GO universe from annotation unions, so check `go_ids` and their order on every extraction.
6. Use identical annotation propagation and evaluation rules for baseline comparisons.
