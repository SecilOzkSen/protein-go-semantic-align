# Residual V3 PFresGO aligned

Install this directory as `src/residual_v3_pfresgo/` in the project. Keep your original V3 files unchanged.

## Inputs

Three NPZ files from the **same frozen Combined retriever checkpoint**:

- calibration training, held-out from retriever training
- validation, used only for checkpoint and global threshold selection
- test, used only after model selection

Each NPZ must contain `scores`, `labels`, `protein_ids`, `go_ids`, `protein_embeddings` as exported by `extract_scores.py`. Ensure the Combined retriever was trained on the retriever-train split, not the calibration proteins. GO order must match.

## Commands

From repository root:

```bash
python -m src.residual_v3_pfresgo.test_smoke

python -m src.residual_v3_pfresgo.fit \
  --calibration /PATH/combined_calibration.npz \
  --valid /PATH/combined_valid.npz \
  --obo /workspace/stargo/datasets/pfresgo/go.obo \
  --annotations /PATH/annot.tsv \
  --original-eval /workspace/stargo/pfresgo_eval.py \
  --branch bp --device cuda \
  --out-dir /workspace/data_pfresgo/outputs/bp/calibration_study/combined/residual_v3_pfresgo

python -m src.residual_v3_pfresgo.evaluate \
  --test /PATH/combined_test.npz \
  --head-dir /workspace/data_pfresgo/outputs/bp/calibration_study/combined/residual_v3_pfresgo \
  --obo /workspace/stargo/datasets/pfresgo/go.obo \
  --annotations /PATH/annot.tsv \
  --original-eval /workspace/stargo/pfresgo_eval.py \
  --branch bp --device cuda \
  --out /workspace/data_pfresgo/outputs/bp/calibration_study/combined/residual_v3_pfresgo/test_results.json
```

The `--annotations` file must contain the proteins used in validation and test. The calibration training labels are taken from its NPZ. The test command must only be run after training and validation selection are complete.

## Design

- Frozen retriever scores and **learned local-window protein vectors** (masked mean of normalized `proj_local` window outputs, followed by L2 normalization). No raw ESM mean pooling.
- V3 distribution statistics + learned-local-vector encoder produce a shared protein-specific scale and boundary.
- Raw sigmoid probability and corrected sigmoid probability are mixed with one trainable global alpha, initialized at 0.2.
- Protein-internal GO ordering is preserved by construction.
- Differentiable ancestor max propagation is applied to labels and predicted probabilities during training.
- Training approximates a global-threshold sweep using nine soft thresholds (0.1 to 0.9), with differentiable soft selection. Soft precision uses a differentiable predicted-positive protein weight; soft recall averages over all proteins. This is a surrogate, not mathematically identical to hard PFresGO Fmax.
- Exact validation uses existing `pfresgo_fast_exact.metrics`, with a parity check against the original evaluator on a subset, and selects the highest Fmax checkpoint subject to macro/micro AP and AUROC >= 98% of raw.
- No cardinality head, no ranking loss, no post-training alpha grid search.

## Important cautions

- The exact metric implementation (`pfresgo_fast_exact.py` and `pfresgo_comparison.py`) is imported from the existing project and was not included in this package.
- Original PFresGO's published sequence-identity test subset and original baseline pickle parity are not established by these scripts.
- The train loss uses mini-batches, while exact validation aggregates over the whole split.
- Verify your NPZ `scores` are raw full-GO scores and not already sigmoid-transformed.
- The design is not guaranteed to improve validation Fmax or preserve test AUPR; selection falls back to raw when no feasible improvement exists.

## Export the three score matrices (required)

Use the same Combined checkpoint and its matching YAML config for every split.

```bash
python -m src.residual_v3_pfresgo.extract_scores --config /PATH/combined.yaml --checkpoint /PATH/combined_checkpoint.pt --ids /PATH/calibration_train_ids.txt --out /PATH/combined_calibration.npz
python -m src.residual_v3_pfresgo.extract_scores --config /PATH/combined.yaml --checkpoint /PATH/combined_checkpoint.pt --ids /PATH/validation_ids.txt --out /PATH/combined_valid.npz
python -m src.residual_v3_pfresgo.extract_scores --config /PATH/combined.yaml --checkpoint /PATH/combined_checkpoint.pt --ids /PATH/test_ids.txt --out /PATH/combined_test.npz
```

**Important:** This code calls `trainer._score_full_go` for the exact scoring path and separately calls `trainer.model.encode_local_windows` for learned local representations. Verify that your running trainer uses `score_fine_grained` and that the loaded checkpoint matches the supplied `alignment_model.py`. The exported summary vector is a representation of local windows, not a vector whose dot product reproduces SmoothMIL scores.

## Checkpoint and fallback policy

Training uses nine differentiable global thresholds (0.1..0.9) as a surrogate for the exact PFresGO 100-threshold validation sweep. `--sweep-temperature` controls the soft selection (default 0.05). These training thresholds are shared across all proteins, not protein-specific.

Patience begins only after the first feasible improvement over raw validation Fmax. If none is found, training runs for all requested epochs and saves `selection.json` with `raw_fallback=true`. In that case `head.pt` is intentionally absent, and `evaluate.py` uses the raw retriever with the validation-selected raw threshold.

## Per-epoch diagnostics

Every `EPOCH` line and `history.json` now include `alpha`, `boundary_mean/std/min/max`, `scale_mean/std/min/max`, `residual_mean_abs`, `residual_max_abs`, `grad_alpha`, `grad_boundary`, and `grad_scale`. Boundary, scale and residual statistics are calculated over the entire validation set; gradients are mean pre-clipping L2 norms across training batches. `history.json` is updated after every epoch, so interrupted runs retain their diagnostics. `grad_boundary` includes the global boundary parameter. These metrics diagnose parameter compensation, but do not by themselves prove identifiability.
