# MZSGO pure temporal benchmark patch

This patch adds the first two reproducible pieces of the MZSGO experiment:

1. conversion of the released 2023-training/2025-temporal split into the
   existing protein-GO store layout;
2. benchmark-compatible temporal metrics plus retrieval-specific diagnostics
   over existing retriever or reranker candidate dumps.
3. branch-specific retriever/reranker config generation and strict temporal
   evaluation commands.

## 1. Prepare and audit the released split

```bash
python src/script/mzsgo/prepare_mzsgo_temporal.py \
  --mzsgo-root /workspace/MZSGO \
  --out-root /workspace/MZSGO_temporal
```

The command intentionally fails unless the released counts are reproduced:

| Branch | Training proteins | Temporal proteins | Temporal labels |
|---|---:|---:|---:|
| MF | 50,874 | 555 | 35 |
| BP | 51,618 | 3,687 | 34 |
| CC | 51,743 | 17 | 6 |

It also rejects protein-ID overlap, exact-sequence overlap and temporal-label
leakage into training. It does not claim to reproduce the unpublished DIAMOND
command. The released `zero_shot_below30.txt` split is treated as the benchmark
artifact, and an independent DIAMOND audit remains a separate step.

The preparation step also writes `sequences/all_train_temporal.fasta`, which
can be passed directly to the existing `extract_gor2023_esm1b.py` script.

## 2. Generate branch configs

Use the three supplied GOR2023 YAMLs as architecture/hyperparameter bases:

```bash
python src/script/mzsgo/generate_mzsgo_configs.py \
  --base-config-dir /workspace/protein-go-semantic-align/src/configs \
  --mzsgo-root /workspace/MZSGO_temporal \
  --out-dir /workspace/protein-go-semantic-align/src/configs/mzsgo
```

This writes, per branch:

- retriever training config, validation only;
- retriever temporal eval config;
- reranker training config, validation only;
- reranker temporal eval config.

The temporal configs are separate so temporal proteins cannot influence early
stopping or checkpoint selection.

Train one retriever with the existing entry point:

```bash
python -m src.main_pfresgo \
  --config src/configs/mzsgo/mzsgo_mf_train.yaml
```

The generated configs use `best.pt` as a readable checkpoint placeholder for
later commands. Replace it with the actual checkpoint selected on validation,
or create a stable `best.pt` link after training.

Train the reranker after producing train and validation candidate dumps:

```bash
python -m src.script.mzsgo.run_reranker_with_config \
  --config src/configs/mzsgo/mzsgo_mf_reranker.yaml
```

## 3. Generate and evaluate candidate dumps

Use the existing global/local dump script with the generated branch files,
candidate GO IDs and markerless GO text. The output must contain:

- `eval_go_ids.npy`
- `top_go_cols.int32.npy`
- `top_scores.float32.npy`
- `protein_ids.json`
- optionally `top_valid.int8.npy`
- optionally global/local score arrays

The complete final-evaluation sequence for one branch is encoded in
`run_mzsgo_branch.sh`. It creates strict train/validation/temporal dumps,
evaluates the frozen reranker checkpoint, converts its exported scores to the
common dump schema, and reports fused/global/local retriever plus reranker
metrics.

```bash
bash src/script/mzsgo/run_mzsgo_branch.sh \
  mf \
  /workspace/protein-go-semantic-align \
  /workspace/MZSGO_temporal \
  /workspace/MZSGO_temporal/outputs/mf/retriever/ACTUAL_BEST.pt \
  /workspace/MZSGO_temporal/outputs/mf/reranker/ACTUAL_BEST.pt
```

## 4. Evaluate a dump directly

MF example:

```bash
python src/script/mzsgo/evaluate_mzsgo_temporal.py \
  --dump-dir /workspace/candidate_dumps/mzsgo_mf_temporal_top200 \
  --temporal-go-ids /workspace/MZSGO_temporal/processed/mf/temporal_go_ids_mf.json \
  --pid-to-temporal-positives /workspace/MZSGO_temporal/processed/mf/pid_to_go_temporal_mf.json \
  --ks 10 50 100 200 \
  --output /workspace/MZSGO_temporal/results/mf_fused.json
```

Set `--score-kind global` or `--score-kind local` to evaluate the expert score
arrays emitted by the global/local dump script.

The output includes:

- MZSGO-style protein-centric Fmax, AUPR, precision and recall;
- unseen Recall@K;
- candidate coverage@K;
- oracle microF@K;
- macro term recall@K.

Missing temporal labels outside the dumped top-K are assigned score zero. This
keeps retrieval failure visible instead of evaluating the reranker only on
successfully retrieved positives.
