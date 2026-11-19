#!/usr/bin/env bash
set -euo pipefail
cd /workspace/protein-go-semantic-align
unset GLOBAL_POOL_ABLATION
CKPT=/workspace/MZSGO_temporal/outputs/cc/v2_infonce_local_pbr_freq_aware_from_scratch/best_primary.pt
test -f "$CKPT" || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m src.main_cc_val_only_finetune --config ./src/ablations/mzsgo_cc_val_only_finetune.yaml
