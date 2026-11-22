#!/usr/bin/env bash
set -euo pipefail
TARGET_DIR="${1:-/workspace/protein-go-semantic-align}"
if (( $# > 0 )); then shift; fi
if [[ ! -f "$TARGET_DIR/src/ablations/b0_mean_mean/main.py" ]]; then
  printf 'Install this package first with: bash install.sh %s\n' "$TARGET_DIR" >&2
  exit 1
fi
cd -- "$TARGET_DIR"
exec env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
  python -m src.ablations.b0_mean_mean.main --config src/ablations/mzsgo_mf_b0_mean_mean.yaml "$@"
