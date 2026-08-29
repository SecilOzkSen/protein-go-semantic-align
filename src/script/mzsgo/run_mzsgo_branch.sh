#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <mf|bp|cc> <project-root> <mzsgo-temporal-root> <retriever-checkpoint> <reranker-checkpoint>" >&2
  exit 2
fi

branch="$1"
project_root="$2"
benchmark_root="$3"
retriever_checkpoint="$4"
reranker_checkpoint="$5"

case "$branch" in
  mf|bp|cc) ;;
  *) echo "branch must be mf, bp, or cc" >&2; exit 2 ;;
esac

cd "$project_root"
config="src/configs/mzsgo/mzsgo_${branch}_train.yaml"
reranker_config="src/configs/mzsgo/mzsgo_${branch}_reranker_temporal_eval.yaml"
branch_dir="$benchmark_root/processed/$branch"
dump_root="$benchmark_root/candidate_dumps/$branch"
result_root="$benchmark_root/results/$branch"
mkdir -p "$dump_root" "$result_root"

# Strict retrieval, no positive injection. Training/validation dumps are used
# for reranker fitting and checkpoint selection only.
python -m src.script.dump_retriever_candidates_global_local \
  --config "$config" \
  --checkpoint "$retriever_checkpoint" \
  --out_dir "$dump_root/train_top200" \
  --split train --topk 200 --save_top_go_ids --overwrite

python -m src.script.dump_retriever_candidates_global_local \
  --config "$config" \
  --checkpoint "$retriever_checkpoint" \
  --out_dir "$dump_root/valid_top200" \
  --split val --topk 200 --save_top_go_ids --overwrite

# Temporal split is opened only after checkpoint selection.
python -m src.script.dump_retriever_candidates_global_local \
  --config "$config" \
  --checkpoint "$retriever_checkpoint" \
  --out_dir "$dump_root/temporal_top200" \
  --split val \
  --ids_path "$branch_dir/test.txt" \
  --pid2pos_path "$branch_dir/pid_to_positives_${branch}.json" \
  --topk 200 --save_top_go_ids --overwrite

python -m src.script.eval_hiercross_checkpoint \
  --config "$reranker_config" \
  --checkpoint "$reranker_checkpoint" \
  --candidate_dump "$dump_root/temporal_top200" \
  --split val --topk 200 \
  --out_dir "$result_root/reranker_temporal"

python -m src.script.mzsgo.prepare_reranker_dump \
  --input-dir "$result_root/reranker_temporal" \
  --output-dir "$dump_root/reranker_temporal_top200"

# Retriever expert/fused metrics.
for score_kind in fused global local; do
  python -m src.script.mzsgo.evaluate_mzsgo_temporal \
    --dump-dir "$dump_root/temporal_top200" \
    --temporal-go-ids "$branch_dir/temporal_go_ids_${branch}.json" \
    --pid-to-temporal-positives "$branch_dir/pid_to_go_temporal_${branch}.json" \
    --score-kind "$score_kind" --ks 10 50 100 200 \
    --output "$result_root/retriever_${score_kind}.json"
done

python -m src.script.mzsgo.evaluate_mzsgo_temporal \
  --dump-dir "$dump_root/reranker_temporal_top200" \
  --temporal-go-ids "$branch_dir/temporal_go_ids_${branch}.json" \
  --pid-to-temporal-positives "$branch_dir/pid_to_go_temporal_${branch}.json" \
  --score-kind fused --ks 10 50 100 200 \
  --output "$result_root/reranker.json"
