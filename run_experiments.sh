#!/usr/bin/env bash
# Run from the repository root after generating all three configurations.
set -euo pipefail
for experiment in reference experiment0 experiment1; do
  for steps in 1000 2500; do
    model_dir="experiments/${experiment}"
    python experiments/model_main_tf2.py \
      --model_dir="$model_dir" \
      --pipeline_config_path="$model_dir/pipeline_new.config" \
      --num_train_steps="$steps" --checkpoint_every_n=500 \
      > "$model_dir/train-${steps}.log" 2>&1
    python evaluate_once.py --model-dir "$model_dir" --steps "$steps" \
      > "$model_dir/eval-${steps}.log" 2>&1
  done
done
