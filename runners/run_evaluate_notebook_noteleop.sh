#!/usr/bin/env bash
python ./../src/scripts/evaluate.py \
  --policy act \
  --where client \
  --teleop none \
  --remote-ip 10.98.56.119 \
  --robot-id jetson-bot \
  --hf-model-id Bobik553/jetson-bot_policy-blue_cubes_in_red-NEO-1 \
  --hf-eval-dataset-base-id Bobik553/jetson-bot_blue-block-on-box_eval-help-NEO-PC-noteleop \
  --display-data false \
  --image-writer-threads 1 \
  "$@"
