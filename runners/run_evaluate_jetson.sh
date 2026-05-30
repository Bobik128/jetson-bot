#!/usr/bin/env bash
python ./../src/scripts/evaluate.py \
  --policy act \
  --where jetson \
  --teleop dummy \
  --remote-ip 127.0.0.1 \
  --robot-id jetson-bot \
  --hf-model-id Bobik553/jetson-bot_policy-blue_cubes_in_red-NEO-1 \
  --hf-eval-dataset-base-id Bobik553/jetson-bot_blue-block-on-box_eval-help-NEO \
  "$@"
