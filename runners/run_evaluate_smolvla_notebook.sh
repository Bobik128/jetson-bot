#!/usr/bin/env bash
python ./../src/scripts/evaluate.py \
  --policy smolvla \
  --where client \
  --teleop dummy \
  --remote-ip 10.98.56.119 \
  --robot-id jetson-bot \
  --hf-model-id Bobik553/jetson-bot_policy-smolvla-blue_on_red \
  --hf-eval-dataset-base-id Bobik553/jetson-bot_blue-block-on-box_eval-help \
  "$@"
