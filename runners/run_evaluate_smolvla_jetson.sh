#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

python ./../src/scripts/evaluate.py \
  --policy smolvla \
  --where jetson \
  --teleop dummy \
  --remote-ip 127.0.0.1 \
  --robot-id jetson-bot \
  --hf-model-id Bobik553/jetson-bot_policy-smolvla-blue_on_red \
  --hf-eval-dataset-base-id Bobik553/jetson-bot_blue-block-on-box_eval-help \
  "$@"
