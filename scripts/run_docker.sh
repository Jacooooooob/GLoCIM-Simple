#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run the GLORY container with GPU and mounted volumes.
# Usage examples:
#   bash scripts/run_docker.sh                     # default small config
#   bash scripts/run_docker.sh +val_mode=false     # override Hydra args
#   bash scripts/run_docker.sh model=GLORY dataset=MINDlarge

IMG_NAME=${IMG_NAME:-glocim:cu121}
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/logs" "$ROOT_DIR/checkpoint" "$ROOT_DIR/outputs"

docker run --rm -it \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all} \
  -e PROJECT_ROOT=/app \
  -e WANDB_MODE=offline \
  -v "$ROOT_DIR/data":/app/data \
  -v "$ROOT_DIR/logs":/app/logs \
  -v "$ROOT_DIR/checkpoint":/app/checkpoint \
  -v "$ROOT_DIR/outputs":/app/outputs \
  "$IMG_NAME" "$@"

