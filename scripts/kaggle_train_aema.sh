#!/usr/bin/env bash
set -euo pipefail

# Kaggle usage:
#   git clone -b codex/kaggle-aema-training https://github.com/TranTungDuong1611/DAOD-RGB2IR.git
#   cd DAOD-RGB2IR
#   bash scripts/kaggle_train_aema.sh
#
# Optional env overrides:
#   DATA_ROOT=/kaggle/input/flir-aligned/align
#   OUTPUT_DIR=/kaggle/working/daod_rgb2ir_aema
#   TOTAL_ITERS=35000
#   STOP_AT=10000
#   BATCH_SIZE=2

DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/daod_rgb2ir_aema}"
TOTAL_ITERS="${TOTAL_ITERS:-35000}"
CHUNK_ITERS="${CHUNK_ITERS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-2}"
MIN_SIZE="${MIN_SIZE:-512}"
MAX_SIZE="${MAX_SIZE:-640}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
VIS_EVERY="${VIS_EVERY:-5000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
ADV_WEIGHT="${ADV_WEIGHT:-0.2}"

export OUTPUT_DIR
mkdir -p "${OUTPUT_DIR}"

if [[ -z "${STOP_AT:-}" ]]; then
  CURRENT_STEP=0
  if [[ -f "${OUTPUT_DIR}/latest.pt" ]]; then
    CURRENT_STEP="$(python - <<'PY'
import os
import torch
path = os.path.join(os.environ["OUTPUT_DIR"], "latest.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print(int(ckpt.get("global_step", 0)))
PY
)"
  fi
  STOP_AT="$((CURRENT_STEP + CHUNK_ITERS))"
  if [[ "${STOP_AT}" -gt "${TOTAL_ITERS}" ]]; then
    STOP_AT="${TOTAL_ITERS}"
  fi
fi

DATA_ARGS=()
if [[ -n "${DATA_ROOT}" ]]; then
  DATA_ARGS+=(--data_root "${DATA_ROOT}")
fi

python -m pip install -q -r requirements.txt

set -o pipefail
python -u example_flir.py \
  "${DATA_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --log_file "${OUTPUT_DIR}/training_full_aema.log" \
  --metrics_file "${OUTPUT_DIR}/metrics_history.json" \
  --device cuda \
  --model fcos \
  --from_coco \
  --total_iters "${TOTAL_ITERS}" \
  --stop_at "${STOP_AT}" \
  --phase1_end 15000 \
  --phase2_end 20000 \
  --phase3_end 25000 \
  --batch_size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --min_size "${MIN_SIZE}" \
  --max_size "${MAX_SIZE}" \
  --eval_every "${EVAL_EVERY}" \
  --vis_every "${VIS_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --adv_weight "${ADV_WEIGHT}" \
  --ema_mode aema \
  --aema_fast_alpha 0.997 \
  --aema_slow_alpha 0.9996 \
  --aema_top_ratio 0.10 \
  --aema_update_interval 2 \
  --auto_resume \
  2>&1 | tee -a "${OUTPUT_DIR}/training_full_aema.log"
