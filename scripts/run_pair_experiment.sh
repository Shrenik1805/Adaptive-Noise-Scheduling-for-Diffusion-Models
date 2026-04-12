#!/usr/bin/env bash
set -euo pipefail

# End-to-end paired experiment runner:
# 1) Trains adaptive model
# 2) Trains fixed-cosine model
# 3) Selects best checkpoints by lowest validation FID
# 4) Runs fair paired evaluation
# 5) Produces a compact summary artifact

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VENV_PYTHON="${VENV_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-2e-4}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-10000}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-1000}"
REPEATS="${REPEATS:-3}"
RUN_TAG="${RUN_TAG:-default}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"

ADAPTIVE_CKPT_DIR="${ADAPTIVE_CKPT_DIR:-./checkpoints_adaptive_${RUN_TAG}}"
FIXED_CKPT_DIR="${FIXED_CKPT_DIR:-./checkpoints_fixed_${RUN_TAG}}"
ADAPTIVE_SAMPLE_DIR="${ADAPTIVE_SAMPLE_DIR:-./samples_adaptive_${RUN_TAG}}"
FIXED_SAMPLE_DIR="${FIXED_SAMPLE_DIR:-./samples_fixed_${RUN_TAG}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-./adaptive_diffusion/analysis_${RUN_TAG}}"

echo "Starting paired experiment with:"
echo "  DEVICE=${DEVICE}"
echo "  EPOCHS=${EPOCHS}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  LR=${LR}"
echo "  WANDB_MODE=${WANDB_MODE}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  SEED=${SEED}"
echo "  NUM_WORKERS=${NUM_WORKERS}"

export WANDB_MODE

echo "Training adaptive model..."
"${VENV_PYTHON}" -m adaptive_diffusion.train \
  --schedule-mode adaptive \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --num-workers "${NUM_WORKERS}" \
  --checkpoint-dir "${ADAPTIVE_CKPT_DIR}" \
  --sample-dir "${ADAPTIVE_SAMPLE_DIR}"

echo "Training fixed-cosine baseline..."
"${VENV_PYTHON}" -m adaptive_diffusion.train \
  --schedule-mode fixed_cosine \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --num-workers "${NUM_WORKERS}" \
  --checkpoint-dir "${FIXED_CKPT_DIR}" \
  --sample-dir "${FIXED_SAMPLE_DIR}"

find_best_ckpt() {
  local ckpt_dir="$1"
  local best_ckpt
  best_ckpt="$("${VENV_PYTHON}" - <<'PY' "${ckpt_dir}"
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
pattern = re.compile(r"epoch_\\d+_fid_([0-9]+(?:\\.[0-9]+)?)\\.pt$")
best_path = None
best_fid = None
for path in ckpt_dir.glob("epoch_*_fid_*.pt"):
    match = pattern.search(path.name)
    if match is None:
        continue
    fid = float(match.group(1))
    if best_fid is None or fid < best_fid:
        best_fid = fid
        best_path = path

if best_path is not None:
    print(str(best_path))
PY
)"
  if [[ -z "${best_ckpt}" ]]; then
    echo "No checkpoint found in ${ckpt_dir}" >&2
    exit 1
  fi
  echo "${best_ckpt}"
}

ADAPTIVE_BEST_CKPT="$(find_best_ckpt "${ADAPTIVE_CKPT_DIR}")"
FIXED_BEST_CKPT="$(find_best_ckpt "${FIXED_CKPT_DIR}")"

echo "Best adaptive checkpoint: ${ADAPTIVE_BEST_CKPT}"
echo "Best fixed checkpoint: ${FIXED_BEST_CKPT}"

echo "Running paired evaluation..."
"${VENV_PYTHON}" -m adaptive_diffusion.evaluate \
  --adaptive-checkpoint "${ADAPTIVE_BEST_CKPT}" \
  --fixed-checkpoint "${FIXED_BEST_CKPT}" \
  --device "${DEVICE}" \
  --output-dir "${ANALYSIS_DIR}" \
  --num-fid-samples "${NUM_FID_SAMPLES}" \
  --samples-per-class "${SAMPLES_PER_CLASS}" \
  --repeats "${REPEATS}"

echo "Generating summary artifact..."
"${VENV_PYTHON}" scripts/summarize_results.py \
  --analysis-dir "${ANALYSIS_DIR}" \
  --adaptive-checkpoint "${ADAPTIVE_BEST_CKPT}" \
  --fixed-checkpoint "${FIXED_BEST_CKPT}"

echo "Paired experiment completed successfully."
echo "Artifacts:"
echo "  ${ANALYSIS_DIR}/efficiency_frontier.csv"
echo "  ${ANALYSIS_DIR}/per_class_metrics.csv"
echo "  ${ANALYSIS_DIR}/summary_metrics.csv"
echo "  ${ANALYSIS_DIR}/summary_report.md"
