#!/usr/bin/env bash
# Evaluate a single KISS-IMU checkpoint on a list of sequences.
#
# Training already picks the best checkpoint via validation, so this
# script intentionally evaluates one ckpt — pass best_model.ckpt via CKPT.
set -euo pipefail

DATA_DIR=${DATA_DIR:-/storage1/Datasets/kiss_imu_datasets/DiTer_os}
DATA_TYPE=${DATA_TYPE:-diter_os}
EVAL_SEQS=${EVAL_SEQS:-"Forest_new Lawn_lower_night Park_in_day"}
DEVICE=${DEVICE:-cuda:0}
WINDOW=${WINDOW:-200}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
RESULT_DIR=${RESULT_DIR:-${REPO_ROOT}/eval_results}

CKPT=${CKPT:?"set CKPT=path/to/best_model.ckpt"}

eval_seqs_arr=( $EVAL_SEQS )

cd "${REPO_ROOT}/src"

python3 evaluate.py \
    --data-root "${DATA_DIR}" \
    --data-type "${DATA_TYPE}" \
    --eval-seqs "${eval_seqs_arr[@]}" \
    --device "${DEVICE}" \
    --window-size "${WINDOW}" \
    --result-dir "${RESULT_DIR}" \
    --ckpt "${CKPT}" \
    --save-plot
