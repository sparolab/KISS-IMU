#!/usr/bin/env bash
# Evaluate one or more KISS-IMU checkpoints on a list of sequences.
#
# Two modes:
#   (A) point at a single .ckpt :  CKPT=path/to.ckpt
#   (B) point at a directory    :  CKPT_DIR=path/to/ckpt_dir
#
# Run from the repo root.
set -euo pipefail

DATA_DIR=${DATA_DIR:-/storage1/Datasets/kiss_imu_datasets/DiTer_os}
DATA_TYPE=${DATA_TYPE:-diter_os}
EVAL_SEQS=${EVAL_SEQS:-"Forest_new Lawn_lower_night Park_in_day"}
DEVICE=${DEVICE:-cuda:0}
WINDOW=${WINDOW:-200}
RESULT_DIR=${RESULT_DIR:-eval_results}

# choose ONE of the two:
CKPT=${CKPT:-}
CKPT_DIR=${CKPT_DIR:-}

SELECT_METRIC=${SELECT_METRIC:-balanced}      # ape | rpe_trans | rpe_rot | balanced
BALANCE_MODE=${BALANCE_MODE:-euclid}          # euclid | minimax | weighted
BALANCE_W=${BALANCE_W:-'(1.0,1.0,1.0)'}

if [[ -z "$CKPT" && -z "$CKPT_DIR" ]]; then
    echo "[evaluate.sh] either CKPT or CKPT_DIR must be set." >&2
    exit 1
fi

eval_seqs_arr=( $EVAL_SEQS )

cd "$(dirname "$0")/../src"

CMD=( python3 evaluate.py
      --data-root "${DATA_DIR}"
      --data-type "${DATA_TYPE}"
      --eval-seqs "${eval_seqs_arr[@]}"
      --device "${DEVICE}"
      --window-size "${WINDOW}"
      --result-dir "${RESULT_DIR}"
      --select-metric "${SELECT_METRIC}"
      --balance-mode  "${BALANCE_MODE}"
      --balance-weights "${BALANCE_W}"
      --save-plot
)

if [[ -n "$CKPT" ]]; then
    CMD+=( --ckpt "$CKPT" )
else
    CMD+=( --ckpt-dir "$CKPT_DIR" )
fi

"${CMD[@]}"
