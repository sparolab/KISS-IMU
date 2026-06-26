#!/usr/bin/env bash
# Run ICP + PGO inference using a trained KISS-IMU checkpoint.
#
# Saves per-sequence trajectories (.npz + .png) under $OUT_DIR. This is for
# *trajectory artifacts*, not metrics — use scripts/evaluate.sh for RPE/APE.
#
# Override via env vars:
#   CKPT=/abs/path/to/best_model.ckpt
#   DATA_TYPE=diter_os  DATA_DIR=/storage/DiTer_os
#   SEQS="07 09"
#   bash scripts/inference.sh
set -euo pipefail

# ---- paths -----------------------------------------------------------------
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Required. Example:
#   CKPT="${REPO_ROOT}/results/<data_type>/<train_name>/.../best_model.ckpt"
CKPT=${CKPT:-}
if [[ -z "${CKPT}" ]]; then
    echo "[inference.sh] CKPT is not set. Pass it like:" >&2
    echo "    CKPT=/abs/path/to/best_model.ckpt bash scripts/inference.sh" >&2
    exit 1
fi

# ---- dataset ---------------------------------------------------------------
DATA_DIR=${DATA_DIR:-/storage1/Datasets/kiss_imu_datasets/DiTer_os}
DATA_TYPE=${DATA_TYPE:-diter_os}
SEQS=${SEQS:-"Forest_new"}

# ---- inference knobs -------------------------------------------------------
LO_MODEL=${LO_MODEL:-kiss_icp}
LM_WEIGHT=${LM_WEIGHT:-'(1,0.1,1,0.1,0.1)'}
USE_SUBMAP=${USE_SUBMAP:-false}
USE_ADAPTIVE_WEIGHT=${USE_ADAPTIVE_WEIGHT:-false}   # weight ICP by overlap,
                                                    # IMU by integrated cov

DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-5}
NUM_WORKERS=${NUM_WORKERS:-2}

# ---- output ----------------------------------------------------------------
OUT_DIR=${OUT_DIR:-$(dirname "${CKPT}")/inference}
mkdir -p "${OUT_DIR}"

seqs_arr=( $SEQS )

# ---- flag handling ---------------------------------------------------------
[[ "$USE_SUBMAP"          == "true" ]] && SUBMAP_ARG="--use-submap"          || SUBMAP_ARG=""
[[ "$USE_ADAPTIVE_WEIGHT" == "true" ]] && ADAPT_ARG="--use-adaptive-weight"  || ADAPT_ARG=""

# ---- run -------------------------------------------------------------------
cd "${REPO_ROOT}/src"
python3 inference.py \
    --ckpt "${CKPT}" \
    --data-root "${DATA_DIR}" \
    --data-type "${DATA_TYPE}" \
    --seqs ${seqs_arr[@]} \
    --lo-model ${LO_MODEL} \
    --lm-weight "${LM_WEIGHT}" \
    ${SUBMAP_ARG} \
    ${ADAPT_ARG} \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS}

echo "[inference.sh] done. out-dir: ${OUT_DIR}"
