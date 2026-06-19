#!/usr/bin/env bash
# Run the "†" baseline used in the paper:
# raw IMU -> PVGO with no learned correction, no GMM, no freq-gate.
#
# Edit the variables below or override via environment.
set -euo pipefail

DATA_DIR=${DATA_DIR:-/storage1/Datasets/kiss_imu_datasets/DiTer_os}
DATA_TYPE=${DATA_TYPE:-diter_os}
TRAIN_SEQS=${TRAIN_SEQS:-"Forest_new"}
VALID_SEQS=${VALID_SEQS:-"Forest_new"}
INFERENCE_SEQS=${INFERENCE_SEQS:-"Forest_new"}
LO_MODEL=${LO_MODEL:-kiss_icp}

LR=${LR:-1e-05}
EPOCH=${EPOCH:-30}
BATCH_SIZE=${BATCH_SIZE:-5}
DEVICE=${DEVICE:-cuda:0}

LM_WEIGHT=${LM_WEIGHT:-'(1,0.1,1,0.1,0.1)'}
ROT_W=${ROT_W:-1e3}; VEL_W=${VEL_W:-1e0}; POS_W=${POS_W:-1e2}
COV_R_W=${COV_R_W:-1e-4}; COV_V_W=${COV_V_W:-1e-5}; COV_T_W=${COV_T_W:-1e-5}
ROT_COV_S=${ROT_COV_S:-1e-3}; VEL_COV_S=${VEL_COV_S:-1e-3}; POS_COV_S=${POS_COV_S:-1e-3}

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
RESULT_BASE=${RESULT_BASE:-${REPO_ROOT}/results_raw}
mkdir -p "$RESULT_BASE"

train_seqs_arr=( $TRAIN_SEQS )
valid_seqs_arr=( $VALID_SEQS )
infer_seqs_arr=( $INFERENCE_SEQS )

cd "${REPO_ROOT}/src"
python3 raw_pvgo.py \
    --result-dir "${RESULT_BASE}" \
    --data-type ${DATA_TYPE} \
    --batch-size ${BATCH_SIZE} \
    --epoch ${EPOCH} \
    --worker-num 2 \
    --data-root ${DATA_DIR} \
    --train-seqs ${train_seqs_arr[@]} \
    --valid-seqs ${valid_seqs_arr[@]} \
    --inference-seqs ${infer_seqs_arr[@]} \
    --lm-weight ${LM_WEIGHT} \
    --lr ${LR} \
    --device ${DEVICE} \
    --rot-w ${ROT_W} --vel-w ${VEL_W} --pos-w ${POS_W} \
    --cov-r-w ${COV_R_W} --cov-v-w ${COV_V_W} --cov-t-w ${COV_T_W} \
    --rot-cov-scaler ${ROT_COV_S} --vel-cov-scaler ${VEL_COV_S} --pos-cov-scaler ${POS_COV_S} \
    --lo-model ${LO_MODEL} \
    --no-adaptive-weight

echo "[raw_pvgo.sh] done. result-dir: ${RESULT_BASE}"
