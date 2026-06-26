#!/usr/bin/env bash
# Train KISS-IMU (GMM + balance-aware + frequency gate).
#
# Edit the variables below or override them via environment, e.g.:
#   DATA_TYPE=diter_os DATA_DIR=/storage/DiTer_os bash scripts/train.sh
#
# Set USE_GT=true to train against GT poses instead of the ICP/PGO pseudo-
# label. ICP and PGO are then skipped entirely, so iterations are much
# faster — useful for isolating the contribution of GMM reweighting from
# the LO pseudo-label.
#
# Run from the repo root.
set -euo pipefail

# ---- dataset selection -----------------------------------------------------
DATA_DIR=${DATA_DIR:-/storage1/Datasets/kiss_imu_datasets/DiTer_os}
DATA_TYPE=${DATA_TYPE:-diter_os}                   # diter_os | diter++ (add your own in data/seq_dataset.py)
TRAIN_SEQS=${TRAIN_SEQS:-"Forest_new"}
VALID_SEQS=${VALID_SEQS:-"Forest_new"}
LO_MODEL=${LO_MODEL:-kiss_icp}                     # kiss_icp|fast_gicp|small_gicp

# ---- training hyper-params -------------------------------------------------
LR=${LR:-1e-05}
EPOCH=${EPOCH:-30}
BATCH_SIZE=${BATCH_SIZE:-5}
DEVICE=${DEVICE:-cuda:0}

LM_WEIGHT=${LM_WEIGHT:-'(1,0.1,1,0.1,0.1)'}
ROT_W=${ROT_W:-1e3}; VEL_W=${VEL_W:-1e0}; POS_W=${POS_W:-1e0}
COV_R_W=${COV_R_W:-1e-4}; COV_V_W=${COV_V_W:-1e-5}; COV_T_W=${COV_T_W:-1e-5}
ROT_COV_S=${ROT_COV_S:-1e-3}; VEL_COV_S=${VEL_COV_S:-1e-3}; POS_COV_S=${POS_COV_S:-1e-3}

TRAIN_RATIO=${TRAIN_RATIO:-1.0}
GMM_COMP_NUM=${GMM_COMP_NUM:-0}                    # 0 = auto-pick K via BIC
USE_VALIDATION=${USE_VALIDATION:-true}
USE_SUBMAP=${USE_SUBMAP:-false}
USE_GT=${USE_GT:-false}                            # true = GT supervision (no ICP/PGO)

# ---- output layout ---------------------------------------------------------
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)         # absolute path to repo root
RESULT_BASE=${RESULT_BASE:-${REPO_ROOT}/results}    # absolute by default so the
                                                    # output stays at repo root
                                                    # regardless of cwd later
if [[ "$USE_GT" == "true" ]]; then
    TRAIN_NAME="GT+GMM_bs=${BATCH_SIZE}_lr=${LR}_rw=${ROT_W}_vw=${VEL_W}_tw=${POS_W}_crw=${COV_R_W}_cvw=${COV_V_W}_ctw=${COV_T_W}_rcs=${ROT_COV_S}_vcs=${VEL_COV_S}_pcs=${POS_COV_S}_K=${GMM_COMP_NUM}"
else
    TRAIN_NAME="bs=${BATCH_SIZE}_lr=${LR}_rw=${ROT_W}_vw=${VEL_W}_tw=${POS_W}_crw=${COV_R_W}_cvw=${COV_V_W}_ctw=${COV_T_W}_rcs=${ROT_COV_S}_vcs=${VEL_COV_S}_pcs=${POS_COV_S}_K=${GMM_COMP_NUM}"
fi

train_seqs_arr=( $TRAIN_SEQS ); valid_seqs_arr=( $VALID_SEQS )
train_seqs_str=$(IFS='_' ; echo "${train_seqs_arr[*]}")
valid_seqs_str=$(IFS='_' ; echo "${valid_seqs_arr[*]}")

if [[ "$USE_GT" == "true" ]]; then
    # GT supervision skips LO entirely, so leaving LO_MODEL out of the path
    # is honest — and the "use_gt" segment keeps these runs from colliding
    # with regular pseudo-label runs.
    RESULT_DIR="${RESULT_BASE}/${DATA_TYPE}/use_gt/${LM_WEIGHT}/${TRAIN_NAME}/${train_seqs_str}_valid_${valid_seqs_str}/${TRAIN_RATIO}"
else
    RESULT_DIR="${RESULT_BASE}/${DATA_TYPE}/${LM_WEIGHT}/${TRAIN_NAME}/${LO_MODEL}/${train_seqs_str}_valid_${valid_seqs_str}/${TRAIN_RATIO}"
fi
mkdir -p "$RESULT_DIR"

# ---- flag handling ---------------------------------------------------------
# When validation is off, omit --valid-seqs entirely (argparse nargs='+'
# rejects the flag with no values), rather than passing an empty flag.
[[ "$USE_VALIDATION" == "true" ]] && VALID_ARG="--valid-seqs ${valid_seqs_arr[@]}" || VALID_ARG=""
[[ "$USE_SUBMAP"    == "true" ]] && SUBMAP_ARG="--use-submap" || SUBMAP_ARG="--no-submap"
[[ "$USE_GT"        == "true" ]] && USE_GT_ARG="--use-gt"     || USE_GT_ARG=""

# ---- run -------------------------------------------------------------------
cd "$(dirname "$0")/../src"
python3 train.py \
    --result-dir "${RESULT_DIR}" \
    --data-type ${DATA_TYPE} \
    --train-name "${TRAIN_NAME}" \
    --pretrained-model None \
    --batch-size ${BATCH_SIZE} \
    --epoch ${EPOCH} \
    --worker-num 2 \
    --data-root ${DATA_DIR} \
    --train-seqs ${train_seqs_arr[@]} \
    ${VALID_ARG} \
    --lm-weight ${LM_WEIGHT} \
    --lr ${LR} \
    --device ${DEVICE} \
    --rot-w ${ROT_W} --vel-w ${VEL_W} --pos-w ${POS_W} \
    --cov-r-w ${COV_R_W} --cov-v-w ${COV_V_W} --cov-t-w ${COV_T_W} \
    --rot-cov-scaler ${ROT_COV_S} --vel-cov-scaler ${VEL_COV_S} --pos-cov-scaler ${POS_COV_S} \
    --lo-model ${LO_MODEL} \
    --no-adaptive-weight \
    ${SUBMAP_ARG} \
    --train-ratio ${TRAIN_RATIO} \
    --gmm-comp-num ${GMM_COMP_NUM} \
    ${USE_GT_ARG}

echo "[train.sh] done. result-dir: ${RESULT_DIR}"
