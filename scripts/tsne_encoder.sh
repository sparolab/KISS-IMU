#!/usr/bin/env bash
# t-SNE of encoder features colored by GMM component (motion regime).
#
# Loads a trained best_model.ckpt + the GMM saved next to it
# (gmm.joblib), runs the encoder over the chosen sequences, and saves
# tsne_train.png / tsne_eval.png under $OUT_DIR.
#
# Override via env vars:
#   CKPT=...          (required if not the default)
#   DATA_TYPE=diter_os   DATA_DIR=/storage/DiTer_os
#   TRAIN_SEQS="Forest_new"   EVAL_SEQS="Park_in_day"
#   bash scripts/tsne_encoder.sh
set -euo pipefail

# ---- paths -----------------------------------------------------------------
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# Required. Example:
#   CKPT="${REPO_ROOT}/results/<data_type>/<train_name>/.../best_model.ckpt"
CKPT=${CKPT:-}
if [[ -z "${CKPT}" ]]; then
    echo "[tsne_encoder.sh] CKPT is not set. Pass it like:" >&2
    echo "    CKPT=/abs/path/to/best_model.ckpt bash scripts/tsne_encoder.sh" >&2
    exit 1
fi
GMM=${GMM:-}                                              # default: <ckpt-dir>/gmm.joblib

# ---- dataset ---------------------------------------------------------------
DATA_DIR=${DATA_DIR:-/storage1/Datasets/kiss_imu_datasets/DiTer_os}
DATA_TYPE=${DATA_TYPE:-diter_os}
TRAIN_SEQS=${TRAIN_SEQS:-"Forest_new"}
EVAL_SEQS=${EVAL_SEQS:-"Forest_new"}

# ---- runtime knobs ---------------------------------------------------------
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-2}
PERPLEXITY=${PERPLEXITY:-30}
MAX_WINDOWS=${MAX_WINDOWS:-5000}
PER_COMP=${PER_COMP:-10}                        # 0 = auto (see --per-comp help)
MIN_PER_COMP=${MIN_PER_COMP:-10}
PAIR_MODE=${PAIR_MODE:-farthest}                # all | farthest
SEED=${SEED:-0}

# ---- output ----------------------------------------------------------------
OUT_DIR=${OUT_DIR:-$(dirname "${CKPT}")/tsne}
mkdir -p "${OUT_DIR}"

train_seqs_arr=( $TRAIN_SEQS ); eval_seqs_arr=( $EVAL_SEQS )

# ---- run -------------------------------------------------------------------
cd "${REPO_ROOT}/src"

GMM_ARG=""
[[ -n "${GMM}" ]] && GMM_ARG="--gmm ${GMM}"

python3 tsne_encoder.py \
    --ckpt "${CKPT}" \
    ${GMM_ARG} \
    --data-root "${DATA_DIR}" \
    --data-type "${DATA_TYPE}" \
    --train-seqs ${train_seqs_arr[@]} \
    --eval-seqs ${eval_seqs_arr[@]} \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --perplexity ${PERPLEXITY} \
    --max-windows ${MAX_WINDOWS} \
    --per-comp ${PER_COMP} \
    --min-per-comp ${MIN_PER_COMP} \
    --pair-mode ${PAIR_MODE} \
    --seed ${SEED}

echo "[tsne_encoder.sh] done. out-dir: ${OUT_DIR}"
