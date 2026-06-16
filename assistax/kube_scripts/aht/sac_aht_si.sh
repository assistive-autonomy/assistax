#!/bin/bash
# /pvc/scripts/sac_aht_si.sh  (SAC ad-hoc teamwork, scratchitch)
set -Eeuo pipefail
apt-get update && apt-get install -y --no-install-recommends \
    libegl1-mesa libegl-dev libgles2-mesa-dev ffmpeg rsync\
    && rm -rf /var/lib/apt/lists/*
export NETRC=/pvc/.netrc
# --- Parse arguments ---
CONFIG=${1:-sac_aht}
ENV_NAME=${2:-scratchitch}
GPU_ENV_CAPACITY=${3:-24576} # For H200 49152, for A100 80 GB 24576 and for 4090 8192
# --- Logging setup ---
ts(){ date +'%Y-%m-%dT%H:%M:%S%z'; }
mkdir -p /pvc/job-logs
LOG_FILE="/pvc/job-logs/${POD_NAME}-$(ts).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(ts)] Job: $POD_NAME"
echo "[$(ts)] Config: $CONFIG, Env: $ENV_NAME"
# --- Error handler ---
on_err(){
    ec=$?
    echo "[$(ts)] ERROR exit=$ec at line $LINENO: $BASH_COMMAND"
    sleep infinity
}
trap on_err ERR
# --- Workspace isolation ---
WORK_DIR="/pvc/tmp/${POD_NAME}"
mkdir -p "$WORK_DIR"
# Copy source code only, exclude heavy zoo directory (read-only, shared safely via RWX PVC)
rsync -a --exclude='zoo' /pvc/assistax/ "$WORK_DIR/assistax/"
# Symlink zoo back so code sees it in the expected location
ln -s /pvc/assistax/zoo "$WORK_DIR/assistax/zoo"
cd "$WORK_DIR/assistax"
export PYTHONPATH="$WORK_DIR/assistax:${PYTHONPATH:-}"
export UV_CACHE_DIR=/pvc/.uv-cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95 # Set to .90 for A100 and 4090
JOB_ZOO_DIR="/pvc/assistax/zoo"
# --- Symlink Hydra output dirs to persistent storage ---
mkdir -p /pvc/assistax/outputs
rm -rf outputs
ln -s /pvc/assistax/outputs outputs
echo "[$(ts)] Workspace: $WORK_DIR"
echo "[$(ts)] Zoo: $JOB_ZOO_DIR (shared, read-only)"
echo "[$(ts)] Outputs symlinked to: /pvc/assistax/outputs"

uv run --extra cuda12 python assistax/baselines/ZSC/sac_aht.py \
    -cn $CONFIG -m \
    network=ff_nps \
    ++ENV_NAME=$ENV_NAME \
    ++ZOO_PATH=$JOB_ZOO_DIR \
    GPU_ENV_CAPACITY=$GPU_ENV_CAPACITY \
    ++POLICY_LR=0.0000562 \
    ++Q_LR=0.00178 \
    ++ALPHA_LR=0.000252 \
    ++TAU=0.0002728487 \
    ++NUM_SAC_UPDATES=32 \
    ++ROLLOUT_LENGTH=8 \
    ++BATCH_SIZE=512 \
    ++NUM_SEEDS=16 \
    ++EXP_TAGS=[MASAC,FF_NPS,AHT_FINAL]
# --- Cleanup ---
rm -rf "$WORK_DIR"
