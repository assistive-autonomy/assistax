#!/bin/bash
# /pvc/scripts/run_ippo.sh
set -Eeuo pipefail

apt-get update && apt-get install -y --no-install-recommends \
    libegl1-mesa libegl-dev libgles2-mesa-dev \
  && rm -rf /var/lib/apt/lists/*

export NETRC=/pvc/.netrc
# --- Parse arguments ---
CONFIG=${1:-ippo}
ENV_NAME=${2:-bedbathing}
GPU_ENV_CAPACITY=${3:-49152}

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
cp -r /pvc/assistax "$WORK_DIR/"
cd "$WORK_DIR/assistax"

export PYTHONPATH="$WORK_DIR/assistax:${PYTHONPATH:-}"
export UV_CACHE_DIR=/pvc/.uv-cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95 # Set to .90 for A100 and 4090

# --- Symlink Hydra output dirs to persistent storage ---
mkdir -p /pvc/assistax/outputs
rm -rf outputs
ln -s /pvc/assistax/outputs outputs

echo "[$(ts)] Workspace: $WORK_DIR"
echo "[$(ts)] Outputs symlinked to: /pvc/assistax/outputs"

# --- Run training ---
uv run python assistax/baselines/IPPO/ippo_run.py \
    -cn $CONFIG -m \
    network=rnn_nps \
    ++NUM_SEEDS=16 \
    ++ENV_NAME=$ENV_NAME \
    ++TOTAL_TIMESTEPS=4e7 \
    ++GPU_ENV_CAPACITY=$GPU_ENV_CAPACITY \
    ++LR=0.00289 \
    ++UPDATE_EPOCHS=16 \
    ++NUM_MINIBATCHES=4 \
    ++CLIP_EPS=0.06623137 \
    ++ENT_COEF=0.0005725827 \
    ++NUM_STEPS=64 \
    ++EXP_TAGS=[IPPO,RNN_NPS,MARL_FINAL]

# --- Cleanup ---
rm -rf "$WORK_DIR"
