#!/bin/bash
# /pvc/scripts/ippo_ff_teeth_sweep.sh
set -Eeuo pipefail

apt-get update && apt-get install -y --no-install-recommends \
    libegl1-mesa libegl-dev libgles2-mesa-dev ffmpeg\
  && rm -rf /var/lib/apt/lists/*

export NETRC=/pvc/.netrc

# --- Parse arguments ---
CONFIG=${1:-ppo_aht}
ENV_NAME=${2:-bedbathing}
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
cp -r /pvc/assistax "$WORK_DIR/"
cd "$WORK_DIR/assistax"

export PYTHONPATH="$WORK_DIR/assistax:${PYTHONPATH:-}"
export UV_CACHE_DIR=/pvc/.uv-cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95 # Set to .90 for A100 and 4090

JOB_ZOO_DIR="$WORK_DIR/assistax/zoo"

# --- Symlink Hydra output dirs to persistent storage ---
mkdir -p /pvc/assistax/outputs
rm -rf outputs
ln -s /pvc/assistax/outputs outputs

echo "[$(ts)] Workspace: $WORK_DIR"
echo "[$(ts)] Outputs symlinked to: /pvc/assistax/outputs"

#echo "DEBUG: uv run python assistax/baselines/ZSC/crossplay_zoo.py \
#    -cn $CONFIG -m \
#    network=ff_nps \
#    ++NUM_SEEDS=10 \
#    ++NUM_EVAL_EPISODES=32 \
#    ++ENV_NAME=$ENV_NAME \
#    GPU_ENV_CAPACITY=$GPU_ENV_CAPACITY

uv run python assistax/baselines/ZSC/ppo_aht.py \
    -cn $CONFIG -m \
    network=ff_nps \
    ++ENV_NAME=$ENV_NAME \
    ++ZOO_PATH=$JOB_ZOO_DIR \
    GPU_ENV_CAPACITY=$GPU_ENV_CAPACITY \
    ++LR=0.000893 \
    ++UPDATE_EPOCHS=16 \
    ++NUM_MINIBATCHES=8 \
    ++CLIP_EPS=0.041964713 \
    ++ENT_COEF=0.0000638 \
    ++NUM_STEPS=64 \
    ++NUM_SEEDS=16 \
    ++EXP_TAGS=[IPPO,FF_NPS,AHT_FINAL]

# --- Cleanup ---
rm -rf "$WORK_DIR"
