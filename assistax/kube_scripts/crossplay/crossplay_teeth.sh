#!/bin/bash
# /pvc/scripts/ippo_ff_teeth_sweep.sh
set -Eeuo pipefail

# --- Parse arguments ---
CONFIG=${1:-crossplay_zoo}
ENV_NAME=${2:-teethbrushing}
GPU_ENV_CAPACITY=${3:-49152} # For H200 49152, for A100 80 GB 24576 and for 4090 8192

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
mkdir -p /pvc/assistax/multirun
rm -rf multirun
ln -s /pvc/assistax/multirun multirun

echo "[$(ts)] Workspace: $WORK_DIR"
echo "[$(ts)] Outputs symlinked to: /pvc/assistax/multirun"

#echo "DEBUG: uv run python assistax/baselines/ZSC/crossplay_zoo.py \
#    -cn $CONFIG -m \
#    network=ff_nps \
#    ++NUM_SEEDS=10 \
#    ++NUM_EVAL_EPISODES=32 \
#    ++ENV_NAME=$ENV_NAME \
#    GPU_ENV_CAPACITY=$GPU_ENV_CAPACITY 

uv run python assistax/baselines/ZSC/crossplay_zoo.py \
    -cn $CONFIG -m \
    network=ff_nps \
    ++NUM_SEEDS=16 \
    ++NUM_EVAL_EPISODES=32 \
    ++ENV_NAME=$ENV_NAME \
    ++MAX_CROSSPLAY_PAIRS=32 \
    GPU_ENV_CAPACITY=$GPU_ENV_CAPACITY 

# --- Cleanup ---
rm -rf "$WORK_DIR"
