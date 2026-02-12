#!/bin/bash
# /pvc/scripts/run_ippo.sh
set -Eeuo pipefail

# --- Parse arguments ---
CONFIG=${1:-ippo_sweep}
ENV_NAME=${2:-bedbathing}
GPU_ENV_CAPACITY=${3:-49152} # For H200, for A100 80 GB 24576 and for 4090 8192


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

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95 # Set to .90 for A100 and 4090

cd /pvc/assistax
ulimit -n 10000

# --- Run training ---
uv run python assistax/baselines/IPPO/ippo_sweep.py \
    -cn $CONFIG -m \
    network=rnn_nps \
    ++NUM_SEEDS=6 \
    ++ENV_NAME=$ENV_NAME \
    ++TOTAL_TIMESTEPS=4e7 \
    ++NUM_MINIBATCHES=4,8,16 \
    ++UPDATE_EPOCHS=4,8,16 \
    "++SEED=range(0,5)" \
    SWEEP.num_configs=3 \
    GPU_ENV_CAPACITY=$GPU_ENV_CAPACITY 
