#!/bin/bash
# /pvc/scripts/run_ippo.sh
set -Eeuo pipefail

apt-get update && apt-get install -y --no-install-recommends \
    libegl1-mesa libegl-dev libgles2-mesa-dev ffmpeg\
  && rm -rf /var/lib/apt/lists/*

export NETRC=/pvc/.netrc
# --- Parse arguments ---
CONFIG=${1:-ippo}
ENV_NAME=${2:-scratchitch}
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

# --- Run 1: No preference rewards ---
#uv run python assistax/baselines/IPPO/ippo_run.py \
#    -cn $CONFIG -m \
#    network=rnn_nps \
#    ++NUM_SEEDS=8 \
#    ++SEED=0,1 \
#    ++ENV_NAME=$ENV_NAME \
#    ++TOTAL_TIMESTEPS=4e7 \
#    ++GPU_ENV_CAPACITY=$GPU_ENV_CAPACITY \
#    ++LR=0.000439 \
#    ++UPDATE_EPOCHS=8 \
#    ++NUM_MINIBATCHES=4 \
#    ++CLIP_EPS=0.04558986 \
#    ++ENT_COEF=0.000545135 \
#    ++NUM_STEPS=64 \
#    ++EXP_TAGS=[IPPO,RNN_NPS,MARL_FINAL]

echo "[$(ts)] Run 1 (no pref) complete. Starting Run 2 (with pref)..."

# --- Run 2: With preference rewards ---
uv run python assistax/baselines/IPPO/ippo_run.py \
    -cn $CONFIG -m \
    network=rnn_nps \
    ++NUM_SEEDS=8 \
    ++SEED=0,1 \
    ++ENV_NAME=$ENV_NAME \
    ++TOTAL_TIMESTEPS=4e7 \
    ++GPU_ENV_CAPACITY=$GPU_ENV_CAPACITY \
    ++LR=0.000439 \
    ++UPDATE_EPOCHS=8 \
    ++NUM_MINIBATCHES=4 \
    ++CLIP_EPS=0.04558986 \
    ++ENT_COEF=0.000545135 \
    ++NUM_STEPS=64 \
    '++ENV_KWARGS.preference_rewards={preference_weights:{speed_preference:0.25,force_preference:0.35,touch_penalty:-0.03},preference_ranges:{speed_range:[0.06,0.14],force_range:[1.5,3.5]},touch_threshold:0.1,reward_budget:1,overall_weight:1,normalization_mode:budget,variable_names:{speed:ee_speed,force:ee_force,action_magnitude:action_magnitude,contact_forces:contact_forces}}' \
    ++EXP_TAGS=[IPPO,RNN_NPS,MARL_FINAL,PREF]

# --- Cleanup ---
rm -rf "$WORK_DIR"
