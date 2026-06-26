#!/bin/bash
# /pvc/scripts/run_zoo_gen.sh
# FILL-TO-630: bedbathing MAPPO top-up (+210 humans). Mirrors mappo_ff_bb_zoo.sh;
# only SEED (fresh 23/24/25) and the per-job shard dir differ. Seeds 23/24/25 are
# shared with the feeding/teeth MAPPO fills, so those three get the SAME 210 humans.
set -Eeuo pipefail

CONFIG=${1:-mappo_zoo_gen}
ENV_NAME=${2:-bedbathing}
GPU_ENV_CAPACITY=${3:-49152}

ts(){ date +'%Y-%m-%dT%H:%M:%S%z'; }
mkdir -p /pvc/job-logs
LOG_FILE="/pvc/job-logs/${POD_NAME}-$(ts).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(ts)] Job: $POD_NAME"
echo "[$(ts)] Config: $CONFIG, Env: $ENV_NAME"

on_err(){
    ec=$?
    echo "[$(ts)] ERROR exit=$ec at line $LINENO: $BASH_COMMAND"
    sleep infinity
}
trap on_err ERR

# --- Workspace isolation (same as before) ---
WORK_DIR="/pvc/tmp/${POD_NAME}"
mkdir -p "$WORK_DIR"
cp -r /pvc/assistax "$WORK_DIR/"
cd "$WORK_DIR/assistax"
export PYTHONPATH="$WORK_DIR/assistax:${PYTHONPATH:-}"
export UV_CACHE_DIR=/pvc/.uv-cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9

# --- Per-job zoo directory (NOT shared; isolated fill shard) ---
JOB_ZOO_DIR="/pvc/zoo_shards/${ENV_NAME}/${CONFIG}_fill630/zoo"
mkdir -p "$JOB_ZOO_DIR"
echo "[$(ts)] Zoo shard: $JOB_ZOO_DIR"

# --- Run zoo generation ---
uv run --extra cuda12 python assistax/baselines/MAPPO/mappo_zoo_gen.py \
    -cn $CONFIG -m\
    ++ENV_NAME=$ENV_NAME \
    ++ZOO_PATH=$JOB_ZOO_DIR \
    ++PREFERENCE_SWEEP.num_configs=70 \
    ++SEED=23 \
    ++LR=0.00112 \
    ++UPDATE_EPOCHS=16 \
    ++NUM_MINIBATCHES=8 \
    ++CLIP_EPS=0.14979327 \
    ++ENT_COEF=0.0029010084 \
    ++NUM_STEPS=128 \
    ++ENV_KWARGS.disability.joint_restriction_factor=0 \
    ++ENV_KWARGS.disability.joint_strength=0.5 \
    ++ENV_KWARGS.preference_rewards.reward_budget=1


uv run --extra cuda12 python assistax/baselines/MAPPO/mappo_zoo_gen.py \
    -cn $CONFIG -m\
    ++ENV_NAME=$ENV_NAME \
    ++ZOO_PATH=$JOB_ZOO_DIR \
    ++PREFERENCE_SWEEP.num_configs=70 \
    ++SEED=24 \
    ++LR=0.00154 \
    ++UPDATE_EPOCHS=16 \
    ++NUM_MINIBATCHES=4 \
    ++CLIP_EPS=0.2705661 \
    ++ENT_COEF=0.000794748 \
    ++NUM_STEPS=128 \
    ++ENV_KWARGS.disability.joint_restriction_factor=0.5 \
    ++ENV_KWARGS.disability.joint_strength=1 \
    ++ENV_KWARGS.preference_rewards.reward_budget=1

uv run --extra cuda12 python assistax/baselines/MAPPO/mappo_zoo_gen.py \
    -cn $CONFIG -m\
    ++ENV_NAME=$ENV_NAME \
    ++ZOO_PATH=$JOB_ZOO_DIR \
    ++PREFERENCE_SWEEP.num_configs=70 \
    ++SEED=25 \
    ++LR=0.00154 \
    ++UPDATE_EPOCHS=16 \
    ++NUM_MINIBATCHES=4 \
    ++CLIP_EPS=0.2705661 \
    ++ENT_COEF=0.000794748 \
    ++NUM_STEPS=128 \
    ++ENV_KWARGS.disability.joint_restriction_factor=1 \
    ++ENV_KWARGS.disability.joint_strength=0.5 \
    ++ENV_KWARGS.preference_rewards.reward_budget=1

# --- Cleanup workspace (but NOT the zoo shard) ---
rm -rf "$WORK_DIR"
echo "[$(ts)] Done. Zoo shard preserved at: $JOB_ZOO_DIR"
