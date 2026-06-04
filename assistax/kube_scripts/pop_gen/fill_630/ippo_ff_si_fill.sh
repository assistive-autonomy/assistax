#!/bin/bash
# /pvc/scripts/run_zoo_gen.sh
# FILL-TO-630: scratchitch IPPO top-up (+210 humans). Mirrors ippo_ff_si_zoo.sh;
# only SEED (fresh 20/21/22), and the per-job shard dir differ. Combos depend only
# on (SEED, num_configs, sweep ranges), so fresh seeds = new humans (preflight-checked).
set -Eeuo pipefail

CONFIG=${1:-ippo_zoo_gen}
ENV_NAME=${2:-scratchitch}
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
uv run python assistax/baselines/IPPO/ippo_zoo_gen.py \
    -cn $CONFIG -m\
    ++ENV_NAME=$ENV_NAME \
    ++ZOO_PATH=$JOB_ZOO_DIR \
    ++PREFERENCE_SWEEP.num_configs=70 \
    ++SEED=20 \
    ++LR=0.000334 \
    ++UPDATE_EPOCHS=8 \
    ++NUM_MINIBATCHES=16 \
    ++CLIP_EPS=0.16875672 \
    ++ENT_COEF=0.0016069901 \
    ++NUM_STEPS=64 \
    ++ENV_KWARGS.disability.joint_restriction_factor=0 \
    ++ENV_KWARGS.disability.joint_strength=0


uv run python assistax/baselines/IPPO/ippo_zoo_gen.py \
    -cn $CONFIG -m\
    ++ENV_NAME=$ENV_NAME \
    ++ZOO_PATH=$JOB_ZOO_DIR \
    ++PREFERENCE_SWEEP.num_configs=70 \
    ++SEED=21 \
    ++LR=0.000334 \
    ++UPDATE_EPOCHS=8 \
    ++NUM_MINIBATCHES=16 \
    ++CLIP_EPS=0.16875672 \
    ++ENT_COEF=0.0016069901 \
    ++NUM_STEPS=64 \
    ++ENV_KWARGS.disability.joint_restriction_factor=0.5 \
    ++ENV_KWARGS.disability.joint_strength=0.5

uv run python assistax/baselines/IPPO/ippo_zoo_gen.py \
    -cn $CONFIG -m\
    ++ENV_NAME=$ENV_NAME \
    ++ZOO_PATH=$JOB_ZOO_DIR \
    ++PREFERENCE_SWEEP.num_configs=70 \
    ++SEED=22 \
    ++LR=0.000334 \
    ++UPDATE_EPOCHS=8 \
    ++NUM_MINIBATCHES=16 \
    ++CLIP_EPS=0.16875672 \
    ++ENT_COEF=0.0016069901 \
    ++NUM_STEPS=64 \
    ++ENV_KWARGS.disability.joint_restriction_factor=1 \
    ++ENV_KWARGS.disability.joint_strength=1

# --- Cleanup workspace (but NOT the zoo shard) ---
rm -rf "$WORK_DIR"
echo "[$(ts)] Done. Zoo shard preserved at: $JOB_ZOO_DIR"
