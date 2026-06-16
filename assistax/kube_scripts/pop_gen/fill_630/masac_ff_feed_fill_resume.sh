#!/bin/bash
# /pvc/scripts/run_zoo_gen.sh
# RESUME for a partial feeding MASAC fill shard: runs ONLY seeds 27 & 28 (the two
# invocations that didn't complete). Seed 26's 20 teams are already in the shard, so
# this appends the remaining 40 teams -> 60 total, with no duplicates and no recompute.
# Writes to the SAME shard dir as masac_ff_feed_fill.sh. Run only if that shard still
# has just seed 26 (40 rows); delete the shard and use the full script instead otherwise.
set -Eeuo pipefail

CONFIG=${1:-masac_zoo_gen}
ENV_NAME=${2:-feeding}
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
# XLA writes compiled-kernel (CUBIN) temp files to $TMPDIR; the container's default
# /tmp is small and the heavy feeding (NUM_SAC_UPDATES=128, BATCH_SIZE=512) compiles
# exhaust it ("couldn't get temp CUBIN file name"). Point it at the roomy PVC workspace.
export TMPDIR="$WORK_DIR/xla-tmp"
mkdir -p "$TMPDIR"

# --- Per-job zoo directory (NOT shared; SAME shard as masac_ff_feed_fill.sh) ---
JOB_ZOO_DIR="/pvc/zoo_shards/${ENV_NAME}/${CONFIG}_fill630/zoo"
mkdir -p "$JOB_ZOO_DIR"
echo "[$(ts)] Zoo shard: $JOB_ZOO_DIR"

# --- Run zoo generation (seeds 27 & 28 only) ---
uv run --extra cuda12 python assistax/baselines/MASAC/masac_zoo_gen.py \
    -cn $CONFIG -m\
    ++ENV_NAME=$ENV_NAME \
    ++ZOO_PATH=$JOB_ZOO_DIR \
    ++PREFERENCE_SWEEP.num_configs=20 \
    ++TOTAL_TIMESTEPS=10e6 \
    ++SEED=27 \
    ++POLICY_LR=0.0000562 \
    ++Q_LR=0.00178 \
    ++ALPHA_LEARNING_RATE=0.000252\
    ++TAU=0.0002728487 \
    ++NUM_SAC_UPDATES=128 \
    ++ROLLOUT_LENGTH=8 \
    ++BATCH_SIZE=512 \
    ++ENV_KWARGS.disability.joint_restriction_factor=0 \
    ++ENV_KWARGS.disability.joint_strength=1 \
    ++ENV_KWARGS.preference_rewards.reward_budget=3.11

uv run --extra cuda12 python assistax/baselines/MASAC/masac_zoo_gen.py \
    -cn $CONFIG -m\
    ++ENV_NAME=$ENV_NAME \
    ++ZOO_PATH=$JOB_ZOO_DIR \
    ++PREFERENCE_SWEEP.num_configs=20 \
    ++TOTAL_TIMESTEPS=10e6 \
    ++SEED=28 \
    ++POLICY_LR=0.0000562 \
    ++Q_LR=0.00178 \
    ++ALPHA_LEARNING_RATE=0.000252\
    ++TAU=0.0002728487 \
    ++NUM_SAC_UPDATES=128 \
    ++ROLLOUT_LENGTH=8 \
    ++BATCH_SIZE=512 \
    ++ENV_KWARGS.disability.joint_restriction_factor=0 \
    ++ENV_KWARGS.disability.joint_strength=0.5 \
    ++ENV_KWARGS.preference_rewards.reward_budget=3.11

# --- Cleanup workspace (but NOT the zoo shard) ---
rm -rf "$WORK_DIR"
echo "[$(ts)] Done. Zoo shard preserved at: $JOB_ZOO_DIR"
