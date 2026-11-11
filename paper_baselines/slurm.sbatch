#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# The paper's MARL baselines (paper_baselines/run.py), one algorithm, task and seed block per array task, one GPU each.
# Submit from the checkout root (its .venv must have CUDA jax):
#   sbatch --array=0-<n-1> --job-name=pb-<sha> paper_baselines/slurm.sbatch <out root> <seeds per task> <capacity> \
#          <alg:env:seed> [<alg:env:seed> ...]
# Array task i trains entry i (NUM_SEEDS=<seeds per task>, SEED=<seed>) into <out root>/<alg>_<env>/seed<seed>.
set -eu
root=$1 seeds=$2 capacity=$3
shift 3
entry=${@:$((SLURM_ARRAY_TASK_ID + 1)):1}
IFS=: read -r alg env seed <<< "$entry"
out="$root/${alg}_${env}/seed$seed"
mkdir -p "$out" "$root/tmp"
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.9} TMPDIR="$root/tmp"
echo "$(date '+%F %T') $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} $(pwd) $(git rev-parse --short HEAD) $entry" \
     "seeds=$seeds capacity=$capacity" > "$out/job.txt"
status=0
.venv/bin/python -m paper_baselines.run "$alg" "$env" "$out" --seeds "$seeds" --seed "$seed" --capacity "$capacity" \
    > "$out/run.log" 2>&1 || status=$?
echo "$(date '+%F %T') exit $status" >> "$out/job.txt"
exit $status
