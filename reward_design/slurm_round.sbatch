#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# One reward-design training run per array task, one GPU each. Submit from the checkout root:
#   sbatch --array=0-<n-1> --job-name=rd-<round>-<sha> reward_design/slurm_round.sbatch \
#          <out root> <version> <steps> <seeds> <env> [<env> ...] [-- <extra run.py args>]
# Array task i trains env i into <out root>/<env> (run.log there); the clone's .venv must exist (uv sync).
set -eu
root=$1 version=$2 steps=$3 seeds=$4
shift 4
envs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
[ $# -gt 0 ] && shift
env=${envs[$SLURM_ARRAY_TASK_ID]}
out="$root/$env"
mkdir -p "$out"
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "$(date +%H:%M:%S) $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} $env $version $steps $seeds $*" > "$out/job.txt"
status=0
.venv/bin/python -m reward_design.run train "$env" "$version" "$out" --steps "$steps" --seeds "$seeds" "$@" \
    > "$out/run.log" 2>&1 || status=$?
echo "$(date +%H:%M:%S) exit $status" >> "$out/job.txt"
exit $status
