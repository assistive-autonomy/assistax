#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# One reward-design training run per array task, one GPU each. Submit from the checkout root:
#   sbatch --array=0-<n-1> --job-name=rd-<round>-<sha> reward_design/slurm_round.sbatch \
#          <out root> <version> <steps> <seeds> <env[:seed]> [<env[:seed]> ...] [-- <extra run.py args>]
# Array task i trains entry i into <out root>/<env> (or <env>_s<seed>; run.log there). The clone's .venv must exist.
set -eu
root=$1 version=$2 steps=$3 seeds=$4
shift 4
entries=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do entries+=("$1"); shift; done
[ $# -gt 0 ] && shift
entry=${entries[$SLURM_ARRAY_TASK_ID]}
env=${entry%%:*}
seed_args=()
out="$root/$env"
if [ "$entry" != "$env" ]; then seed=${entry#*:}; seed_args=(--seed "$seed"); out="$root/${env}_s$seed"; fi
# Warm start: with INIT_ROOT set at submission, start from $INIT_ROOT/<env> (robot/human.safetensors)
if [ -n "${INIT_ROOT:-}" ]; then seed_args+=(--init "$INIT_ROOT/$env"); fi
mkdir -p "$out"
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "$(date +%H:%M:%S) $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} $entry $version $steps $seeds $*" > "$out/job.txt"
status=0
.venv/bin/python -m reward_design.run train "$env" "$version" "$out" --steps "$steps" --seeds "$seeds" \
    ${seed_args[@]+"${seed_args[@]}"} "$@" > "$out/run.log" 2>&1 || status=$?
echo "$(date +%H:%M:%S) exit $status" >> "$out/job.txt"
exit $status
