#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# v2 population blocks (population/run_block.py), one block per array task, one GPU each.
# Submit from the checkout root (its .venv must have CUDA jax):
#   sbatch --array=0-<n-1>%<max at once> --job-name=pop-<sha> population/slurm_blocks.sbatch <out root> \
#          <task>:<first>-<last> [<task>:<first>-<last> ...]
# Array task i trains entry i into <out root>/<task>/<block>/ and logs to <out root>/logs/. RUN_ARGS (e.g.
# "--smoke -- ++TOTAL_TIMESTEPS=2e5") is passed to run_block, for timing runs only.
set -eu
root=$1
shift
entry=${@:$((SLURM_ARRAY_TASK_ID + 1)):1}
IFS=: read -r task persons <<< "$entry"
mkdir -p "$root/logs" "$root/tmp"
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.9} TMPDIR="$root/tmp"
log="$root/logs/$task-$persons-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
echo "$(date '+%F %T') $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} $(pwd) $(git rev-parse --short HEAD) $entry" > "$log"
status=0
# shellcheck disable=SC2086
.venv/bin/python -m population.run_block population/manifest_v2.json "$task" "$persons" "$root" ${RUN_ARGS:-} \
    >> "$log" 2>&1 || status=$?
echo "$(date '+%F %T') exit $status" >> "$log"
exit $status
