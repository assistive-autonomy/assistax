#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# Human-involvement audit runs: train and record (audit_run.py), one task and seed per array task, one GPU each.
# Submit from the checkout whose assistax should run (its .venv must exist), e.g. a pre-0.1.0 clone:
#   sbatch --array=0-<n-1> --job-name=<name> <path>/slurm_audit.sbatch <path>/audit_run.py <out root> <steps> <env:seed> ...
# Slurm runs a copy of this script, so audit_run.py's path is passed explicitly.
set -eu
audit=$1 root=$2 steps=$3
shift 3
entry=${@:$((SLURM_ARRAY_TASK_ID + 1)):1}
env=${entry%%:*}
seed=${entry#*:}
out="$root/${env}_s$seed"
mkdir -p "$out"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "$(date +%H:%M:%S) $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} $(pwd) $entry $steps" > "$out/job.txt"
status=0
.venv/bin/python "$audit" train "$env" "$out" --steps "$steps" --seeds 1 --seed "$seed" > "$out/run.log" 2>&1 || status=$?
echo "$(date +%H:%M:%S) exit $status" >> "$out/job.txt"
exit $status
