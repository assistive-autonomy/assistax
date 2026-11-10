#!/usr/bin/env bash
# Run reward-design jobs one after another (from the worktree root).
# Usage: bash reward_design/run_queue.sh <job file> <log file>
# Each job line: <train|eval> <env> <version> <out_dir> [extra args]; MEM sets the GPU memory fraction per process.
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM:-0.3}
while read -r mode env version out rest; do
  [ -z "${mode:-}" ] && continue
  mkdir -p "$out"
  echo "$(date +%H:%M:%S) start $mode $env $version $out $rest" >> "$2"
  .venv-cuda/bin/python -m reward_design.run "$mode" "$env" "$version" "$out" $rest > "$out/run.log" 2>&1
  echo "$(date +%H:%M:%S) exit $? $mode $env $version $out" >> "$2"
done < "$1"
echo "$(date +%H:%M:%S) queue finished" >> "$2"
