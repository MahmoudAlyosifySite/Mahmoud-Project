#!/usr/bin/env bash
# Stage B: robustness sweep. CPU, roughly an hour. Safe to run beside a GPU job.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p results/logs
nohup python experiments/robustness_sweep.py "$@" > results/logs/stage_b.log 2>&1 &
echo "started, pid $!; follow with: tail -f results/logs/stage_b.log"
