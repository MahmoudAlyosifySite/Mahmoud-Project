#!/usr/bin/env bash
# Stage C: train a pixel JEPA on the GPU and run the three-way comparison on its latents.
# Runs in the background so it survives a dropped SSH session.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p results/logs
nohup python experiments/pilot_pixels.py --device cuda --steps 20000 "$@" > results/logs/stage_c.log 2>&1 &
echo "started, pid $!; follow with: tail -f results/logs/stage_c.log"
