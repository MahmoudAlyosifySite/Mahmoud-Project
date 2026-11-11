#!/usr/bin/env bash
# Stage A: ideal latents, seeds 1-5. CPU, a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python experiments/pilot_synthetic.py "$@"
