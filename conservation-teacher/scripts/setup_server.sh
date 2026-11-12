#!/usr/bin/env bash
# Create the environment on the GPU server and check that everything works.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[pixels,dev]"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
pytest -q
