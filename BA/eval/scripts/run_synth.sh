#!/bin/bash
# Run one synthesis job in the background-friendly way. Usage: run_synth.sh <venv> <script> <args...>
source /tmp/ba/env.sh
export PKUSEG_HOME=/tmp/ba/pkuseg
cd /tmp/ba
venv=$1; shift
exec /tmp/ba/venvs/$venv/bin/python "$@"
