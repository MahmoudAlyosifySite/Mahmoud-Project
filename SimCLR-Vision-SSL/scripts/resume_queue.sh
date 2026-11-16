#!/bin/bash
# Start (or restart) the GPU queue and its watchdog. Safe to run at any time and more than once:
# finished runs are skipped, and nothing is started twice.
#   bash scripts/resume_queue.sh
# Checkpoints, datasets and queue logs live in $STORE (default /home/jovyan/simclr_store, a persistent volume).
STORE=${STORE:-/home/jovyan/simclr_store}
cd "$(dirname "$0")/.." || exit 1
export STORE DLT_DATA=$STORE/data
mkdir -p "$STORE/logs"
if ps -ww -eo args | grep -v grep | grep -q "scripts/run_phases.sh jobs/phases.txt"; then
  echo "queue: already running"
else
  setsid nohup scripts/run_phases.sh jobs/phases.txt "$STORE/ckpt" "$STORE/logs" 2 > /dev/null 2>&1 < /dev/null &
  echo "queue: started (logs in $STORE/logs)"
fi
if ps -ww -eo args | grep -v grep | grep -q "scripts/queue_watchdog.sh"; then
  echo "watchdog: already running"
else
  setsid nohup scripts/queue_watchdog.sh > /dev/null 2>&1 < /dev/null &
  echo "watchdog: started"
fi
