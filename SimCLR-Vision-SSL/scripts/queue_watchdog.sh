#!/bin/bash
# Every 5 minutes: restart the GPU queue if it died while phases are left (e.g. a killed process), and keep a
# copy of the queue logs in the persistent store. Only one watchdog runs at a time.
# Started by scripts/resume_queue.sh. A server restart kills it too: run scripts/resume_queue.sh again.
STORE=${STORE:-/home/jovyan/simclr_store}
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$STORE/logs"
exec 9> "$STORE/watchdog.lock"
flock -n 9 || exit 0
LOG="$STORE/logs/watchdog.log"
echo "$(date -u +%F_%T) watchdog started" >> "$LOG"
while true; do
  # a queue started from another log directory (e.g. a Claude session scratchpad): mirror its logs
  if [ -n "$EXTRA_LOG_DIR" ] && [ -d "$EXTRA_LOG_DIR" ] && \
     ps -ww -eo args | grep -v grep | grep -q "run_phases.sh jobs/phases.txt .* $EXTRA_LOG_DIR"; then
    cp -a "$EXTRA_LOG_DIR"/. "$STORE/logs/" 2> /dev/null
  fi
  if ! ps -ww -eo args | grep -v grep | grep -q "scripts/run_phases.sh jobs/phases.txt"; then
    left=0
    while read -r f; do
      [ -z "$f" ] && continue
      case "$f" in \#*) continue;; esac
      grep -qxF "$f" "$STORE/logs/phases_done.txt" 2> /dev/null || left=$((left + 1))
    done < jobs/phases.txt
    if [ "$left" -eq 0 ]; then
      echo "$(date -u +%F_%T) all phases finished; watchdog exits" >> "$LOG"
      exit 0
    fi
    echo "$(date -u +%F_%T) queue not running, $left phases left: restarting" >> "$LOG"
    DLT_DATA="$STORE/data" setsid nohup scripts/run_phases.sh jobs/phases.txt "$STORE/ckpt" "$STORE/logs" 2 \
      > /dev/null 2>&1 < /dev/null &
  fi
  sleep 300 9>&-      # the child must not keep the lock
done
