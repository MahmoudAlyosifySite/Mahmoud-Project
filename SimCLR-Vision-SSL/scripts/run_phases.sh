#!/bin/bash
# Run the job files listed in a phases file one after another on one GPU.
# The phases file is read again after every phase, so new phases can be appended while the queue runs.
#   scripts/run_phases.sh jobs/phases.txt <ckpt_dir> <log_dir> [parallel]
PH=$1; CK=$2; LOGS=$3; PAR=${4:-1}
mkdir -p "$LOGS"; DONE="$LOGS/phases_done.txt"; touch "$DONE"
while true; do
  next=""
  while read -r f; do
    [ -z "$f" ] && continue
    case "$f" in \#*) continue;; esac
    grep -qxF "$f" "$DONE" || { next="$f"; break; }
  done < "$PH"
  [ -z "$next" ] && break
  echo "[phases] $(date -u +%F_%T) start $next" >> "$LOGS/phases.log"
  python scripts/run_queue.py "$next" --out results/runs --ckpt_dir "$CK" --parallel "$PAR" >> "$LOGS/$(basename "$next" .txt).log" 2>&1
  echo "$next" >> "$DONE"
  echo "[phases] $(date -u +%F_%T) end   $next" >> "$LOGS/phases.log"
done
echo "[phases] $(date -u +%F_%T) all phases finished" >> "$LOGS/phases.log"
