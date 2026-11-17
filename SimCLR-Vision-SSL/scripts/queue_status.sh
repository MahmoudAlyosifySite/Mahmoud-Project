#!/bin/bash
# One-screen status of the GPU queue:   bash scripts/queue_status.sh
STORE=${STORE:-/home/jovyan/simclr_store}
cd "$(dirname "$0")/.." || exit 1
date -u +"now: %F %H:%M UTC"
if ps -ww -eo args | grep -v grep | grep -q "scripts/run_phases.sh jobs/phases.txt"; then echo "queue: RUNNING"; else echo "queue: NOT RUNNING  ->  bash scripts/resume_queue.sh"; fi
if ps -ww -eo args | grep -v grep | grep -q "scripts/queue_watchdog.sh"; then echo "watchdog: running"; else echo "watchdog: not running"; fi
echo "jobs on the GPU now:"; ps -eo etime,args | grep -E "python .*-m (dlt|scripts)\." | grep -v grep | sed -E 's/.*--name ([^ ]+).*/  \1/' | sort -u 
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2> /dev/null | sed 's/^/GPU: /'
python - <<'PY'
import os, shlex
tot = done = 0; rows = []
for ph in open("jobs/phases.txt").read().split():
    n = d = 0
    for line in open(ph):
        s = line.split("#")[0].strip()
        if not s: continue
        t = shlex.split(s); name = t[t.index("--name") + 1]; n += 1
        d += os.path.exists(f"results/runs/{name}/result.json")
    tot += n; done += d
    rows.append(f"  {os.path.basename(ph)[4:-4]:24s} {d:3d}/{n}")
print(f"runs finished: {done}/{tot}"); print("\n".join(rows))
PY
# failures whose run has not been completed since (a later retry may have succeeded)
grep -h "FAILED" "$STORE"/logs/rev_*.log 2> /dev/null | sed -E 's/.*FAILED ([^ ]+).*/\1/' | sort -u | while read -r n; do
  [ -f "results/runs/$n/result.json" ] || echo "failed (not yet rerun): $n"; done
