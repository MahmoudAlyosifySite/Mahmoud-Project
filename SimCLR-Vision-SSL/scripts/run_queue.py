"""Run a list of jobs on one GPU, a few at a time, skipping finished ones.

Each line of the jobs file is the argument string of ``python -m <module>``
(default module: dlt.train) and must contain ``--name <run name>``. Header lines
``# parallel: N`` and ``# retries: N`` override the command-line settings for that file. A job counts
as finished when ``<out>/<name>/result.json`` exists, so the queue can be
stopped and restarted at any time.

    python scripts/run_queue.py jobs/core.txt --out results/runs --parallel 1
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("jobs")
p.add_argument("--out", default="results/runs")
p.add_argument("--ckpt_dir", default=None)
p.add_argument("--log_dir", default=None)
p.add_argument("--parallel", type=int, default=1)
p.add_argument("--retries", type=int, default=2)
args = p.parse_args()
# a jobs file may override the queue settings with header lines such as "# parallel: 1" or "# retries: 3"
for line in open(args.jobs):
    if line.startswith("#") and ":" in line:
        key, _, val = line[1:].partition(":")
        if key.strip() in ("parallel", "retries") and val.strip().isdigit():
            setattr(args, key.strip(), int(val.strip()))

log_dir = args.log_dir or os.path.join(args.out, "_logs")
os.makedirs(log_dir, exist_ok=True)
jobs = []
for line in open(args.jobs):
    line = line.split("#")[0].strip()
    if not line:
        continue
    toks = shlex.split(line)
    module = "dlt.train"
    if toks[0] == "-m":
        module, toks = toks[1], toks[2:]
    name = toks[toks.index("--name") + 1]
    jobs.append((name, module, toks))


def done(name):
    return os.path.exists(os.path.join(args.out, name, "result.json"))


pending = [j for j in jobs if not done(j[0])]
print(f"[queue] {len(jobs)} jobs, {len(pending)} pending", flush=True)
running, attempts, failed = [], {}, []
t0 = time.time()
while pending or running:
    while pending and len(running) < args.parallel:
        name, module, toks = pending.pop(0)
        cmd = [sys.executable, "-m", module] + toks + ["--out", args.out]
        if args.ckpt_dir:
            cmd += ["--ckpt_dir", args.ckpt_dir]
        log = open(os.path.join(log_dir, name + ".log"), "a")
        running.append((name, module, toks, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, time.time()))
        print(f"[queue] start {name}", flush=True)
    time.sleep(2)
    for item in list(running):
        name, module, toks, proc, log, ts = item
        rc = proc.poll()
        if rc is None:
            continue
        running.remove(item)
        log.close()
        if rc == 0 and done(name):
            left = len(pending) + len(running)
            print(f"[queue] done  {name} ({(time.time() - ts) / 60:.1f} min) | {left} left | "
                  f"elapsed {(time.time() - t0) / 3600:.2f} h", flush=True)
        else:
            attempts[name] = attempts.get(name, 0) + 1
            if attempts[name] <= args.retries:
                print(f"[queue] RETRY {name} (rc={rc})", flush=True)
                pending.insert(0, (name, module, toks))
            else:
                print(f"[queue] FAILED {name} (rc={rc})", flush=True)
                failed.append(name)
print(f"[queue] finished: {len(jobs) - len(failed)} ok, {len(failed)} failed {failed} "
      f"in {(time.time() - t0) / 3600:.2f} h", flush=True)
