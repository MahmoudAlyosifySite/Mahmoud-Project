#!/usr/bin/env bash
# Phase 2 engine experiments, one GPU job at a time. Usage: run_engine.sh <step> [<step> ...]
#   steps: asr ref cs chunk sweep full (synthesis + scoring), wm, bench, bench_noopt
source /tmp/ba/env.sh
E=/home/jovyan/Mahmoud-Project/BA/eval/scripts
B=/tmp/ba/venvs/ba/bin/python
V=/tmp/ba/venvs/eval/bin/python
L=/tmp/ba/logs
cd /tmp
for s in "$@"; do
  if [ "$s" = asr ]; then
    $B $E/asr_check.py > $L/asr_check.log 2>&1 || echo "FAILED asr" >> $L/engine_chain.log
  elif [ "$s" = wm ]; then
    $B $E/watermark_check.py > $L/watermark_check.log 2>&1 || echo "FAILED wm" >> $L/engine_chain.log
  elif [ "$s" = bench ]; then
    $B $E/bench_engine.py > $L/bench.log 2>&1 || echo "FAILED bench" >> $L/engine_chain.log
  elif [ "$s" = bench_noopt ]; then
    $B $E/bench_engine.py --no-optimize > $L/bench_noopt.log 2>&1 || echo "FAILED bench_noopt" >> $L/engine_chain.log
  else
    $B $E/synth_ba.py $s > $L/synth_ba_$s.log 2>&1 || echo "FAILED synth $s" >> $L/engine_chain.log
    $V $E/score_manifest.py $s > $L/score_ba_$s.log 2>&1 || echo "FAILED score $s" >> $L/engine_chain.log
  fi
  echo "DONE $s $(date +%H:%M:%S)" >> $L/engine_chain.log
done
