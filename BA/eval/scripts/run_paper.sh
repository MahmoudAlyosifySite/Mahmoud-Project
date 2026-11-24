#!/usr/bin/env bash
# Paper experiments (step A of BA/paper/PLAN.md), strictly one GPU job at a time.
# Usage: run_paper.sh <step> [<step> ...]
#   synthesis + scoring: seeds_ba seeds_voxcpm2 seeds_cbv3 seeds_namaa abl_nonorm abl_noloan abl_notrans abl_rawref long_ba long_voxcpm2
#   dialect2   classifier 2 on every p_* experiment that exists (dialect_classifier2.py score p_...)
#   wm         watermark attacks (watermark_attacks.py)
# Before each GPU step it waits (at most 2 h, polling every 60 s) until the shared A40 has >= 14 GB free.
# Logs: /tmp/ba/logs/paper_<kind>_<step>.log; progress: /tmp/ba/logs/paper_chain.log
source /tmp/ba/env.sh
E=/home/jovyan/Mahmoud-Project/BA/eval/scripts
V=/tmp/ba/venvs/eval/bin/python
L=/tmp/ba/logs
mkdir -p $L

wait_gpu() {
  for i in $(seq 1 120); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge 14000 ] && return 0
    echo "waiting for GPU memory (free ${free} MiB) $(date +%H:%M:%S)" >> $L/paper_chain.log
    sleep 60
  done
  return 1
}

for s in "$@"; do
  echo "START $s $(date +%H:%M:%S)" >> $L/paper_chain.log
  wait_gpu || { echo "FAILED $s: GPU memory never became free" >> $L/paper_chain.log; continue; }
  case $s in
    dialect2)
      exps=$(cd /home/jovyan/ba-work/engine && ls -d p_* 2>/dev/null | tr '\n' ' ')
      (cd /tmp/ba && $V $E/dialect_classifier2.py score $exps) > $L/paper_dialect2.log 2>&1 || echo "FAILED dialect2" >> $L/paper_chain.log ;;
    wm)
      (cd /tmp/ba && /tmp/ba/venvs/ba/bin/python $E/watermark_attacks.py) > $L/paper_wm.log 2>&1 || echo "FAILED wm" >> $L/paper_chain.log ;;
    *)
      case $s in
        seeds_ba|abl_*|long_ba) venv=ba ;;
        seeds_voxcpm2|long_voxcpm2) venv=vox ;;
        seeds_cbv3|seeds_namaa) venv=cb ;;
        *) echo "FAILED unknown step $s" >> $L/paper_chain.log; continue ;;
      esac
      $E/run_synth.sh $venv $E/synth_paper.py $s > $L/paper_synth_$s.log 2>&1 || echo "FAILED synth $s" >> $L/paper_chain.log
      wait_gpu
      (cd /tmp/ba && $V $E/score_manifest.py p_$s) > $L/paper_score_$s.log 2>&1 || echo "FAILED score $s" >> $L/paper_chain.log ;;
  esac
  echo "DONE $s $(date +%H:%M:%S)" >> $L/paper_chain.log
done
echo "CHAIN END $(date +%H:%M:%S)" >> $L/paper_chain.log
