#!/bin/bash
# Third revision: start (or restart) the three GPU queues of the new runs. Safe to run more than once:
# finished runs are skipped and a queue that is already running is not started again.
#   bash scripts/resume_rev3.sh            # all three queues
#   bash scripts/resume_rev3.sh grid pre   # only some of them
# grid: fine-tuning recipe grids (8 processes); pre: pretraining and what depends on it; r50: ResNet-50 SupCon+U.
STORE=${STORE:-/home/jovyan/simclr_store}
cd "$(dirname "$0")/.." || exit 1
export STORE DLT_DATA=$STORE/data
mkdir -p "$STORE/logs"
# CUDA MPS lets the many small fine-tuning processes run their kernels concurrently instead of time-slicing the
# GPU (about 3x the throughput of 8 recipe workers). The daemon is per user; a server restart kills it.
export CUDA_MPS_PIPE_DIRECTORY=$STORE/mps/pipe CUDA_MPS_LOG_DIRECTORY=$STORE/mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if ! ps -eo args | grep -v grep | grep -q "nvidia-cuda-mps-control -d"; then
  nvidia-cuda-mps-control -d && echo "MPS daemon: started"
fi
for q in ${@:-grid pre r50 stage}; do
  ph=jobs/phases_rev3_$q.txt
  if ps -ww -eo args | grep -v grep | grep -q "scripts/run_phases.sh $ph"; then
    echo "$q: already running"
  else
    setsid nohup scripts/run_phases.sh "$ph" "$STORE/ckpt" "$STORE/logs" 1 > /dev/null 2>&1 < /dev/null &
    echo "$q: started"
  fi
done
