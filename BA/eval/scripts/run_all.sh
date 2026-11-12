#!/bin/bash
# Full Phase 1 pipeline, strictly one GPU job at a time. Every step is resumable
# (finished utterances are skipped), so the script can simply be re-run.
# Logs: /tmp/ba/logs/synth_<system>.log, /tmp/ba/logs/score.log
S=/home/jovyan/Mahmoud-Project/BA/eval/scripts
L=/tmp/ba/logs
R="$S/run_synth.sh"
step() { name=$1; shift; echo "$(date -Is) start $name"; "$@" > $L/$name.log 2>&1; echo "$(date -Is) end $name rc=$?"; }

if [ "$1" != "--skip-cb" ]; then
  step synth_cb-v3   $R cb   $S/synth_chatterbox.py cb-v3 full
  step synth_cb-v2   $R cb   $S/synth_chatterbox.py cb-v2 full
  step synth_namaa   $R cb   $S/synth_chatterbox.py namaa egy
  step synth_lahgtna $R cb   $S/synth_chatterbox.py lahgtna egy
fi
step synth_cb-v3 $R cb $S/synth_chatterbox.py cb-v3 full   # upstream-faithful V3 (no alignment analyzer)
for s in cb-v3 cb-v2 namaa lahgtna; do step pack_$s $R cb $S/synth_chatterbox.py $s pack; done
step synth_voxcpm2        $R vox  $S/synth_voxcpm.py voxcpm2 full
step pack_voxcpm2         $R vox  $S/synth_voxcpm.py voxcpm2 pack
step synth_voxcpm2-prompt $R vox  $S/synth_voxcpm.py voxcpm2-prompt egy_main
step pack_voxcpm2-prompt  $R vox  $S/synth_voxcpm.py voxcpm2-prompt pack
step synth_omnivoice      $R omni $S/synth_omnivoice.py omnivoice egy_en
step pack_omnivoice       $R omni $S/synth_omnivoice.py omnivoice pack
step synth_voicetut       $R omni $S/synth_omnivoice.py voicetut egy_en
step pack_voicetut        $R omni $S/synth_omnivoice.py voicetut pack
# isolated GPU timing re-run (earlier runs overlapped with CPU benchmarks/downloads)
for s in cb-v3 cb-v2; do step speed_$s $R cb $S/synth_chatterbox.py $s speed; done
step speed_voxcpm2 $R vox $S/synth_voxcpm.py voxcpm2 speed
step score                $R eval $S/score.py
step aggregate            $R eval $S/aggregate.py
