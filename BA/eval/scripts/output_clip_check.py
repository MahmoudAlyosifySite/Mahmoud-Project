"""Share of samples at |x| >= 0.999 in raw VoxCPM2 outputs (Phase 1 bake-off, 324 files), to set the per-chunk
'clipped' check of the engine without false alarms. -> results/output_clip_check.json"""
import glob, json, os, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_common import RESULTS
vals, peaks = [], []
for p in sorted(glob.glob("/home/jovyan/ba-work/bakeoff/voxcpm2/*.wav")):
    y, sr = sf.read(p, dtype="float32")
    vals.append(float(np.mean(np.abs(y) >= 0.999))); peaks.append(float(np.max(np.abs(y))))
v = np.array(vals)
out = dict(n=len(v), files_with_any_clipped_sample=int((v > 0).sum()), max_ratio=float(v.max()), p99_ratio=float(np.percentile(v, 99)),
           peak_max=round(max(peaks), 4), peak_median=round(float(np.median(peaks)), 4),
           decision="OUTPUT_CLIP_MAX_RATIO = 0.001: above every raw VoxCPM2 output in this set" if v.max() < 0.001 else "see max_ratio")
json.dump(out, open(f"{RESULTS}/output_clip_check.json", "w"), indent=1)
print(out)
