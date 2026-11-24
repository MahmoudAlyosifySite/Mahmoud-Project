"""Watermark robustness under signal attacks (venv /tmp/ba/venvs/ba; CPU by default, --device cuda to use the GPU).

Watermarked clips: the 60 BA outputs of watermark.json (held-out utterances of engine/full, 48 kHz WAV).
Unwatermarked clips (false positives): the 60 raw VoxCPM2 bake-off outputs with the same utterance ids, and all
230 real-speech calibration clips (calib.jsonl: lahgtna_eg, NileTTS, FLEURS ar_eg / en_us / fr_fr) = 290 clips.
Every attack is applied to every clip, watermarked or not. Detector: ba_tts.watermark.detect (AudioSeal, 16 kHz,
decision P(watermark) >= 0.5) and the 16-bit message match.

Attacks (seeded per clip, so every run is identical):
  none                      the file as written
  white_{20,10,5}dB         Gaussian white noise; SNR against the mean power of the speech frames (frames within
                            30 dB of the 95th-percentile frame level, as synth_ba.degrade)
  babble_{20,10,5}dB        babble: sum of 6 real speakers (the 12 FLEURS same-speaker clips of refs.jsonl, none
                            of them in the false-positive set), each RMS-normalised, random offsets, looped to length
  speed_{+5,-5}pct          resampling (tempo and pitch change together)
  tempo_{+5,-5}pct          ffmpeg atempo (tempo only, pitch kept)
  phone_8k                  4th-order Butterworth band-pass 300-3400 Hz, then resampled to 8 kHz
  mp3_64k                   MP3 64 kbit/s (libmp3lame), decoded
  mp3_64k_x2                MP3 64 kbit/s, decoded, MP3 64 kbit/s again
  mp3_32k                   MP3 32 kbit/s
  mp3_64k_then_aac_64k      MP3 64 kbit/s, decoded, then AAC 64 kbit/s (ffmpeg native aac)
  aac_64k                   AAC 64 kbit/s
  opus_32k, opus_16k        Opus (libopus) at 32 / 16 kbit/s
  phone_mp3_32k_noise20     band-pass 8 kHz + MP3 32 kbit/s + white noise 20 dB (a phone-and-messaging chain)
  excerpt_3s                a 3-second excerpt from a random position (whole clip if shorter)
  gain_-20dB, gain_-10dB    level change
  gain_+6dB_clipped         +6 dB with hard clipping at full scale
  loudnorm_-14LUFS          loudness-normalised to -14 LUFS (pyloudnorm), peak-limited to 0.99

Per-clip results are cached in /home/jovyan/ba-work/paper/watermark/attacks.jsonl (resumable).
Output: BA/eval/results/watermark_attacks.json and .md.
Usage: python watermark_attacks.py [--device cpu|cuda] [--report-only]
"""
import json, os, subprocess, sys, tempfile, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf
from paper_common import PAPER_WORK, RESULTS, EVAL, ENGINE_ROOT, BAKEOFF_ROOT, jl, dump, fmt

CACHE = f"{PAPER_WORK}/watermark/attacks.jsonl"
ATTACKS = ["none", "white_20dB", "white_10dB", "white_5dB", "babble_20dB", "babble_10dB", "babble_5dB",
           "speed_+5pct", "speed_-5pct", "tempo_+5pct", "tempo_-5pct", "phone_8k", "mp3_64k", "mp3_64k_x2", "mp3_32k",
           "mp3_64k_then_aac_64k", "aac_64k", "opus_32k", "opus_16k", "phone_mp3_32k_noise20", "excerpt_3s",
           "gain_-20dB", "gain_-10dB", "gain_+6dB_clipped", "loudnorm_-14LUFS"]


def ffmpeg_chain(y, sr, steps):
    """steps: list of (container_ext, ffmpeg codec args) applied in turn; returns decoded float32 mono + sr."""
    with tempfile.TemporaryDirectory() as d:
        cur = f"{d}/in.wav"
        sf.write(cur, y, sr)
        for k, (ext, args) in enumerate(steps):
            enc = f"{d}/enc{k}.{ext}"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", cur] + args + [enc], check=True)
            dec = f"{d}/dec{k}.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", enc, dec], check=True)
            cur = dec
        z, zsr = sf.read(cur, dtype="float32")
    return (z.mean(1) if z.ndim > 1 else z).astype(np.float32), zsr


MP3 = lambda k: ("mp3", ["-c:a", "libmp3lame", "-b:a", f"{k}k"])  # noqa: E731
AAC = lambda k: ("m4a", ["-c:a", "aac", "-b:a", f"{k}k"])  # noqa: E731
OPUS = lambda k: ("ogg", ["-c:a", "libopus", "-b:a", f"{k}k"])  # noqa: E731


def speech_power(y, sr):
    from ba_tts.audio import frame_rms_db, resample
    r = frame_rms_db(resample(y, sr, 16000))
    act = r > np.percentile(r, 95) - 30
    return float(np.mean((10 ** (r[act] / 20)) ** 2))


def add_noise(y, sr, noise, snr):
    n = noise[:len(y)]
    n = n / (np.sqrt(np.mean(n ** 2)) + 1e-12)
    return (y + n * np.sqrt(speech_power(y, sr) / 10 ** (snr / 10))).astype(np.float32)


class Babble:
    def __init__(self):
        from ba_tts.audio import load_audio, resample
        self.src = []
        for r in jl(f"{EVAL}/refs.jsonl"):
            y, sr = load_audio(r["pair_path"])
            y = resample(y, sr, 48000)
            self.src.append(y / (np.sqrt(np.mean(y ** 2)) + 1e-12))

    def make(self, n, sr, rng):
        from ba_tts.audio import resample
        out = np.zeros(n, np.float32)
        for k in rng.choice(len(self.src), 6, replace=False):
            s = self.src[k] if sr == 48000 else resample(self.src[k], 48000, sr)
            off = int(rng.integers(0, len(s)))
            reps = int(np.ceil((n + off) / len(s))) + 1
            out += np.tile(s, reps)[off:off + n]
        return out


def bandpass_8k(y, sr):
    from scipy.signal import butter, sosfiltfilt
    from ba_tts.audio import resample
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return resample(sosfiltfilt(sos, y).astype(np.float32), sr, 8000), 8000


def attack(name, y, sr, rng, babble):
    from ba_tts.audio import resample
    if name == "none":
        return y, sr
    if name.startswith("white_"):
        return add_noise(y, sr, rng.standard_normal(len(y)).astype(np.float32), float(name[6:-2])), sr
    if name.startswith("babble_"):
        return add_noise(y, sr, babble.make(len(y), sr, rng), float(name[7:-2])), sr
    if name.startswith("speed_"):
        f = 1 + float(name[6:-3]) / 100
        import librosa
        return librosa.resample(y, orig_sr=int(round(sr * f)), target_sr=sr).astype(np.float32), sr
    if name.startswith("tempo_"):
        f = 1 + float(name[6:-3]) / 100
        return ffmpeg_chain(y, sr, [("wav", ["-filter:a", f"atempo={f}"])])
    if name == "phone_8k":
        return bandpass_8k(y, sr)
    if name == "mp3_64k":
        return ffmpeg_chain(y, sr, [MP3(64)])
    if name == "mp3_64k_x2":
        return ffmpeg_chain(y, sr, [MP3(64), MP3(64)])
    if name == "mp3_32k":
        return ffmpeg_chain(y, sr, [MP3(32)])
    if name == "mp3_64k_then_aac_64k":
        return ffmpeg_chain(y, sr, [MP3(64), AAC(64)])
    if name == "aac_64k":
        return ffmpeg_chain(y, sr, [AAC(64)])
    if name.startswith("opus_"):
        return ffmpeg_chain(y, sr, [OPUS(int(name[5:-1]))])
    if name == "phone_mp3_32k_noise20":
        z, zsr = bandpass_8k(y, sr)
        z, zsr = ffmpeg_chain(z, zsr, [MP3(32)])
        return add_noise(z, zsr, rng.standard_normal(len(z)).astype(np.float32), 20.0), zsr
    if name == "excerpt_3s":
        n = 3 * sr
        if len(y) <= n:
            return y, sr
        s = int(rng.integers(0, len(y) - n))
        return y[s:s + n], sr
    if name.startswith("gain_"):
        db = float(name.split("_")[1][:-2])          # "gain_-20dB" -> -20, "gain_+6dB_clipped" -> +6
        return np.clip(y * 10 ** (db / 20), -1.0, 1.0).astype(np.float32), sr
    if name == "loudnorm_-14LUFS":
        from ba_tts.audio import normalize_loudness
        return normalize_loudness(y, sr, -14.0, peak=0.99), sr
    raise ValueError(name)


def clips():
    wm = json.load(open(f"{RESULTS}/watermark.json"))["utterances"]
    out = [(f"{ENGINE_ROOT}/full/{u}.wav", "ba_watermarked", u) for u in wm]
    out += [(f"{BAKEOFF_ROOT}/voxcpm2/{u}.wav", "raw_voxcpm2", u) for u in wm if os.path.exists(f"{BAKEOFF_ROOT}/voxcpm2/{u}.wav")]
    out += [(c["path"], f"real_{c['set']}", c["calib_id"]) for c in jl(f"{EVAL}/calib.jsonl")]
    return out


def run(device):
    import torch
    torch.set_num_threads(8)
    from ba_tts.audio import load_audio
    from ba_tts.watermark import detect
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    done = {(r["path"], r["attack"]) for r in jl(CACHE)}
    babble = Babble()
    with open(CACHE, "a") as f:
        for i, (p, kind, cid) in enumerate(clips()):
            if all((p, a) in done for a in ATTACKS):
                continue
            y, sr = load_audio(p)
            for a in ATTACKS:
                if (p, a) in done:
                    continue
                rng = np.random.default_rng(zlib.crc32(f"{cid}|{a}".encode()))
                try:
                    z, zsr = attack(a, y, sr, rng, babble)
                    d = detect(z, zsr, device=device)
                    rec = dict(path=p, kind=kind, id=cid, attack=a, seconds=round(len(z) / zsr, 2), **d)
                except Exception as e:
                    rec = dict(path=p, kind=kind, id=cid, attack=a, error=repr(e)[:200])
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0:
                print(i, kind, cid, flush=True)


def report():
    rows = jl(CACHE)
    groups = {"ba_watermarked": lambda k: k == "ba_watermarked", "raw_voxcpm2": lambda k: k == "raw_voxcpm2",
              "real_speech": lambda k: k.startswith("real_"), "all_unwatermarked": lambda k: k != "ba_watermarked"}
    res = {}
    for a in ATTACKS:
        res[a] = {}
        for g, sel in groups.items():
            rr = [r for r in rows if r["attack"] == a and sel(r["kind"])]
            ok = [r for r in rr if "error" not in r]
            res[a][g] = dict(n=len(rr), errors=len(rr) - len(ok), detected=sum(1 for r in ok if r["watermarked"]),
                             rate=round(sum(1 for r in ok if r["watermarked"]) / len(ok), 4) if ok else None,
                             mean_score=round(float(np.mean([r["score"] for r in ok])), 4) if ok else None,
                             min_score=round(float(np.min([r["score"] for r in ok])), 4) if ok else None,
                             max_score=round(float(np.max([r["score"] for r in ok])), 4) if ok else None,
                             mean_bits_match=round(float(np.mean([r["bits_match"] for r in ok])), 4) if ok else None,
                             all_bits_match=sum(1 for r in ok if r["bits_match"] == 1.0))
    # Wilson 95% interval for detection / false-positive rates
    def wilson(k, n, z=1.96):
        if not n:
            return None
        p = k / n
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [round(max(0, c - h), 4), round(min(1, c + h), 4)]
    for a in res:
        for g in res[a]:
            v = res[a][g]
            v["rate_ci95_wilson"] = wilson(v["detected"], v["n"] - v["errors"])
    fp_all = sum(res[a]["all_unwatermarked"]["detected"] for a in ATTACKS)
    n_all = sum(res[a]["all_unwatermarked"]["n"] - res[a]["all_unwatermarked"]["errors"] for a in ATTACKS)
    import subprocess as sp
    out = dict(method="AudioSeal via ba_tts.watermark.detect (16 kHz, P(watermark) >= 0.5); attacks as in the script docstring",
               n_watermarked=res["none"]["ba_watermarked"]["n"], n_unwatermarked=res["none"]["all_unwatermarked"]["n"],
               unwatermarked_sets=dict(raw_voxcpm2=res["none"]["raw_voxcpm2"]["n"], real_speech=res["none"]["real_speech"]["n"]),
               attacks=res, false_positives_all_attacks=dict(detected=fp_all, n=n_all, rate=round(fp_all / n_all, 5) if n_all else None,
                                                              ci95_wilson=wilson(fp_all, n_all)),
               ffmpeg=sp.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
               command="/tmp/ba/venvs/ba/bin/python BA/eval/scripts/watermark_attacks.py")
    dump(out, f"{RESULTS}/watermark_attacks.json")
    L = ["# Watermark robustness under attacks\n",
         "Generated by `BA/eval/scripts/watermark_attacks.py` from `watermark_attacks.json`. AudioSeal detector as shipped in "
         "`ba_tts.watermark` (threshold 0.5). Watermarked: the 60 held-out BA outputs of `watermark.json`. Unwatermarked: "
         f"{out['unwatermarked_sets']['raw_voxcpm2']} raw VoxCPM2 outputs (same utterances) + {out['unwatermarked_sets']['real_speech']} real-speech "
         "calibration clips, with the same attacks applied. Rates with Wilson 95% intervals.\n",
         "| attack | detected (BA, n=60) [95% CI] | mean score | message bits all correct | false positives raw VoxCPM2 | false positives real speech | mean score unwatermarked |",
         "|---|---|---|---|---|---|---|"]
    for a in ATTACKS:
        w, rv, rs, u = res[a]["ba_watermarked"], res[a]["raw_voxcpm2"], res[a]["real_speech"], res[a]["all_unwatermarked"]
        L.append(f"| {a} | {w['detected']}/{w['n'] - w['errors']} = {fmt(w['rate'])} [{w['rate_ci95_wilson'][0]:.3f}, {w['rate_ci95_wilson'][1]:.3f}] | "
                 f"{fmt(w['mean_score'])} | {w['all_bits_match']}/{w['n'] - w['errors']} | {rv['detected']}/{rv['n'] - rv['errors']} | "
                 f"{rs['detected']}/{rs['n'] - rs['errors']} | {fmt(u['mean_score'])} |")
    fa = out["false_positives_all_attacks"]
    L.append(f"\nFalse positives over all attacks and all unwatermarked clips: {fa['detected']}/{fa['n']} (rate {fa['rate']}, 95% CI {fa['ci95_wilson']}).")
    errs = sum(res[a][g]["errors"] for a in res for g in ("ba_watermarked", "all_unwatermarked"))
    L.append(f"Attack or detection errors (counted as missing, not as detections): {errs}.\n")
    obs = f"{RESULTS}/watermark_attacks_observations.md"
    if os.path.exists(obs):
        L.append("## Interpretation\n")
        L.append(open(obs).read())
    open(f"{RESULTS}/watermark_attacks.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    if "--report-only" not in sys.argv:
        run(dev)
    report()
