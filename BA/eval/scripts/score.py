"""Score bake-off outputs and real-speech calibration clips (venv /tmp/ba/venvs/eval).

Per audio file it computes:
  asr      Whisper large-v3 (openai/whisper-large-v3, transformers 4.46.3, fp16), language forced
           (arz -> "arabic", en -> "english", fr -> "french"), greedy, long-form via timestamps for > 30 s.
  cer/wer  after normalisation (see norm_ar / norm_latin). Egyptian digit texts: scored against both the
           hand-written spelled-out Egyptian reference (text_spoken) and the digit text; the lower error is
           kept (cer_digits_best). They are excluded from the headline Egyptian CER and reported separately.
           English/French: digits expanded with num2words on both sides.
  sim      cosine(reference clip, output) for ECAPA (speechbrain/spkrec-ecapa-voxceleb) and WavLM-SV
           (microsoft/wavlm-base-plus-sv, x-vector head), 16 kHz.
  dialect  badrex/mms-300m-arabic-dialect-identifier top label (Arabic-text outputs and calibration clips).
  mos      DNSMOS (speechmos 0.0.1.1): P.808 MOS and P.835 OVRL. English-trained predictor; weak for Arabic.
  dur      output length, speech-active length (librosa trim, top_db=35), peak.
Results are cached per file in /home/jovyan/ba-work/bakeoff/scores/<system>.jsonl (resumable).

Usage: python score.py [systems...]   (default: every directory under /home/jovyan/ba-work/bakeoff that has meta.jsonl, plus calib)
"""
import json, os, re, sys, unicodedata, glob
import numpy as np, soundfile as sf, librosa, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matrix import build, refs as load_refs, EVAL, OUT_ROOT

SCORES = f"{OUT_ROOT}/scores"
WHISPER_LANG = {"arz": "arabic", "en": "english", "fr": "french"}
PROFILE = {"cb-v3": "full", "cb-v3-pypi": "full", "cb-v2": "full", "voxcpm2": "full", "namaa": "egy", "lahgtna": "egy",
           "omnivoice": "egy_en", "voicetut": "egy_en", "voxcpm2-prompt": "egy_main"}

# ---------------------------------------------------------------- normalisation
AR_DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def strip_punct(s):
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)


def norm_ar(s):
    s = s.translate(AR_DIGITS)
    s = AR_DIAC.sub("", s).replace("ـ", "")
    s = re.sub("[أإآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    s = strip_punct(s).lower()
    return " ".join(s.split())


def norm_latin(s, lang):
    from num2words import num2words
    s = s.lower().replace("’", "'")
    s = re.sub(r"(\d+)[,\s](\d{3})\b", r"\1\2", s)  # 50,000 -> 50000
    s = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang=lang) + " ", s)
    s = s.replace("-", " ").replace("'", " ")
    s = strip_punct(s)
    return " ".join(s.split())


def err_rates(ref, hyp, lang):
    import jiwer
    if lang == "arz":
        r, h = norm_ar(ref), norm_ar(hyp)
    else:
        r, h = norm_latin(ref, lang), norm_latin(hyp, lang)
    if not r:
        return None, None
    cer = jiwer.cer(r, h) if h else 1.0
    wer = jiwer.wer(r, h) if h else 1.0
    return float(cer), float(wer)


# ---------------------------------------------------------------- models
class Scorer:
    def __init__(self):
        from transformers import pipeline, AutoFeatureExtractor, WavLMForXVector
        from speechbrain.inference.speaker import EncoderClassifier
        self.dev = "cuda"
        self.asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                            torch_dtype=torch.float16, device=0)
        self.ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir="/tmp/ba/hf/sb_ecapa", run_opts={"device": "cuda"})
        self.wfe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to("cuda").eval()
        self.did = pipeline("audio-classification", model="badrex/mms-300m-arabic-dialect-identifier", device=0)
        self._emb_cache = {}

    def asr_text(self, y16, lang):
        gk = {"language": WHISPER_LANG[lang], "task": "transcribe"}
        if len(y16) / 16000 > 30:
            out = self.asr({"raw": y16, "sampling_rate": 16000}, generate_kwargs=gk, return_timestamps=True)
        else:
            out = self.asr({"raw": y16, "sampling_rate": 16000}, generate_kwargs=gk)
        return out["text"].strip()

    @torch.no_grad()
    def embs(self, y16, key=None):
        if key and key in self._emb_cache:
            return self._emb_cache[key]
        if len(y16) < 16000:  # WavLM's x-vector TDNN needs >= ~0.3 s of frames; pad very short outputs with silence
            y16 = np.pad(y16, (0, 16000 - len(y16)))
        e1 =self.ecapa.encode_batch(torch.from_numpy(y16)[None].cuda()).squeeze().cpu().numpy()
        inp = self.wfe(y16, sampling_rate=16000, return_tensors="pt", padding=True).to("cuda")
        e2 = self.wavlm(**inp).embeddings.squeeze().cpu().numpy()
        r = (e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2))
        if key: self._emb_cache[key] = r
        return r

    def dialect(self, y16):
        out = self.did({"raw": y16, "sampling_rate": 16000}, top_k=5)
        return out[0]["label"], {o["label"]: round(float(o["score"]), 4) for o in out}


def load16(p):
    y, sr = sf.read(p, dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y.astype(np.float32)


def audio_stats(y16):
    peak = float(np.max(np.abs(y16))) if len(y16) else 0.0
    if len(y16) == 0 or peak < 1e-4:
        return dict(dur=len(y16) / 16000, active_dur=0.0, peak=peak)
    yt, _ = librosa.effects.trim(y16, top_db=35)
    return dict(dur=round(len(y16) / 16000, 3), active_dur=round(len(yt) / 16000, 3), peak=round(peak, 4))


def mos(y16):
    from speechmos import dnsmos
    if len(y16) < 16000:  # DNSMOS needs >= 1 s; pad with silence
        y16 = np.pad(y16, (0, 16000 - len(y16)))
    peak = float(np.max(np.abs(y16)))
    if peak > 1.0:  # DNSMOS rejects samples outside [-1, 1] (resampling can overshoot); rescale
        y16 = y16 / peak
    d = dnsmos.run(y16, sr=16000)
    return round(float(d["p808_mos"]), 3), round(float(d["ovrl_mos"]), 3)


def score_one(S, path, lang, ref_text, ref_text_alt=None, ref_voice_path=None, do_dialect=False):
    rec = {}
    y = load16(path)
    rec.update(audio_stats(y))
    if rec["active_dur"] < 0.1:
        rec.update(asr="", cer=1.0, wer=1.0)
    else:
        rec["asr"] = S.asr_text(y, lang)
        rec["cer"], rec["wer"] = err_rates(ref_text, rec["asr"], lang)
        if ref_text_alt:
            c2, w2 = err_rates(ref_text_alt, rec["asr"], lang)
            rec["cer_vs_digit_text"], rec["wer_vs_digit_text"] = c2, w2
    if ref_voice_path:
        er = S.embs(load16(ref_voice_path), key=ref_voice_path)
        eo = S.embs(y) if rec["active_dur"] >= 0.1 else None
        rec["sim_ecapa"] = None if eo is None else round(float(er[0] @ eo[0]), 4)
        rec["sim_wavlm"] = None if eo is None else round(float(er[1] @ eo[1]), 4)
    if do_dialect and rec["active_dur"] >= 0.1:
        rec["dialect"], rec["dialect_scores"] = S.dialect(y)
    if rec["active_dur"] >= 0.1:
        rec["mos_p808"], rec["mos_ovrl"] = mos(y)
    return rec


def load_done(p):
    d = {}
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l); d[r["key"]] = r
    return d


def score_system(S, system):
    meta = {json.loads(l)["utt_id"]: json.loads(l) for l in open(f"{OUT_ROOT}/{system}/meta.jsonl")}
    outp = f"{SCORES}/{system}.jsonl"; done = load_done(outp)
    with open(outp, "a") as f:
        for j in build(PROFILE[system]):
            k = j["utt_id"]
            if k in done or k not in meta: continue
            m = meta[k]; wav = f"{OUT_ROOT}/{system}/{k}.wav"
            rec = dict(key=k, system=system)
            if m["error"] or not os.path.exists(wav):
                rec.update(missing=True)
            else:
                ref_text = j["text_spoken"] or j["text"]
                alt = j["text"] if j["text_spoken"] else None
                try:
                    rec.update(score_one(S, wav, j["lang"], ref_text, alt, j["ref_path"], do_dialect=(j["lang"] == "arz")))
                except Exception as e:  # keep going; the row records the error and the report counts it
                    rec.update(score_error=repr(e)[:300])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(system, k, rec.get("cer"), rec.get("sim_ecapa"), rec.get("dialect"), flush=True)


def score_calib(S):
    outp = f"{SCORES}/_calib.jsonl"; done = load_done(outp)
    rows = [json.loads(l) for l in open(f"{EVAL}/calib.jsonl")]
    with open(outp, "a") as f:
        for r in rows:
            if r["calib_id"] in done: continue
            rec = dict(key=r["calib_id"], set=r["set"], lang=r["lang"])
            rec.update(score_one(S, r["path"], r["lang"], r["text"], do_dialect=(r["lang"] == "arz")))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print("calib", r["calib_id"], rec.get("cer"), rec.get("dialect"), flush=True)
    # reference clips: speaker ceiling (ref vs same-speaker pair) / floor (ref vs other refs), and dialect/MOS of refs
    R = load_refs(); out = {"pairs": [], "refs": []}
    for v, r in R.items():
        a = S.embs(load16(r["path"]), key=r["path"]); b = S.embs(load16(r["pair_path"]))
        out["pairs"].append(dict(voice_id=v, lang=r["lang"], ecapa=round(float(a[0] @ b[0]), 4), wavlm=round(float(a[1] @ b[1]), 4)))
        y = load16(r["path"]); rr = dict(voice_id=v)
        rr["mos_p808"], rr["mos_ovrl"] = mos(y)
        if r["lang"] == "ar_eg":
            rr["dialect"], rr["dialect_scores"] = S.dialect(y)
        out["refs"].append(rr)
    vs = sorted(R)
    out["diff"] = []
    for i in range(len(vs)):
        for k in range(i + 1, len(vs)):
            a = S.embs(load16(R[vs[i]]["path"]), key=R[vs[i]]["path"]); b = S.embs(load16(R[vs[k]]["path"]), key=R[vs[k]]["path"])
            out["diff"].append(dict(a=vs[i], b=vs[k], same_lang=R[vs[i]]["lang"] == R[vs[k]]["lang"],
                                    ecapa=round(float(a[0] @ b[0]), 4), wavlm=round(float(a[1] @ b[1]), 4)))
    json.dump(out, open(f"{SCORES}/_refpairs.json", "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(SCORES, exist_ok=True)
    systems = sys.argv[1:] or [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(f"{OUT_ROOT}/*/meta.jsonl"))
                               if os.path.basename(os.path.dirname(p)) in PROFILE]
    S = Scorer()
    if "calib" in systems or not sys.argv[1:]:
        score_calib(S)
    for s in systems:
        if s != "calib":
            score_system(S, s)
