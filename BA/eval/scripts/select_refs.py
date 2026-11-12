"""Pick reference voices from FLEURS test splits (ar_eg, en_us, fr_fr).

FLEURS has no speaker ids, so speakers are told apart with ECAPA speaker
embeddings. Steps per language:
  0. FLEURS ar_eg *test* has only female speakers, so ar_eg male candidates
     come from the ar_eg train split (male rows);
  1. candidates: clips 8-15 s long;
  2. cleanliness: DNSMOS (speechmos) BAK + OVRL, plus a frame-energy SNR proxy;
  3. greedy pick, cleanest first, per gender: a clip is accepted only if its
     ECAPA cosine to every already-chosen clip is < DISSIM_MAX (different
     person) and it has a partner clip (different sentence) with cosine
     >= PAIR_MIN (same person). The partner gives the real same-speaker ceiling.
Writes BA/eval/refs.jsonl and copies audio to /home/jovyan/ba-work/refs/.

Run:  /tmp/ba/venvs/eval/bin/python BA/eval/scripts/select_refs.py
"""
import csv, json, os, shutil, sys
import numpy as np, soundfile as sf, librosa, torch

FLEURS = "/tmp/ba/data/fleurs"
WAVS = "/tmp/ba/data/fleurs_wav"
OUT_AUDIO = "/home/jovyan/ba-work/refs"
REPO_EVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED = {"ar_eg": {"male": 3, "female": 3}, "en_us": {"male": 2, "female": 1}, "fr_fr": {"male": 2, "female": 1}}
DISSIM_MAX = 0.35
PAIR_MIN = 0.60
MIN_OVRL = 2.8  # DNSMOS overall floor for a reference clip
# FLEURS ar_eg test contains only FEMALE speakers (checked: 428/428), so male
# ar_eg voices come from the train split (MALE rows only).
# fr_fr test+dev yield only one clean (DNSMOS OVRL >= MIN_OVRL), distinct female
# speaker with a same-speaker partner, so fr_fr uses 2 male + 1 female.
SPLITS = {"ar_eg": (["test", "train"], "male")}

def load16(p):
    y, sr = sf.read(p, dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y

def snr_proxy(y):
    fr = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0] + 1e-8
    return float(20 * np.log10(np.percentile(fr, 95) / np.percentile(fr, 10)))

def main():
    from speechbrain.inference.speaker import EncoderClassifier
    from speechmos import dnsmos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="/tmp/ba/hf/sb_ecapa", run_opts={"device": dev})
    os.makedirs(OUT_AUDIO, exist_ok=True)
    rows_out, all_log = [], {}
    for cfg, need in NEED.items():
        rows = []
        splits, extra_gender = SPLITS.get(cfg, (["test"], None))
        for split in splits:
          with open(f"{FLEURS}/data/{cfg}/{split}.tsv", encoding="utf-8") as f:
            for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                sid, fn, raw, norm, _ph, nsamp, gender = r[:7]
                if split != "test" and gender.lower() != extra_gender: continue
                dur = int(nsamp) / 16000
                rows.append(dict(sent_id=sid, file=fn, text=raw, gender=gender.lower(), dur=dur, split=split,
                                 path=f"{WAVS}/{cfg}/{split}/{fn}"))
        rows = [r for r in rows if os.path.exists(r["path"])]
        # embed everything (needed to find partners), score candidates
        embs = []
        for r in rows:
            y = load16(r["path"])
            with torch.no_grad():
                e = ecapa.encode_batch(torch.from_numpy(y)[None].to(dev)).squeeze().cpu().numpy()
            embs.append(e / np.linalg.norm(e))
            r["snr"] = snr_proxy(y)
            if 8 <= r["dur"] <= 15:
                d = dnsmos.run(y, sr=16000)
                r["dns_ovrl"], r["dns_bak"] = float(d["ovrl_mos"]), float(d["bak_mos"])
        E = np.stack(embs); S = E @ E.T
        cand = [i for i, r in enumerate(rows) if 8 <= r["dur"] <= 15]
        cand.sort(key=lambda i: -(rows[i]["dns_bak"] + rows[i]["dns_ovrl"]))
        chosen = []
        for g in ("male", "female"):
            k = 0
            for i in cand:
                if k >= need[g]: break
                if rows[i]["gender"] != g: continue
                if rows[i]["dns_ovrl"] < MIN_OVRL: continue
                if any(S[i, j] >= DISSIM_MAX for j in chosen): continue
                # partner: same-speaker, different sentence, 4-20 s
                parts = [j for j in range(len(rows)) if j != i and rows[j]["sent_id"] != rows[i]["sent_id"]
                         and 4 <= rows[j]["dur"] <= 20 and S[i, j] >= PAIR_MIN]
                if not parts: continue
                p = max(parts, key=lambda j: S[i, j])
                chosen.append(i); k += 1
                rows[i]["partner"] = p
            if k < need[g]:
                print(f"WARNING {cfg} {g}: found only {k}", file=sys.stderr)
        n = 0
        for i in chosen:
            r = rows[i]; p = rows[r["partner"]]; n += 1
            vid = f"{cfg}_{r['gender'][0]}{n}"
            dst = f"{OUT_AUDIO}/{vid}.wav"; dstp = f"{OUT_AUDIO}/{vid}_pair.wav"
            sf.write(dst, load16(r["path"]), 16000); sf.write(dstp, load16(p["path"]), 16000)
            rows_out.append(dict(voice_id=vid, lang=cfg, gender=r["gender"], path=dst, duration=round(r["dur"], 2),
                                 transcript=r["text"], fleurs_split=r["split"], fleurs_file=r["file"], fleurs_sent_id=r["sent_id"],
                                 dnsmos_ovrl=round(r["dns_ovrl"], 3), dnsmos_bak=round(r["dns_bak"], 3),
                                 snr_proxy_db=round(r["snr"], 1),
                                 pair_path=dstp, pair_transcript=p["text"], pair_fleurs_file=p["file"],
                                 pair_duration=round(p["dur"], 2), pair_ecapa_cos=round(float(S[i, r["partner"]]), 4)))
        all_log[cfg] = {"n_clips": len(rows), "n_candidates_8_15s": len(cand),
                        "chosen_cross_cos": [[round(float(S[a, b]), 3) for b in chosen] for a in chosen]}
    with open(f"{REPO_EVAL}/refs.jsonl", "w", encoding="utf-8") as f:
        for r in rows_out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{REPO_EVAL}/results/refs_selection.json", "w") as f:
        json.dump(dict(params=dict(DISSIM_MAX=DISSIM_MAX, PAIR_MIN=PAIR_MIN, MIN_OVRL=MIN_OVRL, embedder="speechbrain/spkrec-ecapa-voxceleb",
                                   clean_score="DNSMOS bak+ovrl (speechmos 0.0.1.1)"), per_lang=all_log), f, indent=1)
    print(json.dumps(all_log, indent=1))

if __name__ == "__main__":
    main()
