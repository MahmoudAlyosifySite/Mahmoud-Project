"""Real-speech calibration clips (measurement only, never training).

  lahgtna_eg : 50 Egyptian (`language == "eg"`) rows from the *test* split of
               oddadmix/dialectal-arabic-lahgtna-v2 (only the two test parquet shards are read,
               rows streamed; the train split is never touched). Real colloquial Egyptian speech.
  niletts    : 50 rows of NileTTS metadata_eval.csv (ahmedez10/egyptian-arabic-tts-dataset,
               Apache-2.0), 25 per speaker. Colloquial Egyptian text, synthetic voices.
  fleurs_ar_eg : 50 FLEURS ar_eg test clips (MSA read by Egyptian speakers) not used as references.
  fleurs_en_us / fleurs_fr_fr : 30 clips each (ASR / MOS / duration calibration for en, fr).
Selection is deterministic: first N rows meeting 2-20 s duration, in file order.
Writes /home/jovyan/ba-work/calib/<set>/*.wav and BA/eval/calib.jsonl.
"""
import csv, io, json, os, sys
import numpy as np, soundfile as sf, librosa

EVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "/home/jovyan/ba-work/calib"


def save(set_name, idx, y, sr, text, extra=None):
    os.makedirs(f"{OUT}/{set_name}", exist_ok=True)
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000); sr = 16000
    p = f"{OUT}/{set_name}/{set_name}_{idx:03d}.wav"
    sf.write(p, y, sr)
    r = dict(calib_id=f"{set_name}_{idx:03d}", set=set_name, path=p, text=text, duration=round(len(y) / sr, 2))
    if extra: r.update(extra)
    return r


def main():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rows = []
    # lahgtna eg test
    n = 0
    for shard in ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]:
        if n >= 50: break
        p = hf_hub_download("oddadmix/dialectal-arabic-lahgtna-v2", shard, repo_type="dataset")
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["language", "transcript_text", "audio"]).to_pylist()
            for r in t:
                if r["language"] != "eg" or n >= 50: continue
                y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                d = len(y) / sr
                if not (2 <= d <= 20) or not r["transcript_text"].strip(): continue
                rows.append(save("lahgtna_eg", n, y, sr, r["transcript_text"].strip(), {"lang": "arz", "source_shard": shard}))
                n += 1
            if n >= 50: break
    print("lahgtna_eg", n, flush=True)
    # NileTTS eval
    meta = hf_hub_download("ahmedez10/egyptian-arabic-tts-dataset", "metadata_eval.csv", repo_type="dataset")
    per = {}; n = 0
    with open(meta, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="|"):
            spk = r["speaker_name"]
            if per.get(spk, 0) >= 25: continue
            p = hf_hub_download("ahmedez10/egyptian-arabic-tts-dataset", r["audio_file"], repo_type="dataset")
            y, sr = sf.read(p, dtype="float32")
            if not (2 <= len(y) / sr <= 20): continue
            rows.append(save("niletts", n, y, sr, r["text"].strip(), {"lang": "arz", "speaker": spk, "file": r["audio_file"]}))
            per[spk] = per.get(spk, 0) + 1; n += 1
            if n >= 50: break
    print("niletts", n, flush=True)
    # FLEURS (exclude chosen references and their pairs)
    used = set()
    for l in open(f"{EVAL}/refs.jsonl"):
        r = json.loads(l); used.add(r["fleurs_file"]); used.add(r["pair_fleurs_file"])
    for cfg, N, lang in [("ar_eg", 50, "arz"), ("en_us", 30, "en"), ("fr_fr", 30, "fr")]:
        n = 0
        with open(f"/tmp/ba/data/fleurs/data/{cfg}/test.tsv", encoding="utf-8") as f:
            for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                fn, raw, nsamp = r[1], r[2], int(r[5])
                if fn in used or not (2 <= nsamp / 16000 <= 20): continue
                y, sr = sf.read(f"/tmp/ba/data/fleurs_wav/{cfg}/test/{fn}", dtype="float32")
                rows.append(save(f"fleurs_{cfg}", n, y, sr, raw, {"lang": lang, "file": fn}))
                n += 1
                if n >= N: break
        print(cfg, n, flush=True)
    with open(f"{EVAL}/calib.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
