# BA evaluation (Phase 1 bake-off)

Everything needed to re-run the Phase 1 measurement. Audio and weights never go in git:
generated audio is in `/home/jovyan/ba-work/bakeoff/`, reference and calibration clips in
`/home/jovyan/ba-work/refs/` and `/home/jovyan/ba-work/calib/`, scratch in `/tmp/ba`.

## Layout

| Path | What |
|---|---|
| `texts/egy.jsonl` | 40 colloquial Egyptian sentences (12 everyday, 8 questions, 6 emotion, 8 with digits, 6 long). Digit rows carry `text_spoken`, a hand-written Egyptian spelled-out form used as the ASR reference |
| `texts/egy_diacritised.jsonl` | 10 of them, diacritised for Egyptian pronunciation |
| `texts/codeswitch.jsonl` | 6 Arabic-English code-switched sentences (Latin script inside Arabic) |
| `texts/en.jsonl`, `texts/fr.jsonl` | 20 sentences each, 8-30 words, some with digits |
| `texts/long.jsonl` | one ~120-word passage per language |
| `refs.jsonl` | 12 FLEURS reference voices (6 ar_eg: 3 male from the train split because the ar_eg test split is all female, 3 female; 3 en_us; 3 fr_fr: 2 male + 1 female because only one clean, distinct female fr_fr speaker was found) with a same-speaker second clip each |
| `calib.jsonl` | real-speech calibration clips: 50 lahgtna-v2 `eg` test rows, 50 NileTTS eval rows, 50 FLEURS ar_eg, 30 en_us, 30 fr_fr |
| `scripts/` | all code (below) |
| `results/` | `bakeoff.json`, `bakeoff_utterances.csv`, `bakeoff.md`, `refs_selection.json`, `observations.md` |

## Environments

`source /tmp/ba/env.sh` first (unsets the server's HF token; all downloads are anonymous).
One venv per family, Python 3.11, created with `uv venv -p 3.11 /tmp/ba/venvs/<name>`:

| venv | install |
|---|---|
| `cb` | `uv pip install chatterbox-tts==0.1.7 "setuptools<81"`, then `uv pip uninstall spacy-pkuseg` (setuptools: `perth` watermarker needs `pkg_resources`; pkuseg is only for Chinese and downloads a model from GitHub on first use) |
| `vox` | `uv pip install voxcpm` (2.0.3, torch 2.15 cu132) |
| `omni` | `uv pip install omnivoice voicetut-tts` (0.2.1 / 0.1.1) |
| `eval` | `uv pip install torch==2.6.0 torchaudio==2.6.0 transformers==4.46.3 speechbrain==1.0.3 jiwer soundfile librosa pandas huggingface_hub datasets accelerate speechmos onnxruntime num2words` |

## Re-run

```bash
source /tmp/ba/env.sh
E=/home/jovyan/Mahmoud-Project/BA/eval/scripts
# FLEURS: data/{ar_eg,en_us,fr_fr}/test.tsv + audio/test.tar.gz, plus ar_eg train.tsv/train.tar.gz
#   (male Egyptian voices) and fr_fr dev, downloaded with hf_hub_download into /tmp/ba/data/fleurs
#   and extracted to /tmp/ba/data/fleurs_wav/<cfg>/<split>/
/tmp/ba/venvs/eval/bin/python $E/select_refs.py      # -> refs.jsonl, results/refs_selection.json (GPU, ~2 min)
/tmp/ba/venvs/eval/bin/python $E/prepare_calib.py    # -> calib.jsonl (CPU, downloads)
$E/run_all.sh                                        # synthesis for every system, listening-pack audio, scoring, aggregation
# CPU speed (2 threads), one system at a time, nothing else running:
taskset -c 0,1 $E/run_synth.sh cb  $E/synth_chatterbox.py cb-v3 full --cpu-bench
taskset -c 0,1 $E/run_synth.sh cb  $E/synth_chatterbox.py cb-v2 full --cpu-bench
taskset -c 0,1 $E/run_synth.sh vox $E/synth_voxcpm.py voxcpm2 full --cpu-bench
/tmp/ba/venvs/eval/bin/python $E/aggregate.py        # re-aggregate after CPU numbers exist
/tmp/ba/venvs/eval/bin/python $E/make_pack.py        # listening pack
```

Each synthesis script is resumable: finished utterances (listed in `<system>/meta.jsonl`) are skipped.
Seeds are fixed per utterance: `crc32(utt_id) mod 2^31` (`matrix.py`).

## Test matrix (`scripts/matrix.py`)

Full matrix (cb-v3, cb-v2, voxcpm2), 324 utterances: Egyptian 40 x 3 of 6 voices (rotating, each voice 20 times),
diacritised 10 x 2, code-switch 6 x 3, English 20 x 3, French 20 x 3, cross-language 10 en + 10 fr x 2 Egyptian voices,
long passages 3 x 2. Egyptian fine-tunes (namaa, lahgtna): Egyptian, diacritised, code-switch, Egyptian passage (160).
Reference-only (omnivoice, voicetut): those plus English and the English passage (222). VoxCPM2 prompt+transcript
variant: the 120 main Egyptian utterances. Listening pack: 6 sentences x 2 voices for every system.

## Metrics (`scripts/score.py`, `scripts/aggregate.py`)

See the docstrings: Whisper large-v3 with forced language, CER/WER after normalisation; ECAPA and WavLM-SV cosine;
`badrex/mms-300m-arabic-dialect-identifier`; DNSMOS (speechmos) as the MOS predictor; failure rule F1-F4; G1.

## Phase 2: the BA engine (`ba` system)

venv `ba`: `uv venv -p /opt/conda/bin/python3.12 /tmp/ba/venvs/ba`, then
`uv pip install "torch==2.13.0" "voxcpm==2.0.3" "gradio==6.31.0" spaces audioseal pyloudnorm pytest jiwer`
and `uv pip install -e "BA[test]"` (Python 3.12 and torch 2.13.0 match what Hugging Face ZeroGPU supports).
Generated audio: `/home/jovyan/ba-work/engine/<experiment>/`.

| Script | What |
|---|---|
| `engine_common.py` | tuning / held-out split (fixed before any run), experiment settings |
| `synth_ba.py <exp>` | synthesis with `ba_tts` for `ref`, `cs`, `chunk`, `sweep`, `full` (resumable) |
| `score_manifest.py <exp>` | the Phase 1 scorer on an experiment (venv `eval`) |
| `analyze_engine.py <exp>` | decision rules (in its docstring) -> `reference_thresholds.json`, `codeswitch_policy.json`, `chunk_length.json`, `sweep.json` |
| `asr_check.py` | Whisper large-v3-turbo vs large-v3 on the calibration clips -> `asr_check.json` |
| `watermark_check.py` | AudioSeal detection on BA outputs, MP3 128k/64k, 16 kHz, and on unwatermarked audio -> `watermark.json` |
| `bench_engine.py [--no-optimize]` | load time, generation time, peak VRAM on the A40 -> `bench.json` |
| `engine_report.py` | BA vs raw VoxCPM2 on held-out texts, gate G3 -> `engine.json`, `engine.md` |
| `run_engine.sh <steps>` | runs synthesis + scoring per step, one GPU job at a time |

```bash
E=BA/eval/scripts
$E/run_engine.sh asr ref cs chunk sweep
/tmp/ba/venvs/eval/bin/python $E/analyze_engine.py ref   # likewise cs, chunk, sweep; copy decisions into src/ba_tts/defaults.py
$E/run_engine.sh full
/tmp/ba/venvs/ba/bin/python $E/watermark_check.py
/tmp/ba/venvs/ba/bin/python $E/bench_engine.py; /tmp/ba/venvs/ba/bin/python $E/bench_engine.py --no-optimize
/tmp/ba/venvs/eval/bin/python $E/engine_report.py
```
