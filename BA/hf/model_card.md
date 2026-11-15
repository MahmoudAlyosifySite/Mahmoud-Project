---
license: other
license_name: ba-community-license-1.0
license_link: LICENSE
language:
- ar
- arz
- en
- fr
pipeline_tag: text-to-speech
tags:
- voice-cloning
- text-to-speech
- tts
- egyptian-arabic
- arabic
- english
- french
- zero-shot-voice-cloning
- audio-watermarking
- consent
thumbnail: https://huggingface.co/mahmoudalyosify/BA/resolve/main/assets/social-card.png
---

![BA: Your Voice. Your Identity.](https://huggingface.co/mahmoudalyosify/BA/resolve/main/assets/banner.png)

# BA · Your Voice. Your Identity.

BA speaks any text in your own voice, from a 10–15 second recording of you: **Egyptian Arabic**, **English**, and **French (beta)**. Every output carries an inaudible watermark. Use only your own voice or a voice whose owner has given you explicit permission.

**Demo:** [the BA Space](https://huggingface.co/spaces/mahmoudalyosify/BA): record or upload your voice, tick the consent box, type, listen and download. It runs on ZeroGPU; the first request after a pause takes about 30 s, later ones about 10 s for a short sentence.

## What BA is, and what it is built from

BA v0.1.0 is two things:

1. **The VoxCPM2 weights, unmodified.** BA ships the unmodified VoxCPM2 weights from [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) at commit `7936080b41163874f38610f46f438e2c0757f9fd`. VoxCPM2 is a 2B-parameter, 48 kHz multilingual text-to-speech model by OpenBMB, released under Apache-2.0; all credit for the model goes to its authors. The weight and tokenizer files here are byte-for-byte copies of the files at that commit (see `NOTICE`). BA v0.1 does **not** fine-tune or otherwise change them, which is why this repository declares no `base_model` relation: the Hub's relations (adapter, merge, quantized, finetune) all describe changed weights.
2. **The BA engine** (`ba_tts`, this repository's wheel), which decides how the model is driven:
   - prepares the reference recording (mono, trimmed to speech, loudness-normalised, the cleanest stretch chosen; noisy, clipped or too-short audio is refused with a clear message);
   - transcribes the reference with Whisper large-v3-turbo when you give no transcript, and clones with the reference *and* its transcript;
   - writes digits, times, prices, percentages and symbols as Egyptian, English or French words, and writes common English loanwords in Egyptian text in Arabic script;
   - splits long text at sentence boundaries, checks every chunk (duration against text length, empty or clipped audio) and regenerates failures with a new seed;
   - loudness-normalises the result and embeds an [AudioSeal](https://huggingface.co/facebook/audioseal) watermark.

VoxCPM2 was chosen in a bake-off of eight voice-cloning systems on Egyptian Arabic (below). Of the systems eligible as a base (permissively licensed weights with documented training data), it was the only one whose Egyptian output a dialect classifier labelled as Egyptian most of the time.

## Quick start

Python 3.10 or newer. A CUDA GPU is strongly recommended (peak GPU memory in our runs: 8.11 GB allocated, 10.99 GB reserved); CPU works but is slow.

```bash
pip install https://huggingface.co/mahmoudalyosify/BA/resolve/main/ba_tts-0.1.0-py3-none-any.whl
```

```python
import soundfile as sf
from ba_tts import BA, detect_watermark

ba = BA.from_pretrained("mahmoudalyosify/BA")      # downloads the weights (about 5 GB) on first use
voice = ba.create_voice("my_voice.wav", language="arz")   # 10-15 s of YOUR voice; transcript optional

sr, wav = ba.speak("الميعاد اتغير للساعة 7:30، والتذكرة بقت ب 150 جنيه.", voice, language="arz")
sf.write("out.wav", wav, sr)                        # 48 kHz, watermarked

print(detect_watermark(wav, sr))                    # {'watermarked': True, 'score': ..., 'bits_match': ...}
```

`language` is `"arz"` (Egyptian Arabic), `"en"` or `"fr"`. You can pass several recordings (`create_voice(["a.wav", "b.wav"])`), give the transcript yourself (`transcript="..."`), save a voice for later (`voice.save("me.npz")`, `Voice.load("me.npz")`), and set `seed`, `cfg_value` or `inference_timesteps` in `speak`. Text longer than one chunk is split and joined automatically (up to 5,000 characters).

Command line:

```bash
ba-tts --ref my_voice.wav --lang arz --text "أنا رايح الشغل دلوقتي وهرجع على المغرب." --out out.wav
ba-tts --ref my_voice.wav --lang en --text-file speech.txt --out speech.wav --save-voice me.npz
ba-tts --detect out.wav          # is this file a BA output?
```

## Languages

A language is listed only if the full system passes gate G3 on held-out texts: failure rate at most 5%; intelligibility within a fixed bound; speaker similarity at least 80% of the similarity between two real recordings of the same speaker; and, for Egyptian, a dialect-ID share at least 60% of the share measured on real colloquial Egyptian speech (otherwise it would be listed only as "Arabic").

| Language | Code | Status | Failure rate | Intelligibility | Speaker similarity (ECAPA) | Egyptian dialect share |
|---|---|---|---|---|---|---|
| Egyptian Arabic | `arz` | listed | 0/130 | CER 0.030 (bound 0.171) | 0.769 (bar 0.696) | 0.889 (bar 0.54) |
| English | `en` | listed | 0/64 | WER 0.003 (bound 0.08) | 0.779 (bar 0.699) | – |
| French | `fr` | **beta** | 0/82 | WER 0.030 (bound 0.10) | 0.792 (bar 0.720) | – |

French passed the same gate but is marked beta: it rests on three French reference voices, its generation settings were not tuned separately (it uses the English values), and it is the newest language. Cross-language use (an Egyptian voice speaking English or French) works, with lower similarity (see Limitations).

## Measured results

### Evaluation set

- **Reference voices:** 12 speakers from [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0): 6 Egyptian (`ar_eg`, 3 male and 3 female), 3 American English, 3 French; clean clips of 8–15 s. They are used for measurement only; neither these clips nor anything cloned from them is published.
- **Texts:** 40 colloquial Egyptian sentences (everyday talk, questions, numbers and dates, long sentences, a diacritised copy of ten), 20 English, 20 French, 6 Arabic–English code-switched sentences, and one passage of about 120 words per language. Some texts were used to tune the engine; **every number below excludes them** (held-out texts only). BA and raw VoxCPM2 are compared on the same texts, voices and seeds.
- **Automatic metrics, each calibrated on real speech:**
  - words: CER (Egyptian) and WER (English, French) from Whisper large-v3 after normalisation. On real speech the same ASR gives CER 0.071 on colloquial Egyptian and WER 0.052 (English) and 0.062 (French) on FLEURS;
  - voice: cosine similarity of ECAPA-TDNN speaker embeddings between the reference and the output. Two different real clips of the same speaker score 0.870 (Egyptian), 0.874 (English), 0.900 (French); clips of different speakers 0.084–0.184;
  - dialect: share of outputs that the [`badrex/mms-300m-arabic-dialect-identifier`](https://huggingface.co/badrex/mms-300m-arabic-dialect-identifier) classifier labels Egyptian. colloquial Egyptian speech: 0.90 (lahgtna test set; the gate's reference) and 0.98 (NileTTS); Egyptians reading Modern Standard Arabic (FLEURS): 0.34;
  - naturalness: DNSMOS P.808 (real speech: 3.515–3.990);
  - failures: empty, cut off, or a duration far outside what the text length implies.

### BA engine vs raw VoxCPM2 (held-out texts)

| Condition | n | Error BA | Error raw | Similarity BA | Similarity raw | Egyptian share BA | Egyptian share raw | DNSMOS BA | DNSMOS raw |
|---|---|---|---|---|---|---|---|---|---|
| Egyptian sentences (CER) | 72 | **0.030** | 0.069 | **0.769** | 0.713 | **0.889** | 0.819 | 3.527 | 3.349 |
| Egyptian with digits (CER) | 18 | **0.073** | 0.122 | 0.777 | 0.730 | 0.833 | 0.667 | 3.562 | 3.313 |
| Egyptian, diacritised (CER) | 20 | 0.040 | 0.045 | 0.768 | 0.694 | 0.600 | 0.400 | 3.550 | 3.410 |
| Arabic–English code-switch (CER) | 18 | 0.284 | 0.274 | 0.773 | 0.739 | 0.944 | 0.722 | 3.515 | 3.387 |
| English (WER) | 42 | 0.003 | 0.000 | **0.779** | 0.655 | – | – | 3.861 | 3.702 |
| French (WER) | 60 | 0.030 | 0.030 | 0.792 | 0.759 | – | – | 3.727 | 3.631 |
| Egyptian voice → English (WER) | 20 | 0.000 | 0.000 | 0.730 | 0.639 | – | – | 3.603 | 3.529 |
| Egyptian voice → French (WER) | 20 | 0.062 | 0.137 | 0.696 | 0.570 | – | – | 3.576 | 3.385 |
| ~120-word Egyptian passage (CER) | 2 | 0.040 | 0.033 | 0.823 | 0.762 | 1.000 | 1.000 | 3.558 | 3.522 |
| ~120-word English passage (WER) | 2 | 0.000 | 0.000 | 0.807 | 0.703 | – | – | 3.942 | 3.762 |
| ~120-word French passage (WER) | 2 | 0.010 | 0.010 | 0.839 | 0.797 | – | – | 3.843 | 3.781 |

Failures: none for BA in any condition; raw VoxCPM2 produced 2 empty English outputs. On the code-switch set the plain CER is slightly worse for BA because BA writes loanwords in Arabic script, which changes what Whisper writes; with both scripts mapped to one form, BA scores 0.074 against 0.084 for raw VoxCPM2. Raw VoxCPM2 would fail G3 for English on similarity (0.655); BA passes.

### Why VoxCPM2 (bake-off, Egyptian sentences, 3 voices × 32 sentences)

| System (raw, before the BA engine) | Weights licence | Egyptian CER | Labelled Egyptian | Similarity (ECAPA) | Failure rate (all conditions) |
|---|---|---|---|---|---|
| Chatterbox Multilingual V3 | MIT | 0.035 | 0.208 | 0.706 | 6/324 |
| Chatterbox Multilingual V2 | MIT | 0.056 | 0.167 | 0.739 | 10/324 |
| **VoxCPM2** | Apache-2.0 | 0.065 | **0.823** | 0.714 | 3/324 |

Chatterbox says the right words with the wrong dialect (mostly labelled Levantine, Gulf or MSA). Only permissively licensed weights were eligible; non-commercial systems and community fine-tunes with undocumented training data were measured for reference only.

**Listening check.** The owner, a native Egyptian speaker, rated the raw bake-off outputs blind (system names hidden, order shuffled): one rater, 6 Egyptian sentences per system, before the BA engine. Both Chatterbox bases were rated not Egyptian on 6/6 sentences; VoxCPM2 was never rated not Egyptian (3 Egyptian, 3 partly), which agrees with the dialect classifier on the choice of base. By ear, VoxCPM2 with the reference transcript (BA's default mode) sounded less Egyptian than without it (1 vs 3 fully Egyptian) while getting more words right (5 vs 3 of 6); with 6 clips and one rater this difference is not settled. On the digits sentence the rater noted that in most raw systems the number was not said the Egyptian way; BA's Egyptian number normaliser addresses this, but its effect is so far measured by ASR only. BA's own outputs have not yet been rated by ear.

### Speed and memory (NVIDIA A40, no `torch.compile`, as on ZeroGPU)

| | |
|---|---|
| Model load | 19.9 s |
| Preparing a voice from a 10 s clip | 0.3 s |
| Egyptian sentence, 23 words → 8.25 s of audio | 7.89 s |
| Egyptian passage, 118 words → 39.99 s of audio | 38.03 s |
| English passage, 158 words → 41.22 s of audio | 39.0 s |
| Peak GPU memory (allocated / reserved) | 8.08 / 10.35 GB |

With `torch.compile` (the default on a plain CUDA GPU) the 118-word Egyptian passage took 27.49 s. On 2 CPU threads raw VoxCPM2 ran at about 30× real time in the bake-off, so a GPU is needed for practical use.

### Watermark detection

| Audio | n | Detected |
|---|---|---|
| BA output, 48 kHz WAV | 60 | 60 |
| BA output, MP3 128 kbit/s | 60 | 60 |
| BA output, MP3 64 kbit/s | 60 | 60 |
| BA output, resampled to 16 kHz | 60 | 60 |
| Raw VoxCPM2 output (no watermark) | 60 | 0 |
| Real speech | 53 | 0 |

## How it works

```
your recording(s) ──► reference preparation ──► Whisper large-v3-turbo transcript (if none given)
                                 │                               │
                                 └──────► VoxCPM2 prompt cache (reference audio + transcript)
text ──► language normaliser (numbers, symbols, loanwords) ──► sentence chunks
                                                                 │
          VoxCPM2 generation per chunk, fixed voice and seed ◄───┘
                     │
          chunk check (duration vs text length, empty, clipped) ──► retry with a new seed (up to 2)
                     │
          join with pauses ──► loudness normalisation (−20 LUFS) ──► AudioSeal watermark ──► 48 kHz audio
```

The per-language generation settings (guidance 2.0, 10 steps), the chunk length, the reference noise thresholds and the loanword policy were chosen by measurement on the tuning texts. They are recorded in `ba_config.json`, which also pins the revisions of the ASR model and the watermark weights.

## Limitations

- **No native-listener review of BA's outputs yet.** The only listening test so far rated the raw bake-off outputs (one rater, 6 sentences per system; see Listening check). BA's own outputs, with its normaliser and reference transcript, have not been rated by ear. Every quality number on this card is automatic.
- **Automatic metrics have blind spots.** One dialect classifier decides "Egyptian"; it labels Egyptians reading Modern Standard Arabic as Egyptian only 34% of the time, and the six Egyptian reference voices are FLEURS clips of read MSA, not colloquial speech. DNSMOS was trained on English and is a weak judge of Arabic. Similarity is one embedding model (ECAPA).
- **Digits are measured by ASR.** Whisper writes most spoken numbers back as digits, so the digit-sentence CER shows whether the right number was said, not whether it was said the Egyptian way. The normaliser itself is covered by unit tests.
- **Small sets.** 18 digit sentences, 18 code-switched utterances and 2 passages per language; French rests on 3 voices.
- **Cross-language similarity is lower.** An Egyptian voice speaking French reaches similarity 0.696, exactly 80% of the Egyptian same-speaker ceiling (0.870); speaking English, 0.730. Cross-language use was not part of the gate.
- **Code-switching.** Inside Egyptian text, a fixed list of common English loanwords is written in Arabic script and short upper-case acronyms are spelled letter by letter; other Latin words are left as written and may be mispronounced.
- **Speed.** Timings are on an A40. The Space runs on shared ZeroGPU hardware, where speed and queueing differ. CPU-only use is very slow.
- **Reference quality matters.** Noisy, reverberant, clipped or very short recordings are refused or give a weaker likeness.
- **No fine-tune.** BA v0.1 uses the base weights; anything VoxCPM2 cannot say well, BA cannot either.

## Responsible use

- **Consent.** Use only your own voice, or a voice whose owner has given you explicit permission for this use. Do not impersonate anyone, deceive, defraud, harass, or make non-consensual content. These terms are part of the licence.
- **Watermark.** Every BA output carries an inaudible AudioSeal watermark with the 16-bit message "BA". Anyone can check a file: `ba-tts --detect file.wav`, or `detect_watermark(wav, sr)` in Python. Detection survived MP3 at 64 kbit/s and resampling in our tests; it was not tested against deliberate removal attacks. Do not remove or circumvent it, and do not publish audio made with the watermark switched off (`watermark=False` exists for research measurement).
- **The Space.** Generation is blocked until you tick the consent box. Uploads are processed in a per-request temporary directory that is deleted after the request, together with Gradio's own copies; the voice prepared from them lives only in your browser session's memory. Generated audio sits in Gradio's cache only so the player and download work, and is swept every few minutes. No text or audio is logged; flagging and analytics are off. Nothing you upload is used for training.

## Audio samples (synthetic voices)

These voices are **synthetic**: they belong to no real person. Each reference voice was created with VoxCPM2's voice-design mode from a text description, then cloned by BA from that reference alone (the reference was transcribed automatically, as in the Space). No real person's voice is published here.

| Voice | Reference (voice design) | Egyptian 1 | Egyptian 2 | English | French |
|---|---|---|---|---|---|
| Synthetic man (“an Egyptian man in his thirties, warm, calm voice”) | <audio controls src="https://huggingface.co/mahmoudalyosify/BA/resolve/main/samples/synthetic_man_reference.mp3"></audio> | <audio controls src="https://huggingface.co/mahmoudalyosify/BA/resolve/main/samples/synthetic_man_arz1.mp3"></audio> | <audio controls src="https://huggingface.co/mahmoudalyosify/BA/resolve/main/samples/synthetic_man_arz2.mp3"></audio> | <audio controls src="https://huggingface.co/mahmoudalyosify/BA/resolve/main/samples/synthetic_man_en.mp3"></audio> | <audio controls src="https://huggingface.co/mahmoudalyosify/BA/resolve/main/samples/synthetic_man_fr.mp3"></audio> |
| Synthetic woman (“an Egyptian woman in her thirties, clear, friendly voice”) | <audio controls src="https://huggingface.co/mahmoudalyosify/BA/resolve/main/samples/synthetic_woman_reference.mp3"></audio> | <audio controls src="https://huggingface.co/mahmoudalyosify/BA/resolve/main/samples/synthetic_woman_arz1.mp3"></audio> | <audio controls src="https://huggingface.co/mahmoudalyosify/BA/resolve/main/samples/synthetic_woman_arz2.mp3"></audio> | <audio controls src="https://huggingface.co/mahmoudalyosify/BA/resolve/main/samples/synthetic_woman_en.mp3"></audio> | <audio controls src="https://huggingface.co/mahmoudalyosify/BA/resolve/main/samples/synthetic_woman_fr.mp3"></audio> |

Texts: Egyptian 1 «أنا رايح الشغل دلوقتي وهرجع على المغرب إن شاء الله.» · Egyptian 2 «الميعاد اتغير للساعة 7:30، والتذكرة بقت ب 150 جنيه.» · English “The meeting has been moved to Thursday afternoon, so please update your calendar.” · French « Le concert commence à 21h, mais il vaut mieux arriver un peu en avance. »

Each sample was generated once with the default seed. The design seed of each voice was the first one, in a fixed list, that gave a reference of 8–12 s which the dialect classifier labelled Egyptian. All samples, references included, carry the watermark. The samples illustrate; they are not part of the measurements above.

## Licence

- **BA's own work** (the `ba_tts` code, `ba_config.json`, the normaliser lexicons, this card and the docs, the samples, the BA name and brand assets): **BA Community License 1.0** (`LICENSE`). In short: free for non-commercial use, modification and sharing, with attribution and under the same terms; commercial use only with the owner's written permission (contact through the [Hugging Face profile](https://huggingface.co/mahmoudalyosify)); use a voice only if it is yours or you have the speaker's explicit permission; no impersonation, fraud, deception, harassment or non-consensual content; do not remove or circumvent the watermark; no warranty.
- **Third-party components keep their own licences:** the VoxCPM2 weights are Apache-2.0 (`LICENSE-VoxCPM2`, `NOTICE`), Whisper large-v3-turbo is MIT, AudioSeal is MIT. Nothing in the BA licence restricts the rights those licences give you; you may use the VoxCPM2 weights under Apache-2.0 independently of BA.
- The licence has not yet been reviewed by a lawyer.

## Files in this repository

| File | What |
|---|---|
| `model.safetensors`, `audiovae.pth`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tokenization_voxcpm2.py` | VoxCPM2, unmodified (OpenBMB, Apache-2.0) |
| `ba_config.json` | BA version, pinned revisions (VoxCPM2, Whisper, AudioSeal), per-language defaults |
| `ba_tts-0.1.0-py3-none-any.whl` | the BA engine and CLI |
| `LICENSE`, `LICENSE-VoxCPM2`, `NOTICE` | licences and attribution |
| `samples/`, `assets/` | synthetic-voice samples, banner and social card |

## Citations

```bibtex
@article{voxcpm2_2026,
  title   = {VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning},
  author  = {VoxCPM Team},
  journal = {GitHub},
  year    = {2026}
}
@article{voxcpm2025,
  title   = {VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning},
  author  = {Zhou, Yixuan and Zeng, Guoyang and Liu, Xin and Li, Xiang and Yu, Renjie and Wang, Ziyang and Ye, Runchuan and Sun, Weiyue and Gui, Jiancheng and Li, Kehan and Wu, Zhiyong and Liu, Zhiyuan},
  journal = {arXiv preprint arXiv:2509.24650},
  year    = {2025}
}
@article{radford2022whisper,
  title   = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author  = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal = {arXiv preprint arXiv:2212.04356},
  year    = {2022}
}
@inproceedings{sanroman2024audioseal,
  title     = {Proactive Detection of Voice Cloning with Localized Watermarking},
  author    = {San Roman, Robin and Fernandez, Pierre and Elsahar, Hady and D{\'e}fossez, Alexandre and Furon, Teddy and Tran, Tuan},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024}
}
@article{conneau2022fleurs,
  title   = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author  = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal = {arXiv preprint arXiv:2205.12446},
  year    = {2022}
}
```
