# BA Phase 4: publish and validate — report

Executor run of 2026-11-12, resumed 2026-11-14. Progress log: `/home/jovyan/ba-work/publish/progress.md`.

## Summary

| Item | Status |
|---|---|
| Model repo [`mahmoudalyosify/BA`](https://huggingface.co/mahmoudalyosify/BA) | **Public, not gated.** Current commit `66c979c1c7b5bf0e1358543848b8beed2e455bbd` |
| Wheel `ba_tts-0.1.0-py3-none-any.whl` at the repo root | Installs from the public URL with plain `pip` (validation B) |
| Space `mahmoudalyosify/BA` | **Not created.** The Hub refuses with HTTP 402 (account limit, below). The bundle is staged and was checked locally against the public model repo |
| Validation A (clean venv, private repo, token in-process) | Passed: 6/6 outputs, scores in line with `engine.md`, watermark 6/6 |
| Validation B (clean venv, anonymous) | Model part passed: outputs **bit-identical** to A, watermark 6/6. Space part **not possible** (no Space) |
| Tests | 195 passed (190 original + 5 new), GPU tests included, against the public `mahmoudalyosify/BA` |

## 1. Model repository

URL: https://huggingface.co/mahmoudalyosify/BA (public, `gated: false`, licence `other` / `ba-community-license-1.0`).

Commits:

| Commit | What |
|---|---|
| `481b8487dc49` | initial commit (created by the Hub) |
| `ec718bb787e004640b063ed4575ce91bc2defe53` | BA v0.1.0: weights, `ba_config.json`, wheel, card, licences, samples (uploaded private) |
| `2bad999f3b45f2971a32a00efe02afb6cd5ab519` | card: owner's blind listening check added |
| `62a1dd35cde537ca9d77c64c7b181d1ac7cf0ab8` | card metadata: `base_model` removed; explicit VoxCPM2 credit in card and `NOTICE` |
| `66c979c1c7b5bf0e1358543848b8beed2e455bbd` | card: "the demo Space is not online yet" |

Pinned upstream revisions:

| Component | Repo | Revision |
|---|---|---|
| VoxCPM2 weights (copied, unmodified) | `openbmb/VoxCPM2` | `7936080b41163874f38610f46f438e2c0757f9fd` |
| ASR for reference transcripts | `openai/whisper-large-v3-turbo` | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` |
| Watermark weights | `facebook/audioseal` | `3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba` |

**Unmodified weights, verified.** All seven VoxCPM2 files in the BA repo have the same size, git blob id and LFS sha256 as the upstream files at the pinned commit (`files_metadata` comparison through the Hub API). Not copied: upstream `README.md` (replaced by the BA card) and `.gitattributes`.

Files (public listing at `62a1dd35`; total 4,961,651,308 bytes; `README.md` is now 21.6 kB at `66c979c1`):

| File | Bytes |
|---|---|
| `.gitattributes` | 1,665 |
| `LICENSE` (BA Community License 1.0) | 3,712 |
| `LICENSE-VoxCPM2` (OpenBMB's Apache-2.0 file from the voxcpm 2.0.3 package) | 11,298 |
| `NOTICE` | 1,538 |
| `README.md` (model card) | 21,608 |
| `assets/banner.png` | 16,511 |
| `assets/social-card.png` (card `thumbnail`) | 12,133 |
| `audiovae.pth` | 376,951,122 |
| `ba_config.json` | 2,482 |
| `ba_tts-0.1.0-py3-none-any.whl` (sha256 `c6edc3db…abe7fa`, identical locally and from the public URL) | 45,727 |
| `config.json` | 4,336 |
| `model.safetensors` | 4,580,080,592 |
| `samples/synthetic_{man,woman}_{reference,arz1,arz2,en,fr}.mp3` (10 files) | 50,733 – 148,269 each |
| `special_tokens_map.json` | 1,632 |
| `tokenization_voxcpm2.py` | 2,895 |
| `tokenizer.json` | 3,676,772 |
| `tokenizer_config.json` | 5,059 |

**Card metadata.** `license: other`, `license_name: ba-community-license-1.0`, `license_link: LICENSE`, `language: [ar, arz, en, fr]`, `pipeline_tag: text-to-speech`, tags (voice-cloning, egyptian-arabic, …), `thumbnail`. **No `base_model`.** With it, the Hub inferred and displayed `base_model:finetune:openbmb/VoxCPM2`. The Hub's relations (adapter, merge, quantized, finetune) all describe changed weights, so none fits an unmodified copy. It was removed on the planner's instruction, and the public tags no longer contain it. Credit is explicit in the card text and in `NOTICE`: "BA ships the unmodified VoxCPM2 weights from openbmb/VoxCPM2 at commit 7936080b…". No `library_name` is set: `voxcpm` would advertise a loader that bypasses the BA engine.

**Card numbers.** Every measured number on the card is from `eval/results/engine.md` or `bakeoff.md`. The listening-check paragraph is from `eval/results/listening_test.md`; the tally was re-counted from `listening_test.json` and matches. The sample and validation scores below are **not** on the card.

## 2. The `ba_tts` changes and tests

- `src/ba_tts/config.py` (new): builds and reads `ba_config.json` (format `ba-config-1`): BA version, pinned revisions of the base, ASR and watermark, per-language defaults (from `defaults.py`), reference thresholds. At load time the engine takes the generation settings, the chunk length and the ASR id and revision from it; a plain VoxCPM2 directory falls back to `defaults.py`.
- `BA.from_pretrained()` now defaults to `mahmoudalyosify/BA`. It downloads only the files the engine needs (`allow_patterns`: weights, tokenizer, `ba_config.json`; not the wheel or samples) and accepts `revision`, `token`, `cache_dir` and `local_files_only`. `device` defaults to CUDA if available, else CPU.
- Whisper is now loaded at its pinned revision (`defaults.ASR_REVISION`).
- CLI `--model`, the Space's `BA_MODEL_REPO` default and the GPU tests' default now point to `mahmoudalyosify/BA`.
- `tests/test_config.py` (5 new tests: config mirrors `defaults.py`, round trip, fallback, unknown format rejected, download patterns cover the VoxCPM2 loader).
- `pyproject.toml`: licence text, inline package description (`BA/README.md` is the stale vision document), project URLs. `BA/LICENSE` ships in the wheel's `dist-info/licenses/`.

pytest (`BA_RUN_GPU_TESTS=1`, venv `ba`, A40):
- 2026-11-12, before the repo existed, `BA_MODEL_REPO=openbmb/VoxCPM2`: **195 passed**, 13 warnings, 43.55 s.
- 2026-11-14, default `mahmoudalyosify/BA` (public, anonymous cache): **195 passed**, 13 warnings, 50.02 s.

## 3. Audio samples (synthetic voices only, D7)

Made by `BA/hf/make_samples.py` (GPU) and scored by `BA/hf/score_samples.py` (the Phase 1 scorer). Audio and scores: `/home/jovyan/ba-work/publish/samples/`.

- Two voices from VoxCPM2 voice design: "An Egyptian man in his thirties, warm, calm voice, speaking Egyptian Arabic" and "An Egyptian woman in her thirties, clear, friendly voice, speaking Egyptian Arabic". The reference was loudness-normalised and watermarked, then cloned by BA with no transcript given (Whisper transcribes it, the product's default path).
- **Design-seed rule:** the first seed in a fixed list whose clip is 8–12 s and is labelled Egyptian by the dialect classifier; this gave man 41, woman 11. Candidates and their scores: `samples/candidates/`. The rule was added after a first attempt: the first length-only pick for the man (seed 37) was labelled MSA by the classifier, and his clones read as Gulf/Levantine. Disclosed on the card.
- French sentence changed before the final run. "… ferme plus tôt …" was transcribed as the homophone "plutôt", giving WER 0.133, a metric artefact. The new sentence is the Space's other French example.
- No sample failed the engine's checks; none was dropped. All are within the G3 bound for their language. Watermark detected on all 10 published MP3s (score 1.0, bits 1.0).

| Sample | CER / WER (best of spelled / digit text) | ECAPA vs synthetic ref | Dialect | DNSMOS |
|---|---|---|---|---|
| man reference | 0.009 (vs design text) | – | Egyptian | 3.895 |
| man arz1 / arz2 | 0.020 / 0.051 | 0.778 / 0.814 | Egyptian / Egyptian | 3.779 / 3.638 |
| man en / fr | 0.000 / 0.000 | 0.780 / 0.727 | – | 3.615 / 3.571 |
| woman reference | 0.037 | – | Egyptian | 3.634 |
| woman arz1 / arz2 | 0.000 / 0.122 | 0.866 / 0.764 | Egyptian / Egyptian | 3.405 / 3.559 |
| woman en / fr | 0.000 / 0.000 | 0.684 / 0.724 | – | 3.557 / 3.728 |

## 4. Space `mahmoudalyosify/BA`: not created

Exact errors from `HfApi.create_repo(..., repo_type="space", space_sdk="gradio")` with the owner's token:

1. Without hardware: `402 Payment Required`: "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription."
2. With `space_hardware="zero-a10g"`: `402 Payment Required`: "Free users are limited to 2 ZeroGPU Spaces. Subscribe to PRO for up to 10: https://huggingface.co/settings/billing". Same result on 2026-11-12 and twice on 2026-11-14.

Cause: the account (free, not PRO) already has two Spaces set to ZeroGPU: `mahmoudalyosify/horus-chat-api` (RUNTIME_ERROR) and `mahmoudalyosify/MinimalLM` (SLEEPING). They were read, not touched. D11's premise is correct (free accounts may host 2 ZeroGPU Spaces), but both slots are in use. "Leave the Space on CPU" was not possible either: a new Gradio Space on cpu-basic needs PRO. I did not try workarounds such as creating a static Space and switching its SDK.

**Owner action needed (one of):** move one of those two Spaces off ZeroGPU (Settings → hardware) or delete it; or subscribe to PRO. Then run, from the server:

```bash
source /tmp/ba/env.sh && python BA/hf/publish.py stage && python BA/hf/publish.py upload-space   # venv /tmp/ba/venvs/ba
python BA/hf/publish.py public space && python BA/hf/publish.py runtime
```

`upload-space` creates the Space with `zero-a10g` from the start, sets the plain variable `BA_MODEL_REPO=mahmoudalyosify/BA` (no secrets), and uploads the bundle. Then validation B's Space part can run: `python BA/hf/space_check.py mahmoudalyosify/BA <out_dir>` (consent unticked must be refused; consent ticked must return watermarked audio), scored with `BA/hf/score_samples.py`.

**Staged bundle** (`/tmp/ba/publish/space`, 25 files, 152,649 bytes): `README.md` (2,798), `app.py` (11,074), `theme.py` (8,530), `style.css` (3,814), `header.html` (5,204), `requirements.txt` (327), `packages.txt` (7, ffmpeg), `LICENSE` (3,712), `social-card.png` (12,133), `ba_tts/` (15 files: a copy of `BA/src/ba_tts`, so the Space does not depend on the wheel URL).

**Front matter, checked against the Spaces config reference** (huggingface.co/docs/hub/spaces-config-reference, read 2026-11-12):
- `sdk: gradio`, `sdk_version: 6.31.0` ("all versions of Gradio are supported");
- `python_version: "3.12.12"` (one of the two Python versions ZeroGPU supports; torch 2.13.0 is on its list);
- `short_description` 58 characters (the reference gives no limit; kept within 60);
- `license: other` with `license_name` and `license_link: LICENSE` (in the Space repo);
- `models: [mahmoudalyosify/BA, openai/whisper-large-v3-turbo, facebook/audioseal]`;
- `thumbnail` (documented: "URL for defining a custom thumbnail for social sharing") pointing to the Space repo's `social-card.png`;
- `preload_from_hub`: the BA weight files, plus Whisper and AudioSeal at their pinned commits. The reference says preloading private repos is not supported; the model repo is public now.

**Local check of the staged bundle (a stand-in, not the public Space).** The exact staged folder was run on the A40 with a fresh, anonymous HF cache; weights came from the public repo, and `ba_tts` was the bundled copy. It was called through `gradio_client` (`BA/hf/space_check.py`, FLEURS `ar_eg_f6`, no transcript):

| | Result |
|---|---|
| consent unticked | refused: «لازم تأكد إن ده صوتك …» / "Please confirm that this is your voice or that you have the speaker's explicit permission." |
| request 1 (Egyptian, digits) | 10.65 s wall (first request); 48 kHz; CER 0.098 (vs digit text); ECAPA 0.855; Egyptian; DNSMOS 3.623; watermark detected (1.0 / 1.0) |
| request 2 (same session, voice reused) | 3.39 s; CER 0.059; ECAPA 0.780; Egyptian; DNSMOS 3.575; watermark detected |
| copies of the uploaded reference left in Gradio's cache | 0 |

These match the Phase 3 local test in `engine.md` 6b (ECAPA 0.855 / 0.780, CER 0.098 / 0.059). ZeroGPU-specific behaviour (CUDA emulation at import, `@spaces.GPU` duration, visitor quotas: 2 GPU-minutes a day for anonymous visitors per the ZeroGPU docs) remains untested until the Space exists.

## 5. Validation A: clean environment, private repo

- New venv `/tmp/ba/venvs/cleanA` (Python 3.12, `uv`), empty `HF_HOME`. The wheel was downloaded from the private repo with the token inside Python; the snippet passed `token=` read from the file inside the process. `HF_TOKEN` was unset throughout.
- Installed from PyPI by the wheel's requirements: torch 2.15.0, voxcpm 2.0.3, transformers 5.22.2, audioseal 0.2.1, huggingface_hub 2.4.0. `ba_tts` imported from the venv's site-packages (run with `python -I`).
- `BA.from_pretrained("mahmoudalyosify/BA")` loaded `ba_config.json` (0.1.0, base revision `7936080b…`, Whisper at `41f01f3f…`). Load including the 5 GB download and the torch.compile warm-up: 149 s.
- 6 held-out texts, FLEURS references (`BA/hf/validate_snippet.py`; outputs in `/home/jovyan/ba-work/publish/validation/A/`, not published). All engine checks ok; watermark 6/6 (score 1.0, bits 1.0).

| Utterance | Voice | Error | ECAPA | Dialect | DNSMOS |
|---|---|---|---|---|---|
| egy02 | ar_eg_m1 | CER 0.176 | 0.721 | Egyptian | 3.080 |
| egy03 | ar_eg_m1 | CER 0.024 | 0.761 | Egyptian | 3.299 |
| egy27 (digits) | ar_eg_f6 | CER 0.000 vs digit text (0.295 vs spelled) | 0.795 | Egyptian | 3.993 |
| egy02 | ar_eg_f6 | CER 0.020 | 0.763 | Egyptian | 3.734 |
| en02 | en_us_m1 | WER 0.000 | 0.811 | – | 3.861 |
| fr02 | fr_fr_m1 | WER 0.000 | 0.778 | – | 3.910 |

Against `engine.md` (held-out means): Egyptian sentences CER 0.030, ECAPA 0.769, Egyptian share 0.889; English 0.003 / 0.779; French 0.030 / 0.792. Here, the 3 Egyptian main sentences give mean CER 0.073 (one outlier, 0.176), mean ECAPA 0.748, and 4/4 Egyptian; English and French are at or above the engine figures. The evaluated engine run shows the same kind of per-utterance spread (for example egy27 on ar_eg_m1: plain CER 0.341). Judged within reason for n = 6, and the run proceeded.

## 6. Validation B: clean environment, anonymous

- New venv `/tmp/ba/venvs/cleanB` made with the standard library (`python3.12 -m venv`). Exactly the card's command: `pip install https://huggingface.co/mahmoudalyosify/BA/resolve/main/ba_tts-0.1.0-py3-none-any.whl`. Empty `HF_HOME`, no token in the environment or on disk. The snippet ran exactly as on the card: `BA.from_pretrained("mahmoudalyosify/BA")`, no token.
- Same package versions as A. Load time 90.5 s.
- **All 6 outputs are bit-identical to validation A** (max |A−B| = 0.0, same lengths), so every score equals the A table. Watermark 6/6. Outputs: `/home/jovyan/ba-work/publish/validation/B/`.
- Space part (gradio_client against the public Space): **not run**, because the Space does not exist (section 4). The local bundle check above is the substitute.

## 7. Licence

`LICENSE`, the BA Community License 1.0, plain English and not lawyer-reviewed. It covers BA's own work (the `ba_tts` code, configs, normaliser lexicons, docs, card, samples, brand). It allows non-commercial use, modification and sharing, with attribution and under the same terms. Commercial use needs the owner's prior written permission (contact through the Hugging Face profile; no email). A voice may be used only by its speaker or with the speaker's explicit permission. It forbids impersonation, fraud, deception, harassment, non-consensual content and illegal use, and forbids removing, altering or circumventing the watermark or publishing BA audio made with it off. It gives no trademark use beyond attribution and no warranty, and it ends on breach (with a 30-day cure). It states that third-party components keep their own licences (VoxCPM2 weights Apache-2.0, Whisper MIT, AudioSeal MIT) and that nothing in it restricts the rights those licences grant. The copyright line names the account (`mahmoudalyosify`) because the owner's legal name is not on file; the owner may want to add it.

## 8. Incidents and open items

- **Local Space check hang (2026-11-14).** My first local run waited for Gradio's "Running on local URL" line. That line goes to block-buffered stdout, so it never appeared in the log; the app (and its inductor compile worker) kept holding GPU memory for about 4 h until the planner stopped it. The rerun polled the HTTP port with a 15-minute deadline, ran the app under `timeout` in its own process group with `PYTHONUNBUFFERED=1`, and killed the group at the end (`/tmp/ba/spacecheck/run_check.sh`). The GPU is shared: other users' processes were never touched.
- **Owner:** free a ZeroGPU slot or get PRO, then run the Space commands in section 4. After that the planner can run the Space part of validation B. The card's "Demo" line and the wheel metadata's `Demo` URL point to the Space-to-be.
- **Owner:** revoke the write token once the Space is published (execution plan, section 7).
- **Planner:** nothing was committed or pushed. New files: `BA/LICENSE`, `BA/hf/{publish.py, make_samples.py, score_samples.py, validate_snippet.py, space_check.py, model_card.md}`, `BA/src/ba_tts/config.py`, `BA/tests/test_config.py`, this report. Modified: `BA/pyproject.toml`, `BA/space/{app.py, README.md}`, `BA/src/ba_tts/{__init__, asr, cli, defaults, engine}.py`, `BA/tests/test_gpu_smoke.py`. (`BA/docs/decisions.md` was changed by the planner, not by this run.) `BA/README.md` is still the "vision only, no code yet" document.
- **Not done:** `BA/docs/reproduce.md` (a Phase 3 item); `publish.py` docstring has the commands.
