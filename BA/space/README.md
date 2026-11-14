---
title: BA
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12.12"
app_file: app.py
license: other
license_name: ba-community-license-1.0
license_link: LICENSE
short_description: "Your voice, with consent: Egyptian Arabic, English, French"
thumbnail: https://huggingface.co/spaces/mahmoudalyosify/BA/resolve/main/social-card.png
models:
  - mahmoudalyosify/BA
  - openai/whisper-large-v3-turbo
  - facebook/audioseal
preload_from_hub:
  - mahmoudalyosify/BA config.json,model.safetensors,audiovae.pth,tokenizer.json,tokenizer_config.json,special_tokens_map.json,tokenization_voxcpm2.py,ba_config.json
  - openai/whisper-large-v3-turbo config.json,generation_config.json,model.safetensors,preprocessor_config.json,tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,normalizer.json,added_tokens.json,special_tokens_map.json 41f01f3fe87f28c78e2fbf8b568835947dd65ed9
  - facebook/audioseal generator_base.pth 3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba
tags:
  - voice-cloning
  - text-to-speech
  - egyptian-arabic
  - audio-watermarking
---

# BA · Your Voice. Your Identity.

Record or upload 10-15 seconds of your own voice, tick the consent box, type a text and listen: Egyptian Arabic, English, or French (beta). Model card, measured results and licence: [mahmoudalyosify/BA](https://huggingface.co/mahmoudalyosify/BA).

- **Consent.** Generation is blocked until you confirm the voice is yours or that you have the speaker's explicit permission.
- **Privacy.** Uploads are processed in a temporary directory that is deleted after each request; Gradio's copies of the uploads are deleted at the same time. The voice prepared from them stays only in your browser session's memory, so you can generate again without re-uploading, and is gone when the session ends. Generated audio sits in Gradio's cache only so the player and the download work, and the cache is swept every two minutes (files older than ten minutes are removed). No text or audio is logged. Flagging and analytics are off. Nothing you upload is used for training.
- **Watermark.** Every output carries an inaudible AudioSeal watermark. Check any file with `ba-tts --detect file.wav` (see the model card).
- **Limits.** Up to 600 characters of text per request; one request at a time, queue of 16. The Space runs on shared ZeroGPU hardware, so a request may wait in a queue.

Built on the VoxCPM2 weights (OpenBMB, Apache-2.0, unmodified), Whisper large-v3-turbo (MIT) and AudioSeal (MIT). BA's own code and assets: BA Community License 1.0 (`LICENSE`).

Model weights are loaded from the Hub at start-up; the repository is set by the `BA_MODEL_REPO` variable (default `mahmoudalyosify/BA`). The `ba_tts` package sits next to `app.py` (copied from `BA/src/ba_tts` by `BA/hf/publish.py`).
