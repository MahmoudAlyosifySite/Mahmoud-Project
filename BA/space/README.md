---
title: BA
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12.12"
app_file: app.py
license: other
short_description: Your voice, with your consent. Egyptian Arabic and English
models:
  - openbmb/VoxCPM2
  - openai/whisper-large-v3-turbo
  - facebook/audioseal
preload_from_hub:
  - openbmb/VoxCPM2
  - openai/whisper-large-v3-turbo
  - facebook/audioseal generator_base.pth
---

# BA · Your Voice. Your Identity.

Record or upload 10-15 seconds of your own voice, tick the consent box, type a text and listen.

- **Consent.** Generation is blocked until you confirm the voice is yours or that you have the speaker's explicit permission.
- **Privacy.** Uploads are processed in a temporary directory that is deleted after each request; Gradio's copies of the uploads are deleted at the same time. The voice prepared from them stays only in your browser session's memory, so you can generate again without re-uploading, and is gone when the session ends. Generated audio sits in Gradio's cache only so the player and the download work, and the cache is swept every two minutes (files older than ten minutes are removed). No text or audio is logged. Flagging and analytics are off.
- **Watermark.** Every output carries an inaudible AudioSeal watermark.
- **Limits.** Up to 600 characters of text per request; one request at a time, queue of 16.

Model weights are loaded from the Hub at start-up; the repository is set by the `BA_MODEL_REPO` variable (default `openbmb/VoxCPM2`).
The `ba_tts` package must sit next to `app.py` (the publish script copies it from `BA/src/ba_tts`).
