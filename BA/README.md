<p align="center">
  <img src="assets/logo.svg" alt="BA — Your Voice. Your Identity." width="520">
</p>

# BA — Product Vision & Engineering Mission

> **Your Voice. Your Identity.**

**Status:** v0.1.0 published as a model on Hugging Face: **https://huggingface.co/mahmoudalyosify/BA** (Egyptian Arabic, English, French beta). Demo Space (ZeroGPU): **https://huggingface.co/spaces/mahmoudalyosify/BA**.

**Paper:** *BA: Right Words, Wrong Dialect. Standard Metrics Pick the Wrong Model for Egyptian Arabic Voice Cloning*: sources, PDF and arXiv package script in `paper/` (`paper/REVIEW.md`: internal review and venue plan). **Archive (code, evaluation texts, per-utterance scores, paper sources):** Zenodo, https://doi.org/10.5281/zenodo.24517175.

| Folder | What is in it |
|---|---|
| `src/ba_tts/`, `tests/`, `pyproject.toml` | The BA engine (Python package `ba-tts`) and its 195 tests |
| `eval/` | Bake-off of 8 systems, engine evaluation, paper experiments (seeds, ablations, long form, selection analysis, watermark robustness), owner's listening check; every published number comes from `eval/results/` |
| `space/` | The Gradio demo (ZeroGPU-ready) |
| `hf/` | Publishing and validation scripts for the Hub |
| `docs/` | Execution plan, decision log, brand guide, publish report |
| `assets/` | Logo, banner, social card, avatar and their generators |
| `paper/` | The paper (LaTeX, generated numbers and tables, PDF), plan, review, human-evaluation materials |
| `scripts/` | Zenodo archiving (`zenodo_upload.py`) |

**The mark.** The Arabic letter **ب**, whose name is pronounced *ba*, drawn with a voice wave in its base, and its dot in turquoise. It reads as *BA* to an Arabic reader and as a voice to everyone. Files: `assets/logo.svg` (full logo) and `assets/mark.svg` (square, for avatars and icons). Colours: sapphire `#14254D`, gold `#D9B45A`, turquoise `#3FD0C9`.

---

## Engineering mission

You are tasked with designing and building **BA**, a high-quality AI voice cloning and personalized speech generation model.

You have full engineering autonomy to investigate, research, architect, experiment, implement, optimize, and deploy the best solution.

Do not blindly follow a predefined architecture or model recommendation. Think independently and make engineering decisions based on:

- evidence,
- experiments,
- hardware constraints,
- model capabilities,
- licensing compatibility.

Your responsibility is not simply to implement instructions. Your responsibility is to determine the best way to build BA.

---

## 1. What is BA?

BA is a personalized multilingual AI voice generation project inspired by the ancient Egyptian concept of **"Ba"**, representing an aspect of an individual's identity.

Our tagline:

> **BA — Your Voice. Your Identity.**

The central idea is simple:

1. A person provides recordings of their own voice.
2. They enter any supported text.
3. They select a language.
4. BA generates natural speech that sounds like that person.

We want the generated voice to preserve the speaker's recognizable vocal identity while speaking new sentences naturally.

This is not intended to be a basic text-to-speech wrapper.

We want BA to become a high-quality, independently usable Hugging Face model project.
