"""``ba_config.json``: what a BA model repository pins and the engine defaults it was published with.

The model repository (``mahmoudalyosify/BA``) holds the unmodified VoxCPM2 weights plus this file.
``BA.from_pretrained`` reads it after download: the generation settings and chunk length per
language and the ASR model and revision come from the file; everything else in ``defaults`` is
recorded for reference and taken from ``ba_tts.defaults``. A plain VoxCPM2 directory (no
``ba_config.json``) falls back to ``ba_tts.defaults``.
"""
from __future__ import annotations

import json
import os

from . import defaults as D

CONFIG_NAME = "ba_config.json"
FORMAT = "ba-config-1"
DEFAULT_REPO = "mahmoudalyosify/BA"

# Files the VoxCPM2 loader reads (voxcpm 2.0.3: VoxCPM2Model.from_local reads config.json, the
# LlamaTokenizerFast files, model.safetensors and audiovae.safetensors or audiovae.pth).
# tokenization_voxcpm2.py ships with the weights (tokenizer_config.json names it in auto_map); BA never
# executes it.
BASE_FILES = ["config.json", "model.safetensors", "audiovae.pth", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "tokenization_voxcpm2.py"]
DOWNLOAD_PATTERNS = BASE_FILES + ["audiovae.safetensors", CONFIG_NAME]


def build_config(version: str, base_repo: str, base_revision: str) -> dict:
    """The ``ba_config.json`` written into the model repository, built from ``ba_tts.defaults``."""
    from .watermark import MESSAGE_BITS, REPO as WM_REPO, REVISION as WM_REVISION
    per_language = {}
    for lang in ("arz", "en", "fr"):
        lo, hi = D.CHARS_PER_SEC_BAND[lang]
        per_language[lang] = dict(D.GEN_DEFAULTS[lang], max_chunk_chars=D.MAX_CHUNK_CHARS[lang],
                                  chars_per_sec_band=[lo, hi])
    per_language["arz"]["latin_words"] = D.CODESWITCH_POLICY
    return {
        "format": FORMAT,
        "ba_version": version,
        "base_model": {"repo_id": base_repo, "revision": base_revision, "licence": "Apache-2.0",
                       "weights_modified": False, "files": BASE_FILES},
        "asr": {"repo_id": D.ASR_MODEL, "revision": D.ASR_REVISION, "licence": "MIT",
                "use": "transcribes the reference recording when no transcript is given"},
        "watermark": {"repo_id": WM_REPO, "revision": WM_REVISION, "licence": "MIT",
                      "generator": "generator_base.pth", "detector": "detector_base.pth",
                      "message": "BA", "message_bits": MESSAGE_BITS},
        "languages": {"arz": "Egyptian Arabic", "en": "English", "fr": "French (beta)"},
        "defaults": {
            "per_language": per_language,
            "seed": D.DEFAULT_SEED,
            "max_text_chars": D.MAX_TEXT_CHARS,
            "max_retries_per_chunk": D.MAX_RETRIES,
            "pauses_s": {"sentence": D.PAUSE_SENTENCE, "clause": D.PAUSE_CLAUSE, "paragraph": D.PAUSE_PARAGRAPH},
            "output_lufs": D.OUTPUT_LUFS,
            "reference": {"sample_rate": D.REF_SR, "min_speech_s": D.MIN_SPEECH_S, "target_s": [D.TARGET_MIN_S, D.TARGET_MAX_S],
                          "noise_reject_db": D.REF_SNR_MIN_DB, "noise_warn_db": D.REF_SNR_WARN_DB,
                          "clip_max_ratio": D.REF_CLIP_MAX_RATIO, "lufs": D.REF_LUFS},
        },
        "read_at_load": ["defaults.per_language.*.cfg_value", "defaults.per_language.*.inference_timesteps",
                         "defaults.per_language.*.max_chunk_chars", "asr.repo_id", "asr.revision"],
    }


def load_config(path: str) -> dict | None:
    """Read ``ba_config.json`` from a model directory (or the file itself). None if there is none."""
    p = os.path.join(path, CONFIG_NAME) if os.path.isdir(path) else path
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("format") != FORMAT:
        raise ValueError(f"{p}: unsupported BA config format {cfg.get('format')!r} (this ba_tts reads {FORMAT!r})")
    return cfg


def engine_settings(cfg: dict | None) -> dict:
    """The settings the engine takes from a config; ``ba_tts.defaults`` where the config has none."""
    gen = {k: dict(v) for k, v in D.GEN_DEFAULTS.items()}
    chunk = dict(D.MAX_CHUNK_CHARS)
    asr_model, asr_revision = D.ASR_MODEL, D.ASR_REVISION
    if cfg:
        for lang, v in cfg.get("defaults", {}).get("per_language", {}).items():
            if lang in gen:
                for k in ("cfg_value", "inference_timesteps"):
                    if k in v:
                        gen[lang][k] = v[k]
                if "max_chunk_chars" in v:
                    chunk[lang] = int(v["max_chunk_chars"])
        asr = cfg.get("asr") or {}
        if asr.get("repo_id"):
            asr_model, asr_revision = asr["repo_id"], asr.get("revision")
    return dict(gen_defaults=gen, max_chunk_chars=chunk, asr_model=asr_model, asr_revision=asr_revision)
