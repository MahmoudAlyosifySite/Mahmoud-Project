"""ba_config.json: built from ba_tts.defaults, read back by the engine (no GPU, no network)."""
import json

import pytest

import ba_tts
from ba_tts import config as C
from ba_tts import defaults as D


def _cfg():
    return C.build_config(ba_tts.__version__, "openbmb/VoxCPM2", "0" * 40)


def test_build_config_mirrors_defaults():
    cfg = _cfg()
    assert cfg["format"] == C.FORMAT and cfg["ba_version"] == "0.1.0"
    assert cfg["base_model"]["weights_modified"] is False
    assert cfg["asr"] == {**cfg["asr"], "repo_id": D.ASR_MODEL, "revision": D.ASR_REVISION}
    for lang in ("arz", "en", "fr"):
        pl = cfg["defaults"]["per_language"][lang]
        assert pl["cfg_value"] == D.GEN_DEFAULTS[lang]["cfg_value"]
        assert pl["inference_timesteps"] == D.GEN_DEFAULTS[lang]["inference_timesteps"]
        assert pl["max_chunk_chars"] == D.MAX_CHUNK_CHARS[lang]
        assert tuple(pl["chars_per_sec_band"]) == D.CHARS_PER_SEC_BAND[lang]
    assert cfg["defaults"]["per_language"]["arz"]["latin_words"] == D.CODESWITCH_POLICY
    json.dumps(cfg)     # serialisable


def test_round_trip_and_engine_settings(tmp_path):
    cfg = _cfg()
    cfg["defaults"]["per_language"]["fr"]["inference_timesteps"] = 12
    cfg["defaults"]["per_language"]["en"]["max_chunk_chars"] = 400
    (tmp_path / C.CONFIG_NAME).write_text(json.dumps(cfg), encoding="utf-8")
    st = C.engine_settings(C.load_config(str(tmp_path)))
    assert st["gen_defaults"]["fr"]["inference_timesteps"] == 12
    assert st["gen_defaults"]["arz"] == D.GEN_DEFAULTS["arz"]
    assert st["max_chunk_chars"]["en"] == 400 and st["max_chunk_chars"]["arz"] == D.MAX_CHUNK_CHARS["arz"]
    assert (st["asr_model"], st["asr_revision"]) == (D.ASR_MODEL, D.ASR_REVISION)
    assert D.GEN_DEFAULTS["fr"]["inference_timesteps"] == 10      # module defaults untouched


def test_no_config_falls_back_to_defaults(tmp_path):
    assert C.load_config(str(tmp_path)) is None
    st = C.engine_settings(None)
    assert st["gen_defaults"] == D.GEN_DEFAULTS and st["max_chunk_chars"] == D.MAX_CHUNK_CHARS


def test_unknown_format_rejected(tmp_path):
    (tmp_path / C.CONFIG_NAME).write_text(json.dumps({"format": "ba-config-99"}), encoding="utf-8")
    with pytest.raises(ValueError):
        C.load_config(str(tmp_path))


def test_download_patterns_cover_voxcpm2_loader():
    for f in ("config.json", "model.safetensors", "audiovae.pth", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", C.CONFIG_NAME):
        assert f in C.DOWNLOAD_PATTERNS
    assert ba_tts.DEFAULT_REPO == "mahmoudalyosify/BA"
