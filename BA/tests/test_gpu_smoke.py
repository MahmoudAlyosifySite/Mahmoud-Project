"""End-to-end smoke test on a GPU with the real weights. Skipped unless BA_RUN_GPU_TESTS=1.

    BA_RUN_GPU_TESTS=1 BA_TEST_REF=/path/to/clip.wav pytest -m gpu
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.gpu
RUN = os.environ.get("BA_RUN_GPU_TESTS") == "1"
REF = os.environ.get("BA_TEST_REF", "/home/jovyan/ba-work/refs/ar_eg_m1.wav")


@pytest.fixture(scope="module")
def ba():
    if not RUN:
        pytest.skip("set BA_RUN_GPU_TESTS=1 to run GPU tests")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    from ba_tts import BA
    return BA.from_pretrained(os.environ.get("BA_MODEL_REPO", "openbmb/VoxCPM2"), device="cuda", optimize=False)


def test_create_voice_and_speak(ba, tmp_path):
    import soundfile as sf
    from ba_tts import detect_watermark
    if not os.path.exists(REF):
        pytest.skip("reference clip not found")
    v = ba.create_voice(REF, language="arz")
    assert v.mode == "prompt" and v.transcript
    sr, wav, info = ba.speak("القطر هيقوم الساعة 7:30 الصبح، والتذكرة ب 150 جنيه.", v, "arz", return_info=True)
    assert sr == 48000 and len(wav) > sr and np.isfinite(wav).all()
    assert all(c["ok"] for c in info["chunks"])
    assert "سبعة ونص" in info["normalized_text"]
    assert detect_watermark(wav, sr)["watermarked"]
    sf.write(tmp_path / "out.wav", wav, sr)


def test_in_memory_encoder_matches_file_encoder(ba, tmp_path):
    import soundfile as sf
    import torch
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(16000 * 3)).astype(np.float32)
    p = tmp_path / "x.wav"
    sf.write(p, y, 16000, subtype="FLOAT")
    a = ba.model._encode_wav(str(p), padding_mode="right")
    b = ba._encode(y, "right")
    c = ba._encode(y, "right")
    # the VAE encoder is not bit-exact run to run on GPU (~1e-3 on values up to ~7.5); file and memory
    # paths must agree to within a few times that noise
    noise = float((b - c).abs().max())
    assert float((a - b).abs().max()) <= max(3 * noise, 5e-3)


def test_reference_only_fallback(ba):
    from ba_tts.voice import Voice
    if not os.path.exists(REF):
        pytest.skip("reference clip not found")
    v = ba.create_voice(REF, language="arz")
    v2 = Voice(audio=v.audio, sample_rate=v.sample_rate, transcript=None, language="arz")
    sr, wav = ba.speak("Hello, this is a short test.", v2, "en", watermark=False)
    assert len(wav) > sr // 2
