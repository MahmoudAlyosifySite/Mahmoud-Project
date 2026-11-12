"""Watermark round trip on CPU (downloads facebook/audioseal weights from the Hub on first use)."""
import shutil
import subprocess

import numpy as np
import pytest

from ba_tts import watermark as wm
from tests.test_chunking_reference import speechlike

SR = 48000


@pytest.fixture(scope="module")
def audio():
    return speechlike(6.0, sr=SR, seed=7, pauses=[(2.5, 3.0)])


def test_roundtrip_detects_ba_message(audio):
    marked = wm.embed(audio, SR, device="cpu")
    assert marked.shape == audio.shape
    r = wm.detect(marked, SR, device="cpu")
    assert r["watermarked"] and r["bits_match"] >= 0.9
    assert not wm.detect(audio, SR, device="cpu")["watermarked"]


def test_watermark_is_small(audio):
    marked = wm.embed(audio, SR, device="cpu")
    snr = 10 * np.log10(np.sum(audio ** 2) / np.sum((marked - audio) ** 2))
    assert snr > 20


def test_survives_resampling(audio):
    from ba_tts.audio import resample
    marked = wm.embed(audio, SR, device="cpu")
    assert wm.detect(resample(marked, SR, 16000), 16000, device="cpu")["watermarked"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_survives_mp3_128k(audio, tmp_path):
    import soundfile as sf
    marked = wm.embed(audio, SR, device="cpu")
    src, mp3, back = tmp_path / "a.wav", tmp_path / "a.mp3", tmp_path / "b.wav"
    sf.write(src, marked, SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-b:a", "128k", str(mp3)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), str(back)], check=True)
    y, sr = sf.read(back, dtype="float32")
    assert wm.detect(y, sr, device="cpu")["watermarked"]


def test_restricted_unpickler_blocks_code():
    import io
    import os
    import pickle
    payload = pickle.dumps(os.system)    # a global outside the allow-list
    mod = wm._safe_pickle_module()
    with pytest.raises(pickle.UnpicklingError):
        mod.Unpickler(io.BytesIO(payload)).load()
