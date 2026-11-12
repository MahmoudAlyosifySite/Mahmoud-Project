"""Audio helpers: loading, mono, resampling, loudness, silence trimming, joining."""
from __future__ import annotations

import os
from typing import Union

import numpy as np

from . import defaults as D

AudioInput = Union[str, os.PathLike, tuple, np.ndarray]


def to_float_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
    y = y.astype(np.float32)
    if y.ndim == 2:
        # (n, ch) or (ch, n): average the shorter axis
        y = y.mean(axis=1) if y.shape[0] >= y.shape[1] else y.mean(axis=0)
    return y.reshape(-1)


def load_audio(src: AudioInput, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Path, (sr, array) tuple (Gradio's numpy format) or array + sr -> (float32 mono, sr)."""
    if isinstance(src, tuple) and len(src) == 2:
        a, b = src
        if isinstance(a, (int, np.integer)):
            return to_float_mono(b), int(a)
        return to_float_mono(a), int(b)
    if isinstance(src, np.ndarray):
        if sr is None:
            raise ValueError("a sample rate is required when passing a raw array")
        return to_float_mono(src), int(sr)
    path = os.fspath(src)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        import soundfile as sf
        y, file_sr = sf.read(path, dtype="float32", always_2d=False)
        return to_float_mono(y), int(file_sr)
    except Exception:
        import librosa   # falls back to audioread/ffmpeg for m4a, webm, ...
        y, file_sr = librosa.load(path, sr=None, mono=True)
        return to_float_mono(y), int(file_sr)


def resample(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return y.astype(np.float32)
    import soxr
    return soxr.resample(y, sr, target, quality="HQ").astype(np.float32)


def loudness_lufs(y: np.ndarray, sr: int) -> float:
    import pyloudnorm as pyln
    if len(y) < int(0.4 * sr):   # BS.1770 needs at least one 400 ms block
        y = np.pad(y, (0, int(0.4 * sr) - len(y)))
    return float(pyln.Meter(sr).integrated_loudness(y.astype(np.float64)))


def normalize_loudness(y: np.ndarray, sr: int, target_lufs: float, peak: float = D.PEAK_CEILING) -> np.ndarray:
    """Gain to target integrated loudness (ITU-R BS.1770), then scale down if the peak exceeds ``peak``."""
    if len(y) == 0 or float(np.max(np.abs(y))) < 1e-6:
        return y.astype(np.float32)
    cur = loudness_lufs(y, sr)
    if np.isfinite(cur):
        y = y * (10.0 ** ((target_lufs - cur) / 20.0))
    p = float(np.max(np.abs(y)))
    if p > peak:
        y = y * (peak / p)
    return y.astype(np.float32)


def frame_rms_db(y16: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    """RMS per 25 ms frame (10 ms hop) at 16 kHz, in dB."""
    if len(y16) < frame:
        y16 = np.pad(y16, (0, frame - len(y16)))
    n = 1 + (len(y16) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(y16[idx] ** 2, axis=1)) + 1e-8
    return 20 * np.log10(rms)


def snr_proxy_db(y16: np.ndarray) -> float:
    """95th / 10th percentile of frame RMS, in dB (same definition as eval/scripts/select_refs.py)."""
    r = frame_rms_db(y16)
    return float(np.percentile(r, 95) - np.percentile(r, 10))


def active_duration(y: np.ndarray, sr: int, top_db: float = 35.0) -> float:
    """Speech-active length as the scorer measures it (librosa trim, top_db=35, at 16 kHz)."""
    import librosa
    y16 = resample(y, sr, 16000)
    if len(y16) == 0 or float(np.max(np.abs(y16))) < 1e-4:
        return 0.0
    yt, _ = librosa.effects.trim(y16, top_db=top_db)
    return len(yt) / 16000


def trim_edges(y: np.ndarray, sr: int, top_db: float = 40.0, margin_s: float = 0.05) -> np.ndarray:
    import librosa
    if len(y) == 0 or float(np.max(np.abs(y))) < 1e-4:
        return y
    _, (s, e) = librosa.effects.trim(y, top_db=top_db, frame_length=2048, hop_length=512)
    m = int(margin_s * sr)
    return y[max(0, s - m):min(len(y), e + m)]


def join(chunks: list[np.ndarray], pauses: list[float], sr: int, fade_s: float = D.CROSSFADE_S) -> np.ndarray:
    """Concatenate chunks with a pause after each (last pause ignored); short fades at every edge
    and a crossfade of ``fade_s`` between each chunk and the following silence, so joins never click."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    f = max(1, int(fade_s * sr))
    ramp_in = np.linspace(0.0, 1.0, f, dtype=np.float32)
    out = []
    for i, c in enumerate(chunks):
        c = c.astype(np.float32).copy()
        if len(c) > 2 * f:
            c[:f] *= ramp_in
            c[-f:] *= ramp_in[::-1]
        out.append(c)
        if i < len(chunks) - 1:
            out.append(np.zeros(int(max(pauses[i], 0.0) * sr), dtype=np.float32))
    return np.concatenate(out)
