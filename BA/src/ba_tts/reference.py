"""Reference-audio preparation: mono, 16 kHz, quality checks, choice of the cleanest 6-15 s.

``prepare_reference(recordings)`` returns the selected audio (16 kHz float32, loudness-
normalised) plus a report, or raises ``ReferenceAudioError`` with a message meant for the
end user. Thresholds live in ``defaults.py``; how they were set is in
BA/eval/results/reference_thresholds.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import defaults as D
from .audio import AudioInput, frame_rms_db, load_audio, normalize_loudness, resample

HOP_S = 0.01


class ReferenceAudioError(ValueError):
    """Raised when the reference cannot be used; ``str(e)`` is a user-facing message."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


@dataclass
class Recording:
    index: int
    y16: np.ndarray
    raw_clip_ratio: float
    snr_db: float
    segments: list = field(default_factory=list)     # [(start_s, end_s)] of speech
    speech_s: float = 0.0


def clip_ratio(y: np.ndarray, level: float = D.REF_CLIP_LEVEL) -> float:
    """Share of samples at (near) digital full scale, measured before any gain change."""
    return float(np.mean(np.abs(y) >= level)) if len(y) else 0.0


def speech_segments(y16: np.ndarray, rel_db: float = 35.0, min_gap_s: float = 0.3,
                    min_seg_s: float = 0.15) -> list[tuple[float, float]]:
    """Energy VAD: frames within ``rel_db`` of the 95th-percentile level count as speech;
    gaps shorter than ``min_gap_s`` are bridged."""
    r = frame_rms_db(y16)
    p95, p10 = np.percentile(r, 95), np.percentile(r, 10)
    # above the noise floor, but never stricter than 10 dB below the loud frames (speech without clear pauses)
    thr = min(max(p95 - rel_db, p10 + 6.0), p95 - 10.0)
    act = r > thr
    segs, start = [], None
    for i, a in enumerate(act):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append([start, i]); start = None
    if start is not None:
        segs.append([start, len(act)])
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) * HOP_S < min_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    out = [(s * HOP_S, e * HOP_S + 0.025) for s, e in merged if (e - s) * HOP_S >= min_seg_s]
    return out


def _split_long_segments(segs, rms_db, max_s: float = 8.0, edge_s: float = 2.0):
    """Cut segments longer than ``max_s`` at their quietest frame (away from the edges), so that
    continuous speech without pauses can still be windowed."""
    out = []
    todo = list(segs)
    while todo:
        s, e = todo.pop(0)
        if e - s <= max_s:
            out.append((s, e)); continue
        a, b = int((s + edge_s) / HOP_S), int((e - edge_s) / HOP_S)
        k = a + int(np.argmin(rms_db[a:b])) if b > a else int((s + e) / 2 / HOP_S)
        cut = k * HOP_S
        todo[:0] = [(s, cut), (cut, e)]
    return out


def analyse(src: AudioInput, index: int = 0) -> Recording:
    y, sr = load_audio(src)
    if len(y) > D.MAX_INPUT_S * sr:
        y = y[: int(D.MAX_INPUT_S * sr)]
    raw_clip = clip_ratio(y)
    y16 = resample(y, sr, D.REF_SR)
    r = frame_rms_db(y16) if len(y16) else np.zeros(1)
    snr = float(np.percentile(r, 95) - np.percentile(r, 1))   # noise estimator: reference_thresholds.json "adopted"
    rec = Recording(index, y16, raw_clip, snr)
    if len(y16) == 0 or float(np.max(np.abs(y16))) < 1e-4:
        return rec
    rec.segments = _split_long_segments(speech_segments(y16), r)
    rec.speech_s = float(sum(e - s for s, e in rec.segments))
    return rec


def _render(rec: Recording, segs: list[tuple[float, float]], pad_s: float = 0.2) -> np.ndarray:
    """Cut [first.start - pad, last.end + pad], shortening inner pauses to MAX_INNER_PAUSE_S."""
    sr = D.REF_SR
    pieces = []
    for k, (s, e) in enumerate(segs):
        a = max(0.0, s - (pad_s if k == 0 else 0.0))
        b = min(len(rec.y16) / sr, e + (pad_s if k == len(segs) - 1 else 0.0))
        pieces.append(rec.y16[int(a * sr):int(b * sr)])
        if k < len(segs) - 1:
            gap = segs[k + 1][0] - e
            g = min(gap, D.MAX_INNER_PAUSE_S)
            # keep the real room tone of the pause instead of digital silence
            mid = e + gap / 2
            pieces.append(rec.y16[int((mid - g / 2) * sr):int((mid + g / 2) * sr)])
    return np.concatenate(pieces) if pieces else np.zeros(0, np.float32)


def _windows(rec: Recording, max_s: float):
    """Every run of consecutive speech segments whose rendered length is <= max_s."""
    segs = rec.segments
    for i in range(len(segs)):
        for j in range(i, len(segs)):
            sub = segs[i:j + 1]
            span = (sub[-1][1] - sub[0][0]) + 0.4
            inner = sum(max(0.0, sub[k + 1][0] - sub[k][1]) - min(max(0.0, sub[k + 1][0] - sub[k][1]), D.MAX_INNER_PAUSE_S)
                        for k in range(len(sub) - 1))
            dur = span - inner
            if dur > max_s:
                break
            yield sub, dur


def _score(rec: Recording, dur: float) -> float:
    """Cleaner first (SNR, capped at 40 dB), then closer to ~10-12 s."""
    return min(rec.snr_db, 40.0) + 10.0 * min(dur, 12.0) / 12.0


def _reject_reason(rec: Recording) -> str | None:
    if rec.speech_s < 0.2:
        return "silent"
    if rec.raw_clip_ratio > D.REF_CLIP_MAX_RATIO:
        return "clipped"
    if rec.snr_db < D.REF_SNR_MIN_DB:
        return "noisy"
    return None


MESSAGES = {
    "silent": ("No speech was found in the recording. Please record yourself reading aloud.",
               "مفيش كلام في التسجيل. سجّل نفسك وإنت بتقرا بصوت عالي."),
    "clipped": ("The recording is distorted because it is too loud (clipping). Record again a little farther "
                "from the microphone or with a lower input level.",
                "التسجيل فيه تشويش لأن الصوت عالي زيادة. سجّل تاني وإنت أبعد شوية عن المايك."),
    "noisy": ("There is too much background noise in the recording. Record again in a quiet room.",
              "فيه دوشة كتير في الخلفية. سجّل تاني في مكان هادي."),
    "short": ("Not enough clear speech: at least {min:.0f} seconds are needed and {got:.1f} s were found. "
              "Record 10-15 seconds of yourself reading aloud.",
              "الكلام الواضح قليل: محتاجين {min:.0f} ثواني على الأقل ولقينا {got:.1f}. سجّل من ١٠ لـ ١٥ ثانية وإنت بتقرا بصوت عالي."),
}


def message(code: str, **kw) -> str:
    en, ar = MESSAGES[code]
    return f"{ar}\n{en}".format(**kw)


def prepare_reference(recordings, transcripts: list[str | None] | None = None, joint_transcript: str | None = None):
    """Choose the reference.

    Without transcripts: the best window of 6-15 s within one recording; if no single
    recording has 6 s of usable speech, the best windows of several recordings are joined.
    With transcripts (one per recording): whole recordings are used, so the words still match;
    they must fit in MAX_WITH_TRANSCRIPT_S together, otherwise the transcripts are dropped and
    the reference is re-selected (the caller then transcribes it).

    Returns (audio16, transcript_or_None, report).
    """
    if not isinstance(recordings, (list, tuple)) or (len(recordings) == 2 and isinstance(recordings[0], (int, np.integer))):
        recordings = [recordings]
    recs = [analyse(r, i) for i, r in enumerate(recordings)]
    report = {"recordings": [dict(index=r.index, seconds=round(len(r.y16) / D.REF_SR, 2), speech_s=round(r.speech_s, 2),
                                  snr_db=round(r.snr_db, 1), clip_ratio=round(r.raw_clip_ratio, 5),
                                  rejected=_reject_reason(r)) for r in recs], "notes": []}
    usable = [r for r in recs if _reject_reason(r) is None]
    if not usable:
        reasons = [_reject_reason(r) for r in recs]
        longest = max(r.speech_s for r in recs)
        if "clipped" not in reasons and 0.2 <= longest < D.MIN_SPEECH_S:
            raise ReferenceAudioError(message("short", min=D.MIN_SPEECH_S, got=longest), "short")
        code = next((c for c in ("clipped", "noisy", "silent") if c in reasons), "silent")
        raise ReferenceAudioError(message(code), code)
    for r in usable:
        if r.snr_db < D.REF_SNR_WARN_DB:
            report["notes"].append(f"recording {r.index}: some background noise (estimate {r.snr_db:.0f} dB); "
                                   "the cloned voice may sound less like you. A quieter room helps.")
    total = sum(r.speech_s for r in usable)
    if total < D.MIN_SPEECH_S:
        raise ReferenceAudioError(message("short", min=D.MIN_SPEECH_S, got=total), "short")

    tr = None
    if joint_transcript and joint_transcript.strip():
        # one transcript for all recordings read in order: use them all, whole, or not at all
        whole = [_render(r, r.segments) for r in recs if r.segments]
        dur = sum(len(w) for w in whole) / D.REF_SR
        if len(usable) == len(recs) and len(whole) == len(recs) and D.MIN_SPEECH_S <= dur <= D.MAX_WITH_TRANSCRIPT_S:
            y = _join_refs(whole)
            report.update(mode="transcribed_by_user", used=[r.index for r in recs], seconds=round(len(y) / D.REF_SR, 2))
            return normalize_loudness(y, D.REF_SR, D.REF_LUFS), joint_transcript.strip(), report
        report["notes"].append("transcript dropped: it covers several recordings that cannot all be used whole "
                               f"(at most {D.MAX_WITH_TRANSCRIPT_S:.0f} s together, all clean)")
    if transcripts and any(t for t in transcripts):
        if len(transcripts) != len(recs):
            raise ValueError("Give one transcript per recording (or none).")
        chosen = [r for r in usable if transcripts[r.index]]
        chosen.sort(key=lambda r: -r.snr_db)
        audio, texts, dur = [], [], 0.0
        for r in chosen:
            seg_audio = _render(r, r.segments)
            d = len(seg_audio) / D.REF_SR
            if dur + d > D.MAX_WITH_TRANSCRIPT_S:
                continue
            audio.append(seg_audio); texts.append(transcripts[r.index].strip()); dur += d
            if dur >= D.TARGET_MIN_S:
                break
        if audio and dur >= D.MIN_SPEECH_S:
            y = _join_refs(audio)
            report.update(mode="transcribed_by_user", used=[r.index for r in chosen][:len(audio)], seconds=round(len(y) / D.REF_SR, 2))
            return normalize_loudness(y, D.REF_SR, D.REF_LUFS), " ".join(texts), report
        report["notes"].append("transcript dropped: the recordings with a transcript are too long or too short to use whole")

    # best single window of TARGET_MIN..TARGET_MAX seconds
    best = None
    for r in usable:
        for sub, dur in _windows(r, D.TARGET_MAX_S):
            if dur >= D.TARGET_MIN_S:
                s = _score(r, dur)
                if best is None or s > best[0]:
                    best = (s, r, sub, dur)
    if best is not None:
        _, r, sub, dur = best
        y = _render(r, sub)
        report.update(mode="window", used=[r.index], window_s=[round(sub[0][0], 2), round(sub[-1][1], 2)],
                      seconds=round(len(y) / D.REF_SR, 2))
        return normalize_loudness(y, D.REF_SR, D.REF_LUFS), tr, report

    # no single recording has 6 s: join the best (longest clean) window of each recording
    parts = []
    for r in sorted(usable, key=lambda r: -r.snr_db):
        cand = max(_windows(r, D.TARGET_MAX_S), key=lambda w: w[1], default=None)
        if cand:
            parts.append(_render(r, cand[0]))
        if sum(len(p) for p in parts) / D.REF_SR >= D.TARGET_MAX_S:
            break
    y = _join_refs(parts)
    if len(y) / D.REF_SR > D.TARGET_MAX_S:
        y = y[: int(D.TARGET_MAX_S * D.REF_SR)]
    if len(y) / D.REF_SR < D.TARGET_MIN_S:
        report["notes"].append(f"short reference ({len(y) / D.REF_SR:.1f} s): similarity may be lower; 10-15 s is best")
    report.update(mode="joined", used=[r.index for r in usable], seconds=round(len(y) / D.REF_SR, 2))
    return normalize_loudness(y, D.REF_SR, D.REF_LUFS), tr, report


def _join_refs(parts: list[np.ndarray], gap_s: float = 0.3) -> np.ndarray:
    if not parts:
        return np.zeros(0, np.float32)
    gap = np.zeros(int(gap_s * D.REF_SR), np.float32)
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        if i < len(parts) - 1:
            out.append(gap)
    return np.concatenate(out).astype(np.float32)
