"""Split long text into chunks the model handles well, and check each generated chunk.

split_sentences -> [(sentence, pause_after)]; pack -> chunks of at most max_chars, never
breaking a sentence unless the sentence alone is longer than max_chars (then it is cut at
clause punctuation, then at the space nearest the middle).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from . import defaults as D

SENT_END = re.compile(r"(?<=[.!?؟…])[\"'”»)\]]*\s+")
CLAUSE = re.compile(r"(?<=[,،;؛:])\s+")


@dataclass
class Chunk:
    text: str
    pause_after: float


def split_sentences(text: str) -> list[Chunk]:
    out: list[Chunk] = []
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    for pi, para in enumerate(paras):
        sents = [s.strip() for s in SENT_END.split(para) if s.strip()]
        for si, s in enumerate(sents):
            last_in_para = si == len(sents) - 1
            pause = D.PAUSE_PARAGRAPH if (last_in_para and pi < len(paras) - 1) else D.PAUSE_SENTENCE
            out.append(Chunk(s, pause))
    return out


def _split_long(s: str, max_chars: int) -> list[str]:
    if len(s) <= max_chars:
        return [s]
    parts = [p for p in CLAUSE.split(s) if p.strip()]
    if len(parts) > 1:
        out, cur = [], ""
        for p in parts:
            if cur and len(cur) + 1 + len(p) > max_chars:
                out.append(cur)
                cur = p
            else:
                cur = f"{cur} {p}".strip()
        if cur:
            out.append(cur)
        res = []
        for o in out:
            res.extend(_split_long(o, max_chars) if len(o) > max_chars else [o])
        return res
    # no clause punctuation: cut at the space nearest the middle
    mid = len(s) // 2
    spaces = [m.start() for m in re.finditer(r"\s", s)]
    if not spaces:
        return [s[i:i + max_chars] for i in range(0, len(s), max_chars)]
    cut = min(spaces, key=lambda i: abs(i - mid))
    return _split_long(s[:cut].strip(), max_chars) + _split_long(s[cut:].strip(), max_chars)


def pack(sentences: list[Chunk], max_chars: int) -> list[Chunk]:
    """Greedy packing of whole sentences into chunks of at most ``max_chars`` characters.
    A paragraph break always ends a chunk."""
    pieces: list[Chunk] = []
    for c in sentences:
        sub = _split_long(c.text, max_chars)
        for k, s in enumerate(sub):
            pieces.append(Chunk(s, c.pause_after if k == len(sub) - 1 else D.PAUSE_CLAUSE))
    out: list[Chunk] = []
    for p in pieces:
        if out and out[-1].pause_after < D.PAUSE_PARAGRAPH and len(out[-1].text) + 1 + len(p.text) <= max_chars:
            out[-1] = Chunk(out[-1].text + " " + p.text, p.pause_after)
        else:
            out.append(Chunk(p.text, p.pause_after))
    return out


def chunk_text(text: str, max_chars: int) -> list[Chunk]:
    return pack(split_sentences(text), max_chars)


# ------------------------------------------------------------------ duration check
_AR_DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")


def spoken_chars(text: str, lang: str) -> int:
    """Characters the speech must cover: same normalisation as the Phase 1 scorer (score.py
    norm_ar / norm_latin on already spelled-out text), spaces and punctuation removed."""
    t = text
    if lang == "arz":
        t = _AR_DIAC.sub("", t).replace("ـ", "")
        t = re.sub("[أإآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = t.lower().replace("’", "'").replace("-", " ").replace("'", " ")
    t = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in t)
    return len(t.replace(" ", "").replace("\n", "").replace("\t", ""))


@dataclass
class CheckResult:
    ok: bool
    reason: str | None
    active_s: float
    chars_per_sec: float | None
    peak: float
    clip_ratio: float


def check_chunk(wav: np.ndarray, sr: int, text: str, lang: str) -> CheckResult:
    from .audio import active_duration
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    clip = float(np.mean(np.abs(wav) >= D.OUTPUT_CLIP_LEVEL)) if len(wav) else 0.0
    act = active_duration(wav, sr) if len(wav) else 0.0
    if act < D.MIN_ACTIVE_S or peak < D.MIN_PEAK:
        return CheckResult(False, "empty", act, None, peak, clip)
    if clip > D.OUTPUT_CLIP_MAX_RATIO:
        return CheckResult(False, "clipped", act, None, peak, clip)
    cps = spoken_chars(text, lang) / act
    lo, hi = D.CHARS_PER_SEC_BAND[lang]
    if cps < lo:
        return CheckResult(False, "too_slow", act, cps, peak, clip)
    if cps > hi:
        return CheckResult(False, "too_fast", act, cps, peak, clip)
    return CheckResult(True, None, act, cps, peak, clip)


def band_distance(r: CheckResult, lang: str) -> float:
    """How far a failed chunk is from passing (0 = passes); used to keep the least-bad attempt."""
    if r.reason in ("empty",):
        return 1e9
    if r.reason == "clipped":
        return 1e6 + r.clip_ratio
    lo, hi = D.CHARS_PER_SEC_BAND[lang]
    c = r.chars_per_sec or 0.0
    return max(0.0, lo - c, c - hi)
