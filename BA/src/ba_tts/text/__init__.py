"""Text normalisation per language. ``normalize(text, lang)`` is what the engine calls."""
from __future__ import annotations

import re
import unicodedata

from . import arz, en, fr

LANGS = ("arz", "en", "fr")
ALIASES = {"ar": "arz", "egy": "arz", "eg": "arz", "ar-eg": "arz", "ar_eg": "arz", "arabic": "arz",
           "english": "en", "en-us": "en", "en_us": "en", "french": "fr", "fr-fr": "fr", "fr_fr": "fr"}

# emoji and pictographs, zero-width and bidi control characters
_JUNK = re.compile("[\U0001F000-\U0001FAFF☀-➿️​-‏‪-‮⁦-⁩]")


def canonical_lang(lang: str) -> str:
    k = (lang or "").strip().lower()
    k = ALIASES.get(k, k)
    if k not in LANGS:
        raise ValueError(f"Unsupported language {lang!r}; use one of {LANGS}")
    return k


def clean(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = _JUNK.sub("", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def normalize(text: str, lang: str, latin: str | None = None) -> str:
    """Spell out numbers/symbols for the given language ("arz", "en", "fr").

    ``latin`` is the code-switching policy for Egyptian text ("keep" or "arabic"); None uses the
    measured default in ``ba_tts.defaults``.
    """
    from ..defaults import CODESWITCH_POLICY
    lang = canonical_lang(lang)
    t = clean(text)
    if lang == "arz":
        return arz.normalize(t, latin=latin or CODESWITCH_POLICY)
    if lang == "en":
        return en.normalize(t)
    return fr.normalize(t)
