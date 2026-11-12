"""A cloned voice: the selected reference audio, its transcript and the model conditioning.

Held in memory only. ``voice.save(path)`` writes a single .npz file (no pickle) only when the
caller asks; ``Voice.load(path)`` reads it back.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Voice:
    audio: np.ndarray                 # selected reference, 16 kHz float32 mono, loudness-normalised
    sample_rate: int
    transcript: str | None            # what is said in ``audio`` (None -> reference-only cloning)
    language: str | None              # language of the recording (arz / en / fr), if known
    report: dict = field(default_factory=dict)
    _cache: dict | None = field(default=None, repr=False)   # VoxCPM2 prompt cache (latents), built lazily

    @property
    def mode(self) -> str:
        """"prompt": reference + transcript (VoxCPM2 ref_continuation); "reference": audio only."""
        return "prompt" if self.transcript else "reference"

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.sample_rate

    def save(self, path: str) -> str:
        meta = dict(transcript=self.transcript, language=self.language, report=self.report,
                    sample_rate=self.sample_rate, format="ba-voice-1")
        if not str(path).endswith(".npz"):
            path = f"{path}.npz"
        np.savez_compressed(path, audio=self.audio.astype(np.float32),
                            meta=np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8))
        return path

    @classmethod
    def load(cls, path: str) -> "Voice":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"].tolist()).decode("utf-8"))
            audio = z["audio"].astype(np.float32)
        return cls(audio=audio, sample_rate=int(meta["sample_rate"]), transcript=meta.get("transcript"),
                   language=meta.get("language"), report=meta.get("report", {}))
