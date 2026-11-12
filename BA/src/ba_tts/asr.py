"""Reference transcription when the user gives none: Whisper large-v3-turbo (OpenAI, MIT).

Why this model: permissive licence (MIT), Arabic/English/French in one model, and ~8x fewer
decoder layers than large-v3 (fast enough to run inside the same GPU call as synthesis). Its
accuracy on Egyptian speech is checked in BA/eval/results/asr_check.json.
"""
from __future__ import annotations

import re
import threading

import numpy as np

WHISPER_LANG = {"arz": "arabic", "en": "english", "fr": "french"}
WHISPER_CODE_TO_BA = {"ar": "arz", "en": "en", "fr": "fr"}


class Transcriber:
    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device = device
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=self.dtype).to(device).eval()
        self._lock = threading.Lock()

    def __call__(self, audio16: np.ndarray, language: str | None = None) -> tuple[str, str | None]:
        """Transcribe <= 30 s of 16 kHz audio. Returns (text, language); language is detected
        (and mapped to arz/en/fr when possible) if not given."""
        import torch
        feats = self.processor(audio16, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=self.dtype)
        kw = {"task": "transcribe", "max_new_tokens": 220}
        lang = language
        with self._lock, torch.inference_mode():
            if lang is None:
                tok_id = int(self.model.detect_language(feats)[0])
                code = self.processor.tokenizer.convert_ids_to_tokens(tok_id).strip("<|>")
                lang = WHISPER_CODE_TO_BA.get(code)
                kw["language"] = code
            else:
                kw["language"] = WHISPER_LANG[lang]
            ids = self.model.generate(feats, **kw)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text, lang
