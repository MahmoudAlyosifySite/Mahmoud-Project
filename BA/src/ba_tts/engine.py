"""The BA engine on top of VoxCPM2 (OpenBMB, Apache-2.0).

    from ba_tts import BA
    ba = BA.from_pretrained("openbmb/VoxCPM2", device="cuda")
    voice = ba.create_voice(["me.wav"], language="arz")        # transcript optional
    sr, wav = ba.speak("أنا رايح الشغل الساعة 7:30", voice, language="arz")
"""
from __future__ import annotations

import os
import re
import time
from typing import Sequence

import numpy as np

from . import defaults as D
from .audio import AudioInput, join, normalize_loudness, trim_edges
from .chunking import band_distance, check_chunk, chunk_text
from .reference import ReferenceAudioError, prepare_reference
from .text import canonical_lang, clean, normalize
from .voice import Voice


def on_zerogpu() -> bool:
    return bool(os.environ.get("SPACES_ZERO_GPU"))


def _seed_all(seed: int):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _end_punct(t: str) -> str:
    t = t.strip()
    return t if re.search(r"[.!?؟…]$", t) else t + "."


class BA:
    def __init__(self, tts, device: str, model_id: str, asr_model: str = D.ASR_MODEL):
        self.tts = tts                      # voxcpm.VoxCPM
        self.model = tts.tts_model          # VoxCPM2Model
        self.sample_rate = int(self.model.sample_rate)
        self.device = device
        self.model_id = model_id
        self.asr_model = asr_model
        self._asr = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, model: str = "openbmb/VoxCPM2", device: str = "cuda", optimize: bool | None = None,
                        asr_model: str = D.ASR_MODEL, load_asr: bool = False) -> "BA":
        """``model`` is a Hub repo id or a local directory holding VoxCPM2 weights.
        ``optimize`` (torch.compile) defaults to on for CUDA, off on ZeroGPU and CPU."""
        from voxcpm import VoxCPM
        if optimize is None:
            optimize = str(device).startswith("cuda") and not on_zerogpu()
        tts = VoxCPM.from_pretrained(model, load_denoiser=False, optimize=optimize, device=device)
        ba = cls(tts, device, model, asr_model)
        if load_asr:
            _ = ba.asr
        return ba

    @property
    def asr(self):
        if self._asr is None:
            from .asr import Transcriber
            self._asr = Transcriber(self.asr_model, device=self.device)
        return self._asr

    # ------------------------------------------------------------------ voices
    def create_voice(self, recordings: AudioInput | Sequence[AudioInput], transcript: str | Sequence[str | None] | None = None,
                     language: str | None = None) -> Voice:
        """Build a voice from one or more recordings (paths, (sr, array) tuples).

        ``transcript``: what is said in the recording(s): a string (for one recording, or for all
        recordings read in order) or a list with one entry per recording. If missing, the chosen
        reference is transcribed with Whisper; if that fails, the voice uses reference-only cloning.
        ``language``: language of the recordings (arz / en / fr); detected if missing.
        Raises ``ReferenceAudioError`` (user-facing message) if the audio cannot be used.
        """
        if isinstance(recordings, (str, os.PathLike, np.ndarray)) or (
                isinstance(recordings, tuple) and len(recordings) == 2 and isinstance(recordings[0], (int, np.integer))):
            recordings = [recordings]
        recordings = list(recordings)
        lang = canonical_lang(language) if language else None
        trs = None
        if isinstance(transcript, str) and transcript.strip():
            trs = [transcript] if len(recordings) == 1 else None
            joint = transcript if len(recordings) > 1 else None
        else:
            joint = None
            if transcript is not None and not isinstance(transcript, str):
                trs = [t if (t and str(t).strip()) else None for t in transcript]
        audio, tr, report = prepare_reference(recordings, trs, joint_transcript=joint)
        if tr is None:
            try:
                tr, det = self.asr(audio, lang)
                lang = lang or det
                report["transcript_source"] = "asr"
            except Exception as e:     # transcription is optional: fall back to reference-only cloning
                tr = None
                report["transcript_source"] = f"none (ASR failed: {type(e).__name__})"
            if tr is not None and len(tr.strip()) < 2:
                tr = None
                report["transcript_source"] = "none (ASR returned nothing)"
        else:
            report["transcript_source"] = "user"
        if tr:
            tr = _end_punct(normalize(tr, lang) if lang else clean(tr))
        return Voice(audio=audio, sample_rate=D.REF_SR, transcript=tr, language=lang, report=report)

    def _encode(self, audio16: np.ndarray, padding_mode: str):
        """In-memory twin of VoxCPM2Model._encode_wav (which needs a file path)."""
        import torch
        m = self.model
        with torch.inference_mode():
            a = torch.from_numpy(audio16.astype(np.float32)).unsqueeze(0)
            patch_len = m.patch_size * m.chunk_size
            if a.size(1) % patch_len:
                pad = patch_len - a.size(1) % patch_len
                a = torch.nn.functional.pad(a, (pad, 0) if padding_mode == "left" else (0, pad))
            feat = m.audio_vae.encode(a.to(m.device), m._encode_sample_rate).cpu()
            return feat.view(m.audio_vae.latent_dim, -1, m.patch_size).permute(1, 2, 0)

    def prompt_cache(self, voice: Voice, mode: str | None = None) -> dict:
        mode = mode or voice.mode
        key = (id(self.model), mode)
        if voice._cache and voice._cache.get("key") == key:
            return voice._cache["cache"]
        cache = {"ref_audio_feat": self._encode(voice.audio, "right")}
        if mode == "prompt":
            if not voice.transcript:
                raise ValueError("prompt mode needs a transcript")
            cache.update(prompt_text=voice.transcript + " ", audio_feat=self._encode(voice.audio, "left"),
                         mode="ref_continuation")
        else:
            cache["mode"] = "reference"
        voice._cache = {"key": key, "cache": cache}
        return cache

    # ------------------------------------------------------------------ synthesis
    def _generate_chunk(self, text: str, cache: dict, cfg: float, steps: int) -> np.ndarray:
        from voxcpm.model.utils import next_and_close
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        gen = self.model._generate_with_prompt_cache(
            target_text=text, prompt_cache=cache, min_len=2, max_len=4096, inference_timesteps=steps,
            cfg_value=cfg, retry_badcase=True, retry_badcase_max_times=3, retry_badcase_ratio_threshold=6.0,
            streaming=False)
        wav, _, _ = next_and_close(gen)
        return wav.squeeze(0).float().cpu().numpy().astype(np.float32)

    def speak(self, text: str, voice: Voice, language: str, seed: int | None = None, cfg_value: float | None = None,
              inference_timesteps: int | None = None, watermark: bool = True, normalize_text: bool = True,
              max_chunk_chars: int | None = None, mode: str | None = None, return_info: bool = False):
        """Text -> (sample_rate, float32 waveform). Long text is split into sentence chunks; each chunk
        is generated with the same voice and seed, checked (duration vs. text length, empty,
        clipped) and regenerated with another seed up to ``MAX_RETRIES`` times."""
        lang = canonical_lang(language)
        if not text or not text.strip():
            raise ValueError("Text is empty.")
        if len(text) > D.MAX_TEXT_CHARS:
            raise ValueError(f"Text is too long ({len(text)} characters; the limit is {D.MAX_TEXT_CHARS}).")
        g = D.GEN_DEFAULTS[lang]
        cfg = float(cfg_value if cfg_value is not None else g["cfg_value"])
        steps = int(inference_timesteps if inference_timesteps is not None else g["inference_timesteps"])
        seed = D.DEFAULT_SEED if seed is None else int(seed)
        t0 = time.time()
        norm = normalize(text, lang) if normalize_text else clean(text)
        chunks = chunk_text(norm, max_chunk_chars or D.MAX_CHUNK_CHARS[lang])
        cache = self.prompt_cache(voice, mode)
        pieces, info_chunks = [], []
        for ch in chunks:
            best = None
            attempts = []
            for k in range(D.MAX_RETRIES + 1):
                s = seed + k
                _seed_all(s)
                wav = self._generate_chunk(ch.text, cache, cfg, steps)
                res = check_chunk(wav, self.sample_rate, ch.text, lang)
                attempts.append(dict(seed=s, ok=res.ok, reason=res.reason, active_s=round(res.active_s, 3),
                                     chars_per_sec=None if res.chars_per_sec is None else round(res.chars_per_sec, 2)))
                d = band_distance(res, lang) if not res.ok else 0.0
                if best is None or d < best[0]:
                    best = (d, wav, res)
                if res.ok:
                    break
            _, wav, res = best
            pieces.append(trim_edges(wav, self.sample_rate))
            info_chunks.append(dict(text=ch.text, ok=res.ok, attempts=attempts))
        out = join(pieces, [c.pause_after for c in chunks], self.sample_rate)
        out = normalize_loudness(out, self.sample_rate, D.OUTPUT_LUFS)
        if watermark:
            from . import watermark as wm
            out = wm.embed(out, self.sample_rate, device=self.device)
            peak = float(np.max(np.abs(out))) if len(out) else 0.0
            if peak > 0.999:
                out = out * (0.999 / peak)
        info = dict(language=lang, normalized_text=norm, mode=mode or voice.mode, cfg_value=cfg,
                    inference_timesteps=steps, seed=seed, chunks=info_chunks, watermark=watermark,
                    seconds=round(len(out) / self.sample_rate, 3), gen_seconds=round(time.time() - t0, 3))
        if return_info:
            return self.sample_rate, out.astype(np.float32), info
        return self.sample_rate, out.astype(np.float32)


__all__ = ["BA", "Voice", "ReferenceAudioError", "on_zerogpu"]
