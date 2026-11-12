"""Imperceptible watermark on every BA output: AudioSeal (Meta; code MIT on PyPI, weights MIT
on the Hub at facebook/audioseal).

AudioSeal works at 16 kHz. BA audio is 48 kHz, so the watermark signal is computed on a
16 kHz copy, upsampled and added to the full-band audio; the content above 8 kHz is left
untouched. A fixed 16-bit message ("BA" in ASCII) is embedded. Detection resamples any input
to 16 kHz.

The checkpoints are pickles. Instead of AudioSeal's own loader (``torch.load(weights_only=False)``)
they are read with a restricted unpickler that only rebuilds tensors and plain containers;
configuration objects inside the file are replaced by inert stubs and the architecture comes
from the model cards shipped in the ``audioseal`` package.
"""
from __future__ import annotations

import collections
import pickle
import threading
import types

import numpy as np

REPO = "facebook/audioseal"
REVISION = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba"
MESSAGE_BITS = [int(b) for b in format(ord("B"), "08b") + format(ord("A"), "08b")]
DETECT_THRESHOLD = 0.5
_lock = threading.Lock()
_models: dict = {}


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass

    def __setitem__(self, k, v):
        pass

    def append(self, v):
        pass

    def extend(self, v):
        pass


class _DefaultDict(dict):
    def __init__(self, *a, **k):
        super().__init__()


def _safe_pickle_module():
    import torch
    allowed = {("collections", "OrderedDict"): collections.OrderedDict,
               ("collections", "defaultdict"): _DefaultDict,
               ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
               ("torch", "FloatStorage"): torch.FloatStorage}
    builtins_ok = {"dict": dict, "list": list, "long": int, "int": int}

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in allowed:
                return allowed[(module, name)]
            if module in ("builtins", "__builtin__") and name in builtins_ok:
                return builtins_ok[name]
            if module.startswith("omegaconf.") or (module, name) == ("typing", "Any"):
                return _Stub
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    return types.SimpleNamespace(Unpickler=Unpickler, load=pickle.load, __name__="ba_safe_pickle")


def _load(kind: str, device: str):
    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from audioseal.builder import (AudioSealDetectorConfig, AudioSealWMConfig, create_detector,
                                   create_generator)
    from audioseal.loader import (AudioSeal, convert_state_dict_for_scriptable_model,
                                  load_local_model_config, load_state_dict)
    fname, card, ctype, create = {
        "generator": ("generator_base.pth", "audioseal_wm_16bits", AudioSealWMConfig, create_generator),
        "detector": ("detector_base.pth", "audioseal_detector_16bits", AudioSealDetectorConfig, create_detector),
    }[kind]
    path = hf_hub_download(REPO, fname, revision=REVISION)
    ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_safe_pickle_module())
    sd = ck["best_state"]["model"] if "best_state" in ck else ck["model"]
    cfg = OmegaConf.to_container(load_local_model_config(card))
    cfg.pop("checkpoint", None)
    model = create(AudioSeal.parse_config(cfg, config_type=ctype, nbits=16))
    load_state_dict(model, state_dict=convert_state_dict_for_scriptable_model(sd))
    return model.to(device).eval()


def _get(kind: str, device: str):
    key = (kind, device)
    with _lock:
        if key not in _models:
            _models[key] = _load(kind, device)
        return _models[key]


def _device(device: str | None) -> str:
    if device:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed(wav: np.ndarray, sr: int, device: str | None = None) -> np.ndarray:
    """Return ``wav`` with the BA watermark added (same length and sample rate)."""
    import torch
    from .audio import resample
    dev = _device(device)
    gen = _get("generator", dev)
    x16 = resample(wav.astype(np.float32), sr, 16000)
    from audioseal.libs.moshi.utils.compile import no_compile   # eager: no torch.compile (ZeroGPU has none)
    with torch.inference_mode(), no_compile():
        x = torch.from_numpy(x16)[None, None].to(dev)
        msg = torch.tensor(MESSAGE_BITS, device=dev)[None]
        wm16 = gen.get_watermark(x, sample_rate=16000, message=msg).squeeze().float().cpu().numpy()
    wm = resample(wm16, 16000, sr)
    n = min(len(wm), len(wav))
    out = wav.astype(np.float32).copy()
    out[:n] += wm[:n]
    return out


def detect(wav: np.ndarray, sr: int, device: str | None = None) -> dict:
    """{"watermarked": bool, "score": P(watermark), "bits_match": share of message bits that are BA's}."""
    import torch
    from .audio import resample
    dev = _device(device)
    det = _get("detector", dev)
    x16 = resample(np.asarray(wav, dtype=np.float32), sr, 16000)
    from audioseal.libs.moshi.utils.compile import no_compile
    with torch.inference_mode(), no_compile():
        x = torch.from_numpy(x16)[None, None].to(dev)
        score, msg = det.detect_watermark(x, sample_rate=16000)
    score = float(score if not hasattr(score, "item") else score.item())
    bits = (msg.squeeze().float().cpu().numpy() > 0.5).astype(int).tolist()
    match = float(np.mean([a == b for a, b in zip(bits, MESSAGE_BITS)]))
    return {"watermarked": score >= DETECT_THRESHOLD, "score": round(score, 4), "bits_match": round(match, 4)}
