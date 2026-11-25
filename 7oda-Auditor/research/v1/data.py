"""v1 input tensor (SPEC 14.1), datasets and augmentations.

Tensor contract (SPEC 14.1), float32 ``[10, 256]`` built from a processed
signal ``p`` with ``s = p.alignment.lag``. Time index ``u`` reads grid index
``n = u + s``; where ``n`` is outside ``[0, N)`` every channel is 0:

====== ==========================================================
0-2    ``accelHP[c][n] / 0.02``
3-5    ``gyroHP[c][n] / 0.05``
6-8    ``gravity[c][n]`` (device-motion gravity, g)
9      ``template[u]`` (lag-0 template; 0 if ``u >= N``)
====== ==========================================================

All values are clipped to ``[-50, 50]``.

Two ways to build it:

* reference path: ``tensor_from_processed(auditor_ref.signal.process(rec))``,
  exactly the objects the specification describes;
* fast path: ``tensor_from_recording(rec)`` with ``v1.sigproc.process_fast``,
  used for bulk data. ``tests/test_v1_data.py`` proves both agree.

Augmentations work on batches on the GPU (``Augmenter``). Mirror augmentation
(``mirror_tensor`` / ``mirror_hand``) is for the hand head only: callers must
never give a mirrored sample an identity label (``add_mirrored`` sets it to
``IGNORE``).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict, field
from typing import Iterable, Sequence

import numpy as np
import torch

from . import sigproc

# ---------------------------------------------------------------------------
# Constants of the contract
# ---------------------------------------------------------------------------

T = 256
C = 10
ACCEL_SCALE = 0.02
GYRO_SCALE = 0.05
CLIP = 50.0
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "Gx", "Gy", "Gz", "template")
SIGNAL_CHANNELS = 9            # channels 0..8 carry the recording, 9 the challenge
PATTERN_START_U = 25           # preRoll * fs: the aligned pattern starts here

HANDS = ("right", "left", "twoHanded")
HAND_INDEX = {h: i for i, h in enumerate(HANDS)}
ACCESSORIES = ("none", "watch", "ring", "bracelet", "other")
ACCESSORY_INDEX = {a: i for i, a in enumerate(ACCESSORIES)}
POSTURES = ("sitting", "standing", "freshGrip", "unspecified")
POSTURE_INDEX = {p: i for i, p in enumerate(POSTURES)}
IGNORE = -1

# Mirror across the phone's y-z plane (x -> -x). Accelerations and gravity
# are polar vectors: x flips. Angular velocity is an axial vector, which
# transforms as det(M) * M * w = diag(1, -1, -1) * w: y and z flip.
MIRROR_SIGN = np.array([-1, 1, 1,     # ax ay az
                        1, -1, -1,    # gx gy gz
                        -1, 1, 1,     # Gx Gy Gz
                        1], dtype=np.float32)   # template

# "actuator-aware" mirror: the Taptic Engine pushes along +x in the phone frame
# whatever hand holds the phone, so a mirrored world also has its excitation
# reversed. For a linear (or odd-nonlinear) system the vibration response then
# flips sign: vibration channels transform by -M (accel: diag(1,-1,-1), gyro:
# diag(-1,1,1)), while gravity, which the actuator does not cause, mirrors as M.
MIRROR_SIGN_ACTUATOR = np.array([1, -1, -1,
                                 -1, 1, 1,
                                 -1, 1, 1,
                                 1], dtype=np.float32)
MIRROR_MODES = {"reflect": MIRROR_SIGN, "actuator": MIRROR_SIGN_ACTUATOR}


# ---------------------------------------------------------------------------
# Tensor assembly (shared by both paths)
# ---------------------------------------------------------------------------

def assemble_tensor(accel_hp, gyro_hp, gravity, template, lag: int, n: int) -> np.ndarray:
    """SPEC 14.1 from the arrays of a processed signal. Returns float32 [10, 256]."""
    x = np.zeros((C, T), dtype=np.float64)
    u = np.arange(T)
    src = u + int(lag)
    valid = (src >= 0) & (src < n)
    if np.any(valid):
        uv = u[valid]
        sv = src[valid]
        x[0:3, uv] = np.asarray(accel_hp, dtype=np.float64)[:, sv] / ACCEL_SCALE
        x[3:6, uv] = np.asarray(gyro_hp, dtype=np.float64)[:, sv] / GYRO_SCALE
        x[6:9, uv] = np.asarray(gravity, dtype=np.float64)[:, sv]
        tv = valid & (u < n)
        tmpl = np.asarray(template, dtype=np.float64)
        x[9, u[tv]] = tmpl[u[tv]]
    np.clip(x, -CLIP, CLIP, out=x)
    return x.astype(np.float32)


def _get(obj, *names):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"none of {names} found on {type(obj).__name__}")


def processed_lag(p) -> int:
    """Alignment lag of an ``auditor_ref`` processed signal (or a FastProcessed)."""
    try:
        al = _get(p, "alignment")
        return int(_get(al, "lag"))
    except AttributeError:
        return int(_get(p, "lag"))


def tensor_from_processed(p) -> np.ndarray:
    """Reference path: SPEC 14.1 from ``auditor_ref.signal.process(rec)``."""
    n = int(_get(p, "n"))
    return assemble_tensor(
        np.asarray(_get(p, "accel_hp", "accelHP")),
        np.asarray(_get(p, "gyro_hp", "gyroHP")),
        np.asarray(_get(p, "gravity")),
        np.asarray(_get(p, "template")),
        processed_lag(p), n)


def tensor_from_recording(rec: dict, return_processed: bool = False):
    """Fast path: SPEC 14.1 from a Recording dict (SPEC 4) via ``v1.sigproc``."""
    fp = sigproc.process_fast(rec)
    x = assemble_tensor(fp.accel_hp, fp.gyro_hp, fp.gravity, fp.template, fp.lag, fp.n)
    return (x, fp) if return_processed else x


def tensor_from_recording_reference(rec: dict) -> np.ndarray:
    """Reference path through ``auditor_ref.signal.process`` (slow)."""
    from auditor_ref import signal as ref_signal  # imported lazily: other owner

    return tensor_from_processed(ref_signal.process(rec))


# ---------------------------------------------------------------------------
# Labels and datasets
# ---------------------------------------------------------------------------

def label_fields(rec: dict) -> dict:
    lab = rec.get("label") or {}
    return {
        "personID": str(lab.get("personID", "")),
        "personName": str(lab.get("personName", "")),
        "hand": str(lab.get("hand", "right")),
        "accessory": str(lab.get("accessory", "none")),
        "posture": str(lab.get("posture", "unspecified")),
        "sessionID": str(lab.get("sessionID", "")),
    }


@dataclass
class TensorSet:
    """In-memory dataset of v1 tensors and integer labels.

    ``person`` indexes ``person_ids``; ``session`` indexes ``session_ids``.
    ``identity`` is the training class (person, hand) or IGNORE.
    """

    x: np.ndarray                      # [M, 10, 256] float16 or float32
    person: np.ndarray                 # [M] int32
    hand: np.ndarray                   # [M] int8 (HANDS)
    accessory: np.ndarray              # [M] int8 (ACCESSORIES)
    posture: np.ndarray                # [M] int8 (POSTURES)
    session: np.ndarray                # [M] int32
    ncc: np.ndarray                    # [M] float32
    person_ids: list = field(default_factory=list)
    session_ids: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # optional per-sample arrays

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def identity(self) -> np.ndarray:
        """Class index for (person, hand); twoHanded samples get IGNORE."""
        ident = self.person.astype(np.int64) * 2 + self.hand.astype(np.int64)
        ident[self.hand == HAND_INDEX["twoHanded"]] = IGNORE
        return ident

    def subset(self, idx) -> "TensorSet":
        idx = np.asarray(idx)
        return TensorSet(self.x[idx], self.person[idx], self.hand[idx], self.accessory[idx],
                         self.posture[idx], self.session[idx], self.ncc[idx],
                         list(self.person_ids), list(self.session_ids),
                         {k: v[idx] for k, v in self.extra.items()})

    def save(self, path: str) -> None:
        arrays = dict(x=self.x, person=self.person, hand=self.hand, accessory=self.accessory,
                      posture=self.posture, session=self.session, ncc=self.ncc,
                      person_ids=np.array(self.person_ids, dtype=object),
                      session_ids=np.array(self.session_ids, dtype=object))
        for k, v in self.extra.items():
            arrays["extra_" + k] = v
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "TensorSet":
        z = np.load(path, allow_pickle=True)
        extra = {k[6:]: z[k] for k in z.files if k.startswith("extra_")}
        return TensorSet(z["x"], z["person"], z["hand"], z["accessory"], z["posture"],
                         z["session"], z["ncc"], list(z["person_ids"]), list(z["session_ids"]),
                         extra)

    @staticmethod
    def concat(sets: Sequence["TensorSet"]) -> "TensorSet":
        """Concatenate sets, remapping person and session ids to a shared index."""
        pid_index: dict = {}
        sid_index: dict = {}
        parts = {k: [] for k in ("x", "person", "hand", "accessory", "posture", "session", "ncc")}
        extra_keys = set.intersection(*(set(s.extra) for s in sets)) if sets else set()
        extras = {k: [] for k in extra_keys}
        for s in sets:
            pmap = np.array([pid_index.setdefault(p, len(pid_index)) for p in s.person_ids] or [0])
            smap = np.array([sid_index.setdefault(q, len(sid_index)) for q in s.session_ids] or [0])
            parts["x"].append(s.x)
            parts["person"].append(pmap[s.person].astype(np.int32))
            parts["session"].append(smap[s.session].astype(np.int32))
            for k in ("hand", "accessory", "posture", "ncc"):
                parts[k].append(getattr(s, k))
            for k in extra_keys:
                extras[k].append(s.extra[k])
        return TensorSet(np.concatenate(parts["x"]), np.concatenate(parts["person"]),
                         np.concatenate(parts["hand"]), np.concatenate(parts["accessory"]),
                         np.concatenate(parts["posture"]), np.concatenate(parts["session"]),
                         np.concatenate(parts["ncc"]), list(pid_index), list(sid_index),
                         {k: np.concatenate(v) for k, v in extras.items()})


def tensorset_from_recordings(recs: Iterable[dict], min_ncc: float | None = None,
                              dtype=np.float16, reference: bool = False) -> TensorSet:
    """Build tensors and labels for labelled recordings (simulator or app export).

    ``min_ncc`` drops recordings the quality gate would reject for weak
    alignment (SPEC 6.3 rule 5, default threshold 0.30) when given.
    """
    xs, person, hand, acc, post, sess, nccs = [], [], [], [], [], [], []
    pid_index: dict = {}
    sid_index: dict = {}
    for rec in recs:
        if reference:
            from auditor_ref import signal as ref_signal
            p = ref_signal.process(rec)
            x = tensor_from_processed(p)
            ncc = float(_get(_get(p, "alignment"), "ncc"))
        else:
            x, fp = tensor_from_recording(rec, return_processed=True)
            ncc = fp.ncc
        if min_ncc is not None and ncc < min_ncc:
            continue
        lab = label_fields(rec)
        xs.append(x.astype(dtype))
        person.append(pid_index.setdefault(lab["personID"], len(pid_index)))
        hand.append(HAND_INDEX.get(lab["hand"], 0))
        acc.append(ACCESSORY_INDEX.get(lab["accessory"], len(ACCESSORIES) - 1))
        post.append(POSTURE_INDEX.get(lab["posture"], POSTURE_INDEX["unspecified"]))
        sess.append(sid_index.setdefault(lab["sessionID"], len(sid_index)))
        nccs.append(ncc)
    if not xs:
        empty = np.zeros((0, C, T), dtype=dtype)
        return TensorSet(empty, *(np.zeros(0, dtype=t) for t in
                                  (np.int32, np.int8, np.int8, np.int8, np.int32, np.float32)))
    return TensorSet(np.stack(xs), np.array(person, np.int32), np.array(hand, np.int8),
                     np.array(acc, np.int8), np.array(post, np.int8), np.array(sess, np.int32),
                     np.array(nccs, np.float32), list(pid_index), list(sid_index))


def load_jsonl(path: str) -> list:
    """Recordings from an app export (JSON Lines, SPEC 4)."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tensorset_from_jsonl(paths: Sequence[str], purposes=("enrollment", "accessory", "check", "test"),
                         min_ncc: float | None = 0.30) -> TensorSet:
    """Labelled study recordings from one or more app exports."""
    recs = []
    for p in paths:
        for r in load_jsonl(p):
            if r.get("label") and r.get("purpose", "check") in purposes:
                recs.append(r)
    return tensorset_from_recordings(recs, min_ncc=min_ncc)


# ---------------------------------------------------------------------------
# Mirror augmentation (hand head only)
# ---------------------------------------------------------------------------

def mirror_tensor(x, mode: str = "reflect"):
    """Mirror a tensor (numpy or torch, ``[..., 10, 256]``) across the y-z plane.

    ``reflect`` (the proposal's rule): every vector channel is reflected.
    ``actuator``: gravity is reflected, the vibration channels get the
    reflection with the excitation reversed (see ``MIRROR_SIGN_ACTUATOR``).
    """
    sign_np = MIRROR_MODES[mode]
    if isinstance(x, torch.Tensor):
        sign = torch.as_tensor(sign_np, dtype=x.dtype, device=x.device)
        return x * sign[:, None]
    return np.asarray(x) * sign_np[:, None].astype(np.asarray(x).dtype)


def mirror_hand(h):
    """Swap right and left; twoHanded stays (numpy or torch integer labels)."""
    if isinstance(h, torch.Tensor):
        out = h.clone()
        out[h == 0] = 1
        out[h == 1] = 0
        return out
    h = np.asarray(h)
    out = h.copy()
    out[h == 0] = 1
    out[h == 1] = 0
    return out


def add_mirrored(x: torch.Tensor, identity: torch.Tensor, hand: torch.Tensor, fraction: float,
                 generator: torch.Generator | None = None, mode: str = "reflect"):
    """Append mirrored copies of a random ``fraction`` of the batch.

    The copies carry swapped hand labels and ``IGNORE`` identity labels, so
    they train the hand head only. Returns ``(x, identity, hand, is_mirror)``.
    """
    b = x.shape[0]
    k = int(round(b * fraction))
    is_mirror = torch.zeros(b, dtype=torch.bool, device=x.device)
    if k <= 0:
        return x, identity, hand, is_mirror
    idx = torch.randperm(b, device=x.device, generator=generator)[:k]
    xm = mirror_tensor(x[idx], mode)
    hm = mirror_hand(hand[idx])
    im = torch.full_like(identity[idx], IGNORE)
    return (torch.cat([x, xm]), torch.cat([identity, im]), torch.cat([hand, hm]),
            torch.cat([is_mirror, torch.ones(k, dtype=torch.bool, device=x.device)]))


# ---------------------------------------------------------------------------
# Batch augmentations (GPU)
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    time_jitter: int = 1                 # signal shifted by -k..k samples vs the template
    gain_sigma: float = 0.15             # log-normal gain per sensor (accel, gyro)
    axis_gain_sigma: float = 0.05        # extra log-normal gain per axis
    noise_accel: float = 0.05            # max std of added noise, accel channels (tensor units)
    noise_gyro: float = 0.03             # max std, gyro channels
    noise_gravity: float = 0.003         # max std, gravity channels (g)
    case_prob: float = 0.3               # probability of simulated case damping
    case_atten: tuple = (0.6, 1.0)       # gain on channels 0-5
    case_smooth: tuple = (0.0, 0.25)     # a of the zero-phase low-pass [a, 1-2a, a]
    dropout_prob: float = 0.3            # probability that a sample gets sample dropout
    dropout_rate: tuple = (0.0, 0.03)    # fraction of time steps replaced by neighbour mean
    enabled: bool = True

    @staticmethod
    def none() -> "AugmentConfig":
        return AugmentConfig(enabled=False)

    @staticmethod
    def light() -> "AugmentConfig":
        """For simulated data, which already varies per trial, grip and session:
        no case damping and small gains, so amplitude cues of the hand survive."""
        return AugmentConfig(gain_sigma=0.03, axis_gain_sigma=0.02, case_prob=0.0, noise_accel=0.03,
                             noise_gyro=0.02, dropout_prob=0.2)

    @staticmethod
    def named(name: str) -> "AugmentConfig":
        return {"full": AugmentConfig, "light": AugmentConfig.light, "none": AugmentConfig.none}[name]()

    def to_dict(self) -> dict:
        return asdict(self)


def _uniform(shape, lo, hi, device, gen):
    return lo + (hi - lo) * torch.rand(shape, device=device, generator=gen)


class Augmenter:
    """Applies the training augmentations of the proposal to ``[B, 10, 256]``.

    Order: time jitter, case damping (attenuation + low-pass on channels 0-5),
    gain jitter, sample dropout, sensor noise. The template channel (9) is
    never changed; padding regions that are exactly zero stay zero except for
    added noise inside the valid window.
    """

    def __init__(self, cfg: AugmentConfig, seed: int | None = None, device="cpu"):
        self.cfg = cfg
        self.gen = None
        if seed is not None:
            self.gen = torch.Generator(device=device)
            self.gen.manual_seed(seed)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if not cfg.enabled:
            return x
        g = self.gen
        dev = x.device
        b, c, t = x.shape
        x = x.float().clone()
        valid = (x.abs().sum(dim=1, keepdim=True) > 0).float()  # [B,1,T] inside the grid

        # 1. time jitter of the signal channels relative to the template
        if cfg.time_jitter > 0:
            shift = torch.randint(-cfg.time_jitter, cfg.time_jitter + 1, (b,), device=dev, generator=g)
            src = torch.arange(t, device=dev)[None, :] - shift[:, None]          # [B,T]
            inside = (src >= 0) & (src < t)
            src = src.clamp(0, t - 1)
            sig = torch.gather(x[:, :SIGNAL_CHANNELS], 2,
                               src[:, None, :].expand(b, SIGNAL_CHANNELS, t))
            x[:, :SIGNAL_CHANNELS] = sig * inside[:, None, :].float()

        # 2. simulated case damping: attenuation + zero-phase 3-tap low-pass
        if cfg.case_prob > 0:
            on = (torch.rand(b, device=dev, generator=g) < cfg.case_prob).float()
            att = _uniform((b,), *cfg.case_atten, dev, g)
            a = _uniform((b,), *cfg.case_smooth, dev, g)
            att = 1.0 + on * (att - 1.0)
            a = on * a
            s = x[:, :6]
            left = torch.nn.functional.pad(s[:, :, :-1], (1, 0))
            right = torch.nn.functional.pad(s[:, :, 1:], (0, 1))
            sm = a[:, None, None] * (left + right) + (1 - 2 * a)[:, None, None] * s
            x[:, :6] = sm * att[:, None, None]

        # 3. gain jitter per sensor and per axis
        if cfg.gain_sigma > 0 or cfg.axis_gain_sigma > 0:
            ga = torch.exp(cfg.gain_sigma * torch.randn((b, 1, 1), device=dev, generator=g))
            gg = torch.exp(cfg.gain_sigma * torch.randn((b, 1, 1), device=dev, generator=g))
            ax = torch.exp(cfg.axis_gain_sigma * torch.randn((b, 6, 1), device=dev, generator=g))
            x[:, 0:3] = x[:, 0:3] * ga * ax[:, 0:3]
            x[:, 3:6] = x[:, 3:6] * gg * ax[:, 3:6]

        # 4. random sample dropout, filled like the resampler fills a short gap
        if cfg.dropout_prob > 0 and cfg.dropout_rate[1] > 0:
            on = (torch.rand(b, 1, device=dev, generator=g) < cfg.dropout_prob)
            rate = _uniform((b, 1), *cfg.dropout_rate, dev, g)
            drop = (torch.rand((b, t), device=dev, generator=g) < rate) & on & (valid[:, 0] > 0)
            drop[:, 0] = False
            drop[:, -1] = False
            s = x[:, :SIGNAL_CHANNELS]
            nb = 0.5 * (torch.roll(s, 1, dims=2) + torch.roll(s, -1, dims=2))
            x[:, :SIGNAL_CHANNELS] = torch.where(drop[:, None, :], nb, s)

        # 5. sensor noise (inside the valid window only)
        if cfg.noise_accel > 0 or cfg.noise_gyro > 0 or cfg.noise_gravity > 0:
            sa = _uniform((b, 1, 1), 0.0, cfg.noise_accel, dev, g)
            sg = _uniform((b, 1, 1), 0.0, cfg.noise_gyro, dev, g)
            sG = _uniform((b, 1, 1), 0.0, cfg.noise_gravity, dev, g)
            noise = torch.randn((b, SIGNAL_CHANNELS, t), device=dev, generator=g)
            scale = torch.cat([sa.expand(b, 3, 1), sg.expand(b, 3, 1), sG.expand(b, 3, 1)], dim=1)
            x[:, :SIGNAL_CHANNELS] = x[:, :SIGNAL_CHANNELS] + noise * scale * valid

        return x.clamp_(-CLIP, CLIP)


# ---------------------------------------------------------------------------
# GPU-resident batches
# ---------------------------------------------------------------------------

class DeviceData:
    """A TensorSet copied to a device, with random batch sampling.

    Keeps ``x`` in float16 on the device (a few GB for millions of samples
    fit in the A40's memory), so training needs no DataLoader for data that
    was simulated ahead of time.
    """

    def __init__(self, ts: TensorSet, device, identity: np.ndarray | None = None):
        self.device = torch.device(device)
        if isinstance(ts.x, torch.Tensor):
            self.x = ts.x.to(self.device, torch.float16)
        else:
            self.x = torch.from_numpy(np.ascontiguousarray(ts.x, dtype=np.float16)).to(self.device)
        ident = ts.identity() if identity is None else identity
        self.identity = torch.from_numpy(ident.astype(np.int64)).to(self.device)
        self.hand = torch.from_numpy(ts.hand.astype(np.int64)).to(self.device)
        self.person = torch.from_numpy(ts.person.astype(np.int64)).to(self.device)
        self.n = len(ts)

    def batch(self, idx: torch.Tensor):
        return (self.x[idx].float(), self.identity[idx], self.hand[idx], self.person[idx])

    def epoch_batches(self, batch_size: int, generator: torch.Generator | None = None,
                      drop_last: bool = True):
        perm = torch.randperm(self.n, device=self.device, generator=generator)
        stop = self.n - (self.n % batch_size) if drop_last else self.n
        for i in range(0, stop, batch_size):
            yield perm[i:i + batch_size]
