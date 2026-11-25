"""Random numbers (SPEC section 2): SplitMix64 seeding and xoshiro256**.

All 64-bit arithmetic wraps modulo 2**64 (``& MASK64`` after every multiply,
add and left shift), exactly like Swift's ``&+``, ``&*`` and ``<<`` on
``UInt64``.
"""

from __future__ import annotations

import math
import re

MASK64 = 0xFFFFFFFFFFFFFFFF
TWO_POW_M53 = 2.0 ** -53  # 0x1.0p-53

_SEED_RE = re.compile(r"^0x[0-9a-fA-F]{1,16}$")


class SplitMix64:
    """SplitMix64 (SPEC 2.1). ``SplitMix64(0).next() == 0xE220A8397B1DCDAF``."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


def _rotl(x: int, k: int) -> int:
    return ((x << k) & MASK64) | (x >> (64 - k))


class Xoshiro256:
    """xoshiro256** seeded through SplitMix64 (SPEC 2.2)."""

    __slots__ = ("s",)

    def __init__(self, seed: int) -> None:
        sm = SplitMix64(seed)
        self.s = [sm.next(), sm.next(), sm.next(), sm.next()]

    def next(self) -> int:
        s = self.s
        result = (_rotl((s[1] * 5) & MASK64, 7) * 9) & MASK64
        t = (s[1] << 17) & MASK64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        return result

    def uniform(self) -> float:
        """``Double(next() >> 11) * 0x1.0p-53``, in [0, 1)."""
        return float(self.next() >> 11) * TWO_POW_M53

    def uniform_range(self, a: float, b: float) -> float:
        """``a + (b - a) * uniform()``."""
        return a + (b - a) * self.uniform()

    def int_in_range(self, lo: int, hi: int) -> int:
        """Inclusive integer: ``lo + min(Int(floor(uniform() * Double(hi - lo + 1))), hi - lo)``."""
        span = hi - lo
        return lo + min(int(math.floor(self.uniform() * float(span + 1))), span)

    # Convenience for the synthetic generator (not part of the spec).
    def normal(self) -> float:
        """Standard normal deviate (Box-Muller, one value per two uniforms)."""
        u1 = self.uniform()
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


def seed_to_str(seed: int) -> str:
    """Lowercase hex with ``0x`` prefix and exactly 16 digits (SPEC 2.2)."""
    return "0x%016x" % (int(seed) & MASK64)


def str_to_seed(text: str) -> int | None:
    """Parse a seed string written by :func:`seed_to_str`; ``None`` if invalid."""
    if not isinstance(text, str) or not _SEED_RE.match(text):
        return None
    return int(text[2:], 16)
