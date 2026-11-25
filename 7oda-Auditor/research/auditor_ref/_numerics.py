"""Small numeric helpers shared by the reference modules.

All reductions here are written so that their semantics are unambiguous for
the Swift port:

- ``seq_sum`` is a plain left-to-right sum (no pairwise summation), which is
  what ``reduce(0, +)`` or a ``for`` loop gives in Swift.
- ``median`` follows numpy: the middle value for odd counts, ``(a + b) / 2``
  of the two middle values for even counts.
- ``quantile_linear`` reproduces ``numpy.quantile(..., method="linear")``
  bit for bit, including numpy's two-sided interpolation formula.

``TieAudit`` records every discrete comparison made on computed values
(argmax/argmin choices and threshold tests) together with its relative
margin, so the fixture generator can reject cases whose outcome could flip
under last-bit rounding differences between implementations (SPEC 11).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def seq_sum(values: Iterable[float]) -> float:
    """Left-to-right sum of ``values`` starting from 0.0."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    # numpy's accumulate is strictly sequential (unlike add.reduce).
    return float(np.add.accumulate(arr)[-1])


def mean(values: Iterable[float]) -> float:
    """``seq_sum(values) / count``; 0.0 for an empty input.

    The spec never takes the mean of an empty set with the default
    configuration; returning 0.0 keeps degenerate inputs finite.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return seq_sum(arr) / float(arr.size)


def median(values: Iterable[float]) -> float:
    """Median with numpy semantics; 0.0 for an empty input."""
    arr = np.sort(np.asarray(values, dtype=np.float64).ravel(), kind="stable")
    n = arr.size
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(arr[mid])
    return float((arr[mid - 1] + arr[mid]) / 2.0)


def quantile_linear(values: Iterable[float], q: float) -> float:
    """``numpy.quantile(values, q)`` with the default "linear" method.

    Algorithm (numpy >= 1.22):
    ``h = (n - 1) * q``; ``i = floor(h)``; ``g = h - i``;
    ``a = sorted[i]``, ``b = sorted[min(i + 1, n - 1)]`` (both the last value
    when ``h >= n - 1``, both the first when ``h < 0``);
    result ``a + (b - a) * g`` if ``g < 0.5`` else ``b - (b - a) * (1 - g)``.
    """
    arr = np.sort(np.asarray(values, dtype=np.float64).ravel(), kind="stable")
    n = arr.size
    if n == 0:
        raise ValueError("quantile of an empty sequence")
    h = float(n - 1) * float(q)
    if h >= n - 1:
        return float(arr[n - 1])
    if h < 0:
        return float(arr[0])
    i = math.floor(h)
    g = h - float(i)
    a = float(arr[i])
    b = float(arr[i + 1])
    diff = b - a
    if g >= 0.5:
        return b - diff * (1.0 - g)
    return a + diff * g


def solve_partial_pivot(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Pivot row = the row with the largest ``|A[r][col]|`` at or below the
    diagonal (first such row on ties). Elimination subtracts
    ``f * row_col`` with ``f = A[r][col] / A[col][col]``; back substitution
    is ``x[r] = (b[r] - sum_{c>r} A[r][c] x[c]) / A[r][r]`` with the sum
    accumulated left to right.
    """
    k = len(rhs)
    a = [[float(v) for v in row] for row in matrix]
    b = [float(v) for v in rhs]
    for col in range(k):
        piv = col
        best = abs(a[col][col])
        for r in range(col + 1, k):
            val = abs(a[r][col])
            if val > best:
                best = val
                piv = r
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            b[col], b[piv] = b[piv], b[col]
        pivot = a[col][col]
        if pivot == 0.0:
            raise ZeroDivisionError("singular matrix in solve_partial_pivot")
        for r in range(col + 1, k):
            f = a[r][col] / pivot
            for c in range(col, k):
                a[r][c] = a[r][c] - f * a[col][c]
            b[r] = b[r] - f * b[col]
    x = [0.0] * k
    for r in range(k - 1, -1, -1):
        s = b[r]
        for c in range(r + 1, k):
            s = s - a[r][c] * x[c]
        x[r] = s / a[r][r]
    return x


class TieAudit:
    """Collects discrete comparisons and their relative margins.

    ``note(label, a, b, scale)`` records the comparison of ``a`` with ``b``;
    its relative margin is ``|a - b| / max(|a|, |b|, scale)`` (infinite when
    either side is infinite, 0 when both are equal). ``scale`` gives a
    natural magnitude for comparisons against 0.
    """

    def __init__(self) -> None:
        self.entries: list[tuple[str, float, float, float]] = []

    def note(self, label: str, a: float, b: float, scale: float = 0.0) -> None:
        a = float(a)
        b = float(b)
        if a == 0.0 and b == 0.0:
            # Exact zeros come from exact-zero inputs and are computed
            # identically everywhere; they cannot flip under rounding.
            margin = math.inf
        elif math.isinf(a) or math.isinf(b):
            margin = math.inf if a != b else 0.0
        else:
            denom = max(abs(a), abs(b), float(scale))
            margin = 0.0 if denom == 0.0 else abs(a - b) / denom
        self.entries.append((label, a, b, margin))

    def min_margin(self) -> float:
        if not self.entries:
            return math.inf
        return min(e[3] for e in self.entries)

    def worst(self) -> tuple[str, float, float, float] | None:
        if not self.entries:
            return None
        return min(self.entries, key=lambda e: e[3])


def note(audit: TieAudit | None, label: str, a: float, b: float, scale: float = 0.0) -> None:
    """Record a comparison on ``audit`` if one is given."""
    if audit is not None:
        audit.note(label, a, b, scale)


def note_argmax(audit: TieAudit | None, label: str, values: Sequence[float], chosen: int, scale: float = 0.0) -> None:
    """Record the margin between the chosen maximum and the runner-up."""
    if audit is None or len(values) < 2:
        return
    best = float(values[chosen])
    others = [float(v) for i, v in enumerate(values) if i != chosen]
    audit.note(label, best, max(others), scale)
