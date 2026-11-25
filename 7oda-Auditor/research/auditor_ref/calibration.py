"""On-table calibration (SPEC section 8).

``build(processed_list, reports, device, now)`` returns the calibration dict
in its JSON form::

    {"version": 1, "createdAt": "...Z", "tableFeatures": [36],
     "quietGyroRMS": ..., "quietAccelRMS": ..., "sampleRate": ...,
     "ncc": ..., "snr": ..., "patternsUsed": ..., "device": "..."}

or ``None`` if fewer than 12 patterns passed the ``onTable`` quality gate.
"""

from __future__ import annotations

import datetime as _dt
from typing import Sequence

from ._numerics import median
from .features import extract_absolute
from .recording import format_date
from .signal import Processed

MIN_PASSING = 12
PATTERN_COUNT = 20
FAILURE_MESSAGE = "Calibration failed: keep the phone still on a hard table"


def build(
    processed_list: Sequence[Processed],
    reports: Sequence[dict],
    device: str,
    now: _dt.datetime | str | None = None,
    min_passing: int = MIN_PASSING,
) -> dict | None:
    """Calibration from the patterns whose report passed (SPEC 8).

    ``tableFeatures[k]`` is the median (numpy semantics) of the absolute
    feature ``k`` over the passing patterns; ``quietGyroRMS``,
    ``quietAccelRMS``, ``sampleRate``, ``ncc`` and ``snr`` are medians of the
    corresponding quality metrics.
    """
    if len(processed_list) != len(reports):
        raise ValueError("processed_list and reports differ in length")
    used = [(p, r) for p, r in zip(processed_list, reports) if r["passed"]]
    if len(used) < min_passing:
        return None
    feats = [extract_absolute(p) for p, _ in used]
    table = [median([float(f[k]) for f in feats]) for k in range(len(feats[0]))]

    def med(key: str) -> float:
        return median([float(r["metrics"][key]) for _, r in used])

    return {
        "version": 1,
        "createdAt": format_date(now),
        "tableFeatures": table,
        "quietGyroRMS": med("quietGyroRMS"),
        "quietAccelRMS": med("quietAccelRMS"),
        "sampleRate": med("sampleRate"),
        "ncc": med("ncc"),
        "snr": med("snr"),
        "patternsUsed": len(used),
        "device": device,
    }
