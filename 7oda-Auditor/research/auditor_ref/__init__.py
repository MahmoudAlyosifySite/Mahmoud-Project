"""Python reference implementation of the 7oda Auditor pipeline.

This package follows ``docs/SPEC.md`` literally (sections 2-9, API layout of
section 10.2). It is the referee between the Swift implementation in
``AuditorCore`` and the research code: ``scripts/make_fixtures.py`` writes the
parity fixtures of SPEC section 11 from it.

Only numpy is used in the reference code paths.

Modules, in pipeline order:

- ``rng``          SplitMix64 and xoshiro256** (section 2)
- ``challenge``    random haptic pattern (section 3)
- ``recording``    raw recording JSON helpers (section 4)
- ``dsp``          biquads, zero-phase filter, resampling (sections 5.2-5.4)
- ``signal``       grid, envelopes, template, alignment (section 5)
- ``quality``      quality gate (section 6)
- ``features``     v0 features, featureVersion 2 (section 7)
- ``features_v1``  featureVersion 1, kept for comparisons
- ``calibration``  on-table calibration (section 8)
- ``model``        shared-covariance Gaussian model (section 9.2)
- ``identifier``   prototypes, hand head, sequential decision (section 9)
- ``synth``        small deterministic synthetic recordings for tests/fixtures
"""

SPEC_VERSION = "v0 MVP"
FEATURE_VERSION = 2

__all__ = [
    "rng",
    "challenge",
    "recording",
    "dsp",
    "signal",
    "quality",
    "features",
    "calibration",
    "model",
    "identifier",
    "synth",
    "SPEC_VERSION",
    "FEATURE_VERSION",
]
