import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conserve.probes import ball_count, encoder_probes, occupancy  # noqa: E402
from conserve.worlds import ScenarioConfig, make_dataset, world_for  # noqa: E402


def _clips(n, seed):
    sc = ScenarioConfig(T=40, t_event_low=10, t_event_high=20)
    return make_dataset("permanence", "possible", n, sc, seed), world_for("permanence", sc)


def test_occupancy_shape_and_range():
    clips, cfg = _clips(2, 0)
    occ = occupancy(clips[0], cfg)
    assert occ.shape == (40, 144)
    assert occ.min() >= 0.0 and occ.max() <= 1.0 and occ.sum(1).min() > 0


def test_probe_is_perfect_for_a_latent_that_holds_the_targets():
    train, cfg = _clips(30, 1)
    val, _ = _clips(10, 2)
    latent = lambda c: np.concatenate([ball_count(c), occupancy(c, cfg)], axis=1)  # noqa: E731
    out = encoder_probes([latent(c) for c in train], train, [latent(c) for c in val], val, cfg)
    assert out["probe_r2_count"] > 0.99 and out["probe_r2_occupancy"] > 0.99
    assert out["probe_r2"] > 0.99


def test_probe_is_low_for_noise():
    train, cfg = _clips(20, 3)
    val, _ = _clips(8, 4)
    rng = np.random.default_rng(0)
    noise = lambda c: rng.normal(size=(len(c.alive), 16))  # noqa: E731
    out = encoder_probes([noise(c) for c in train], train, [noise(c) for c in val], val, cfg)
    assert out["probe_r2"] < 0.1
