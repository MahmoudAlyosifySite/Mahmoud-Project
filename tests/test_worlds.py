import numpy as np
import pytest

from conserve.worlds import BilliardsConfig, FAMILIES, ScenarioConfig, make_trajectory, simulate, step
from conserve.worlds.render import render

SC = ScenarioConfig()


@pytest.mark.parametrize("jitter", [0.0, 12.0])
def test_lawful_dynamics_conserve_energy_and_count(jitter):
    cfg = BilliardsConfig(wall_jitter_deg=jitter)
    traj = simulate(cfg, 400, np.random.default_rng(0), n_balls=3)
    energy = traj.kinetic_energy
    assert traj.wall_hit.sum() > 5, "the clip should contain bounces"
    assert np.ptp(energy) / energy.mean() < 1e-12
    assert np.ptp(traj.count) == 0


def test_simulation_is_reproducible():
    a = simulate(BilliardsConfig(wall_jitter_deg=12.0), 100, np.random.default_rng(7), n_balls=2)
    b = simulate(BilliardsConfig(wall_jitter_deg=12.0), 100, np.random.default_rng(7), n_balls=2)
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.vel, b.vel)


def test_step_does_not_modify_its_inputs():
    traj = simulate(BilliardsConfig(), 5, np.random.default_rng(1), n_balls=3)
    pos, vel, alive = traj.pos[0].copy(), traj.vel[0].copy(), traj.alive[0].copy()
    step(pos, vel, alive, BilliardsConfig())
    assert np.array_equal(pos, traj.pos[0]) and np.array_equal(vel, traj.vel[0])


@pytest.mark.parametrize("family", FAMILIES)
def test_possible_and_improbable_clips_are_lawful(family):
    rng = np.random.default_rng(3)
    for condition in ("possible", "improbable"):
        for _ in range(5):
            traj = make_trajectory(family, condition, SC, rng)
            energy = traj.kinetic_energy
            assert np.ptp(energy) / energy.mean() < 1e-12, (family, condition)
            assert np.ptp(traj.count) == 0, (family, condition)


def test_improbable_clip_has_a_ball_faster_than_any_in_training():
    rng = np.random.default_rng(4)
    traj = make_trajectory("energy", "improbable", SC, rng)
    speed = np.linalg.norm(traj.vel[0], axis=1).max()
    assert speed > BilliardsConfig().speed_high * 1.5


def test_energy_jump_changes_energy_at_the_event_only():
    traj = make_trajectory("energy", "impossible", SC, np.random.default_rng(5))
    energy, t = traj.kinetic_energy, traj.t_event
    assert SC.t_event_low <= t < SC.t_event_high
    assert np.ptp(energy[:t]) / energy[0] < 1e-12
    assert np.ptp(energy[t:]) / energy[-1] < 1e-12
    assert abs(energy[t] - energy[t - 1]) / energy[0] > 0.05


def test_vanish_removes_exactly_one_ball():
    traj = make_trajectory("permanence", "impossible", SC, np.random.default_rng(6))
    t = traj.t_event
    assert traj.count[t - 1] - traj.count[t] == 1
    assert np.ptp(traj.count[:t]) == 0 and np.ptp(traj.count[t:]) == 0


def test_teleport_moves_a_ball_but_keeps_energy():
    traj = make_trajectory("continuity", "impossible", SC, np.random.default_rng(7))
    t, ball = traj.t_event, traj.meta["ball"]
    jump = np.linalg.norm(traj.pos[t, ball] - traj.pos[t - 1, ball])
    assert jump > SC.teleport_min_dist * 0.9
    assert np.ptp(traj.kinetic_energy) / traj.kinetic_energy.mean() < 1e-12


def test_bounce_gain_adds_energy_at_a_wall():
    traj = make_trajectory("stochastic", "impossible", SC, np.random.default_rng(8))
    t, ball = traj.t_event, traj.meta["ball"]
    assert traj.wall_hit[t, ball]
    speed_before = np.linalg.norm(traj.vel[t - 1, ball])
    speed_after = np.linalg.norm(traj.vel[t, ball])
    assert speed_after / speed_before == pytest.approx(SC.gain_factor, rel=1e-9)
    assert t <= SC.T - SC.settle


def test_render_shows_one_disc_per_ball():
    cfg = BilliardsConfig()
    traj = simulate(cfg, 3, np.random.default_rng(9), n_balls=2)
    frames = render(traj, cfg, size=48)
    assert frames.shape == (3, 48, 48) and frames.dtype == np.uint8
    area = (frames[0] > 127).sum() / 48**2
    assert area == pytest.approx(2 * np.pi * cfg.radius**2, rel=0.25)
