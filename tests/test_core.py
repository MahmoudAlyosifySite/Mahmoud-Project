import numpy as np
import pytest

from conserve.events import clean_mask, fit_event_threshold
from conserve.latents import frame_latent, phase_space
from conserve.metrics import auroc, bootstrap_ci
from conserve.pipeline import PilotConfig, fit_latents, fit_world, ground_truth_r2
from conserve.runner import run_seed
from conserve.scores import level_shift, oracle_errors, pe_scores
from conserve.worlds import BilliardsConfig, ScenarioConfig, make_dataset, make_trajectory, simulate, world_for

SC = ScenarioConfig()
SMALL = PilotConfig(n_train=120, n_val=40, n_test=30, predictor_iters=40)


def test_auroc_known_cases():
    assert auroc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auroc([0, 1, 2], [3, 4, 5]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert auroc([0, 2], [1, 1]) == 0.5
    lo, hi = bootstrap_ci([3, 4, 5, 6], [0, 1, 2, 3], np.random.default_rng(0), n=200)
    assert 0.5 < lo <= hi <= 1.0


def test_phase_space_and_clean_mask_are_aligned():
    cfg = BilliardsConfig()
    traj = simulate(cfg, 200, np.random.default_rng(0), n_balls=1)
    z = frame_latent(traj)
    u = phase_space(z)
    clean = clean_mask(z, fit_event_threshold([z]))
    assert u.shape == (len(z) - 1, 2 * z.shape[1]) and clean.shape == (len(z) - 1,)
    hits = np.flatnonzero(traj.wall_hit.any(1))
    assert len(hits) > 0
    assert not clean[hits - 1].any(), "the step that straddles a bounce must be unclean"
    # on clean steps the finite difference is the true velocity
    ball = int(np.flatnonzero(traj.alive[0])[0])
    n = cfg.max_balls
    fd = u[:, z.shape[1] + n + ball]  # change in x of that ball
    assert np.allclose(fd[clean], traj.vel[1:, ball, 0][clean])


def test_longer_span_needs_more_event_free_steps():
    traj = simulate(BilliardsConfig(), 200, np.random.default_rng(1), n_balls=2)
    z = frame_latent(traj)
    thr = fit_event_threshold([z])
    one, four = clean_mask(z, thr, span=1), clean_mask(z, thr, span=4)
    assert four.sum() < one.sum()
    assert not (four & ~one).any()
    assert not four[:3].any()


def test_level_shift_fires_on_a_lasting_change_and_ignores_spikes():
    rng = np.random.default_rng(0)
    flat = 0.01 * rng.normal(size=(100, 2))
    clean = np.ones(100, bool)
    quiet = level_shift(flat, clean)["inv"]
    stepped = flat.copy()
    stepped[60:, 1] += 1.0
    hit = level_shift(stepped, clean)
    assert hit["inv"] > 10 * max(quiet, 0.1)
    assert abs(hit["inv_t"] - 60) <= 6 and hit["inv_k"] == 1  # located to within half a window
    spiky = flat.copy()
    spiky[50] += 100.0
    spiky_clean = clean.copy()
    spiky_clean[50] = False
    assert level_shift(spiky, spiky_clean)["inv"] < 1.0
    assert level_shift(np.zeros((100, 0)), clean)["inv"] == 0.0


def test_pe_scores():
    errors = np.array([1.0, 1.0, 1.0, 1.0, 9.0])
    out = pe_scores(errors)
    assert out["pe_max"] == 9.0 and out["pe_mean"] == pytest.approx(2.6)
    assert out["pe_rel"] > 1.0


def test_oracle_error_is_zero_on_lawful_steps_and_positive_at_a_violation():
    cfg = world_for("energy", SC)
    lawful = make_trajectory("energy", "improbable", SC, np.random.default_rng(2))
    assert oracle_errors(lawful, cfg).max() == 0.0
    broken = make_trajectory("energy", "impossible", SC, np.random.default_rng(2))
    errors = oracle_errors(broken, cfg)
    assert errors[broken.t_event - 1] > 0.0
    assert np.count_nonzero(errors) == 1


def test_invariants_recover_energy_and_count_without_labels():
    fw = fit_world("energy", SC, SMALL, seed=0)
    assert fw.head.k >= 2
    r2 = ground_truth_r2(fw, make_dataset("energy", "possible", 30, SC, 999), SMALL, np.random.default_rng(1))
    assert r2["energy"] > 0.999 and r2["count"] > 0.999


def test_no_invariants_are_found_in_unrelated_noise():
    rng = np.random.default_rng(0)
    zs = [rng.normal(size=(120, 4)) for _ in range(80)]
    zs_val = [rng.normal(size=(120, 4)) for _ in range(30)]
    fl = fit_latents(zs, zs_val, SMALL, seed=0, with_predictor=False)
    assert fl.head.k == 0


def test_two_encoders_of_one_world_agree_on_its_invariants():
    from dataclasses import replace

    from conserve.agreement import canonical_correlations
    from conserve.pipeline import invariant_values

    pc = replace(SMALL, alpha=0.0)  # each seed gets its own rotation of the latent and its own training clips
    fw_a, fw_b = fit_world("energy", SC, pc, seed=1), fit_world("energy", SC, pc, seed=2)
    clips = make_dataset("energy", "possible", 30, SC, 555)
    va = invariant_values(fw_a, clips, pc, np.random.default_rng(0))
    vb = invariant_values(fw_b, clips, pc, np.random.default_rng(0))
    both = [ca & cb for (_, ca), (_, cb) in zip(va, vb)]
    a = np.concatenate([v[m] for (v, _), m in zip(va, both)])
    b = np.concatenate([v[m] for (v, _), m in zip(vb, both)])
    shared = canonical_correlations(a, b)
    assert (shared > 0.99).sum() >= 2, shared  # energy and ball count
    noise = np.random.default_rng(3).normal(size=a.shape)
    assert canonical_correlations(a, noise).max() < 0.2


def test_pilot_runs_and_invariants_beat_prediction_error_on_ideal_latents():
    out = run_seed(0, SC, SMALL, ["energy"])
    aurocs = out["families"]["energy"]["aurocs"]
    key = "impossible_vs_improbable"
    assert aurocs["inv"][key] > 0.95
    assert aurocs["inv"][key] > max(aurocs[s][key] for s in ("pe_max", "pe_mean", "pe_rel")) + 0.15
    assert aurocs["oracle_pe"][key] == 1.0
