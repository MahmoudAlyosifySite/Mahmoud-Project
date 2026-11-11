import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conserve.jepa import TinyJEPA, sigreg, train  # noqa: E402
from conserve.worlds import BilliardsConfig, simulate  # noqa: E402
from conserve.worlds.render import render  # noqa: E402


def test_sigreg_ranks_gaussian_below_lopsided_below_collapsed():
    torch.manual_seed(0)
    gaussian = torch.randn(2048, 16)
    uniform = (torch.rand(2048, 16) - 0.5) * 12**0.5  # unit variance, wrong shape
    collapsed = torch.zeros(2048, 16) + 0.01 * torch.randn(2048, 16)
    g, u, c = (float(sigreg(x)) for x in (gaussian, uniform, collapsed))
    assert g < u < c


def test_tiny_jepa_trains_encodes_and_scores():
    cfg = BilliardsConfig()
    rng = np.random.default_rng(0)
    frames = np.stack([render(simulate(cfg, 20, rng, n_balls=2), cfg, 48) for _ in range(6)])
    torch.manual_seed(0)
    model = TinyJEPA(dim=16, size=48)
    log = train(model, torch.from_numpy(frames), steps=3, batch=8, lam=0.1, lr=1e-3, log_every=1)
    assert len(log) == 3 and all(np.isfinite(row["loss"]) for row in log)
    z = model.encode(frames[0], torch.device("cpu"))
    assert z.shape == (20, 16) and z.dtype == np.float64
    errors = model.prediction_errors(z, torch.device("cpu"))
    assert errors.shape == (18,) and np.isfinite(errors).all()
