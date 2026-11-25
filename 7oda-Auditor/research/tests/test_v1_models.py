"""v1 models: shapes, parameter budgets, augmentations, losses, export helpers."""
import importlib.util

import numpy as np
import pytest
import torch

from v1 import data as D
from v1 import losses as L
from v1 import models as M
from v1.export_coreml import fold_batchnorm, palettize_model, palettize_tensor, WEIGHT_THRESHOLD


def _x(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 10, 256, generator=g)
    x[:, 9] = x[:, 9].abs()
    return x


@pytest.mark.parametrize("kind", ["teacher", "student"])
def test_forward_shapes_and_norm(kind):
    m = M.build_model(kind).eval()
    with torch.no_grad():
        out = m(_x())
    assert out["embedding"].shape == (4, 128) and out["hand"].shape == (4, 3)
    np.testing.assert_allclose(out["embedding"].norm(dim=1).numpy(), 1.0, atol=1e-5)
    e, h = M.ExportWrapper(m)(_x())
    assert e.shape == (4, 128) and h.shape == (4, 3)


def test_parameter_budgets():
    n_student = M.count_params(M.Student())
    n_teacher = M.count_params(M.Teacher())
    assert 100_000 <= n_student <= 300_000, n_student
    assert 8_000_000 <= n_teacher <= 13_000_000, n_teacher


def test_challenge_conditioning_changes_output():
    m = M.Student().eval()
    for mod in m.modules():          # FiLM starts at identity; give it weight to test the path
        if isinstance(mod, M.FiLM):
            torch.nn.init.normal_(mod.proj.weight, std=0.1)
    x = _x()
    x2 = x.clone()
    x2[:, 9] = torch.roll(x2[:, 9], 7, dims=1)
    with torch.no_grad():
        assert not torch.allclose(m(x)["embedding"], m(x2)["embedding"])


# --- mirror augmentation ----------------------------------------------------

def test_mirror_flips_sideways_axis_and_matching_gyro_axes():
    x = _x(3)
    xm = D.mirror_tensor(x)
    flipped = [0, 4, 5, 6]
    for c in range(10):
        want = -x[:, c] if c in flipped else x[:, c]
        assert torch.equal(xm[:, c], want), c
    assert torch.equal(D.mirror_tensor(xm), x)
    np.testing.assert_array_equal(D.mirror_tensor(x.numpy()), xm.numpy())
    xa = D.mirror_tensor(x, "actuator")          # gravity reflected, vibration reflected and reversed
    for c in range(10):
        want = -x[:, c] if c in (1, 2, 3, 6) else x[:, c]
        assert torch.equal(xa[:, c], want), c
    assert torch.equal(D.mirror_tensor(xa, "actuator"), x)
    h = torch.tensor([0, 1, 2, 0])
    assert D.mirror_hand(h).tolist() == [1, 0, 2, 1]
    assert D.mirror_hand(np.array([0, 1, 2])).tolist() == [1, 0, 2]


def test_mirrored_copies_never_carry_identity():
    x = _x(8)
    ident = torch.arange(8)
    hand = torch.tensor([0, 1] * 4)
    xa, ia, ha, is_m = D.add_mirrored(x, ident, hand, 0.5, generator=torch.Generator().manual_seed(0))
    assert xa.shape[0] == 12 and is_m.sum() == 4
    assert torch.equal(xa[:8], x) and torch.equal(ia[:8], ident) and torch.equal(ha[:8], hand)
    assert torch.all(ia[is_m] == D.IGNORE)
    # every copy is the mirror of some original, with the swapped hand label
    for xm, hm in zip(xa[is_m], ha[is_m]):
        j = [k for k in range(8) if torch.equal(D.mirror_tensor(x[k]), xm)]
        assert len(j) == 1 and hm.item() == 1 - hand[j[0]].item()
    # and the identity loss ignores them
    arc = M.ArcFace(128, 8)
    emb = torch.nn.functional.normalize(torch.randn(12, 128), dim=1)
    full, _ = arc(emb, ia)
    only, _ = arc(emb[:8], ia[:8])
    assert torch.allclose(full, only)


def test_augmenter_keeps_template_and_shape():
    x = _x(16)
    x[:, :, 240:] = 0          # padding region
    aug = D.Augmenter(D.AugmentConfig(case_prob=1.0, dropout_prob=1.0), seed=0)
    y = aug(x)
    assert y.shape == x.shape
    assert torch.equal(y[:, 9], x[:, 9])
    assert torch.all(y[:, :, 242:] == 0)        # no noise outside the valid window (jitter moves 1)
    assert float((y[:, :9] - x[:, :9]).abs().mean()) > 0
    assert y.abs().max() <= 50


def test_time_jitter_only_shifts_signal_by_one_sample():
    cfg = D.AugmentConfig(gain_sigma=0, axis_gain_sigma=0, noise_accel=0, noise_gyro=0, noise_gravity=0,
                          case_prob=0, dropout_prob=0, time_jitter=1)
    x = _x(32)
    y = D.Augmenter(cfg, seed=1)(x)
    for i in range(32):
        ok = [s for s in (-1, 0, 1) if torch.allclose(y[i, :9, 2:-2], torch.roll(x[i, :9], s, dims=1)[:, 2:-2])]
        assert len(ok) == 1


# --- losses -----------------------------------------------------------------

def test_arcface_margin_and_ignore():
    torch.manual_seed(0)
    arc = M.ArcFace(128, 5, margin=0.0, scale=4)
    w = torch.nn.functional.normalize(arc.weight.detach(), dim=1)
    lab = torch.tensor([0, 1, 2, 3, 4])
    l0, _ = arc(w, lab)                          # embeddings on their class centres
    assert l0 < 0.2
    arc.set_margin(0.3)
    l1, _ = arc(w, lab)
    assert l1 > l0
    l2, _ = arc(w, torch.tensor([0, 1, -1, -1, -1]))
    l3, _ = arc(w[:2], lab[:2])
    assert torch.allclose(l2, l3)
    lz, _ = arc(w, torch.full((5,), -1))
    assert float(lz) == 0.0


def test_ssl_losses():
    g = torch.Generator().manual_seed(0)
    m = L.patch_mask(64, 8, 0.3, generator=g)
    assert m.shape == (64, 256)
    assert torch.all(m.float().sum(1) == 8 * round(0.3 * 32))
    x = _x(64)
    xm = L.apply_mask(x, m)
    assert torch.all(xm[:, :9][m[:, None, :].expand(-1, 9, -1)] == 0) and torch.equal(xm[:, 9], x[:, 9])
    assert float(L.masked_recon_loss(x[:, :9], x, m)) == 0.0
    pred = x[:, :9].clone()
    pred[~m[:, None, :].expand(-1, 9, -1)] += 5.0            # error outside the mask is not counted
    assert float(L.masked_recon_loss(pred, x, m)) == 0.0
    z = torch.randn(32, 16, generator=g)
    assert L.nt_xent(z, z) < L.nt_xent(z, torch.randn(32, 16, generator=g))


def test_distillation_losses():
    t = torch.nn.functional.normalize(torch.randn(32, 128), dim=1)
    assert float(L.cosine_distill(t, t)) < 1e-6
    assert float(L.relational_distill(t, t)) < 1e-9
    assert float(L.relational_distill(3 * t, t)) < 1e-9          # scale-free
    assert float(L.similarity_distill(t, t)) < 1e-9
    s = torch.nn.functional.normalize(torch.randn(32, 128), dim=1)
    assert float(L.cosine_distill(s, t)) > 0.5 and float(L.relational_distill(s, t)) > 0
    lg = torch.randn(8, 3)
    assert float(L.logit_distill(lg, lg)) < 1e-6


# --- export helpers (no coremltools needed) ----------------------------------

def test_fold_batchnorm_and_palettize():
    torch.manual_seed(0)
    s = M.Student()
    s.train()
    with torch.no_grad():                     # give BN non-trivial statistics
        for _ in range(3):
            s(_x(64, seed=_))
    s.eval()
    x = _x(8, seed=9)
    folded = fold_batchnorm(s)
    assert not any(isinstance(mod, torch.nn.BatchNorm1d) for mod in folded.modules())
    with torch.no_grad():
        e0, h0 = M.ExportWrapper(s)(x)
        e1, h1 = M.ExportWrapper(folded)(x)
    assert float((e0 - e1).abs().max()) < 1e-4 and float((h0 - h1).abs().max()) < 1e-3
    names = palettize_model(folded)
    assert names
    for name, p in folded.named_parameters():
        if name in names:
            assert p.numel() > WEIGHT_THRESHOLD and np.unique(p.detach().numpy()).size <= 256
    with torch.no_grad():
        e2, _ = M.ExportWrapper(folded)(x)
    assert float((e0 - e2).abs().max()) < 0.05
    w = np.random.default_rng(0).normal(size=5000)
    q = palettize_tensor(w)
    assert np.unique(q).size <= 256 and np.abs(q - w).mean() < 0.01 and np.abs(q - w).max() < 1.0


@pytest.mark.skipif(importlib.util.find_spec("coremltools") is None, reason="coremltools not installed")
def test_coreml_conversion_contract(tmp_path):
    import coremltools as ct
    from v1.export_coreml import convert, export_ready

    ready, _ = export_ready(M.Student().eval())
    ml = convert(ready, str(tmp_path / "H.mlpackage"), {"version": "test"})
    spec = ml.get_spec()
    assert [i.name for i in spec.description.input] == ["x"]
    assert sorted(o.name for o in spec.description.output) == ["embedding", "hand"]
    assert list(spec.description.input[0].type.multiArrayType.shape) == [1, 10, 256]
    assert spec.specificationVersion >= 8          # iOS 17
