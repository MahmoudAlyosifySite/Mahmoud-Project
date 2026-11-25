"""SPEC section 2: SplitMix64, xoshiro256**, uniform/int helpers, seed strings."""

import math

import numpy as np
import pytest

from auditor_ref.rng import MASK64, SplitMix64, Xoshiro256, seed_to_str, str_to_seed


def test_splitmix_known_vector():
    sm = SplitMix64(0)
    assert sm.next() == 0xE220A8397B1DCDAF
    # published continuation of the seed-0 stream
    assert [sm.next() for _ in range(4)] == [
        0x6E789E6AA1B965F4, 0x06C45D188009454F, 0xF88BB8A8724C81EC, 0x1B39896A51A8749B,
    ]


def test_splitmix_matches_numpy_uint64_emulation():
    def ref(seed, count):
        s = np.uint64(seed)
        out = []
        with np.errstate(over="ignore"):
            for _ in range(count):
                s = s + np.uint64(0x9E3779B97F4A7C15)
                z = s
                z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                out.append(int(z ^ (z >> np.uint64(31))))
        return out

    for seed in (0, 1, 1234, MASK64, 0x8000000000000000):
        sm = SplitMix64(seed)
        assert [sm.next() for _ in range(8)] == ref(seed, 8)


def test_xoshiro_reference_state_vector():
    # Reference output of xoshiro256** for the raw state (1, 2, 3, 4).
    r = Xoshiro256(0)
    r.s = [1, 2, 3, 4]
    assert [r.next() for _ in range(10)] == [
        11520, 0, 1509978240, 1215971899390074240, 1216172134540287360,
        607988272756665600, 16172922978634559625, 8476171486693032832,
        10595114339597558777, 2904607092377533576,
    ]


def test_xoshiro_seeding_uses_four_splitmix_outputs():
    sm = SplitMix64(42)
    r = Xoshiro256(42)
    assert r.s == [sm.next() for _ in range(4)]


def test_xoshiro_determinism_and_seed_sensitivity():
    a = Xoshiro256(1234)
    b = Xoshiro256(1234)
    c = Xoshiro256(1235)
    sa = [a.next() for _ in range(100)]
    assert sa == [b.next() for _ in range(100)]
    assert sa != [c.next() for _ in range(100)]
    assert all(0 <= v <= MASK64 for v in sa)


def test_uniform_range_and_resolution():
    r = Xoshiro256(7)
    vals = [r.uniform() for _ in range(20000)]
    assert min(vals) >= 0.0 and max(vals) < 1.0
    assert abs(np.mean(vals) - 0.5) < 0.01
    # every value is a multiple of 2**-53
    assert all(math.ldexp(v, 53).is_integer() for v in vals[:200])


def test_uniform_extreme_output_is_below_one():
    class Fixed(Xoshiro256):
        def next(self):
            return MASK64

    assert Fixed(0).uniform() == (2.0 ** 53 - 1) * 2.0 ** -53 < 1.0


def test_uniform_range_formula():
    a = Xoshiro256(99)
    b = Xoshiro256(99)
    for _ in range(50):
        assert a.uniform_range(0.08, 0.15) == 0.08 + (0.15 - 0.08) * b.uniform()


@pytest.mark.parametrize("lo,hi", [(6, 8), (0, 0), (-5, 5), (1, 100), (0, 1)])
def test_int_in_range_bounds(lo, hi):
    r = Xoshiro256(lo * 1000 + hi)
    vals = [r.int_in_range(lo, hi) for _ in range(3000)]
    assert min(vals) >= lo and max(vals) <= hi
    if hi - lo <= 10:
        assert set(vals) == set(range(lo, hi + 1))


def test_int_in_range_formula_and_clamp():
    a = Xoshiro256(5)
    b = Xoshiro256(5)
    for _ in range(100):
        assert a.int_in_range(6, 8) == 6 + min(int(math.floor(b.uniform() * 3.0)), 2)


def test_seed_strings():
    assert seed_to_str(1234) == "0x00000000000004d2"
    assert seed_to_str(MASK64) == "0xffffffffffffffff"
    assert seed_to_str(0) == "0x0000000000000000"
    for s in (0, 1, 1234, MASK64, 0xDEADBEEFCAFEBABE):
        assert str_to_seed(seed_to_str(s)) == s
        assert len(seed_to_str(s)) == 18
    assert str_to_seed("0xDEADBEEFCAFEBABE") == 0xDEADBEEFCAFEBABE
    assert str_to_seed("1234") is None
    assert str_to_seed("0x") is None
    assert str_to_seed("0x1" + "0" * 16) is None
