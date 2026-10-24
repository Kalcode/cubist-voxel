import numpy as np

from cubist.dsl import building
from cubist.dsl.noise import fbm, hash3, hash_unit, mix_seed, perlin, value_noise

PTS = np.random.default_rng(0).uniform(-20, 20, size=(500, 3))


def test_hash_values_are_pinned():
    # Changing these silently changes every seeded asset; bump deliberately if ever needed.
    got = hash3(np.array([[0, 0, 0], [1, 2, 3], [-5, 7, -9]]), seed=42).tolist()
    assert got == PINNED


PINNED = [3482966847, 679469061, 3271408132]


def test_hash_unit_is_uniform_enough():
    u = hash_unit(np.indices((20, 20, 20)).reshape(3, -1).T, seed=7)
    assert 0 <= u.min() and u.max() < 1
    assert abs(u.mean() - 0.5) < 0.02


def test_noise_is_deterministic_and_seeded():
    for fn in (perlin, value_noise):
        a, b, c = fn(PTS, 1), fn(PTS, 1), fn(PTS, 2)
        assert np.array_equal(a, b)
        assert not np.allclose(a, c)
        assert np.all(np.abs(a) <= 1.0 + 1e-9)


def test_perlin_is_zero_on_the_lattice():
    lattice = np.array([[0, 0, 0], [3, -2, 5], [10, 10, 10]], dtype=float)
    assert np.allclose(perlin(lattice, 5), 0)


def test_noise_is_continuous():
    step = PTS + 1e-4
    assert np.max(np.abs(perlin(PTS, 3) - perlin(step, 3))) < 1e-2


def test_fbm_stays_in_range():
    v = fbm(PTS, 9, octaves=4)
    assert np.all(np.abs(v) <= 1.0 + 1e-9)


def test_mix_seed_depends_on_the_build_seed():
    with building(seed=0):
        a = mix_seed(5)
    with building(seed=1):
        b = mix_seed(5)
    with building(seed=0):
        c = mix_seed(5)
    assert a != b and a == c
