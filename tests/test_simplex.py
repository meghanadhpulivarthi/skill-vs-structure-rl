import numpy as np
from src.simplex import project_to_simplex


def test_already_on_simplex_is_unchanged():
    w = np.array([0.2, 0.3, 0.5])
    out = project_to_simplex(w)
    np.testing.assert_allclose(out, w, atol=1e-9)


def test_projection_sums_to_one_and_nonnegative():
    v = np.array([3.0, -1.0, 0.5, 2.0])
    out = project_to_simplex(v)
    assert np.all(out >= -1e-12)
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-9)


def test_negative_inputs_are_clipped_to_zero_mass():
    # a very negative entry should receive zero weight
    v = np.array([1.0, 1.0, -50.0])
    out = project_to_simplex(v)
    assert out[2] == 0.0
