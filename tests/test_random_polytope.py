import numpy as np
import pytest

from polypart.geometry import Polytope
from polypart.polytopes import sample_circumscribed_polytope


def test_requires_enough_halfspaces():
    d = 3
    m = 3  # needs at least d + 1 = 4
    with pytest.raises(ValueError):
        sample_circumscribed_polytope(d=d, m=m)


def test_returns_polytope_with_vertices():
    d = 3
    m = 10
    P = sample_circumscribed_polytope(d=d, m=m, seed=123)

    assert isinstance(P, Polytope)
    assert P.vertices is not None
    assert P.vertices.ndim == 2
    assert P.vertices.shape[1] == d
    assert P.vertices.shape[0] > 0


def test_seed_reproducibility():
    d = 4
    m = 20
    P1 = sample_circumscribed_polytope(d=d, m=m, seed=42)
    P2 = sample_circumscribed_polytope(d=d, m=m, seed=42)

    # A and b should match exactly for the same seed
    assert np.array_equal(P1.A, P2.A)
    assert np.array_equal(P1.b, P2.b)

    # Vertices should also match if extreme is deterministic
    assert np.array_equal(P1.vertices, P2.vertices)


def test_radius_changes_halfspace_offsets_not_normals():
    d = 3
    m = 15
    seed = 7

    P1 = sample_circumscribed_polytope(d=d, m=m, radius=1.0, seed=seed)
    P2 = sample_circumscribed_polytope(d=d, m=m, radius=2.0, seed=seed)

    # Same normals, different offsets
    assert np.array_equal(P1.A, P2.A)
    assert np.all(P1.b == 1.0)
    assert np.all(P2.b == 2.0)
