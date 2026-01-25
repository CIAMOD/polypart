import pytest

from polypart import build_incenu_tree, build_partition_tree, number_of_regions
from polypart.core.geometry import Hyperplane
from polypart.generators.arrangements import (
    get_braid_arrangement,
    get_moduli_arrangement,
    get_random_arrangement,
    get_resonance_arrangement,
)
from polypart.generators.polytopes import (
    get_centered_hypercube,
    get_hypercube,
    get_product_of_simplices,
    get_simplex,
    sample_circumscribed_polytope,
    sample_poisson_zero_cell_polytope,
)

# --- Generators Fixtures ---


@pytest.fixture
def run_params():
    return {"decimals": 3, "seed": 42}


def safe_run_comparison(polytope, arrangement):
    """Run all 3 algos and assert they return same number of regions."""
    # PolyPart
    _, n_pp = build_partition_tree(arrangement, polytope, record_stats=True)
    # IncEnu
    _, n_ie = build_incenu_tree(arrangement, polytope)
    # DelRes
    n_dr = number_of_regions(arrangement, polytope)

    assert n_pp == n_ie == n_dr, f"Mismatch: PP={n_pp}, IE={n_ie}, DR={n_dr}"
    return n_pp


# --- Test Cases ---


def test_hypercube_braid():
    d = 3
    P = get_hypercube(d)
    A = get_braid_arrangement(d)
    n = safe_run_comparison(P, A)
    assert n == 6


def test_centered_hypercube_resonance():
    d = 3
    P = get_centered_hypercube(d, r=2)
    A = get_resonance_arrangement(d)
    safe_run_comparison(P, A)


def test_simplex_random():
    d = 3
    P = get_simplex(d)
    P.extreme()
    A = get_random_arrangement(P, m=5, seed=42)
    safe_run_comparison(P, A)


def test_product_simplices_braid():
    # Product of 2 simplices of dim 1. Total dim = 2.
    P = get_product_of_simplices(n=2, d=1)
    A = get_braid_arrangement(2)
    safe_run_comparison(P, A)


def test_circumscribed_polytope_braid():
    d = 3
    P = sample_circumscribed_polytope(d, m=10, seed=123)
    P.extreme()
    A = get_braid_arrangement(d)
    safe_run_comparison(P, A)


def test_moduli_n1():
    for r in range(2, 7):
        dim = r - 1
        # We need a polytope of this dimension.
        P = get_simplex(dim)
        P.extreme()
        A = get_moduli_arrangement(1, r, d=0, use_epsilons=True)

        safe_run_comparison(P, A)


def test_moduli_r2():
    for n in range(1, 6):
        dim = n
        P = get_simplex(dim)
        P.extreme()
        A = get_moduli_arrangement(n, 2, d=0, use_epsilons=True)

        safe_run_comparison(P, A)


def test_poisson_zero_cell_braid():
    d = 4
    P = sample_poisson_zero_cell_polytope(
        d, intensity=1.0, window_radius=10.0, seed=999
    )
    P.extreme()
    A = get_braid_arrangement(d)
    safe_run_comparison(P, A)


def test_ppart_strategies():
    d = 3
    P = get_hypercube(d)
    A = get_braid_arrangement(d)
    _, n1 = build_partition_tree(A, P, strategy="v-entropy", record_stats=True)
    _, n2 = build_partition_tree(A, P, strategy="random", record_stats=True)
    assert n1 == n2 == 6


def test_degenerate_arrangement():
    # Test with degenerate arrangement (multiple hyperplanes intersecting at same point inside P)
    d = 2
    P = get_hypercube(d)
    # Lines x=0.5, y=0.5. Intersection (0.5, 0.5) is inside P=[0,1]^2.

    h1 = Hyperplane([1, 0], 0.5)  # x = 0.5
    h2 = Hyperplane([0, 1], 0.5)  # y = 0.5
    h3 = Hyperplane([1, 1], 1.0)  # x+y = 1. Passes through (0.5, 0.5).
    A = [h1, h2, h3]

    safe_run_comparison(P, A)


def test_incenu_delres_no_support_braid():
    """Test incenu and delres with braid arrangements without providing support."""
    for d in range(2, 5):
        A = get_braid_arrangement(d)

        # Test incenu without support
        tree_ie, n_ie = build_incenu_tree(A)
        assert n_ie > 0
        assert tree_ie.root is not None

        # Test delres without support
        n_dr = number_of_regions(A)
        assert n_dr > 0

        # They should agree on the number of regions
        assert n_ie == n_dr

        # The actual number depends on the braid arrangement implementation
        # Just verify they're positive and consistent
        assert n_ie >= 1
        assert n_dr >= 1


def test_incenu_delres_empty_list_support():
    """Test incenu and delres with empty list as support (equivalent to None)."""
    d = 2
    A = get_braid_arrangement(d)

    # Test with None support
    tree_none, n_none = build_incenu_tree(A)
    n_dr_none = number_of_regions(A)

    # Test with empty list support
    tree_empty, n_empty = build_incenu_tree(A, [])
    n_dr_empty = number_of_regions(A, [])

    # Results should be identical
    assert n_none == n_empty
    assert n_dr_none == n_dr_empty
