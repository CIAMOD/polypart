"""Generators for hyperplane arrangements and polytopes."""

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

__all__ = [
    "get_braid_arrangement",
    "get_centered_hypercube",
    "get_hypercube",
    "get_moduli_arrangement",
    "get_product_of_simplices",
    "get_random_arrangement",
    "get_resonance_arrangement",
    "get_simplex",
    "sample_circumscribed_polytope",
    "sample_poisson_zero_cell_polytope",
]
