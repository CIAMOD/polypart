"""Utility functions for polypart.

Due to circular dependencies with core.geometry, sampling functions
should be imported directly from their modules:
    from polypart.utils.io import load_tree, save_tree
    from polypart.utils.solvers import IncEnuCDDBackend
    from polypart.utils.sampling import sample_point_in_polytope
"""

# Note: Do NOT import from sampling here to avoid circular imports
# (core.geometry -> utils.volume -> utils.__init__ -> sampling -> core.geometry)
# Import sampling functions directly: from polypart.utils.sampling import ...

__all__: list[str] = []
