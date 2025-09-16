# polypart

polypart provides a tool to count partitions of convex polytopes (in H-representation) after being
split by a set of affine hyperplanes. The package uses exact rational arithmetic (Fraction)
to avoid numerical issues and relies on `pycddlib` to convert between H-representation
and V-representation (vertices). The main feature is to build a partition decision tree
to both count the number of regions and classify points into their respective regions.

## Features

- Represent polytopes and hyperplanes using exact rational arithmetic (Fraction).
- Split polytopes with affine hyperplanes and build a partition decision tree.
- Serialize/deserialize the partition trees to JSON for downstream analysis or
  visualization.

## Requirements

- Python == 3.10
- numpy
- pycddlib (pycddlib uses GMP; platform-specific installation may be required)

Requirements are listed in `pyproject.toml` under `[project].dependencies`.

## Installation

It's recommended to use a virtual environment. From the project root run:

```powershell
# Create and activate a venv (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# Install the package in editable mode (installs runtime deps)
python -m pip install -e .
```

Note about `pycddlib`: on some platforms (notably Windows) you may need
to install a prebuilt wheel or use conda to obtain a compatible build with
GMP support. If `pip install -e .` fails at `pycddlib`, consider `conda install -c conda-forge pycddlib`.

## Quickstart (unit square example)

The `examples/square.py` script demonstrates building a simple partition of
the unit square by two axis-aligned hyperplanes and how to save and plot the
result.

Minimal programmatic usage is:

```python
from fractions import Fraction
from polypart.geometry import Polytope, Hyperplane
from polypart.ppart import build_partition_tree
from polypart.io import save_tree

# unit square: 0 <= x <= 1, 0 <= y <= 1 expressed as A x <= b
A = [[-1, 0], [1, 0], [0, -1], [0, 1]]
b = [0, 1, 0, 1]
square = Polytope(A, b)
square.extreme()  # compute vertices (V-representation)

h1 = Hyperplane.from_coefficients([1, 0, Fraction(1, 3)])
h2 = Hyperplane.from_coefficients([0, 1, Fraction(1, 3)])

tree, n_parts = build_partition_tree(square, [h1, h2])
save_tree(tree, "data/square_partitions.json")
print("Number of regions:", n_parts)
```

Run the example from the project root:

```powershell
python -m examples.square
```

This will create `data/square_partitions.json` and `images/square_partitions.png`.

## Examples

- `examples/square.py`
  - Demonstrates a 2D unit square partitioned by two hyperplanes.
  - Builds the partition tree via `build_partition_tree`, saves the tree with
    `polypart.io.save_tree`, and produces a scatter plot coloring points by
    region.

- `examples/moduli.py`
  - A more specialized script used to compute combinatorial moduli for a
    mathematics research workflow. It constructs a simplex polytope and
    enumerates candidate partitioning hyperplanes (using combinatorial
    generation utilities in the same script), then builds and saves the
    partition tree(s) into `data/moduli_n{n}_r{r}.json`.
  - This example is computationally heavier and intended as a reference for
    research use cases; see the top of the file for parameters and usage.

## Library API (overview)

polypart exposes a small set of modules. This overview highlights the most
commonly used classes and functions.

- `polypart.geometry`
  - `Polytope(A, b)` — Construct from H-representation (A x <= b). Use
    `extreme()` to compute and cache vertices (V-rep) using pycddlib.
  - `Hyperplane(normal, offset)` / `Hyperplane.from_coefficients([...])`
    — Represent affine hyperplanes `normal · x = offset` and the halfspace
    convention `normal · x <= offset`.

- `polypart.ppart`
  - `PartitionNode`, `PartitionTree` — Tree node and tree containers used to
    represent the partition decision tree.
  - `build_partition_tree(polytope, hyperplanes)` — Construct a partition
    tree by recursively splitting polytope regions with candidate hyperplanes.
    Returns `(tree, n_regions)`.

- `polypart.io`
  - `save_tree(tree, path)` — Serialize a PartitionTree to a JSON file.
  - `load_tree(path)` — Deserialize back to a `PartitionTree` (polytope fields
    will be `None`; the structural data needed for classification is restored).

- `polypart.ftyping`
  - Utilities for exact arithmetic conversion: `to_fraction`,
    `as_fraction_vector`, `as_fraction_matrix`.

## Running tests

The project uses pytest. From the project root (inside an activated venv):

```powershell
python -m pytest -q
```

If tests fail with import errors for `pycddlib` or `cdd`, verify that
`pycddlib` was installed correctly and that GMP is available to the system.

## Notes and platform caveats

- Exact rational arithmetic implies that polytope vertex computations use
  `pycddlib` which depends on GMP. On Linux/macOS the library is commonly
  available via package managers or pip wheels. On Windows you may prefer
  `conda` (conda-forge) to obtain a working `pycddlib` build.
- The `examples/moduli.py` script can be slow or memory-intensive depending
  on parameters (`n`, `r`), as it enumerates combinatorial configurations.

## Contributing

Contributions are welcome. Please open issues or pull requests with small,
focused changes. If you add public API surface, include tests and update this
README accordingly.

## License

This project is licensed under the MIT License — see `LICENSE` for details.
