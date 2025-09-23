<p align="left">
  <a href="https://pypi.org/project/polypart/">
    <img src="https://img.shields.io/pypi/v/polypart.svg" alt="PyPI">
  </a>
  <a href="https://pepy.tech/projects/polypart">
    <img src="https://static.pepy.tech/badge/polypart" alt="">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-%5E3.10-blue" alt="Python">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/ciamod/polypart">
    <img src="https://img.shields.io/badge/GitHub-Repo-black?logo=github" alt="GitHub Repo">
  </a>
</p>
<p align="left">
  <img width="465" alt="polypart-logo" src="https://raw.githubusercontent.com/CIAMOD/polypart/develop/images/polypart-logo.png" />
</p>

PolyPart is a Python package for **partitioning d-dimensional convex polytopes** by affine hyperplane arrangements. The algorithm works by iteratively slicing the polytope with hyperplanes and organizing the resulting subdivisions into a **decision tree**, which enables:

- Counting the exact number of regions
- Efficient point classification into the corresponding region

All computations use exact rational arithmetic, which ensures robustness and eliminates floating-point errors. The implementation builds on [`pycddlib`](https://pypi.org/project/pycddlib/) with GMP to provide efficient and reliable polyhedral computations.

## 🚀 Quick Start

### Prerequisites

- Python >= 3.10.6
- numpy >= 1.24.4
- pycddlib >= 3.0.2

### Installation

1. Install the package from PyPI

```sh
pip install polypart
```

2. Import the package in your Python code
```python
import polypart
```

### Basic Usage

```python
from fractions import Fraction
from polypart.geometry import Polytope, Hyperplane
from polypart.ppart import build_partition_tree
from polypart.io import save_tree

# 1. Create initial polytope object in H-representation
# unit square: 0 <= x <= 1, 0 <= y <= 1
A = [[-1, 0], [1, 0], [0, -1], [0, 1]]
b = [0, 1, 0, 1]
square = Polytope(A, b) # expressed as A x <= b
square.extreme()  # compute vertices (V-representation)

h1 = Hyperplane.from_coefficients([1, 0, Fraction(1, 3)]) # x = 1/3
h2 = Hyperplane.from_coefficients([0, 1, Fraction(1, 3)]) # y = 1/3

tree, n_parts = build_partition_tree(square, [h1, h2])
save_tree(tree, "data/square_partitions.json")
print("Number of regions:", n_parts)

# Output: 'Number of regions: 4'
```

## 🎯 Examples

Complete Jupyter notebooks providing guided, reproducible demonstrations of how to use PolyPart are available in the [examples folder](https://github.com/ciamod/polypart/examples):

- [**Partitioning_Unit_Square.ipynb**](https://github.com/CIAMOD/polypart/tree/master/examples/Partitioning_Unit_Square.ipynb)
 
  Demonstrates the algorithm on a simple, visual case: partitioning the unit square with three hyperplanes. Includes a plot of the resulting regions and new random points classified into their respective regions using the partition tree.

- [**Moduli_Stability_Chambers.ipynb**](https://github.com/CIAMOD/polypart/tree/master/examples/Moduli_Stability_Chambers.ipynb)
  A higher-dimensional research application on moduli spaces of parabolic vector bundles.

## 📝 Changelog

### [Unreleased] - 2025-09-22

- Added optimization strategy 'v-entropy' to `build_partition_tree` resulting in more balanced trees, faster point classification while still and shorter runtime for building the tree.
- Added ´stats´ function to `PartitionTree` class to compute and return various statistics about the tree structure (e.g., number of nodes, average depth, max depth).
- Fixed avg_depth calculation in `save_tree` function.
- Improved documentation and expanded example "Moduli_Stability_Chambers.ipynb" comparing with both strategies.
- Added `keywords` and `project.urls` to `pyproject.toml` for better PyPI metadata.


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- ***Sergio Herreros Pérez***, Institute for Research in Technology, ICAI, Comillas Pontifical University
- ***José Portela González***, Department of Quantitative Methods and Institute for Research in Technology, ICADE, Comillas Pontifical University
- ***David Alfaya Sánchez***, Department of Applied Mathematics and Institute for Research in Technology, ICAI, Comillas Pontifical University
- ***Jaime Pizarroso Gonzalo***, Department of Telematics and Computing and Institute for Research in Technology, ICAI, Comillas Pontifical University, and Santalucía Chair of Analytics for Education.

## 🙌 Acknowledgments

This research was supported by project CIAMOD (Applications of computational methods and artificial intelligence to the study of moduli spaces, project PP2023_9) funded by Convocatoria de Financiación de Proyectos de Investigación Propios 2023, Universidad Pontificia Comillas, and by grants PID2022-142024NB-I00 and RED2022-134463-T funded by MCIN/AEI/10.13039/501100011033.

Find more about the CIAMOD project in the [project webpage](https://ciamod.github.io/) and the [IIT proyect webpage](https://www.iit.comillas.edu/publicacion/proyecto/en/CIAMOD/Aplicaciones_de_m%c3%a9todos_computacionales_y_de_inteligencia_artificial_al_estudio_de_espacios_de_moduli).

Special thanks to everyone who contributed to the project:

- David Alfaya Sánchez (PI), Department of Applied Mathematics and Institute for Research in Technology, ICAI, Comillas Pontifical University
- Javier Rodrigo Hitos, Department of Applied Mathematics, ICAI, Comillas Pontifical University
- Luis Ángel Calvo Pascual, Department of Quantitative Methods, ICADE, Comillas Pontifical University
- Anitha Srinivasan, Department of Quantitative Methods, ICADE, Comillas Pontifical University
- José Portela González, Department of Quantitative Methods, ICADE, IIT, Comillas Pontifical University
- Jaime Pizarroso Gonzalo, Department of Telematics and Computing and Institute for Research in Technology, ICAI, Comillas Pontifical University
- Tomás Luis Gómez de Quiroga, Institute of Mathematical Sciences, UAM-UCM-UC3M-CSIC
- Daniel Sánchez Sánchez, Student of the Degree in Mathematical Engineering and Artificial Intelligence, Institute for Research in Technology, ICAI, Comillas Pontifical University
- Alejandro Martínez de Guinea García, Student of the Degree in Mathematical Engineering and Artificial Intelligence, Institute for Research in Technology, ICAI, Comillas Pontifical University
- Sergio Herreros Pérez, Student of the Degree in Mathematical Engineering and Artificial Intelligence, Institute for Research in Technology, ICAI, Comillas Pontifical University

## 📚 References

- *David Alfaya, Sergio Herreros, Jaime Pizarroso, José Portela, and Javier Rodrigo*, **"A Computational Analysis of Ismorphism Classes of Moduli Spaces of Parabolic Vector Bundles"**, *in preparation*, 2025.
- *David Alfaya, Sergio Herreros, Jaime Pizarroso and José Portela*, **"PolyPart: Exact Partitioning of Convex Polytopes uisng Decision Trees"**, *in preparation*, 2025.
