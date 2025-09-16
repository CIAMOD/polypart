# PolyPart 0.1.0

<p align="center">
  <img src="https://github.com/user-attachments/assets/aa6f0c16-2ae5-4d86-92f7-86d47ac6596f" />
</p>

<p align="center">
   <span>

   [![pypi](https://img.shields.io/pypi/v/motives.svg)](https://pypi.python.org/pypi/motives)
   [![PyPI Downloads](https://static.pepy.tech/badge/motives)](https://pepy.tech/projects/motives)
   [![python](https://img.shields.io/badge/python-%5E3.10-blue)]()
   [![os](https://img.shields.io/badge/OS-Ubuntu%2C%20Mac%2C%20Windows-purple)]()
   </span>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/ciamod/polypart)


PolyPart is a Python package for counting partitions of convex polytopes by a set of affine hyperplanes. The package uses exact rational arithmetic (Fraction) to avoid numerical issues and relies on `pycddlib` to efficiently convert between H-representation and V-representation (vertices).

In particular, we build a partition decision tree to both count the number of regions and classify new points into their respective regions.


## Citation

If you use the "polypart" package in your work, please cite the paper

>Sergio Herreros, David Alfaya, José Portela and Jaime Pizarroso. [In Progress](https://arxiv.org/abs/XXXX.XXXXX). _arXiv:XXXX.XXXXX_, 2025.

## Getting Started

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

## Usage

The minimal workflow to create a partition of a polytope by hyperplanes is as follows:

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

## Examples

Complete Jupyter notebooks providing guided, reproducible demonstrations of how to use PolyPart are available in the [examples folder](https://github.com/ciamod/polypart/examples):

- **Partitioning_Unit_Square.ipynb**  
  Demonstrates the algorithm on a simple, visual case: partitioning the unit square with three hyperplanes.

- **Moduli_Stability_Chambers.ipynb**  
  A higher-dimensional research application on moduli spaces of parabolic vector bundles.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- ***Sergio Herreros Pérez***, Graduate in Mathematical Engineering and Artificial Intelligence, ICAI, Comillas Pontifical University
- ***José Portela González***, Department of Quantitative Methods, Comillas Pontifical University
- ***David Alfaya Sánchez***, Department of Applied Mathematics and Institute for Research in Technology, ICAI, Comillas Pontifical University
- ***Jaime Pizarroso Gonzalo***, Department of Telematics and Computing and Institute for Research in Technology, ICAI, Comillas Pontifical University

## Acknowledgments

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

## References

