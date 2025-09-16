<p align="center">
  <img width="100%" alt="polypart-logo" src="https://private-user-images.githubusercontent.com/94929744/490196351-a9d7a7fd-89e2-436c-ab4b-2663d78091d6.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTgwNDgyMzgsIm5iZiI6MTc1ODA0NzkzOCwicGF0aCI6Ii85NDkyOTc0NC80OTAxOTYzNTEtYTlkN2E3ZmQtODllMi00MzZjLWFiNGItMjY2M2Q3ODA5MWQ2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MTYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTE2VDE4Mzg1OFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNhZTY3NDcwZmVhZTRjMjYzNDViMWJlYzE2MmUyZTVkNWE1YTdkYjUzYmJiZTc5MTE0ZWE2MmNmMmZkMDkxYTgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.8FifR8DOKU6fs2gyjUhfQkSMeW8xUTqMtR9uSBq_R04" />
</p>


[![pypi](https://img.shields.io/pypi/v/motives.svg)](https://pypi.python.org/pypi/motives)
[![PyPI Downloads](https://static.pepy.tech/badge/motives)](https://pepy.tech/projects/motives)
[![python](https://img.shields.io/badge/python-%5E3.10-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/ciamod/polypart)


PolyPart is a Python library for **partitioning d-dimensional convex polytopes** by affine hyperplanes by building a ***decision tree***, outputing the exact number of regions and allowing efficient point classification into the resulting regions.

All computations are carried out with exact rational arithmetic ensuring robustness and avoiding floating-point errors. Under the hood, `pycddlib` with GMP is used for efficient and exact polyhedral computations.


## Citation

If you use the "polypart" package in your work, please cite the paper

>Sergio Herreros, David Alfaya, José Portela and Jaime Pizarroso. [In Progress](https://arxiv.org/abs/XXXX.XXXXX). _arXiv:XXXX.XXXXX_, 2025.

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

- **Partitioning_Unit_Square.ipynb**  
  Demonstrates the algorithm on a simple, visual case: partitioning the unit square with three hyperplanes. Includes a plot of the resulting regions and new random points classified into their respective regions using the partition tree.

- **Moduli_Stability_Chambers.ipynb**  
  A higher-dimensional research application on moduli spaces of parabolic vector bundles.


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- ***Sergio Herreros Pérez***, Graduate in Mathematical Engineering and Artificial Intelligence, ICAI, Comillas Pontifical University
- ***José Portela González***, Department of Quantitative Methods, Comillas Pontifical University
- ***David Alfaya Sánchez***, Department of Applied Mathematics and Institute for Research in Technology, ICAI, Comillas Pontifical University
- ***Jaime Pizarroso Gonzalo***, Department of Telematics and Computing and Institute for Research in Technology, ICAI, Comillas Pontifical University

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

