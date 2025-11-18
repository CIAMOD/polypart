from polypart.geometry import Hyperplane, Polytope
from polypart.ppart import build_partition_tree


def test_square_partition_4_cells():
    # unit square in 2D: inequalities x>=0, x<=1, y>=0, y<=1 -> expressed as A x <= b
    A = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    b = [0, 1, 0, 1]
    square = Polytope(A, b)
    square.extreme()

    # two axis-aligned hyperplanes: x = 0.2 and y = 0.2
    h1 = Hyperplane.from_coefficients([1, 0, 0.2])
    h2 = Hyperplane.from_coefficients([0, 1, 0.2])
    h3 = Hyperplane.from_coefficients([-1, 1, 0.0])

    _, n_partitions = build_partition_tree(square, [h1, h2, h3])

    assert n_partitions == 6
