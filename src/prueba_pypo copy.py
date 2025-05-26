import numpy as np
from pypoman import (
    compute_polytope_vertices,
    compute_polytope_halfspaces,
)  # pypoman->cdd->double description method
import cdd

points = np.array(
    [
        np.array([0.09354189, 0.71191172, 0.48691181]),
        np.array([0.0999657, 0.72929686, 0.44351312]),
        np.array([0.09257199, 0.7111259, 0.4840692]),
        np.array([0.11184999, 0.71455827, 0.5073961]),
        np.array([0.10787734, 0.71068151, 0.50520338]),
        np.array([0.09991535, 0.72436869, 0.43971949]),
        np.array([0.10011718, 0.72415858, 0.43972567]),
        np.array([0.12424356, 0.72736896, 0.46203143]),
        np.array([0.10787959, 0.71067264, 0.50519023]),
        np.array([0.09707062, 0.70046439, 0.47083138]),
        np.array([0.09476829, 0.69836908, 0.45745356]),
        np.array([0.09477092, 0.69837167, 0.45745352]),
        np.array([0.09367225, 0.7012202, 0.46735247]),
        np.array([0.09467687, 0.69838722, 0.45735069]),
        np.array([0.09470106, 0.69850074, 0.45726384]),
        np.array([0.09982322, 0.72369025, 0.43984195]),
        np.array([0.10070341, 0.71515373, 0.50317246]),
        np.array([0.13527711, 0.73802257, 0.47922593]),
        np.array([0.12018379, 0.72271447, 0.51199993]),
        np.array([0.1119239, 0.7322001, 0.51039059]),
        np.array([0.12140621, 0.75211311, 0.47853525]),
        np.array([0.11202067, 0.73310208, 0.50930094]),
    ]
)
A = [
    [3.03922519, -5.51461381, 4.24358443],
    [-7.31969543, -9.42434647, 5.10274511],
    [8.80917169, 9.07857154, 8.2972878],
    [-7.63670345, 3.93474331, 2.57885694],
    [8.69896814, -8.97771075, -9.31264066],
    [3.78035357, 3.49633356, -4.59212357],
    [7.93752262, -8.05105813, -0.10504745],
    [-3.15225785, -1.36617207, 8.12650464],
    [6.64885282, -5.75321779, -6.36350066],
    [-8.8488248, 0.99057765, -1.16938997],
    [-6.99294781, 5.05303797, 0.98911729],
    [-4.30319011, -9.26226105, 2.19128668],
    [-1.03813678, -1.21875943, -7.53307205],
]
b = [
    -1.44739737,
    -4.90941397,
    11.86814547,
    3.3425141,
    -9.70546992,
    0.89110369,
    -4.91843895,
    2.7945672,
    -6.29877221,
    -0.6807936,
    3.42479628,
    -5.87387085,
    -4.29899556,
]

A, b = np.array(A), np.array(b).reshape(-1, 1)

normal = np.array([-7.93752262, 8.05105813, 0.10504745])
offset = 4.918438948518409

# Evaluate vertices of the polytope on normal
values = points @ normal
print(f"Min value: {np.min(values)}, Max value: {np.max(values)}, b: {offset}")

# Plot polytope and hyperplane
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="b", marker="o")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Polytope and Hyperplane")
# Plot the hyperplane
xx, yy = np.meshgrid(
    np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 10),
    np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 10),
)
zz = (offset - normal[0] * xx - normal[1] * yy) / normal[2]
ax.plot_surface(xx, yy, zz, alpha=0.5, color="r")
ax.set_xlim(np.min(points[:, 0]), np.max(points[:, 0]))
ax.set_ylim(np.min(points[:, 1]), np.max(points[:, 1]))
ax.set_zlim(np.min(points[:, 2]), np.max(points[:, 2]))
plt.show()


# Compute the halfspaces of the polytope
real_vertices = np.array(compute_polytope_vertices(A, b))
print("Real Vertices of the polytope:")
print(real_vertices)

A1 = np.concatenate((A, normal.reshape(1, -1)), axis=0)
b1 = np.concatenate((b, np.array([offset]).reshape(-1, 1)), axis=0)
# Compute the vertices of the polytope
vertices = compute_polytope_vertices(A1, b1)
print("Vertices of the polytope with h1:")
print(vertices)

A2 = np.concatenate((A, -normal.reshape(1, -1)), axis=0)
b2 = np.concatenate((b, -np.array([offset]).reshape(-1, 1)), axis=0)
# Compute the vertices of the polytope
vertices2 = compute_polytope_vertices(A2, b2)
print("Vertices of the polytope with h2:")
print(vertices2)
