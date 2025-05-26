# Polytope & Hyperplane Plotter
# --------------------------------------
# This script visualises a 3‑D polytope (given as six vertices)
# together with a supporting hyperplane.
#
# • Blue points  – the polytope’s vertices
# • Cyan faces   – the convex hull (if SciPy is available)
# • Red surface  – the hyperplane n·x = b
#
# Requirements: numpy, matplotlib (+‑‑optional‑‑> SciPy for the hull)
# Run:  python polytope_hyperplane_plot.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (keeps 3‑D active)

# -------------------------------------------------
# Data
# -------------------------------------------------
points = np.array(
    [
        [0.23481936, 0.39538012, 0.68206639],
        [0.26038136, 0.40860470, 0.66073333],
        [0.25920424, 0.41091447, 0.65988989],
        [0.22993435, 0.40457437, 0.67881085],
        [0.26047210, 0.41092590, 0.66262064],
        [0.23720119, 0.40515930, 0.68745593],
    ]
)

normal = np.array([4.70142088, 6.06961861, -4.35930855])
offset = 0.7133942451297228  # Hyperplane equation: normal·x = offset

# -------------------------------------------------
# Helper: safe convex hull (optional SciPy)
# -------------------------------------------------
hull = None
try:
    from scipy.spatial import ConvexHull  # type: ignore

    if len(points) >= 4:
        hull = ConvexHull(points)
except ModuleNotFoundError:
    print("SciPy not found – convex hull faces will be skipped.")

# -------------------------------------------------
# Plotting
# -------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Scatter vertices
ax.scatter(
    points[:, 0], points[:, 1], points[:, 2], s=40, label="Vertices", depthshade=True
)

# Optional: convex‑hull faces for nicer polytope look
if hull is not None:
    for simplex in hull.simplices:
        tri = points[simplex]
        ax.plot_trisurf(
            tri[:, 0],
            tri[:, 1],
            tri[:, 2],
            alpha=0.25,
            edgecolor="k",
            linewidth=0.3,
            antialiased=True,
            shade=False,
            color="cyan",
        )

# Create mesh for the hyperplane
# Determine grid extent slightly beyond the polytope bounding box
pad = 0.05
mins = points.min(axis=0) - pad
maxs = points.max(axis=0) + pad

# Choose two axes where the normal is not (almost) zero so we can solve for the third
nz = np.abs(normal) > 1e-8
if not nz.any():
    raise ValueError("Normal vector is zero – cannot define a plane.")

# Solve for Z by default (if normal[2] ≠ 0) else Y, else X
if abs(normal[2]) > 1e-8:
    # Plane as z = (b - a*x - c*y)/d
    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], 20),
        np.linspace(mins[1], maxs[1], 20),
    )
    zz = (offset - normal[0] * xx - normal[1] * yy) / normal[2]
elif abs(normal[1]) > 1e-8:
    # y = (b - a*x - d*z)/c
    xx, zz = np.meshgrid(
        np.linspace(mins[0], maxs[0], 20),
        np.linspace(mins[2], maxs[2], 20),
    )
    yy = (offset - normal[0] * xx - normal[2] * zz) / normal[1]
else:
    # x = (b - c*y - d*z)/a
    yy, zz = np.meshgrid(
        np.linspace(mins[1], maxs[1], 20),
        np.linspace(mins[2], maxs[2], 20),
    )
    xx = (offset - normal[1] * yy - normal[2] * zz) / normal[0]

ax.plot_surface(xx, yy, zz, alpha=0.5, color="red", label="Hyperplane")

# Axes labels & cosmetics
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Polytope & Supporting Hyperplane")
# ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
