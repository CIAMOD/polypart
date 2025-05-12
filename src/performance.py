from tree import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from time import perf_counter


def random_polytope(n: int, d: int, bounds: tuple = (0, 1)) -> tuple:
    """
    Generate a random polytope in d dimensions.

    Parameters
    ----------
    n : int
        The number of points in point cloud to generate the polytope.
    d : int
        The number of dimensions.
    bounds : tuple, optional
        The bounds for the random points, by default (0, 1)

    Returns
    -------
    tuple
        A tuple containing the vertices of the polytope and the inequalities, and the inequalities.
    """

    # Generate random points in d dimensions
    points = np.random.uniform(bounds[0], bounds[1], (n, d))

    # Compute the convex hull of the points
    hull = ConvexHull(points)

    # Return the vertices of the convex hull
    vertices = points[hull.vertices]
    inequalities = hull.equations
    A = inequalities[:, :-1]
    b = -inequalities[:, -1]
    return vertices, (A, b)


def generate_random_planes(n: int, d: int, vertices: np.ndarray) -> list:
    """
    Generate n random hyperplanes in d dimensions, intersecting at the vertices of the polytope.
    Parameters
    ----------
    n : int
        The number of hyperplanes to generate.
    d : int
        The number of dimensions.
    vertices : np.ndarray
        The vertices of the polytope.
    Returns
    -------
    list

    """
    hyperplanes = []
    for _ in range(n):
        # Random coefficients
        coeffs = np.random.uniform(-10, 10, d)
        # Random bias in between min and max of vertices
        bias = np.random.uniform(np.min(vertices @ coeffs), np.max(vertices @ coeffs))
        # Create hyperplane
        hyperplanes.append((coeffs, bias))

    return hyperplanes


def plot_sliced_polytope_2d(vertices, hyperplanes):
    """
    Plot the polytope and hyperplanes in 2D.
    Parameters
    ----------
    vertices : np.ndarray
        The vertices of the polytope.
    inequalities : tuple
        The inequalities of the polytope.
    hyperplanes : list
        The hyperplanes to plot.
    """
    _, ax = plt.subplots()
    ax.set_aspect("equal")

    # Plot the polytope
    poly = Polygon(vertices, alpha=0.5, color="blue")
    ax.add_patch(poly)

    # Plot the hyperplanes
    for coeffs, bias in hyperplanes:
        x = np.linspace(-1, 1, 100)
        y = (bias - coeffs[0] * x) / coeffs[1]
        ax.plot(x, y, color="red", alpha=0.5)

    minx, maxx = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    miny, maxy = np.min(vertices[:, 1]), np.max(vertices[:, 1])

    # 5% padding
    xpad = 0.05 * (maxx - minx)
    ypad = 0.05 * (maxy - miny)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)
    plt.show()


def compute_polytope_chambers(
    d: int, inequalities: np.ndarray, hyperplanes: list, verbose: bool = False
) -> list:
    """
    Compute the number of chambers in the polytope using a trivial algorithm.
    """

    start = perf_counter()
    if verbose:
        print(f"Computing chambers for d={d} and {len(hyperplanes)} hyperplanes...")
    simplex = Polytope(*inequalities)
    simplex.extreme()

    if verbose:
        print(f"Computing chambers...")
    chambers = [simplex]
    for hyperplane in hyperplanes:
        new_chambers = []
        for chamber in chambers:
            if not hyperplane_intersects_polytope(chamber, hyperplane):
                new_chambers.append(chamber)
                continue
            poly1, poly2 = cut_polytope_by_hyperplane_fast(chamber, hyperplane)

            new_chambers.append(poly1)
            new_chambers.append(poly2)

        chambers = new_chambers

    if verbose:
        print(f"Found {len(chambers)} chambers in {perf_counter() - start:.2f} s")
    return chambers


# vertices, inequalities = random_polytope(10, 2)
# hyperplanes = generate_random_planes(5, 2, vertices)

# # Plot the polytope and hyperplanes in 2D

# # Compute the chambers of the polytope
# chambers = compute_polytope_chambers(2, inequalities, hyperplanes)
# print(f"Number of chambers: {len(chambers)}")

# plot_sliced_polytope_2d(vertices, hyperplanes)


def performance_test(
    n: int, d: int, min_hyperplanes: int, max_hyperplanes: int, increment: int = 1
):
    """
    Run a performance test on the polytope slicing algorithm.
    """
    # Generate random polytope
    vertices, inequalities = random_polytope(n, d)
    hyperplanes = generate_random_planes(5, d, vertices)

    times = []
    n_chambers = []
    for i in range(min_hyperplanes, max_hyperplanes + 1, increment):
        hyperplanes = generate_random_planes(i, d, vertices)
        start = perf_counter()
        chambers = compute_polytope_chambers(d, inequalities, hyperplanes)
        times.append(perf_counter() - start)
        n_chambers.append(len(chambers))

    # Plot the results
    plt.plot(range(min_hyperplanes, max_hyperplanes + 1, increment), times)
    plt.xlabel("Number of hyperplanes")
    plt.ylabel("Time (s)")
    plt.title(f"Performance test for n={n}, d={d}")
    plt.show()

    plt.plot(
        range(min_hyperplanes, max_hyperplanes + 1, increment),
        n_chambers,
    )
    plt.xlabel("Number of hyperplanes")
    plt.ylabel("Number of chambers")
    plt.title(f"Number of chambers for n={n}, d={d}")
    plt.show()


n, d = 10, 2
performance_test(n, d, 1, 200, 10)
