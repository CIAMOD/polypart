#!/usr/bin/env python3
"""Plot empirical vs theoretical ratio using one subplot per experiment type.

This script scans experiment JSON files in data/clusterv2, computes:
- empirical ratio: polypart_time / incenu_time
- theoretical ratio:
    (|C| * |V| * (log2(m) + |F|)) / (|C| * (m + |F|)^2)

Since |C| cancels, the x-axis ratio is:
    |V| * (log2(m) + |F|) / (m + |F|)^2

Unlike the single-axes script, this version creates one subplot per experiment
family and colors points by dimension in each subplot.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ExperimentPoint:
    """Single validated experiment sample used for plotting."""

    family: str
    file_path: Path
    dimension: int
    m_hyperplanes: int
    time_ratio: float
    complexity_ratio: float


REQUIRED_KEYS = {
    "polypart_time",
    "incenu_time",
    "n_vertices",
    "n_facets",
    "m_hyperplanes",
}

PREFERRED_FAMILY_ORDER = [
    "braid",
    "moduli_n1",
    "moduli_n1_random",
    "moduli_r2",
    "random",
]


def _to_float(value: object) -> float:
    """Convert a JSON value to float and raise ValueError when impossible."""
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"Unsupported numeric value type: {type(value)}")


def _extract_dimension(payload: dict[str, object], json_path: Path) -> int:
    """Extract dimension from payload or infer it from path as fallback."""
    if "dim" in payload:
        return int(_to_float(payload["dim"]))

    if "dimension" in payload:
        return int(_to_float(payload["dimension"]))

    for part in reversed(json_path.parts):
        match = re.search(r"-d(\d+)$", part)
        if match:
            return int(match.group(1))

    raise KeyError("Missing dimension key (dim/dimension) and cannot infer from path")


def _extract_point(json_path: Path, data_dir: Path) -> ExperimentPoint:
    """Parse and validate one experiment JSON file into an ExperimentPoint."""
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("JSON root is not an object")

    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        raise KeyError(f"Missing required keys: {sorted(missing)}")

    polypart_time = _to_float(payload["polypart_time"])
    incenu_time = _to_float(payload["incenu_time"])
    n_vertices = _to_float(payload["n_vertices"])
    n_facets = _to_float(payload["n_facets"])
    m_hyperplanes = _to_float(payload["m_hyperplanes"])
    dimension = _extract_dimension(payload, json_path)

    if incenu_time <= 0.0:
        raise ValueError("incenu_time must be > 0")
    if polypart_time <= 0.0:
        raise ValueError("polypart_time must be > 0")
    if n_vertices <= 0.0:
        raise ValueError("n_vertices must be > 0")
    if n_facets < 0.0:
        raise ValueError("n_facets must be >= 0")
    if m_hyperplanes <= 0.0:
        raise ValueError("m_hyperplanes must be > 0 for log2(m)")

    # numerator = n_vertices * (math.log2(m_hyperplanes) + n_facets)
    numerator = (
        n_vertices * math.log2(m_hyperplanes) * dimension + n_facets**2 * dimension**2
    )
    denominator = (m_hyperplanes + n_facets) * m_hyperplanes * dimension
    if denominator <= 0.0:
        raise ValueError("Invalid complexity denominator")

    time_ratio = polypart_time / incenu_time
    complexity_ratio = numerator / denominator

    try:
        family = json_path.relative_to(data_dir).parts[0]
    except ValueError:
        family = "unknown"

    # Discard random experiments of dimension 7
    # if family == "random" and dimension == 7:
    #     raise ValueError("Skipping random experiments of dimension 7")

    return ExperimentPoint(
        family=family,
        file_path=json_path,
        dimension=dimension,
        m_hyperplanes=int(m_hyperplanes),
        time_ratio=time_ratio,
        complexity_ratio=complexity_ratio,
    )


def load_points(data_dir: Path) -> tuple[list[ExperimentPoint], list[tuple[Path, str]]]:
    """Load all valid experiment points and collect skipped files with reasons."""
    points: list[ExperimentPoint] = []
    skipped: list[tuple[Path, str]] = []

    for json_path in data_dir.rglob("*.json"):
        if not json_path.name.startswith("experiment_"):
            continue

        try:
            point = _extract_point(json_path, data_dir)
        except (json.JSONDecodeError, KeyError, ValueError) as error:
            skipped.append((json_path, str(error)))
            continue

        points.append(point)

    return points, skipped


def estimate_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Estimate k in y = kx via least squares with zero intercept."""
    denominator = float(np.dot(xs, xs))
    if denominator <= 0.0:
        raise ValueError("Cannot estimate slope because all x values are zero")
    return float(np.dot(xs, ys) / denominator)


def _ordered_families(points: list[ExperimentPoint]) -> list[str]:
    """Return families in preferred order, then append unseen families sorted."""
    families = {point.family for point in points}

    ordered = [family for family in PREFERRED_FAMILY_ORDER if family in families]
    remainder = sorted(family for family in families if family not in set(ordered))
    return ordered + remainder


def build_subplots(
    points: list[ExperimentPoint], output_path: Path, show: bool
) -> dict[str, float]:
    """Create and save per-family subplots colored by dimension."""
    families = _ordered_families(points)
    slopes: dict[str, float] = {}

    n_families = len(families)
    n_cols = 3 if n_families > 3 else n_families
    n_rows = int(math.ceil(n_families / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    all_dimensions = sorted({point.dimension for point in points})
    cmap = plt.get_cmap("viridis", len(all_dimensions))
    dimension_color = {
        dimension: cmap(index) for index, dimension in enumerate(all_dimensions)
    }
    random_m_values = sorted(
        {point.m_hyperplanes for point in points if point.family == "random"}
    )
    random_cmap = plt.get_cmap("plasma", max(1, len(random_m_values)))
    random_m_color = {
        m_value: random_cmap(index) for index, m_value in enumerate(random_m_values)
    }

    for index, family in enumerate(families):
        axis = axes_flat[index]
        family_points = [point for point in points if point.family == family]

        xs = np.array([point.complexity_ratio for point in family_points], dtype=float)
        ys = np.array([point.time_ratio for point in family_points], dtype=float)
        slope = estimate_slope(xs, ys)
        slopes[family] = slope

        if family == "random":
            family_m_values = sorted({point.m_hyperplanes for point in family_points})
            for m_value in family_m_values:
                m_points = [
                    point for point in family_points if point.m_hyperplanes == m_value
                ]
                axis.scatter(
                    [point.complexity_ratio for point in m_points],
                    [point.time_ratio for point in m_points],
                    s=28,
                    alpha=0.78,
                    color=random_m_color[m_value],
                    label=f"m={m_value} (n={len(m_points)})",
                )
        else:
            family_dimensions = sorted({point.dimension for point in family_points})
            for dimension in family_dimensions:
                dim_points = [
                    point for point in family_points if point.dimension == dimension
                ]
                axis.scatter(
                    [point.complexity_ratio for point in dim_points],
                    [point.time_ratio for point in dim_points],
                    s=28,
                    alpha=0.78,
                    color=dimension_color[dimension],
                    label=f"d={dimension} (n={len(dim_points)})",
                )

        x_min = float(np.min(xs))
        x_max = float(np.max(xs))
        if x_min == x_max:
            x_min *= 0.95
            x_max *= 1.05
        x_line = np.linspace(x_min, x_max, 200)
        axis.plot(
            x_line,
            slope * x_line,
            color="black",
            linewidth=1.8,
            label=f"fit: y = {slope:.4g}x",
        )

        axis.set_title(f"{family}")
        axis.set_xlabel("|V| * (log2(m) + |F|) / (m + |F|)^2")
        axis.set_ylabel("polypart_time / incenu_time")
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
        axis.legend(loc="best", fontsize=8)

    for index in range(n_families, len(axes_flat)):
        fig.delaxes(axes_flat[index])

    fig.suptitle(
        "clusterv2: empirical vs theoretical ratio by experiment type (color = dimension)",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)

    if show:
        plt.show()

    plt.close(fig)
    return slopes


def build_random_dimension_subplots(
    points: list[ExperimentPoint], output_path: Path, show: bool
) -> dict[int, float]:
    """Create and save random-only subplots with one panel per dimension."""
    random_points = [point for point in points if point.family == "random"]
    if not random_points:
        return {}

    dimensions = sorted({point.dimension for point in random_points})
    m_values = sorted({point.m_hyperplanes for point in random_points})
    m_cmap = plt.get_cmap("plasma", len(m_values))
    m_color = {m_value: m_cmap(index) for index, m_value in enumerate(m_values)}
    slopes: dict[int, float] = {}

    n_dims = len(dimensions)
    n_cols = 3 if n_dims > 3 else n_dims
    n_rows = int(math.ceil(n_dims / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for index, dimension in enumerate(dimensions):
        axis = axes_flat[index]
        dim_points = [point for point in random_points if point.dimension == dimension]

        xs = np.array([point.complexity_ratio for point in dim_points], dtype=float)
        ys = np.array([point.time_ratio for point in dim_points], dtype=float)
        slope = estimate_slope(xs, ys)
        slopes[dimension] = slope

        dim_m_values = sorted({point.m_hyperplanes for point in dim_points})
        for m_value in dim_m_values:
            m_points = [point for point in dim_points if point.m_hyperplanes == m_value]
            axis.scatter(
                [point.complexity_ratio for point in m_points],
                [point.time_ratio for point in m_points],
                s=28,
                alpha=0.8,
                color=m_color[m_value],
                label=f"m={m_value} (n={len(m_points)})",
            )

        x_min = float(np.min(xs))
        x_max = float(np.max(xs))
        if x_min == x_max:
            x_min *= 0.95
            x_max *= 1.05
        x_line = np.linspace(x_min, x_max, 200)
        axis.plot(
            x_line,
            slope * x_line,
            color="black",
            linewidth=1.8,
            label=f"fit: y = {slope:.4g}x",
        )

        axis.set_title(f"random | d={dimension}")
        axis.set_xlabel("|V| * (log2(m) + |F|) / (m + |F|)^2")
        axis.set_ylabel("polypart_time / incenu_time")
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
        axis.legend(loc="best", fontsize=8)

    for index in range(n_dims, len(axes_flat)):
        fig.delaxes(axes_flat[index])

    fig.suptitle(
        "clusterv2 random family: empirical vs theoretical ratio (one subplot per d)",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)

    if show:
        plt.show()

    plt.close(fig)
    return slopes


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot empirical/runtime ratio vs theoretical ratio using one subplot "
            "per clusterv2 experiment family and color by dimension."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "clusterv2",
        help="Directory containing clusterv2 experiment folders (default: data/clusterv2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("images") / "clusterv2_time_vs_complexity_ratio_subplots.png",
        help="Output image path (default: images/clusterv2_time_vs_complexity_ratio_subplots.png).",
    )
    parser.add_argument(
        "--random-output",
        type=Path,
        default=Path("images") / "clusterv2_random_time_vs_complexity_ratio_by_d.png",
        help=(
            "Output image path for random-only subplots "
            "(default: images/clusterv2_random_time_vs_complexity_ratio_by_d.png)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    parser.add_argument(
        "--max-skipped-report",
        type=int,
        default=10,
        help="Maximum number of skipped-file reasons to print.",
    )
    parser.add_argument(
        "--skip-m-leq",
        type=int,
        default=10,
        help=(
            "Skip points with m_hyperplanes <= this value before plotting "
            "(default: 10)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run data loading and subplot generation."""
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    points, skipped = load_points(args.data_dir)
    if not points:
        raise RuntimeError(
            "No valid experiment JSON files were found. Ensure files are named "
            "experiment_*.json and include required keys."
        )

    original_count = len(points)
    points = [point for point in points if point.m_hyperplanes > args.skip_m_leq]
    filtered_count = original_count - len(points)
    if not points:
        raise RuntimeError(
            "No points remain after m-threshold filtering. "
            f"Current threshold is m <= {args.skip_m_leq}."
        )

    slopes = build_subplots(points, args.output, args.show)
    random_slopes = build_random_dimension_subplots(
        points, args.random_output, args.show
    )
    families = _ordered_families(points)

    print(f"Loaded {original_count} valid experiment files from {args.data_dir}")
    print(
        f"Filtered out {filtered_count} points with m <= {args.skip_m_leq}; "
        f"{len(points)} points remain."
    )
    print(f"Families ({len(families)}): {', '.join(families)}")
    print("Estimated proportionality constants (fit y = kx):")
    for family in families:
        print(f"  - {family}: k = {slopes[family]:.6g}")
    print(f"Saved subplot figure to: {args.output}")
    if random_slopes:
        print("Random-only constants by dimension (fit y = kx):")
        for dimension in sorted(random_slopes):
            print(f"  - d={dimension}: k = {random_slopes[dimension]:.6g}")
        print(f"Saved random-only subplot figure to: {args.random_output}")
    else:
        print("No random-family points found; random-only figure was not generated.")

    if skipped:
        print(f"Skipped {len(skipped)} files due to missing/invalid data.")
        for skipped_path, reason in skipped[: args.max_skipped_report]:
            print(f"  - {skipped_path}: {reason}")


if __name__ == "__main__":
    main()
