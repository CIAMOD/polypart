"""
Compute moduli spaces partition trees and isomorphism classes.

This script computes partition trees for various moduli space parameters (n, r),
then computes isomorphism graphs under two different symmetry groups:
1. Symmetric curves: using generate_d_invariant_transformations (includes pullback)
2. Asymmetric curves: using generate_d_invariant_transformations_no_pullback (no pullback)
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from polypart import (
    PartitionGraph,
    StorageLevel,
    SymmetryGroup,
    build_partition_tree,
    save_tree,
)
from polypart.apps.moduli import (
    basic_transformation,
    generate_d_invariant_transformations,
    generate_d_invariant_transformations_no_pullback,
)
from polypart.core.typing import FractionVector, as_fraction_vector
from polypart.generators import get_moduli_arrangement, get_product_of_simplices

# Output directory for all results
OUTPUT_DIR = "data/moduli"

# Define which (n, r) pairs to compute
CASES = [
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    # (1, 8),
    # (1, 9),
    (2, 2),
    (2, 3),
    (2, 4),
    # (2, 5),
    (3, 2),
    (3, 3),
    (4, 2),
    # (4, 3),
    (5, 2),
    (6, 2),
    # (7, 2),
]


def _apply_wrapper(
    vector: FractionVector,
    sigma: tuple[int, ...],
    s: int,
    H: tuple[int, ...],
    n: int,
    r: int,
    d: int,
) -> FractionVector:
    """Reshape flat seed vector, apply transformation, flatten back."""
    zeros = as_fraction_vector(np.zeros(n)).reshape(n, 1)
    alphas = np.hstack((zeros, vector.reshape(n, r - 1))).reshape(1, n, r)
    d_arr = np.array([d])

    new_alphas, _ = basic_transformation(alphas, d_arr, sigma, s, H)

    return new_alphas[0, :, 1:].flatten()


def get_moduli_symmetries(
    n: int, r: int, d: int, no_pullback: bool = False
) -> SymmetryGroup:
    """Create SymmetryGroup from moduli space transformations.

    Args:
        n: Number of parabolic points.
        r: Rank of vector bundles.
        d: Degree of vector bundles.
        no_pullback: If True, use transformations without pullback (asymmetric curves).

    Returns:
        SymmetryGroup with the appropriate transformations.
    """
    transforms = {}
    if no_pullback:
        generator = generate_d_invariant_transformations_no_pullback(n, r, d)
    else:
        generator = generate_d_invariant_transformations(n, r, d)

    for i, (sigma, s, H) in enumerate(generator):
        name = f"T_{i}_sig{sigma}_s{s}_H{H}"
        func = partial(_apply_wrapper, sigma=sigma, s=s, H=H, n=n, r=r, d=d)
        transforms[name] = func

    return SymmetryGroup(transforms)


def compute_moduli_case(
    n: int, r: int, d: int, output_dir: Path, verbose: bool = True
) -> dict:
    """Compute partition tree and isomorphism classes for a single (n, r) pair.

    Args:
        n: Number of parabolic points.
        r: Rank of vector bundles.
        d: Degree of vector bundles.
        output_dir: Directory to save results.
        verbose: Whether to print progress messages.

    Returns:
        Dictionary with results and metrics.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Computing moduli space for n={n}, r={r}, d={d}")
        print(f"{'=' * 60}")

    # Create output directory
    case_dir = output_dir / f"moduli_n{n}_r{r}"
    case_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "n": n,
        "r": r,
        "d": d,
        "dimension": n * (r - 1),
    }

    # Step 1: Build polytope and arrangement
    if verbose:
        print("Building polytope and arrangement...")
    t0 = perf_counter()
    simplex = get_product_of_simplices(n, r - 1)
    simplex.extreme()
    hyperplanes = get_moduli_arrangement(n, r, d=d)
    t_setup = perf_counter() - t0

    results["n_hyperplanes"] = len(hyperplanes)
    results["time_setup"] = t_setup

    if verbose:
        print(f"  Dimension: {n * (r - 1)}")
        print(f"  Hyperplanes: {len(hyperplanes)}")
        print(f"  Setup time: {t_setup:.3f}s")

    # Step 2: Build partition tree
    if verbose:
        print("Building partition tree...")
    t0 = perf_counter()
    tree, n_chambers = build_partition_tree(
        hyperplanes, simplex, strategy="v-entropy", remove_redundancies=True
    )
    t_tree = perf_counter() - t0

    results["n_chambers"] = n_chambers
    results["time_tree"] = t_tree

    if verbose:
        print(f"  Chambers: {n_chambers}")
        print(f"  Time: {t_tree:.3f}s")

    # Step 3: Save tree
    tree_path = case_dir / "tree.json"
    if verbose:
        print(f"Saving tree to {tree_path.name}...")
    save_tree(tree, str(tree_path))

    # Step 4: Compute isomorphism classes for symmetric curves
    if verbose:
        print("Computing isomorphism classes (symmetric curves)...")
    t0 = perf_counter()
    symmetries_sym = get_moduli_symmetries(n, r, d, no_pullback=False)
    results["n_symmetries_symmetric"] = len(symmetries_sym)

    graph_sym = PartitionGraph(tree, symmetries_sym)
    n_classes_sym = graph_sym.reduce(storage=StorageLevel.STABILIZERS, verbose=False)
    t_iso_sym = perf_counter() - t0

    results["n_classes_symmetric"] = n_classes_sym
    results["time_isomorphism_symmetric"] = t_iso_sym

    if verbose:
        print(f"  Symmetries: {len(symmetries_sym)}")
        print(f"  Classes: {n_classes_sym}")
        print(f"  Time: {t_iso_sym:.3f}s")

    # Save graph for symmetric case
    graph_sym_path = case_dir / "graph_symmetric.json"
    if verbose:
        print(f"Saving symmetric graph to {graph_sym_path.name}...")
    save_partition_graph(graph_sym, graph_sym_path)

    # Step 5: Compute isomorphism classes for asymmetric curves
    if verbose:
        print("Computing isomorphism classes (asymmetric curves)...")
    t0 = perf_counter()
    symmetries_asym = get_moduli_symmetries(n, r, d, no_pullback=True)
    results["n_symmetries_asymmetric"] = len(symmetries_asym)

    graph_asym = PartitionGraph(tree, symmetries_asym)
    n_classes_asym = graph_asym.reduce(storage=StorageLevel.STABILIZERS, verbose=False)
    t_iso_asym = perf_counter() - t0

    results["n_classes_asymmetric"] = n_classes_asym
    results["time_isomorphism_asymmetric"] = t_iso_asym

    if verbose:
        print(f"  Symmetries: {len(symmetries_asym)}")
        print(f"  Classes: {n_classes_asym}")
        print(f"  Time: {t_iso_asym:.3f}s")

    # Save graph for asymmetric case
    graph_asym_path = case_dir / "graph_asymmetric.json"
    if verbose:
        print(f"Saving asymmetric graph to {graph_asym_path.name}...")
    save_partition_graph(graph_asym, graph_asym_path)

    # Step 6: Save metrics
    results["time_total"] = t_setup + t_tree + t_iso_sym + t_iso_asym

    metrics_path = case_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"Metrics saved to {metrics_path.name}")
        print(f"Total time: {results['time_total']:.3f}s")

    return results


def save_partition_graph(graph: PartitionGraph, path: Path) -> None:
    """Save partition graph to JSON file.

    Args:
        graph: PartitionGraph to save.
        path: Path to save to.
    """
    data = {
        "n_classes": len(graph.classes),
        "classes": [
            {
                "id": eq_class.id,
                "size": eq_class.size,
                "stabilizers": eq_class.stabilizers
                if hasattr(eq_class, "stabilizers")
                else [],
            }
            for eq_class in graph.classes
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_markdown_report(all_results: list[dict], output_path: Path) -> None:
    """Generate Markdown report with summary tables.

    Args:
        all_results: List of result dictionaries from compute_moduli_case.
        output_path: Path to save the report.
    """
    # Organize results by n and r
    data = {}
    for result in all_results:
        n, r = result["n"], result["r"]
        if n not in data:
            data[n] = {}
        data[n][r] = result

    # Get all n and r values
    n_values = sorted(data.keys())
    r_values = sorted(set(r for n_data in data.values() for r in n_data.keys()))

    with open(output_path, "w") as f:
        f.write("# Moduli Spaces Computation Report\n\n")
        f.write("This report summarizes the computation of partition trees and ")
        f.write(
            "isomorphism classes for moduli spaces of parabolic vector bundles.\n\n"
        )

        # Table 1: Number of chambers
        f.write("## Number of Chambers\n\n")
        f.write("<table>\n")
        f.write("<tr>\n<th></th>\n")
        for r in r_values:
            f.write(f"<th>r={r}</th>\n")
        f.write("</tr>\n")

        for n in n_values:
            f.write(f"<tr>\n<th>n={n}</th>\n")
            for r in r_values:
                if r in data[n]:
                    f.write(f"<td>{data[n][r]['n_chambers']}</td>\n")
                else:
                    f.write("<td>-</td>\n")
            f.write("</tr>\n")

        f.write("</table>\n\n")

        # Table 2: Isomorphism classes (symmetric curves)
        f.write("## Isomorphism Classes (Symmetric Curves)\n\n")
        f.write("Using `generate_d_invariant_transformations` (includes pullback)\n\n")
        f.write("<table>\n")
        f.write("<tr>\n<th></th>\n")
        for r in r_values:
            f.write(f"<th>r={r}</th>\n")
        f.write("</tr>\n")

        for n in n_values:
            f.write(f"<tr>\n<th>n={n}</th>\n")
            for r in r_values:
                if r in data[n]:
                    f.write(f"<td>{data[n][r]['n_classes_symmetric']}</td>\n")
                else:
                    f.write("<td>-</td>\n")
            f.write("</tr>\n")

        f.write("</table>\n\n")

        # Table 3: Isomorphism classes (asymmetric curves)
        f.write("## Isomorphism Classes (Asymmetric Curves)\n\n")
        f.write(
            "Using `generate_d_invariant_transformations_no_pullback` (no pullback)\n\n"
        )
        f.write("<table>\n")
        f.write("<tr>\n<th></th>\n")
        for r in r_values:
            f.write(f"<th>r={r}</th>\n")
        f.write("</tr>\n")

        for n in n_values:
            f.write(f"<tr>\n<th>n={n}</th>\n")
            for r in r_values:
                if r in data[n]:
                    f.write(f"<td>{data[n][r]['n_classes_asymmetric']}</td>\n")
                else:
                    f.write("<td>-</td>\n")
            f.write("</tr>\n")

        f.write("</table>\n\n")

        # Additional details section
        f.write("## Computation Details\n\n")
        for n in n_values:
            f.write(f"### n = {n}\n\n")
            f.write(
                "| r | dim | hyperplanes | chambers | sym_classes | asym_classes | time(s) |\n"
            )
            f.write(
                "|---|-----|-------------|----------|-------------|--------------|--------|\n"
            )

            for r in r_values:
                if r in data[n]:
                    res = data[n][r]
                    f.write(f"| {r} | {res['dimension']} | {res['n_hyperplanes']} | ")
                    f.write(f"{res['n_chambers']} | {res['n_classes_symmetric']} | ")
                    f.write(
                        f"{res['n_classes_asymmetric']} | {res['time_total']:.2f} |\n"
                    )

            f.write("\n")


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """Compute predefined set of moduli cases."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing {len(CASES)} moduli cases")
    print(f"Output directory: {output_dir}")

    all_results = []
    for i, (n, r) in enumerate(CASES, 1):
        print(f"[{i}/{len(CASES)}] Processing n={n}, r={r}...")
        try:
            result = compute_moduli_case(n, r, 0, output_dir, verbose=True)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: Failed to compute n={n}, r={r}: {e}")

    # Generate final report
    if all_results:
        report_path = output_dir / "report.md"
        print(f"\nGenerating final report at {report_path}...")
        generate_markdown_report(all_results, report_path)
        print(f"\nComputation complete! Report saved to {report_path}")
    else:
        print("No successful computations to report.")


if __name__ == "__main__":
    main()
