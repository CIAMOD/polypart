import argparse

from .core import ArrangementClass, Experiment, PolytopeClass
from .runner import run_experiments


def main():
    parser = argparse.ArgumentParser(description="Run a single polytope experiment.")
    parser.add_argument(
        "-d", type=int, required=True, help="Dimension of the polytope."
    )
    parser.add_argument(
        "-p",
        type=str,
        required=True,
        help="Polytope class (e.g., 'cube', 'moduli_n1', 'moduli_r2', 'random').",
    )
    # --bb flag for bbox mode
    parser.add_argument(
        "--bb", action="store_true", help="Use bounding box mode for polytopes."
    )
    parser.add_argument(
        "-a",
        type=str,
        required=True,
        help="Arrangement class (e.g., 'braid', 'moduli_n1', 'moduli_r2', 'random').",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=None,
        help="Number of hyperplanes (required for random arrangement).",
    )
    parser.add_argument(
        "--degen_ratio",
        type=float,
        default=0.0,
        help="Degeneracy ratio (required for random arrangement).",
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="Number of runs to perform."
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Algorithms to exclude (e.g., 'ppart', 'incenu', 'delres').",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./data",
        help="Output folder for results (default: ./data)",
    )

    args = parser.parse_args()

    P_class = PolytopeClass(args.p)
    A_class = ArrangementClass(args.a, m=args.m, degen_ratio=args.degen_ratio)
    experiment = Experiment(P_class, A_class, d=args.d)

    # Reuse batch runner even for single experiment for consistency
    run_experiments(
        experiment,
        n_runs=args.n_runs,
        max_workers=3,
        folder=args.output,
        exclude_algorithms=args.exclude,
    )


if __name__ == "__main__":
    main()
