import argparse
import importlib.util
import sys
from pathlib import Path

from .runner import run_experiments


def load_experiment_list_from_py(filepath):
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location("polypart_experiments_config", path)
    module = importlib.util.module_from_spec(spec)

    # Ensure the config file can import its own helpers
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)

    if not hasattr(module, "experiments"):
        raise ValueError("Config file must define a variable 'experiments' list")
    return module.experiments


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiments from config."
    )
    parser.add_argument("config", type=str, help="Path to Python config file")
    parser.add_argument(
        "-r", "--runs", type=int, default=10, help="Runs per experiment"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Concurrent worker processes (defaults to CPUs-2)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./data/cluster",
        help="Output folder (default: ./data/cluster)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Algorithms to exclude (e.g., 'ppart', 'incenu', 'delres').",
    )

    args = parser.parse_args()

    experiments = load_experiment_list_from_py(args.config)
    run_experiments(
        experiments,
        n_runs=args.runs,
        max_workers=args.workers,
        folder=args.output,
        exclude_algorithms=args.exclude,
    )


if __name__ == "__main__":
    main()
