import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

from polypart.experiments.parallel import (
    run_parallel_batch,
)


def load_experiment_list_from_py(filepath):
    """Dynamically loads the 'experiments' list from a python file."""
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location("dynamic_config", path)
    module = importlib.util.module_from_spec(spec)

    # Add the config file's directory to sys.path so it can handle its own imports
    sys.path.insert(0, str(path.parent))

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error loading config file: {e}")
        raise

    if not hasattr(module, "experiments"):
        raise ValueError(
            f"File {filepath} must define a list variable named 'experiments'"
        )

    return module.experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PolyPart experiments in parallel."
    )

    # Argument 1: The config file path
    parser.add_argument(
        "config",
        type=str,
        help="Path to the python config file defining 'experiments = [...]'",
    )

    # Argument 2: Number of runs per experiment
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=10,
        help="Number of times to run each experiment (default: 10)",
    )

    # Argument 3: Max Workers (Concurrent Experiments)
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent experiments (default: (CPUs-2)//3)",
    )

    # Argument 4: Output Folder
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="../data/cluster",
        help="Folder to save experiment results (default: ../data/cluster)",
    )

    args = parser.parse_args()

    # Load and Run
    try:
        exps = load_experiment_list_from_py(args.config)
        run_parallel_batch(
            exps, n_runs=args.runs, max_workers=args.workers, folder=args.output
        )
    except Exception as e:
        print(f"Fatal Error: {e}")
        traceback.print_exc()
