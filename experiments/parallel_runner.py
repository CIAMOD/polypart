import os

# --- RIGOROUS SETUP: FORCE SINGLE THREADING ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ----------------------------------------------

import argparse
import importlib.util
import multiprocessing
import multiprocessing.pool
import sys
import traceback
from multiprocessing import cpu_count
from pathlib import Path

from tqdm import tqdm


# --- CUSTOM CLASS TO ALLOW NESTED POOLS ---
class NoDaemonProcess(multiprocessing.Process):
    """A process that is not daemonic, allowing it to spawn children."""

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class NestedPool(multiprocessing.pool.Pool):
    """A Pool that creates non-daemonic processes."""

    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(NestedPool, self).__init__(*args, **kwargs)


# ------------------------------------------


def experiment_batch_worker(task_data):
    """
    Worker function to execute a single EXPERIMENT run.

    This function will:
    1. Receive an Experiment object.
    2. Call 'run_parallel_isolated'.
    3. That method will spawn 3 internal processes (PolyPart, IncEnu, DelRes).
    4. Data is aggregated and saved by the Experiment class itself.
    """
    exp, run_idx, seed, decimals = task_data

    try:
        # Call the new method we added to experiments.py
        # We don't need the return value here, just the success status
        results = exp.run_parallel_isolated(decimals=decimals, seed=seed)
        exp.save(results, folder="../data/cluster")
        return {"status": "success", "name": exp.dirname(), "run": run_idx}

    except Exception as e:
        # Capture error without crashing the whole batch
        return {
            "status": "error",
            "name": exp.dirname(),
            "run": run_idx,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def run_parallel_batch(experiments, n_runs=10, max_workers=None):
    """
    Orchestrates the parallel execution.
    """
    total_cores = cpu_count()

    # --- WORKER CALCULATION LOGIC ---
    # Each experiment now acts as a "Parent" that spawns 3 "Children".
    # To avoid oversubscribing the CPU, we divide available cores by 3.
    # Example: 112 Cores -> ~36 Concurrent Experiments -> 108 Active Processes
    if max_workers is None:
        # Leave 2 cores for OS/overhead, divide rest by 3
        max_workers = max(1, (total_cores - 2) // 3)

    print("--- Starting Rigorous Parallel Batch ---")
    print(f"Cluster Cores: {total_cores}")
    print(f"Concurrent Experiments (Workers): {max_workers}")
    print(f"Est. Total Active Processes: {max_workers * 3}")

    # 1. Flatten the workload
    tasks = []
    for exp in experiments:
        for run in range(n_runs):
            # Task = (Experiment, run_id, seed, decimals)
            task_args = (
                exp,
                run,
                42 + run,  # Deterministic seed
                3,  # decimals
            )
            tasks.append(task_args)

    print(f"Total experiments to run: {len(tasks)}")

    # 2. Submit tasks using multiprocessing.Pool
    results_summary = {"success": 0, "error": 0}

    # maxtasksperchild=1 is good practice here to keep the parent runner clean,
    # even though the heavy lifting happens in the sub-sub-processes.
    with NestedPool(processes=max_workers, maxtasksperchild=1) as pool:
        # imap_unordered allows us to update the progress bar as soon as ANY job finishes
        for res in tqdm(
            pool.imap_unordered(experiment_batch_worker, tasks), total=len(tasks)
        ):
            if res["status"] == "success":
                results_summary["success"] += 1
            else:
                results_summary["error"] += 1
                # Log errors immediately
                with open("batch_errors.log", "a") as f:
                    f.write(f"FAILED: {res['name']} | Run {res['run']}\n")
                    f.write(f"Error: {res['error']}\n")
                    f.write(res["traceback"])
                    f.write("\n" + "=" * 30 + "\n")

    print(
        f"\nBatch Complete. Success: {results_summary['success']}, Errors: {results_summary['error']}"
    )
    if results_summary["error"] > 0:
        print("Check 'batch_errors.log' for details.")


def load_experiments_from_file(filepath):
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

    args = parser.parse_args()

    # Load and Run
    try:
        exps = load_experiments_from_file(args.config)
        run_parallel_batch(exps, n_runs=args.runs, max_workers=args.workers)
    except Exception as e:
        print(f"Fatal Error: {e}")
        traceback.print_exc()
