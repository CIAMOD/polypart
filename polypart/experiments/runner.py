import multiprocessing
import multiprocessing.pool
import os
import traceback
from multiprocessing import cpu_count
from time import perf_counter

from tqdm import tqdm

from polypart.delres import number_of_regions
from polypart.inc_enu import build_tree_inc_enu
from polypart.ppart import build_partition_tree

# --- SUPPORTED ALGORITHMS ---
ALGORITHMS: tuple[str, ...] = ("ppart", "incenu", "delres")


# --- CROSS-PLATFORM MEMORY IMPORT ---
try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False
    try:
        import psutil

        HAS_PSUTIL = True
    except ImportError:
        HAS_PSUTIL = False
# ------------------------------------


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


def _run_single_algorithm_isolated(args):
    """
    Regenerates the environment and runs ONE algorithm.
    Returns timings, memory, result count, and geometry stats.
    """
    # Unpack arguments
    P, A, bbox_mode, d, decimals, seed, algo_name = args

    # --- 2. Run Specific Algorithm ---
    start_time = perf_counter()
    num_regions = None
    if algo_name == "ppart":
        T, num_regions = build_partition_tree(P, A, "v-entropy")
    elif algo_name == "incenu":
        _, num_regions = build_tree_inc_enu(A, [] if bbox_mode else P)
    elif algo_name == "delres":
        num_regions = number_of_regions(A, [] if bbox_mode else P)
    end_time = perf_counter()

    # --- Collect Stats of polypart if applicable ---
    stats = T.stats() if algo_name == "ppart" else None

    # --- 3. Measure Peak RAM ---
    if HAS_RESOURCE:
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    elif HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / (1024.0 * 1024.0)
    else:
        mem_usage = -1.0  # Unknown

    return {
        "algo": algo_name,
        "time": round(end_time - start_time, 6),
        "peak_ram_mb": round(mem_usage, 2),
        "num_regions": num_regions,
        "stats": stats,
    }


def run_single_experiment(
    P_class,
    A_class,
    d: int | None = None,
    decimals: int = 3,
    seed: int | None = None,
    exclude_algorithms: list[str] | None = None,
    n_processes: int | None = None,
    log_folder: str | None = None,
) -> dict:
    """
    Runs a single experiment in isolated processes for each algorithm.
    Returns a dictionary with results from each algorithm.
    """
    if exclude_algorithms is None:
        exclude_algorithms = []
    algorithms = set(ALGORITHMS) - set(exclude_algorithms)
    assert len(algorithms) > 0, "No algorithms left to run."

    if n_processes is None:
        n_processes = len(algorithms)

    assert 1 <= n_processes <= len(algorithms), (
        f"n_processes must be between 1 and {len(algorithms)}"
    )
    # --- 0. Ensure log folder exists ---
    if log_folder is not None:
        os.makedirs(log_folder, exist_ok=True)

    # --- 1. Regenerate Geometry (Measure Overhead) ---
    t0 = perf_counter()
    P = P_class.get_polytope(d, decimals=decimals, seed=seed)
    P.remove_redundancies()
    P.extreme()
    t1 = perf_counter()
    p_creation_time = t1 - t0

    t0 = perf_counter()
    A = A_class.get_arrangement(d, P, decimals=decimals, seed=seed)
    t1 = perf_counter()
    a_creation_time = t1 - t0

    # --- 2. Run Algorithms in Isolated Processes ---
    tasks = []
    for algo in algorithms:
        tasks.append((P, A, P_class.bbox_mode, d, decimals, seed, algo))

    # maxtasksperchild=1 ensures each runs in a fresh process (clean RAM)
    with multiprocessing.pool.Pool(processes=n_processes, maxtasksperchild=1) as pool:
        # results_list = pool.map(_run_single_algorithm_isolated, tasks)
        # Loop and log single results as they complete
        results_list = []
        for res in pool.imap_unordered(_run_single_algorithm_isolated, tasks):
            results_list.append(res)
            if log_folder is not None:
                with open(os.path.join(log_folder, "runtime.log"), "a") as f:
                    # Save res without stats to reduce size
                    log_res = res.copy()
                    del log_res["stats"]
                    f.write(f"{log_res}\n")

    # --- 3. Result Consistency Check ---
    # Extract region counts
    region_counts = [res["num_regions"] for res in results_list]

    # Check for Null results (Algorithm Failure)
    if any(not isinstance(r, int) for r in region_counts):
        failed_map = {res["algo"]: res["num_regions"] for res in results_list}
        raise RuntimeError(
            f"One or more algorithms failed to return a region count: {failed_map}"
        )

    # Check for Mismatches (Correctness Failure)
    first_count = region_counts[0]
    assert all(r == first_count for r in region_counts), (
        f"Region count mismatch between algorithms: {region_counts}"
    )

    # --- 4. Construct Final Result ---
    final_result = {
        # Standard metrics
        "polypart_time": None,
        "incenu_time": None,
        "delres_time": None,
        "num_regions": first_count,  # Validated above
        "dim": d,
        # Geometry Stats
        "n_vertices": P.n_vertices,
        "n_facets": P.n_inequalities,
        "m_hyperplanes": len(A),
        # Peak RAM Stats
        "polypart_peak_ram_mb": None,
        "incenu_peak_ram_mb": None,
        "delres_peak_ram_mb": None,
        # Additional Profiling
        "p_creation_time": round(p_creation_time, 6),
        "a_creation_time": round(a_creation_time, 6),
        "ppart_stats": None,
        "decimals": decimals,
        "seed": seed,
    }

    # Populate Algorithm-Specific Data
    for res in results_list:
        algo = res["algo"]
        if algo == "ppart":
            final_result["polypart_time"] = res["time"]
            final_result["polypart_peak_ram_mb"] = res["peak_ram_mb"]
            final_result["ppart_stats"] = res["stats"]
        elif algo == "incenu":
            final_result["incenu_time"] = res["time"]
            final_result["incenu_peak_ram_mb"] = res["peak_ram_mb"]
        elif algo == "delres":
            final_result["delres_time"] = res["time"]
            final_result["delres_peak_ram_mb"] = res["peak_ram_mb"]

    return final_result


def experiment_worker(task_data):
    """
    Worker function to execute a single EXPERIMENT run.

    This function will:
    1. Receive an Experiment object.
    2. Call 'run_parallel_isolated'.
    3. That method will spawn 3 internal processes (PolyPart, IncEnu, DelRes).
    4. Data is aggregated and saved by the Experiment class itself.
    """
    exp, run_idx, seed, decimals, folder, exclude_algorithms = task_data

    try:
        results = run_single_experiment(
            P_class=exp.P_class,
            A_class=exp.A_class,
            d=exp.d,
            decimals=decimals,
            seed=seed,
            exclude_algorithms=exclude_algorithms,
            log_folder=os.path.join(folder, exp.dirname()),
        )
        exp.save(results, folder=folder)
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


def run_experiments(
    experiments: list,
    n_runs: int = 10,
    max_workers: int | None = None,
    folder="./data",
    exclude_algorithms: list[str] | None = None,
):
    """
    Orchestrates the parallel execution.
    Each experiment acts as a "Parent" that spawns N "Children".
    To avoid oversubscribing the CPU, we divide available cores by 3.
    Example: 112 Cores -> ~36 Concurrent Experiments -> 108 Active Processes
    """
    # Single experiment
    if not isinstance(experiments, list):
        experiments = [experiments]

    if exclude_algorithms is None:
        exclude_algorithms = []
    algorithms = set(ALGORITHMS) - set(exclude_algorithms)
    n_algs = len(algorithms)
    if n_algs == 0:
        raise ValueError("No algorithms left to run after exclusions.")

    # Available CPU cores
    total_cores = cpu_count()
    # Determine max workers
    if max_workers is None:
        max_workers = max(1, (total_cores - 2))

    # Parent process count
    pool_size = max_workers // (1 + n_algs)  # Parent + n_algs children

    print("--- Starting Rigorous Parallel Batch ---")
    print(f"Available Cores: {total_cores}")
    print(f"Concurrent Experiments: {pool_size}")
    print(
        f"Total Active Processes: {pool_size * (1 + n_algs)} ({pool_size} parents + {pool_size * n_algs} children)"
    )

    # 0. Prepare the folder
    os.makedirs(folder, exist_ok=True)

    # 1. Flatten the workload
    tasks = []
    for exp in experiments:
        existing_runs = exp.count_existing_runs(folder=folder)
        for run in range(n_runs):
            # Task = (Experiment, run_id, seed, decimals)
            task_args = (
                exp,
                run,
                42 + existing_runs + run,  # Unique seed
                3,  # decimals
                folder,
                exclude_algorithms,
            )
            tasks.append(task_args)

    print(f"Total experiments to run: {len(tasks)}")

    # 2. Submit tasks using multiprocessing.Pool
    results_summary = {"success": 0, "error": 0}

    # maxtasksperchild=1 to prevent memory leaks in long batches
    with NestedPool(processes=pool_size, maxtasksperchild=1) as pool:
        # Update the progress bar as soon as ANY job finishes
        for res in tqdm(
            pool.imap_unordered(experiment_worker, tasks), total=len(tasks)
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
        print("ERROR! -> Check 'batch_errors.log' for details <-")
