import argparse
import json
import os
import time
import uuid
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from polypart.arrangements import (
    get_braid_arrangement,
    get_moduli_arrangement,
    get_random_arrangement,
)
from polypart.delres import number_of_regions
from polypart.geometry import Polytope
from polypart.inc_enu import build_tree_inc_enu
from polypart.polytopes import (
    get_centered_hypercube,
    get_product_of_simplices,
    sample_poisson_zero_cell_polytope,
)
from polypart.ppart import build_partition_tree

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


class PolytopeClass:
    def __init__(self, name: str):
        self.name = name

    def get_polytope(
        self, d: int, decimals: int | None = None, seed: int | None = None
    ) -> Polytope:
        if self.name == "random":
            return sample_poisson_zero_cell_polytope(
                d, 1.0, 50, decimals=decimals, seed=seed
            )
        elif self.name == "cube":
            return get_centered_hypercube(d, 1.0)
        elif self.name == "moduli_n1":
            return get_product_of_simplices(1, d)
        elif self.name == "moduli_r2":
            return get_product_of_simplices(d, 1)
        else:
            raise ValueError(f"Unknown polytope class: {self.name}")

    def __str__(self):
        return self.name


class ArrangementClass:
    def __init__(
        self, name: str, m: int | None = None, degen_ratio: float | None = None
    ):
        self.name = name
        self.m = m
        self.degen_ratio = degen_ratio

    def get_arrangement(
        self,
        d: int,
        P: Polytope | None,
        decimals: int | None = None,
        seed: int | None = None,
    ) -> list:
        if self.name == "random":
            assert self.degen_ratio is not None, (
                "Degeneracy ratio must be specified for random arrangement."
            )
            return get_random_arrangement(
                P, self.m, self.degen_ratio, decimals=decimals, seed=seed
            )
        elif self.name == "braid":
            return get_braid_arrangement(d)
        elif self.name == "moduli_n1":
            return get_moduli_arrangement(1, d + 1, 0)
        elif self.name == "moduli_r2":
            return get_moduli_arrangement(d, 2, 0)
        else:
            raise ValueError(f"Unknown arrangement class: {self.name}")

    def __str__(self):
        if self.name == "random":
            return f"random_m{self.m}_degen{self.degen_ratio}"
        elif self.name == "moduli":
            return f"moduli_n{self.n}_r{self.r}"
        else:
            return self.name


def _run_single_algorithm_isolated(args):
    """
    Regenerates the environment and runs ONE algorithm.
    Returns timings, memory, result count, and geometry stats.
    """
    # Unpack arguments
    p_class, a_class, d, decimals, seed, algo_name = args

    # --- 1. Regenerate Geometry (Measure Overhead) ---
    t0 = time.perf_counter()
    P = p_class.get_polytope(d, decimals=decimals, seed=seed)
    P.remove_redundancies()
    P.extreme()
    t1 = time.perf_counter()
    p_creation_time = t1 - t0

    t0 = time.perf_counter()
    A = a_class.get_arrangement(d, P, decimals=decimals, seed=seed)
    t1 = time.perf_counter()
    a_creation_time = t1 - t0

    # --- 2. Run Specific Algorithm ---
    start_time = time.perf_counter()
    num_regions = None
    stats = None

    if algo_name == "ppart":
        T, num_regions = build_partition_tree(P, A, "v-entropy")
        stats = T.stats()

    elif algo_name == "incenu":
        _, num_regions = build_tree_inc_enu(A, P)

    elif algo_name == "delres":
        num_regions = number_of_regions(A, P)

    end_time = time.perf_counter()

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
        # Capture geometry stats from this worker
        "n_vertices": P.n_vertices,
        "n_facets": P.n_inequalities,
        "m_hyperplanes": len(A),
        "p_creation_time": round(p_creation_time, 6),
        "a_creation_time": round(a_creation_time, 6),
        # Hash for P as the str of list of vertices
        "p_hash": hash(tuple(map(tuple, P.vertices))),
        # Hash for A as the str of list of Hyperplane (h.normal and h.offset)
        "a_hash": hash(tuple((tuple(h.normal), h.offset) for h in A)),
    }


class Experiment:
    def __init__(
        self,
        P_class: PolytopeClass,
        A_class: ArrangementClass,
        d: int | None = None,
    ):
        self.P_class = P_class
        self.A_class = A_class
        self.d = d

        self._last_P = None
        self._last_A = None
        self._last_results = None

    def dim(self) -> int:
        return self.d if self.d is not None else self.P_class.n * (self.P_class.r - 1)

    def run(
        self,
        decimals: int | None = None,
        seed: int | None = None,
        tqdm_bar=None,
        exclude_algorithms: list = [],
    ) -> dict:
        # Append info to tqdm bar if provided
        if tqdm_bar is not None:
            tqdm_bar.set_postfix_str(self.dirname())

        P = self.P_class.get_polytope(self.d, decimals=decimals, seed=seed)
        P.extreme()
        P.remove_redundancies()
        A = self.A_class.get_arrangement(self.d, P, decimals=decimals, seed=seed)
        self._last_P = P
        self._last_A = A
        # Run our algorithm
        if "ppart" not in exclude_algorithms:
            if tqdm_bar is not None:
                tqdm_bar.set_postfix_str(f"{self.dirname()} | PolyPart running")
            start_time = perf_counter()
            T_ppart, num_regions_polypart = build_partition_tree(P, A, "v-entropy")
            end_time = perf_counter()
            elapsed_time_polypart = round(end_time - start_time, 6)
            ppart_stats = T_ppart.stats()
        else:
            elapsed_time_polypart = None
            num_regions_polypart = None
            ppart_stats = None

        # Run IncEnu
        if "incenu" not in exclude_algorithms:
            if tqdm_bar is not None:
                tqdm_bar.set_postfix_str(f"{self.dirname()} | IncEnu running")
            start_time = perf_counter()
            _, num_regions_incenu = build_tree_inc_enu(A, P)
            end_time = perf_counter()
            elapsed_time_incenu = round(end_time - start_time, 6)
        else:
            elapsed_time_incenu = None
            num_regions_incenu = None

        # Run numer_of_regions
        if "delres" not in exclude_algorithms:
            if tqdm_bar is not None:
                tqdm_bar.set_postfix_str(f"{self.dirname()} | DelRes running")
            start_time = perf_counter()
            num_regions = number_of_regions(A, P)
            end_time = perf_counter()
            elapsed_time_num_regions = round(end_time - start_time, 6)
        else:
            elapsed_time_num_regions = None
            num_regions = None

        if not exclude_algorithms:
            assert num_regions_polypart == num_regions_incenu == num_regions, (
                f"Number of regions mismatch between methods: "
                f"PolyPart={num_regions_polypart}, IncEnu={num_regions_incenu}, "
                f"DelRes={num_regions}"
            )

        result = {
            "polypart_time": elapsed_time_polypart,
            "incenu_time": elapsed_time_incenu,
            "delres_time": elapsed_time_num_regions,
            "num_regions": num_regions,
            "n_vertices": P.n_vertices,
            "n_facets": P.n_inequalities,
            "m_hyperplanes": len(A),
            "ppart_stats": ppart_stats,
            "decimals": decimals,
        }
        self._last_results = result
        return result

    def run_parallel_isolated(self, decimals=3, seed=None) -> dict:
        """
        Runs ppart, incenu, and delres in 3 separate processes.
        Verifies geometric consistency and result consistency strictly.
        """
        algorithms = ["ppart", "incenu", "delres"]

        # Prepare arguments
        tasks = []
        for algo in algorithms:
            tasks.append((self.P_class, self.A_class, self.d, decimals, seed, algo))

        # Run with a pool of 3 workers
        # maxtasksperchild=1 ensures each runs in a fresh process (clean RAM)
        with Pool(processes=3, maxtasksperchild=1) as pool:
            results_list = pool.map(_run_single_algorithm_isolated, tasks)

        # --- 1. Geometric Consistency Check ---
        # Assert P and A are the same across all runs
        base_info = results_list[0]
        for _, res in enumerate(results_list[1:]):
            algo_name = res["algo"]
            assert res["n_vertices"] == base_info["n_vertices"], (
                f"Geometry mismatch (Vertices) in {algo_name}: {res['n_vertices']} != {base_info['n_vertices']}"
            )
            assert res["n_facets"] == base_info["n_facets"], (
                f"Geometry mismatch (Facets) in {algo_name}: {res['n_facets']} != {base_info['n_facets']}"
            )
            assert res["m_hyperplanes"] == base_info["m_hyperplanes"], (
                f"Geometry mismatch (Hyperplanes) in {algo_name}: {res['m_hyperplanes']} != {base_info['m_hyperplanes']}"
            )

        # --- 1.1 Hash Consistency Check ---
        for _, res in enumerate(results_list[1:]):
            algo_name = res["algo"]
            assert res["p_hash"] == base_info["p_hash"], (
                f"Geometry mismatch (Polytope Hash) in {algo_name}: {res['p_hash']} != {base_info['p_hash']}"
            )
            assert res["a_hash"] == base_info["a_hash"], (
                f"Geometry mismatch (Arrangement Hash) in {algo_name}: {res['a_hash']} != {base_info['a_hash']}"
            )

        # --- 2. Result Consistency Check ---
        # Extract region counts
        region_counts = [res["num_regions"] for res in results_list]

        # Check for Null results (Algorithm Failure)
        if any(r is None for r in region_counts):
            # Create a map of algo -> result to verify which one failed
            failed_map = {res["algo"]: res["num_regions"] for res in results_list}
            raise RuntimeError(
                f"One or more algorithms failed to return a region count: {failed_map}"
            )

        # Check for Mismatches (Correctness Failure)
        first_count = region_counts[0]
        assert all(r == first_count for r in region_counts), (
            f"Region count mismatch between algorithms: {region_counts}"
        )

        # --- 3. Construct Final Result ---
        final_result = {
            # Standard metrics
            "polypart_time": None,
            "incenu_time": None,
            "delres_time": None,
            "num_regions": first_count,  # Validated above
            # Geometry Stats (Validated above)
            "n_vertices": base_info["n_vertices"],
            "n_facets": base_info["n_facets"],
            "m_hyperplanes": base_info["m_hyperplanes"],
            # Peak RAM Stats
            "polypart_peak_ram_mb": None,
            "incenu_peak_ram_mb": None,
            "delres_peak_ram_mb": None,
            # Additional Profiling
            "p_creation_time": base_info["p_creation_time"],
            "a_creation_time": base_info["a_creation_time"],
            "ppart_stats": None,
            "decimals": decimals,
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

    def dirname(self) -> str:
        return f"{self.P_class}-{self.A_class}-d{self.dim()}"

    def save(self, results: dict, folder: str = "../data"):
        path = Path(folder) / self.dirname()
        # Ensure the folder exists or create it
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"experiment_{timestamp}_{unique_id}.json"
        filepath = path / filename
        # Check if file already exists, add a counter if so
        counter = 1
        while filepath.exists():
            filename = f"experiment_{timestamp}_{counter}_{unique_id}.json"
            filepath = path / filename
            counter += 1
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)

    def load(self, folder: str = "../data") -> list:
        path = Path(folder) / self.dirname()
        results = []
        if not path.exists():
            raise FileNotFoundError(f"No results found for experiment at {path}")
        for filename in os.listdir(path):
            if filename.endswith(".json"):
                filepath = path / filename
                with open(filepath, "r") as f:
                    data = json.load(f)
                    results.append(data)
        if results == []:
            raise FileNotFoundError(f"No result jsons found for experiment at {path}")
        return results

    def __str__(self):
        d = self.d if self.d is not None else self.P_class.n * (self.P_class.r - 1)
        return f"Experiment(P={self.P_class}, A={self.A_class}, d={d})"


def run_experiments(
    experiments: list[Experiment] | Experiment,
    n_runs: int = 10,
    folder: str | None = "../data",
    exclude_algorithms: list = [],
):
    if isinstance(experiments, Experiment):
        experiments = [experiments]
    # Show number of run in tqdm as label
    tqdm_experiments = tqdm(experiments)
    for exp in tqdm_experiments:
        for run in range(n_runs):
            tqdm_experiments.set_description(f"(run {run + 1}/{n_runs})")
            results = exp.run(
                decimals=3,
                seed=42 + run,
                tqdm_bar=tqdm_experiments,
                exclude_algorithms=exclude_algorithms,
            )
            exp.save(results, folder=folder)


def summarize_results(
    experiments: list[Experiment] | Experiment, folder: str = "../data"
):
    def mean_std(times: list[float]) -> tuple[float, float]:
        if any(t is None for t in times):
            return None, None
        mean = np.mean(times)
        std = np.std(times)
        return round(mean, 6), round(std, 6)

    if isinstance(experiments, Experiment):
        experiments = [experiments]

    # Collect stats of experiments
    for exp in experiments:
        try:
            all_results = exp.load(folder=folder)
        except FileNotFoundError as e:
            print("Skipping experiment:", e)
            continue
        # Compute average times, std of times and probability of time_polypart<time_incenu
        polypart_times = [
            res["polypart_time"]
            for res in all_results
            if res["polypart_time"] is not None
        ]
        incenu_times = [
            res["incenu_time"] for res in all_results if res["incenu_time"] is not None
        ]
        delres_times = [
            res["delres_time"] for res in all_results if res["delres_time"] is not None
        ]
        num_regions = [
            res["num_regions"] for res in all_results if res["num_regions"] is not None
        ]
        prob_polypart_better = np.mean(
            [1 if p < i else 0 for p, i in zip(polypart_times, incenu_times)]
        )
        avg_num_regions, std_num_regions = mean_std(num_regions)
        mean_polypart_time, std_polypart_time = mean_std(polypart_times)
        mean_incenu_time, std_incenu_time = mean_std(incenu_times)
        mean_delres_time, std_delres_time = mean_std(delres_times)
        prob_polypart_better_str = (
            f"{prob_polypart_better:.3%}" if prob_polypart_better is not None else "N/A"
        )
        print(
            f"Experiment: {exp.dirname()}\n"
            f"  Avg num regions: {avg_num_regions:.2f} ± {std_num_regions:.2f}\n"
            f"  PolyPart time: {mean_polypart_time} ± {std_polypart_time} ({len(polypart_times)} runs)\n"
            f"  IncEnu time:   {mean_incenu_time} ± {std_incenu_time} ({len(incenu_times)} runs)\n"
            f"  DelRes time:   {mean_delres_time} ± {std_delres_time} ({len(delres_times)} runs)\n"
            f"  P(PolyPart < IncEnu): {prob_polypart_better_str}\n"
        )


def plot_experiment_summary(experiment: Experiment):
    """
    Summary plot of all runs of an experiment, showing overall statistics
    and moments plots per depth.
    """
    try:
        all_results = experiment.load()
    except FileNotFoundError:
        print(f"No experiment found with path {experiment.dirname()}")
        return

    # Filter all_results without per_depth_moments_vertices
    all_results = [
        res
        for res in all_results
        if res["ppart_stats"] is not None
        and "per_depth_moments_vertices" in res["ppart_stats"]
    ]
    if len(all_results) == 0:
        print(f"No per-depth moments data found for experiment {experiment.dirname()}")
        return

    # Collect times per depth
    alphas = all_results[-1]["ppart_stats"]["per_depth_moments_vertices"].keys()
    all_per_depth_moments_candidates = {alpha: {} for alpha in alphas}
    all_per_depth_moments_inequalities = {alpha: {} for alpha in alphas}
    all_per_depth_moments_vertices = {alpha: {} for alpha in alphas}
    for res in all_results:
        ppart_stats = res["ppart_stats"]
        moments_candidates = ppart_stats["per_depth_moments_candidates"]
        moments_inequalities = ppart_stats["per_depth_moments_inequalities"]
        moments_vertices = ppart_stats["per_depth_moments_vertices"]
        for alpha in alphas:
            if alpha in moments_candidates:
                for depth, moment in moments_candidates[alpha].items():
                    if depth not in all_per_depth_moments_candidates[alpha]:
                        all_per_depth_moments_candidates[alpha][depth] = []
                    all_per_depth_moments_candidates[alpha][depth].append(moment)
            else:
                continue
            if alpha in moments_inequalities:
                for depth, moment in moments_inequalities[alpha].items():
                    if depth not in all_per_depth_moments_inequalities[alpha]:
                        all_per_depth_moments_inequalities[alpha][depth] = []
                    all_per_depth_moments_inequalities[alpha][depth].append(moment)
            else:
                continue
            if alpha in moments_vertices:
                for depth, moment in moments_vertices[alpha].items():
                    if depth not in all_per_depth_moments_vertices[alpha]:
                        all_per_depth_moments_vertices[alpha][depth] = []
                    all_per_depth_moments_vertices[alpha][depth].append(moment)
            else:
                continue

    # Data Extraction
    # -------------------------------------------------------------------------
    # Extract raw lists, ensuring we handle None values safely
    regions_list = [
        res["num_regions"] for res in all_results if res.get("num_regions") is not None
    ]

    # Times (PolyPart, IncEnu, and the newly requested DelRes)
    polypart_times = [
        res["polypart_time"]
        for res in all_results
        if res.get("polypart_time") is not None
    ]
    incenu_times = [
        res["incenu_time"] for res in all_results if res.get("incenu_time") is not None
    ]
    delres_times = [
        res["delres_time"] for res in all_results if res.get("delres_time") is not None
    ]

    # Calculate Region Statistics for the subtitle
    avg_regions = np.mean(regions_list) if regions_list else 0
    std_regions = np.std(regions_list) if regions_list else 0

    # Plotting
    # -------------------------------------------------------------------------
    # 2x2 Grid: Efficient use of space
    fig, axs = plt.subplots(
        2, 2, figsize=(18, 12)
    )  # Increased height slightly for better aspect ratio

    fig.suptitle(
        f"Experiment Summary: {experiment.dirname()} (runs={len(all_results)})",
        fontsize=18,
        y=0.95,
    )

    # --- 1. Top-Left: Time Box Plot (Sorted) ---
    ax_time = axs[0, 0]

    # Structure data for sorting: (Data List, Label, Color)
    raw_time_data = [
        (polypart_times, "PolyPart", "#1f77b4"),  # Blue
        (incenu_times, "IncEnu", "#ff7f0e"),  # Orange
        (delres_times, "DelRes", "#2ca02c"),  # Green
    ]
    # Print mean times for debugging
    for data, label, _ in raw_time_data:
        if data:
            mean_time = np.mean(data)
            print(f"{label} mean time: {mean_time:.6f} seconds over {len(data)} runs")
        else:
            print(f"{label} has no data.")

    # Sort by mean value (ascending) -> Fastest first
    # Handle empty lists by assigning infinity so they drop to the bottom
    sorted_time_data = sorted(
        raw_time_data, key=lambda x: np.mean(x[0]) if len(x[0]) > 0 else float("inf")
    )

    # Unpack sorted data
    plot_data = [item[0] for item in sorted_time_data]
    plot_labels = [item[1] for item in sorted_time_data]
    plot_colors = [item[2] for item in sorted_time_data]

    # Create horizontal boxplot
    bplot = ax_time.boxplot(
        plot_data,
        vert=False,
        patch_artist=True,
        labels=plot_labels,
        widths=0.6,
        # Show the Mean Line
        showmeans=True,
        meanline=True,
        # Style the Mean
        meanprops={"color": "black", "linewidth": 2.5, "linestyle": "-"},
        # Style the Median
        medianprops={"color": "red", "linewidth": 1, "linestyle": "--"},
    )

    # Apply colors to the sorted boxes
    for patch, color in zip(bplot["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Invert Y-axis so the first item (fastest/lowest mean) is at the TOP
    ax_time.invert_yaxis()

    # Set Title with Region Stats
    region_stats_str = f"Avg regions: ${int(avg_regions)} \pm {int(std_regions)}$"
    ax_time.set_title(f"Execution Times ({region_stats_str})", fontsize=14)
    ax_time.set_xlabel("Time (seconds)")
    ax_time.grid(True, axis="x", linestyle="--", alpha=0.5)

    # --- 2, 3, 4. The Moment Plots ---
    # Define configuration to map data to specific subplots
    # Order: Top-Right -> Bottom-Left -> Bottom-Right
    moment_configs = [
        (all_per_depth_moments_candidates, "$|A_k|$", axs[0, 1]),
        (all_per_depth_moments_inequalities, "$|P_k|$", axs[1, 0]),
        (all_per_depth_moments_vertices, "$|V_k|$", axs[1, 1]),
    ]

    for all_per_depth_moments, ylabel, ax in moment_configs:
        for alpha in alphas:
            # Sort depths numerically
            depths = sorted(all_per_depth_moments[alpha].keys(), key=lambda x: int(x))

            if not depths:
                continue

            # Calculate stats
            avg_moments = [
                np.mean(all_per_depth_moments[alpha][depth]) for depth in depths
            ]
            std_moments = [
                np.std(all_per_depth_moments[alpha][depth]) for depth in depths
            ]

            # Plot Line
            ax.plot(
                depths,
                avg_moments,
                label=f"$\\alpha={alpha}$",
                marker="o",
                markersize=4,
            )

            # Plot Std Dev Shade
            ax.fill_between(
                depths,
                np.array(avg_moments) - np.array(std_moments),
                np.array(avg_moments) + np.array(std_moments),
                alpha=0.15,
            )

        ax.set_title(f"Per-Depth Moments of {ylabel}", fontsize=14)
        ax.set_xlabel("Depth")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(True, alpha=0.3)

    # Final Layout Adjustment
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)
    plt.tight_layout(pad=3.0)
    plt.show()


# recieve sys args to create an experiment and run it n_runs times
# Usages:
# python experiment.py -d 3 -p cube -a braid -m 5 (optional) -r 0.2 (otional) -n 10 (optional, default 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run polytope partition experiments.")
    parser.add_argument(
        "-d", type=int, required=True, help="Dimension of the polytope."
    )
    parser.add_argument(
        "-p",
        type=str,
        required=True,
        help="Polytope class (e.g., 'cube', 'moduli_n1', 'moduli_r2', 'random').",
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
        "-r",
        type=float,
        default=0.0,
        help="Degeneracy ratio (required for random arrangement).",
    )
    parser.add_argument(
        "-n", type=int, default=1, help="Number of runs to perform (default: 1)."
    )

    args = parser.parse_args()

    P_class = PolytopeClass(args.p)
    A_class = ArrangementClass(args.a, m=args.m, degen_ratio=args.r)
    experiment = Experiment(P_class, A_class, d=args.d)
    run_experiments(experiment, n_runs=args.n)
    summarize_results(experiment)
