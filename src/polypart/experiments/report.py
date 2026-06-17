import ast
import math
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .core import Experiment


def print_results_summary(
    experiments: list[Experiment] | Experiment, folder: str = "./data"
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
        all_results = exp.load(folder=folder)
        if len(all_results) == 0:
            print(f"No results found for experiment {exp.dirname()}")
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
        # Calculate time per region for each run to ensure accurate standard deviations
        polypart_per_region = [
            t / r for t, r in zip(polypart_times, num_regions) if r > 0
        ]
        incenu_per_region = [t / r for t, r in zip(incenu_times, num_regions) if r > 0]
        delres_per_region = [t / r for t, r in zip(delres_times, num_regions) if r > 0]

        # Calculate statistics
        avg_num_regions, std_num_regions = mean_std(num_regions)
        mean_polypart_time, std_polypart_time = mean_std(polypart_times)
        mean_incenu_time, std_incenu_time = mean_std(incenu_times)
        mean_delres_time, std_delres_time = mean_std(delres_times)

        # Calculate time per region statistics
        mean_polypart_per, std_polypart_per = mean_std(polypart_per_region)
        mean_incenu_per, std_incenu_per = mean_std(incenu_per_region)
        mean_delres_per, std_delres_per = mean_std(delres_per_region)

        prob_polypart_better_str = (
            f"{prob_polypart_better:.3%}" if prob_polypart_better is not None else "N/A"
        )

        print(
            f"Experiment: {exp.dirname()}\n"
            f"  Avg num regions: {avg_num_regions:.2f} ± {std_num_regions:.2f}\n"
            f"  PolyPart time: {mean_polypart_time:.4f} ± {std_polypart_time:.4f} ({len(polypart_times)} runs)\n"
            f"  IncEnu time:   {mean_incenu_time:.4f} ± {std_incenu_time:.4f} ({len(incenu_times)} runs)\n"
            f"  DelRes time:   {mean_delres_time:.4f} ± {std_delres_time:.4f} ({len(delres_times)} runs)\n"
            f"  PolyPart time/region: {mean_polypart_per:.4f} ± {std_polypart_per:.6f}\n"
            f"  IncEnu time/region:   {mean_incenu_per:.4f} ± {std_incenu_per:.6f}\n"
            f"  DelRes time/region:   {mean_delres_per:.4f} ± {std_delres_per:.6f}\n"
            f"  P(PolyPart < IncEnu): {prob_polypart_better_str}\n"
        )


def plot_experiment_summary(experiment: Experiment, folder: str = "./data"):
    """
    Summary plot of all runs of an experiment, showing overall statistics
    and moments plots per depth.
    Includes a background histogram of average nodes per depth.
    """
    all_results = experiment.load(folder=folder)
    if len(all_results) == 0:
        print(f"No results found for experiment {experiment.dirname()}")
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
    # We grab the keys (alphas) from the last result as a template
    alphas = all_results[-1]["ppart_stats"]["per_depth_moments_vertices"].keys()

    all_per_depth_moments_candidates = {alpha: {} for alpha in alphas}
    all_per_depth_moments_inequalities = {alpha: {} for alpha in alphas}
    all_per_depth_moments_vertices = {alpha: {} for alpha in alphas}

    # NEW: Container for node counts per depth
    all_per_depth_nodes = {}

    for res in all_results:
        ppart_stats = res["ppart_stats"]

        # Extract Moments
        moments_candidates = ppart_stats["per_depth_moments_candidates"]
        moments_inequalities = ppart_stats["per_depth_moments_inequalities"]
        moments_vertices = ppart_stats["per_depth_moments_vertices"]

        for alpha in alphas:
            if alpha in moments_candidates:
                for depth, moment in moments_candidates[alpha].items():
                    if depth not in all_per_depth_moments_candidates[alpha]:
                        all_per_depth_moments_candidates[alpha][depth] = []
                    all_per_depth_moments_candidates[alpha][depth].append(moment)

            if alpha in moments_inequalities:
                for depth, moment in moments_inequalities[alpha].items():
                    if depth not in all_per_depth_moments_inequalities[alpha]:
                        all_per_depth_moments_inequalities[alpha][depth] = []
                    all_per_depth_moments_inequalities[alpha][depth].append(moment)

            if alpha in moments_vertices:
                for depth, moment in moments_vertices[alpha].items():
                    if depth not in all_per_depth_moments_vertices[alpha]:
                        all_per_depth_moments_vertices[alpha][depth] = []
                    all_per_depth_moments_vertices[alpha][depth].append(moment)

        # NEW: Extract Node Counts
        # Assuming per_depth_nodes is a dict {depth: count} inside stats
        if "per_depth_nodes" in ppart_stats:
            for depth, count in ppart_stats["per_depth_nodes"].items():
                if depth not in all_per_depth_nodes:
                    all_per_depth_nodes[depth] = []
                all_per_depth_nodes[depth].append(count)

    # Prepare Node Histogram Data
    if all_per_depth_nodes:
        # Sort depths numerically
        sorted_node_depths = sorted(all_per_depth_nodes.keys(), key=lambda x: int(x))
        avg_nodes_per_depth = [
            np.mean(all_per_depth_nodes[d]) for d in sorted_node_depths
        ]
        max_avg_node = max(avg_nodes_per_depth) if avg_nodes_per_depth else 1
    else:
        sorted_node_depths = []
        avg_nodes_per_depth = []
        max_avg_node = 1

    # Data Extraction
    # -------------------------------------------------------------------------
    regions_list = [
        res["num_regions"] for res in all_results if res.get("num_regions") is not None
    ]

    # Times
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

    # Stats
    avg_regions = np.mean(regions_list) if regions_list else 0
    std_regions = np.std(regions_list) if regions_list else 0

    # Plotting
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    fig.suptitle(
        f"Experiment Summary: {experiment.dirname()} (runs={len(all_results)})",
        fontsize=18,
        y=0.95,
    )

    # --- 1. Top-Left: Time Box Plot (Sorted) ---
    ax_time = axs[0, 0]
    raw_time_data = [
        (polypart_times, "PolyPart", "#1f77b4"),
        (incenu_times, "IncEnu", "#ff7f0e"),
        (delres_times, "DelRes", "#2ca02c"),
    ]
    sorted_time_data = sorted(
        raw_time_data, key=lambda x: np.mean(x[0]) if len(x[0]) > 0 else float("inf")
    )
    plot_data = [item[0] for item in sorted_time_data]
    plot_labels = [item[1] for item in sorted_time_data]
    plot_colors = [item[2] for item in sorted_time_data]

    bplot = ax_time.boxplot(
        plot_data,
        vert=False,
        patch_artist=True,
        labels=plot_labels,
        widths=0.6,
        showmeans=True,
        meanline=True,
        meanprops={"color": "black", "linewidth": 2.5, "linestyle": "-"},
        medianprops={"color": "red", "linewidth": 1, "linestyle": "--"},
    )
    for patch, color in zip(bplot["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_time.invert_yaxis()
    ax_time.plot([], [], color="black", linewidth=2.5, label="Mean")
    ax_time.plot([], [], color="red", linewidth=1, linestyle="--", label="Median")
    ax_time.legend(loc="upper right", fontsize="small")

    region_stats_str = f"Avg regions: ${int(avg_regions)} \pm {int(std_regions)}$"
    ax_time.set_title(f"Execution Times ({region_stats_str})", fontsize=14)
    ax_time.set_xlabel("Time (seconds)")
    ax_time.grid(True, axis="x", linestyle="--", alpha=0.5)

    # --- 2, 3, 4. The Moment Plots with Histogram Overlay ---
    moment_configs = [
        (all_per_depth_moments_candidates, "$|A_k|$", axs[0, 1]),
        (all_per_depth_moments_inequalities, "$|P_k|$", axs[1, 0]),
        (all_per_depth_moments_vertices, "$|V_k|$", axs[1, 1]),
    ]

    for all_per_depth_moments, ylabel, ax in moment_configs:
        # --- NEW: Plot Node Histogram on Secondary Axis ---
        if sorted_node_depths:
            ax2 = ax.twinx()
            # Plot bars
            ax2.bar(
                sorted_node_depths,
                avg_nodes_per_depth,
                color="gray",
                alpha=0.15,  # High transparency
                width=0.8,
                label="Avg Nodes",
            )
            # Set ylim to 3x the max value to force bars to the bottom 1/3
            ax2.set_ylim(0, max_avg_node * 3)

            # Style the secondary axis to be subtle
            ax2.set_ylabel("Avg Node Count", color="gray", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
            ax2.grid(False)  # Turn off grid for secondary axis to avoid clutter

            # Ensure the primary plot (lines) stays on top of the bars
            ax.set_zorder(10)
            ax.patch.set_visible(False)  # Make primary axis background transparent
            ax2.set_zorder(1)

        # --------------------------------------------------

        for alpha in alphas:
            depths = sorted(all_per_depth_moments[alpha].keys(), key=lambda x: int(x))
            if not depths:
                continue

            try:
                avg_moments = [np.mean(all_per_depth_moments[alpha][d]) for d in depths]
                std_moments = [np.std(all_per_depth_moments[alpha][d]) for d in depths]
            except Exception:
                continue

            ax.plot(
                depths,
                avg_moments,
                label=f"$\\alpha={alpha}$",
                marker="o",
                markersize=4,
            )
            ax.fill_between(
                depths,
                np.array(avg_moments) - np.array(std_moments),
                np.array(avg_moments) + np.array(std_moments),
                alpha=0.15,
            )

        ax.set_title(f"Per-Depth Moments of {ylabel}", fontsize=14)
        ax.set_xlabel("Depth")
        ax.set_ylabel(ylabel)

        # Combine legends? The ax2 legend might be separate.
        # Let's just keep the main legend clean.
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(True, alpha=0.3)

        xticks = ax.get_xticks()
        if len(xticks) > 20:
            ax.set_xticks(xticks[::5])

    plt.tight_layout(pad=3.0)
    plt.savefig(f"./figures/{experiment}.pdf")
    plt.show()


def human_format(num, pos=None):
    """Formatter to display large numbers compactly (e.g., 10K, 1M)."""
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.1f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


def plot_experiment_summary_paper(
    experiment,
    folder: str = "./data",
    remove_titles: bool = False,
):
    """
    Summary plot of all runs of an experiment.
    Plots the mean and standard deviation per depth for candidates, inequalities, and vertices.
    Optimized for a 1x3 layout for paper figures, including a background node histogram.
    """
    all_results = experiment.load(folder=folder)
    if len(all_results) == 0:
        print(f"No results found for experiment {experiment.dirname()}")
        return

    # Filter all_results to ensure per-depth stats exist
    all_results = [
        res
        for res in all_results
        if res.get("ppart_stats") is not None
        and "per_depth_moments_vertices" in res["ppart_stats"]
    ]
    if len(all_results) == 0:
        print(f"No per-depth moments data found for experiment {experiment.dirname()}")
        return

    # Data structures to aggregate across all runs
    depth_candidates = {}
    depth_inequalities = {}
    depth_vertices = {}
    all_per_depth_nodes = {}

    for res in all_results:
        ppart_stats = res["ppart_stats"]

        moments_candidates = ppart_stats.get("per_depth_moments_candidates", {})
        moments_inequalities = ppart_stats.get("per_depth_moments_inequalities", {})
        moments_vertices = ppart_stats.get("per_depth_moments_vertices", {})

        # We assume alpha=1 or alpha="1" represents the standard mean.
        # Safely extract the correct key based on how the dictionary is keyed.
        alpha_key = 1 if 1 in moments_candidates else "1"

        mc = moments_candidates.get(alpha_key, {})
        mi = moments_inequalities.get(alpha_key, {})
        mv = moments_vertices.get(alpha_key, {})

        # Aggregate metrics
        for depth, val in mc.items():
            depth_candidates.setdefault(depth, []).append(val)
        for depth, val in mi.items():
            depth_inequalities.setdefault(depth, []).append(val)
        for depth, val in mv.items():
            depth_vertices.setdefault(depth, []).append(val)

        # Aggregate Node Counts
        if "per_depth_nodes" in ppart_stats:
            for depth, count in ppart_stats["per_depth_nodes"].items():
                all_per_depth_nodes.setdefault(depth, []).append(count)

    # Prepare Node Histogram Data
    if all_per_depth_nodes:
        sorted_node_depths = sorted(all_per_depth_nodes.keys(), key=lambda x: int(x))
        avg_nodes_per_depth = [
            np.mean(all_per_depth_nodes[d]) for d in sorted_node_depths
        ]
        max_avg_node = max(avg_nodes_per_depth) if avg_nodes_per_depth else 1
    else:
        sorted_node_depths = []
        avg_nodes_per_depth = []
        max_avg_node = 1

    # Plotting
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.5))

    moment_configs = [
        (depth_candidates, "$|A_k|$", axs[0]),
        (depth_inequalities, "$|P_k|$", axs[1]),
        (depth_vertices, "$|V_k|$", axs[2]),
    ]

    for i, (metric_data, ylabel, ax) in enumerate(moment_configs):
        # --- Secondary Axis: Node Histogram ---
        if sorted_node_depths:
            ax2 = ax.twinx()
            ax2.bar(
                sorted_node_depths,
                avg_nodes_per_depth,
                color="gray",
                alpha=0.15,
                width=0.8,
            )
            ax2.set_ylim(0, max_avg_node * 3)

            # Only show the secondary label/ticks on the final (far right) plot
            if i == 2:
                ax2.set_ylabel(
                    "Avg Node Count",
                    color="gray",
                    fontsize=10,
                    labelpad=15,
                    rotation=270,
                )
                ax2.tick_params(axis="y", labelcolor="gray", labelsize=9)
                ax2.yaxis.set_major_formatter(FuncFormatter(human_format))
            else:
                ax2.set_yticklabels([])
                ax2.tick_params(axis="y", length=0)

            ax2.grid(False)
            ax.set_zorder(10)
            ax.patch.set_visible(False)
            ax2.set_zorder(1)

        # --- Primary Axis: Mean & Std Dev Lines ---
        depths = sorted(metric_data.keys(), key=lambda x: int(x))
        if not depths:
            continue

        avg_vals = [np.mean(metric_data[d]) for d in depths]
        std_vals = [np.std(metric_data[d]) for d in depths]

        # Plot the mean line
        (line,) = ax.plot(
            depths,
            avg_vals,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label="Mean",
            color="#1f77b4",  # Matplotlib standard blue
        )

        # Shade the standard deviation
        ax.fill_between(
            depths,
            np.array(avg_vals) - np.array(std_vals),
            np.array(avg_vals) + np.array(std_vals),
            alpha=0.2,
            color="#1f77b4",
            label="$\pm 1$ Std Dev",
        )

        if not remove_titles:
            ax.set_title(f"Per-Depth Evolution of {ylabel}", fontsize=12)

        ax.set_xlabel("Depth", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)

        # Add legend only to the first subplot to save space
        if i == 0:
            ax.legend(loc="upper right", fontsize="small", framealpha=0.9)

        ax.grid(True, alpha=0.3)

        # Prevent overlapping X-ticks
        xticks = ax.get_xticks()
        if len(xticks) > 10:
            ax.set_xticks(xticks[:: max(1, len(xticks) // 10)])

    plt.tight_layout(pad=0.5)

    plt.savefig(f"./figures/{experiment.dirname()}.pdf", bbox_inches="tight")
    plt.show()


def plot_random_report(experiments: list, folder: str = "./data"):
    """
    Plot Time (primary axis, lines) and Peak Memory (secondary axis, triple bars)
    per number of hyperplanes for each dimension in a grid of subplots.
    """
    if not experiments:
        print("No experiments provided.")
        return

    # 1. Data Structure: data[dim][m] = {alg: {'time': [], 'mem': []}}
    data_by_dim = defaultdict(
        lambda: defaultdict(
            lambda: {
                "ppart": {"time": [], "mem": []},
                "incenu": {"time": [], "mem": []},
                "delres": {"time": [], "mem": []},
            }
        )
    )

    # 2. Load and Organize Data
    num_runs_by_dim = {}
    for exp in experiments:
        d = exp.d
        results = exp.load(folder=folder)
        if not results:
            continue

        for res in results:
            m = res.get("m_hyperplanes")
            if m is None:
                continue

            # Mapping for keys
            algs = {
                "ppart": ("polypart_time", "polypart_peak_ram_mb"),
                "incenu": ("incenu_time", "incenu_peak_ram_mb"),
                "delres": ("delres_time", "delres_peak_ram_mb"),
            }

            for key, (t_key, m_key) in algs.items():
                if res.get(t_key) is not None:
                    data_by_dim[d][m][key]["time"].append(res[t_key])
                if res.get(m_key) is not None:
                    data_by_dim[d][m][key]["mem"].append(res[m_key])

        # Track the number of runs for each dimension
        num_runs = len(results)
        if d in num_runs_by_dim:
            if num_runs_by_dim[d] != num_runs:
                # raise ValueError(
                #     f"Inconsistent number of runs for dimension {d}: "
                #     f"found {num_runs_by_dim[d]} and {num_runs}."
                # )
                print(
                    f"Warning: Inconsistent number of runs for dimension {d} and m={m} "
                    f"(found {num_runs} instead of {num_runs_by_dim[d]})."
                )
        else:
            num_runs_by_dim[d] = num_runs

    # 3. Setup Plot Grid
    dims = sorted(data_by_dim.keys())
    if not dims:
        print("No valid results found to plot.")
        return

    num_dims = len(dims)
    nrows = math.ceil(num_dims**0.5)
    ncols = math.ceil(num_dims / nrows)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )

    styles = {
        "ppart": {"c": "#1f77b4", "marker": "o-", "label": "PolyPart"},
        "incenu": {"c": "#ff7f0e", "marker": "s-", "label": "IncEnu"},
        "delres": {"c": "#2ca02c", "marker": "^-", "label": "DelRes"},
    }

    # 4. Plot for each Dimension
    for i, d in enumerate(dims):
        row, col = i // ncols, i % ncols
        ax1 = axs[row][col]
        m_dict = data_by_dim[d]
        sorted_ms = sorted(m_dict.keys())

        indices = np.arange(len(sorted_ms))
        bar_width = 0.25

        # --- Axis 2: Memory (Triple Bar Plot in Background) ---
        ax2 = ax1.twinx()
        max_mem_subplot = 0

        for j, (alg, s) in enumerate(styles.items()):
            m_means = [
                np.mean(m_dict[m][alg]["mem"]) if m_dict[m][alg]["mem"] else 0
                for m in sorted_ms
            ]
            max_mem_subplot = max(max_mem_subplot, max(m_means) if m_means else 0)

            ax2.bar(
                indices + (j - 1) * bar_width,
                m_means,
                bar_width,
                color=s["c"],
                alpha=0.3,
                label=f"{s['label']} Mem",
            )

        # Scale ax2 so bars stay in the bottom of the plot
        ax2.set_ylim(0, (max_mem_subplot if max_mem_subplot > 0 else 1) * 2.5)
        ax2.set_ylabel("Memory (MB)", color="grey", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="grey", labelsize=8)

        # --- Axis 1: Time (Lines with Shaded Std Dev) ---
        for alg, s in styles.items():
            t_data = [m_dict[m][alg]["time"] for m in sorted_ms]
            t_means = np.array([np.mean(vals) if vals else np.nan for vals in t_data])
            t_stds = np.array([np.std(vals) if vals else np.nan for vals in t_data])

            ax1.plot(
                indices,
                t_means,
                s["marker"],
                color=s["c"],
                label=s["label"],
                linewidth=1.5,
            )
            mask = ~np.isnan(t_means) & ~np.isnan(t_stds)
            ax1.fill_between(
                indices[mask],
                t_means[mask] - t_stds[mask],
                t_means[mask] + t_stds[mask],
                color=s["c"],
                alpha=0.15,
                edgecolor=None,
            )

        # --- Formatting ---
        num_runs = num_runs_by_dim[d]
        ax1.set_yscale("log")
        ax1.set_title(f"Dimension $d={d}$ (Runs={num_runs})", fontsize=12)
        ax1.set_xticks(indices)
        ax1.set_xticklabels(sorted_ms, fontsize=9)
        ax1.set_xlabel("Hyperplanes ($m$)", fontsize=10)
        ax1.set_ylabel("Time (s)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Combined Legend (only on the first subplot)
        if i == 0:
            lines, labels = ax1.get_legend_handles_labels()
            bars, b_labels = ax2.get_legend_handles_labels()
            ax1.legend(
                lines + bars,
                labels + b_labels,
                loc="upper left",
                ncol=2,
                fontsize="x-small",
            )

    # Hide unused subplots
    for i in range(num_dims, nrows * ncols):
        axs[i // ncols][i % ncols].axis("off")

    plt.tight_layout()
    plt.savefig("./figures/random_report.pdf")
    plt.show()


def plot_times_per_m_across_dim(experiments: list[Experiment], folder: str = "./data"):
    """
    Plot time per number of hyperplanes for each dimension in a subplot.
    Aggregates runs from multiple experiments to calculate mean times.
    """
    if not experiments:
        print("No experiments provided.")
        return

    # Data Structure: data[dim][m] = {'polypart': [], 'incenu': [], 'delres': []}
    data_by_dim = defaultdict(
        lambda: defaultdict(lambda: {"ppart": [], "incenu": [], "delres": []})
    )

    # 1. Load and Organize Data
    for exp in experiments:
        # We determine dimension from the experiment object
        d = exp.d

        # Load results from disk
        results = exp.load(folder=folder)
        if not results:
            continue

        for res in results:
            # We assume 'm_hyperplanes' is consistent within an experiment,
            # but getting it from the result dict is safer.
            m = res.get("m_hyperplanes")
            if m is None:
                continue

            # Append times (filtering out None values)
            if res.get("polypart_time") is not None:
                data_by_dim[d][m]["ppart"].append(res["polypart_time"])
            if res.get("incenu_time") is not None:
                data_by_dim[d][m]["incenu"].append(res["incenu_time"])
            if res.get("delres_time") is not None:
                data_by_dim[d][m]["delres"].append(res["delres_time"])

    # 2. Setup Plot
    dims = sorted(data_by_dim.keys())
    if not dims:
        print("No valid results found to plot.")
        return

    num_dims = len(dims)
    nrows = math.ceil((num_dims) ** 0.5)
    ncols = math.floor((num_dims + 1) ** 0.5)
    _, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3 * nrows),
        squeeze=False,
    )
    # squeeze=False ensures axs is always a 2D array [[ax1, ax2...]] even if 1 subplot

    # 3. Plot for each Dimension
    for i, d in enumerate(dims):
        row = i // ncols
        col = i % ncols
        ax = axs[row][col]
        m_dict = data_by_dim[d]

        # Sort by m (x-axis)
        sorted_ms = sorted(m_dict.keys())

        # Arrays for plotting
        ms = []
        means_ppart = []
        stds_ppart = []
        means_incenu = []
        stds_incenu = []
        means_delres = []
        stds_delres = []

        for m in sorted_ms:
            times = m_dict[m]
            # Only include points where we have data
            if times["ppart"] or times["incenu"] or times["delres"]:
                ms.append(m)
                # Calculate means and stds, defaulting to NaN if list is empty
                means_ppart.append(
                    np.mean(times["ppart"]) if times["ppart"] else np.nan
                )
                stds_ppart.append(np.std(times["ppart"]) if times["ppart"] else np.nan)
                means_incenu.append(
                    np.mean(times["incenu"]) if times["incenu"] else np.nan
                )
                stds_incenu.append(
                    np.std(times["incenu"]) if times["incenu"] else np.nan
                )
                means_delres.append(
                    np.mean(times["delres"]) if times["delres"] else np.nan
                )
                stds_delres.append(
                    np.std(times["delres"]) if times["delres"] else np.nan
                )
            else:
                print(f"Warning: No timing data for d={d}, m={m} in experiment")
                continue

        # Plot Lines with Shaded Std Dev
        ax.plot(ms, means_ppart, "o-", label="PolyPart", color="#1f77b4")
        ax.fill_between(
            ms,
            np.array(means_ppart) - np.array(stds_ppart),
            np.array(means_ppart) + np.array(stds_ppart),
            color="#1f77b4",
            alpha=0.2,
        )

        ax.plot(ms, means_incenu, "s-", label="IncEnu", color="#ff7f0e")
        ax.fill_between(
            ms,
            np.array(means_incenu) - np.array(stds_incenu),
            np.array(means_incenu) + np.array(stds_incenu),
            color="#ff7f0e",
            alpha=0.2,
        )

        ax.plot(ms, means_delres, "^-", label="DelRes", color="#2ca02c")
        ax.fill_between(
            ms,
            np.array(means_delres) - np.array(stds_delres),
            np.array(means_delres) + np.array(stds_delres),
            color="#2ca02c",
            alpha=0.2,
        )

        # Log Scale Y-Axis
        ax.set_yscale("log")

        # Formatting
        ax.set_title(f"Dimension $d={d}$")
        ax.set_xlabel("Number of Hyperplanes ($m$)")
        ax.set_ylabel("Time (s)")

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_memory_per_m_across_dim(experiments: list[Experiment], folder: str = "./data"):
    """
    Plot peak memory usage per number of hyperplanes for each dimension in a subplot.
    Aggregates runs from multiple experiments to calculate mean memory usage.
    """
    if not experiments:
        print("No experiments provided.")
        return

    # Data Structure: data[dim][m] = {'polypart': [], 'incenu': [], 'delres': []}
    data_by_dim = defaultdict(
        lambda: defaultdict(lambda: {"ppart": [], "incenu": [], "delres": []})
    )

    # 1. Load and Organize Data
    for exp in experiments:
        # We determine dimension from the experiment object
        d = exp.d

        # Load results from disk
        results = exp.load(folder=folder)
        if not results:
            continue

        for res in results:
            # We assume 'm_hyperplanes' is consistent within an experiment,
            # but getting it from the result dict is safer.
            m = res.get("m_hyperplanes")
            if m is None:
                continue

            # Append peak RAM usages (filtering out None values)
            if res.get("polypart_peak_ram_mb") is not None:
                data_by_dim[d][m]["ppart"].append(res["polypart_peak_ram_mb"])
            if res.get("incenu_peak_ram_mb") is not None:
                data_by_dim[d][m]["incenu"].append(res["incenu_peak_ram_mb"])
            if res.get("delres_peak_ram_mb") is not None:
                data_by_dim[d][m]["delres"].append(res["delres_peak_ram_mb"])

    # 2. Setup Plot
    dims = sorted(data_by_dim.keys())
    if not dims:
        print("No valid results found to plot.")
        return

    num_dims = len(dims)
    nrows = math.ceil((num_dims) ** 0.5)
    ncols = math.floor((num_dims + 1) ** 0.5)
    _, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3 * nrows),
        squeeze=False,
    )
    # squeeze=False ensures axs is always a 2D array [[ax1, ax2...]] even if 1 subplot

    # 3. Plot for each Dimension
    for i, d in enumerate(dims):
        row = i // ncols
        col = i % ncols
        ax = axs[row][col]
        m_dict = data_by_dim[d]
        # Sort by m (x-axis)
        sorted_ms = sorted(m_dict.keys())
        # Arrays for plotting
        ms = []
        means_ppart = []
        stds_ppart = []
        means_incenu = []
        stds_incenu = []
        means_delres = []
        stds_delres = []
        for m in sorted_ms:
            ram_usages = m_dict[m]
            # Only include points where we have data
            if ram_usages["ppart"] or ram_usages["incenu"] or ram_usages["delres"]:
                ms.append(m)
                # Calculate means and stds, defaulting to NaN if list is empty
                means_ppart.append(
                    np.mean(ram_usages["ppart"]) if ram_usages["ppart"] else np.nan
                )
                stds_ppart.append(
                    np.std(ram_usages["ppart"]) if ram_usages["ppart"] else np.nan
                )
                means_incenu.append(
                    np.mean(ram_usages["incenu"]) if ram_usages["incenu"] else np.nan
                )
                stds_incenu.append(
                    np.std(ram_usages["incenu"]) if ram_usages["incenu"] else np.nan
                )
                means_delres.append(
                    np.mean(ram_usages["delres"]) if ram_usages["delres"] else np.nan
                )
                stds_delres.append(
                    np.std(ram_usages["delres"]) if ram_usages["delres"] else np.nan
                )
            else:
                print(f"Warning: No RAM data for d={d}, m={m} in experiment")
                continue
        # Plot Lines with Shaded Std Dev
        ax.plot(ms, means_ppart, "o-", label="PolyPart", color="#1f77b4")
        ax.fill_between(
            ms,
            np.array(means_ppart) - np.array(stds_ppart),
            np.array(means_ppart) + np.array(stds_ppart),
            color="#1f77b4",
            alpha=0.2,
        )
        ax.plot(ms, means_incenu, "s-", label="IncEnu", color="#ff7f0e")
        ax.fill_between(
            ms,
            np.array(means_incenu) - np.array(stds_incenu),
            np.array(means_incenu) + np.array(stds_incenu),
            color="#ff7f0e",
            alpha=0.2,
        )
        ax.plot(ms, means_delres, "^-", label="DelRes", color="#2ca02c")
        ax.fill_between(
            ms,
            np.array(means_delres) - np.array(stds_delres),
            np.array(means_delres) + np.array(stds_delres),
            color="#2ca02c",
            alpha=0.2,
        )
        # Log Scale Y-Axis
        ax.set_yscale("log")
        # Formatting
        ax.set_title(f"Dimension $d={d}$")
        ax.set_xlabel("Number of Hyperplanes ($m$)")
        ax.set_ylabel("Peak RAM Usage (MB)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    plt.tight_layout()
    plt.show()


# Moduli Hyperplanes Lookup Table
_MODULI_HYPERPLANES_LOOKUP = {
    (1, 2): 0,
    (1, 3): 1,
    (1, 4): 3,
    (1, 5): 11,
    (1, 6): 21,
    (1, 7): 65,
    (1, 8): 129,
    (1, 9): 307,
    (2, 2): 1,
    (2, 3): 9,
    (2, 4): 41,
    (2, 5): 215,
    (2, 6): 799,
    (2, 7): 3927,
    (2, 8): 15049,
    (3, 2): 4,
    (3, 3): 45,
    (3, 4): 344,
    (3, 5): 3075,
    (3, 6): 21379,
    (4, 2): 12,
    (4, 3): 189,
    (4, 4): 2540,
    (4, 5): 39875,
    (5, 2): 32,
    (5, 3): 729,
    (5, 4): 17840,
    (6, 2): 80,
    (6, 3): 2673,
    (7, 2): 192,
    (8, 2): 448,
}


# def _get_moduli_aggregated_stats(experiment_obj: Experiment, folder):
#     """
#     Extract aggregated timing and memory statistics from a single experiment.

#     Returns a dict with mean/std for each algorithm's time and memory usage,
#     plus the total number of results.
#     """
#     results = experiment_obj.load(folder=folder)

#     def calc_stat(data):
#         return (np.mean(data), np.std(data)) if data else (0, 0)

#     # Extraction with safety filters
#     poly_times = [
#         r["polypart_time"] for r in results if r.get("polypart_time") is not None
#     ]
#     inc_times = [r["incenu_time"] for r in results if r.get("incenu_time") is not None]
#     del_times = [r["delres_time"] for r in results if r.get("delres_time") is not None]

#     poly_mems = [
#         r.get("polypart_peak_ram_mb")
#         for r in results
#         if r.get("polypart_peak_ram_mb") is not None
#     ]
#     inc_mems = [
#         r.get("incenu_peak_ram_mb")
#         for r in results
#         if r.get("incenu_peak_ram_mb") is not None
#     ]
#     del_mems = [
#         r.get("delres_peak_ram_mb")
#         for r in results
#         if r.get("delres_peak_ram_mb") is not None
#     ]

#     return {
#         "poly_time": calc_stat(poly_times),
#         "poly_mem": calc_stat(poly_mems),
#         "inc_time": calc_stat(inc_times),
#         "inc_mem": calc_stat(inc_mems),
#         "del_time": calc_stat(del_times),
#         "del_mem": calc_stat(del_mems),
#         "num_results": len(results),
#     }


# def _process_moduli_experiments(exp_list, setting_type, folder):
#     """
#     Process a list of moduli experiments and aggregate statistics.

#     Args:
#         exp_list: List of Experiment objects
#         setting_type: Either "n1" (fixed n=1) or "r2" (fixed r=2)
#         folder: Data folder path

#     Returns:
#         Dictionary with aggregated data ready for plotting
#     """
#     data = {
#         "x_vals": [],
#         "labels": [],
#         "poly_t_m": [],
#         "poly_t_s": [],
#         "poly_m_m": [],
#         "inc_t_m": [],
#         "inc_t_s": [],
#         "inc_m_m": [],
#         "del_t_m": [],
#         "del_t_s": [],
#         "del_m_m": [],
#         "num_results": [],
#     }
#     sorted_exps = sorted(exp_list, key=lambda e: e.d)

#     for exp in sorted_exps:
#         d = exp.d
#         n, r = (1, d + 1) if setting_type == "n1" else (d, 2)
#         m = _MODULI_HYPERPLANES_LOOKUP.get((n, r), "?")
#         stats = _get_moduli_aggregated_stats(exp, folder)

#         data["x_vals"].append(r if setting_type == "n1" else n)
#         # Vertical stack for labels
#         label = f"n={n}, r={r}\nd={d}\nm={m}"
#         data["labels"].append(label)

#         for alg in ["poly", "inc", "del"]:
#             data[f"{alg}_t_m"].append(stats[f"{alg}_time"][0])
#             data[f"{alg}_t_s"].append(stats[f"{alg}_time"][1])
#             data[f"{alg}_m_m"].append(stats[f"{alg}_mem"][0])

#         data["num_results"].append(stats["num_results"])

#     # Check if the number of results is consistent across experiments
#     if len(set(data["num_results"])) != 1:
#         raise ValueError(
#             f"Inconsistent number of results across experiments: {data['num_results']}"
#         )

#     return data


# def _create_moduli_integrated_figure(data, title):
#     """
#     Create an integrated figure showing both time (lines) and memory (bars).

#     Args:
#         data: Processed experiment data dictionary
#         title: Figure title string
#     """
#     _, ax1 = plt.subplots(figsize=(8, 5))

#     indices = np.arange(len(data["x_vals"]))
#     width = 0.22  # Padding between grouped bars

#     styles = {
#         "poly": {"c": "#1f77b4", "marker": "o-", "label": "PolyPart"},
#         "inc": {"c": "#ff7f0e", "marker": "s-", "label": "IncEnu"},
#         "del": {"c": "#2ca02c", "marker": "^-", "label": "DelRes"},
#     }

#     # --- Axis 2: Memory (Bars) ---
#     ax2 = ax1.twinx()
#     for i, (alg, s) in enumerate(styles.items()):
#         ax2.bar(
#             indices + (i - 1) * width,
#             data[f"{alg}_m_m"],
#             width,
#             color=s["c"],
#             alpha=0.3,
#             label=f"{s['label']} Memory",
#         )

#     # Scaling to keep bars in the bottom 40% of the plot
#     all_mem_vals = data["poly_m_m"] + data["inc_m_m"] + data["del_m_m"]
#     max_mem = max(all_mem_vals) if all_mem_vals else 1
#     ax2.set_ylim(0, max_mem * 2.5)
#     ax2.set_ylabel("Peak RAM Usage (MB)", color="grey", alpha=0.7)
#     ax2.tick_params(axis="y", labelcolor="grey")

#     # --- Axis 1: Time (Lines) ---
#     for alg, s in styles.items():
#         m_val = np.array(data[f"{alg}_t_m"])
#         std_val = np.array(data[f"{alg}_t_s"])
#         ax1.plot(
#             indices, m_val, s["marker"], color=s["c"], label=s["label"], linewidth=2
#         )
#         ax1.fill_between(
#             indices, m_val - std_val, m_val + std_val, color=s["c"], alpha=0.15
#         )

#     ax1.set_yscale("log")
#     ax1.set_ylabel("Computation Time (s)")
#     ax1.set_xticks(indices)
#     ax1.set_xticklabels(data["labels"], fontsize=9)
#     ax1.grid(True, linestyle="--", alpha=0.6)

#     # Add number of runs to the title
#     num_runs = data["num_results"][0]
#     ax1.set_title(f"{title} (Runs={num_runs})", fontsize=14, pad=20)

#     # Combined legend
#     lines, l_labels = ax1.get_legend_handles_labels()
#     bars, b_labels = ax2.get_legend_handles_labels()
#     ax1.legend(
#         lines + bars,
#         l_labels + b_labels,
#         loc="upper left",
#         ncol=2,
#         fontsize="small",
#     )

#     plt.tight_layout()
#     plt.show()


def _load_runtime_log(path: Path) -> dict:
    """
    Load and parse runtime.log file to extract timing and memory data.

    Returns a dict with lists of times and memory for each algorithm found.
    """
    log_path = path / "runtime.log"
    if not log_path.exists():
        raise FileNotFoundError(f"No runtime.log found in {path}")

    data = {
        "ppart": {"times": [], "mems": []},
        "incenu": {"times": [], "mems": []},
        "delres": {"times": [], "mems": []},
    }

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = ast.literal_eval(line)
                algo = entry.get("algo")
                if algo in data:
                    if entry.get("time") is not None:
                        data[algo]["times"].append(entry["time"])
                    if entry.get("peak_ram_mb") is not None:
                        data[algo]["mems"].append(entry["peak_ram_mb"])
            except (ValueError, SyntaxError):
                continue

    return data


def _get_moduli_aggregated_stats(experiment_obj, folder):
    """
    Extract aggregated timing and memory statistics from a single experiment.

    Returns a dict with mean/std for each algorithm's time and memory usage,
    plus the total number of results and a flag indicating if data came from runtime.log.
    """
    results = experiment_obj.load(folder=folder)
    from_runtime_log = False
    missing_algo = None

    def calc_stat(data):
        return (np.mean(data), np.std(data)) if data else (0, 0)

    if results:
        # Normal case: we have .json results
        poly_times = [
            r["polypart_time"] for r in results if r.get("polypart_time") is not None
        ]
        inc_times = [
            r["incenu_time"] for r in results if r.get("incenu_time") is not None
        ]
        del_times = [
            r["delres_time"] for r in results if r.get("delres_time") is not None
        ]

        poly_mems = [
            r.get("polypart_peak_ram_mb")
            for r in results
            if r.get("polypart_peak_ram_mb") is not None
        ]
        inc_mems = [
            r.get("incenu_peak_ram_mb")
            for r in results
            if r.get("incenu_peak_ram_mb") is not None
        ]
        del_mems = [
            r.get("delres_peak_ram_mb")
            for r in results
            if r.get("delres_peak_ram_mb") is not None
        ]

        num_results = len(results)
    else:
        # Fallback: load from runtime.log
        path = Path(folder) / experiment_obj.dirname()
        runtime_data = _load_runtime_log(path)
        from_runtime_log = True

        poly_times = runtime_data["ppart"]["times"]
        inc_times = runtime_data["incenu"]["times"]
        del_times = runtime_data["delres"]["times"]

        poly_mems = runtime_data["ppart"]["mems"]
        inc_mems = runtime_data["incenu"]["mems"]
        del_mems = runtime_data["delres"]["mems"]

        # Determine which algorithm is missing (the one currently running)
        # The missing algo will have fewer entries or none at all
        counts = {
            "ppart": len(poly_times),
            "incenu": len(inc_times),
            "delres": len(del_times),
        }

        if counts:
            max_count = max(counts.values())
            # Find algorithms with fewer runs (likely the one that's still running or hasn't started)
            for algo, count in counts.items():
                if count < max_count or count == 0:
                    missing_algo = algo
                    break

        # Use the max count as num_results for runtime.log case
        num_results = max(counts.values()) if counts else 0

    return {
        "poly_time": calc_stat(poly_times),
        "poly_mem": calc_stat(poly_mems),
        "inc_time": calc_stat(inc_times),
        "inc_mem": calc_stat(inc_mems),
        "del_time": calc_stat(del_times),
        "del_mem": calc_stat(del_mems),
        "num_results": num_results,
        "from_runtime_log": from_runtime_log,
        "missing_algo": missing_algo,
    }


def _process_moduli_experiments(exp_list, setting_type, folder):
    """
    Process a list of moduli experiments and aggregate statistics.

    Args:
        exp_list: List of Experiment objects
        setting_type: Either "n1" (fixed n=1) or "r2" (fixed r=2)
        folder: Data folder path

    Returns:
        Dictionary with aggregated data ready for plotting
    """
    data = {
        "x_vals": [],
        "labels": [],
        "poly_t_m": [],
        "poly_t_s": [],
        "poly_m_m": [],
        "inc_t_m": [],
        "inc_t_s": [],
        "inc_m_m": [],
        "del_t_m": [],
        "del_t_s": [],
        "del_m_m": [],
        "num_results": [],
        "from_runtime_log": [],
        "missing_algo": [],
    }
    sorted_exps = sorted(exp_list, key=lambda e: e.d)

    for exp in sorted_exps:
        d = exp.d
        n, r = (1, d + 1) if setting_type == "n1" else (d, 2)
        m = _MODULI_HYPERPLANES_LOOKUP.get((n, r), "?")
        stats = _get_moduli_aggregated_stats(exp, folder)

        data["x_vals"].append(r if setting_type == "n1" else n)
        # Vertical stack for labels
        label = f"n={n}, r={r}\nd={d}\nm={m}"
        data["labels"].append(label)

        for alg in ["poly", "inc", "del"]:
            data[f"{alg}_t_m"].append(stats[f"{alg}_time"][0])
            data[f"{alg}_t_s"].append(stats[f"{alg}_time"][1])
            data[f"{alg}_m_m"].append(stats[f"{alg}_mem"][0])

        data["num_results"].append(stats["num_results"])
        data["from_runtime_log"].append(stats["from_runtime_log"])
        data["missing_algo"].append(stats["missing_algo"])

    # Check if the number of results is consistent across experiments (now just a warning)
    unique_counts = set(data["num_results"])
    if len(unique_counts) != 1:
        warnings.warn(
            f"Inconsistent number of results across experiments: {data['num_results']}"
        )

    return data


def _create_moduli_integrated_figure(data, title):
    """
    Create an integrated figure showing both time (lines) and memory (bars).

    For experiments loaded from runtime.log with a missing algorithm,
    the line will stop at the previous point and show a star marker.

    Args:
        data: Processed experiment data dictionary
        title: Figure title string
    """
    _, ax1 = plt.subplots(figsize=(8, 5))

    indices = np.arange(len(data["x_vals"]))
    width = 0.22  # Padding between grouped bars

    styles = {
        "poly": {
            "c": "#1f77b4",
            "marker": "o",
            "label": "PolyPart",
            "algo_key": "ppart",
        },
        "inc": {"c": "#ff7f0e", "marker": "s", "label": "IncEnu", "algo_key": "incenu"},
        "del": {"c": "#2ca02c", "marker": "^", "label": "DelRes", "algo_key": "delres"},
    }

    # --- Axis 2: Memory (Bars) ---
    ax2 = ax1.twinx()
    for i, (alg, s) in enumerate(styles.items()):
        ax2.bar(
            indices + (i - 1) * width,
            data[f"{alg}_m_m"],
            width,
            color=s["c"],
            alpha=0.3,
            label=f"{s['label']} Memory",
        )

    # Scaling to keep bars in the bottom 40% of the plot
    all_mem_vals = data["poly_m_m"] + data["inc_m_m"] + data["del_m_m"]
    max_mem = max(all_mem_vals) if all_mem_vals else 1
    ax2.set_ylim(0, max_mem * 2.5)
    ax2.set_ylabel("Peak RAM Usage (MB)", color="grey", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="grey")

    # --- Axis 1: Time (Lines) ---
    for alg, s in styles.items():
        m_val = np.array(data[f"{alg}_t_m"])
        std_val = np.array(data[f"{alg}_t_s"])

        # Check if the last experiment has this algorithm missing
        last_idx = len(indices) - 1
        last_missing = data["missing_algo"][last_idx]
        last_from_log = data["from_runtime_log"][last_idx]

        if last_from_log and last_missing == s["algo_key"]:
            # This algorithm is missing in the last experiment
            # Plot up to the second-to-last point with normal markers
            if last_idx > 0:
                ax1.plot(
                    indices[: last_idx - 1],
                    m_val[: last_idx - 1],
                    f"{s['marker']}-",
                    color=s["c"],
                    label=s["label"],
                    linewidth=2,
                )
                ax1.fill_between(
                    indices[:last_idx],
                    m_val[:last_idx] - std_val[:last_idx],
                    m_val[:last_idx] + std_val[:last_idx],
                    color=s["c"],
                    alpha=0.15,
                )
                # Add a line segment to the last valid point without marker
                ax1.plot(
                    indices[last_idx - 2 : last_idx],
                    m_val[last_idx - 2 : last_idx],
                    "-",
                    color=s["c"],
                    linewidth=2,
                )
                # Add a star at the last valid point to indicate "in progress"
                ax1.plot(
                    indices[last_idx - 1],
                    m_val[last_idx - 1],
                    "x",
                    color=s["c"],
                    markersize=10,
                    # markeredgecolor="black",
                    markeredgewidth=2,
                    zorder=10,
                )
        else:
            # Normal case: plot all points
            ax1.plot(
                indices,
                m_val,
                f"{s['marker']}-",
                color=s["c"],
                label=s["label"],
                linewidth=2,
                markersize=7 if s["marker"] == "^" else 6,
            )
            ax1.fill_between(
                indices, m_val - std_val, m_val + std_val, color=s["c"], alpha=0.15
            )

    ax1.set_yscale("log")
    ax1.set_ylabel("Computation Time (s)")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(data["labels"], fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Add number of runs to the title (use max for display when inconsistent)
    num_runs = max(data["num_results"]) if data["num_results"] else 0
    title_suffix = f" (Runs={num_runs})"

    # Indicate if last experiment is from runtime.log
    if data["from_runtime_log"][-1]:
        title_suffix += " [Last: In Progress]"

    # ax1.set_title(f"{title}{title_suffix}", fontsize=14, pad=20)

    # Combined legend
    lines, l_labels = ax1.get_legend_handles_labels()
    bars, b_labels = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + bars,
        l_labels + b_labels,
        loc="upper left",
        ncol=2,
        fontsize="small",
    )

    plt.tight_layout()
    plt.savefig(f"./figures/moduli_{title}_report.pdf")
    plt.show()


def plot_moduli_n1_report(experiments, folder="./data"):
    """
    Generate integrated figure for Moduli Spaces with fixed n=1.
    Shows Time (lines) and Peak Memory (bars) on the same plot.

    Args:
        experiments: List of Experiment objects with fixed n=1
        folder: Data folder path
    """
    data = _process_moduli_experiments(experiments, "n1", folder)
    _create_moduli_integrated_figure(data, "n1")


def plot_moduli_r2_report(experiments, folder="./data"):
    """
    Generate integrated figure for Moduli Spaces with fixed r=2.
    Shows Time (lines) and Peak Memory (bars) on the same plot.

    Args:
        experiments: List of Experiment objects with fixed r=2
        folder: Data folder path
    """
    data = _process_moduli_experiments(experiments, "r2", folder)
    _create_moduli_integrated_figure(data, "r2")


def plot_permutahedron_report(experiments, folder):
    """
    Generates an integrated figure for the Permutahedron experiment.
    Shows Time (lines) and Peak Memory (bars) on the same plot.
    """

    def get_aggregated_stats(experiment_obj):
        results = experiment_obj.load(folder=folder)

        def calc_stat(data):
            return (np.mean(data), np.std(data)) if data else (0, 0)

        # Extraction with safety filters
        poly_times = [
            r["polypart_time"] for r in results if r.get("polypart_time") is not None
        ]
        inc_times = [
            r["incenu_time"] for r in results if r.get("incenu_time") is not None
        ]
        del_times = [
            r["delres_time"] for r in results if r.get("delres_time") is not None
        ]

        poly_mems = [
            r.get("polypart_peak_ram_mb")
            for r in results
            if r.get("polypart_peak_ram_mb") is not None
        ]
        inc_mems = [
            r.get("incenu_peak_ram_mb")
            for r in results
            if r.get("incenu_peak_ram_mb") is not None
        ]
        del_mems = [
            r.get("delres_peak_ram_mb")
            for r in results
            if r.get("delres_peak_ram_mb") is not None
        ]

        return {
            "poly_time": calc_stat(poly_times),
            "poly_mem": calc_stat(poly_mems),
            "inc_time": calc_stat(inc_times),
            "inc_mem": calc_stat(inc_mems),
            "del_time": calc_stat(del_times),
            "del_mem": calc_stat(del_mems),
            "num_results": len(results),
        }

    # --- Preprocessing ---
    data = {
        "dims": [],
        "poly_t_m": [],
        "poly_t_s": [],
        "poly_m_m": [],
        "inc_t_m": [],
        "inc_t_s": [],
        "inc_m_m": [],
        "del_t_m": [],
        "del_t_s": [],
        "del_m_m": [],
        "num_results": [],
    }

    # Sort experiments by dimension d
    sorted_exps = sorted(experiments, key=lambda e: e.d)

    for exp in sorted_exps:
        stats = get_aggregated_stats(exp)
        data["dims"].append(exp.d)

        for alg in ["poly", "inc", "del"]:
            data[f"{alg}_t_m"].append(stats[f"{alg}_time"][0])
            data[f"{alg}_t_s"].append(stats[f"{alg}_time"][1])
            data[f"{alg}_m_m"].append(stats[f"{alg}_mem"][0])

        data["num_results"].append(stats["num_results"])

    # Check if the number of runs is consistent across experiments
    if len(set(data["num_results"])) != 1:
        raise ValueError(
            f"Inconsistent number of runs across experiments: {data['num_results']}"
        )

    num_runs = data["num_results"][0]

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(8, 5))

    indices = np.arange(len(data["dims"]))
    width = 0.22  # Width for grouped bars

    styles = {
        "poly": {"c": "#1f77b4", "marker": "o-", "label": "PolyPart"},
        "inc": {"c": "#ff7f0e", "marker": "s-", "label": "IncEnu"},
        "del": {"c": "#2ca02c", "marker": "^-", "label": "DelRes"},
    }

    # Axis 2: Memory Bars (Background)
    ax2 = ax1.twinx()
    for i, (alg, s) in enumerate(styles.items()):
        ax2.bar(
            indices + (i - 1) * width,
            data[f"{alg}_m_m"],
            width,
            color=s["c"],
            alpha=0.3,
            label=f"{s['label']} Memory",
        )

    all_mem_vals = data["poly_m_m"] + data["inc_m_m"] + data["del_m_m"]
    max_mem = max(all_mem_vals) if all_mem_vals else 1
    ax2.set_ylim(0, max_mem * 2.5)
    ax2.set_ylabel("Peak RAM Usage (MB)", color="grey", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="grey")

    # Axis 1: Time Lines (Foreground)
    for alg, s in styles.items():
        m_val = np.array(data[f"{alg}_t_m"])
        std_val = np.array(data[f"{alg}_t_s"])
        ax1.plot(
            indices, m_val, s["marker"], color=s["c"], label=s["label"], linewidth=2
        )
        ax1.fill_between(
            indices, m_val - std_val, m_val + std_val, color=s["c"], alpha=0.15
        )

    ax1.set_yscale("log")
    ax1.set_ylabel("Computation Time (s)")
    ax1.set_xlabel("Dimension ($d$)")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(
        [f"d={d}\nm={d * (d - 1) // 2}" for d in data["dims"]], fontsize=9
    )
    ax1.grid(True, linestyle="--", alpha=0.6)
    # ax1.set_title(
    #     f"Permutahedron Arrangement (Runs={num_runs})",
    #     fontsize=12,
    #     pad=15,
    # )

    # Combined legend
    lines, l_labels = ax1.get_legend_handles_labels()
    bars, b_labels = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + bars, l_labels + b_labels, loc="upper left", ncol=2, fontsize="small"
    )

    plt.tight_layout()
    # save fig as pdf in ./figures
    plt.savefig("./figures/permutahedron_report.pdf")
    plt.show()


def build_results_table(experiments: list, folder: str = "./data") -> list[dict]:
    """
    Aggregates results from experiments into a structured list of dictionaries.
    Each dictionary represents a row in the final table.
    """
    table_rows = []

    for exp in experiments:
        results = exp.load(folder=folder)
        if not results:
            print(f"Skipping {exp.dirname()}: No results found.")
            continue

        # 1. Extract m_hyperplanes (assert if inconsistent)
        m_values = {
            res["m_hyperplanes"]
            for res in results
            if res.get("m_hyperplanes") is not None
        }
        assert len(m_values) == 1, (
            f"Inconsistent number of hyperplanes across runs in {exp.dirname()}"
        )

        row = {
            "name": exp.A_class.name,
            "d": exp.d,
            "m": m_values.pop(),
        }

        # 2. Extract Statistics for Regions (|C|)
        regions = {
            res["num_regions"] for res in results if res.get("num_regions") is not None
        }
        assert len(regions) == 1, (
            f"Inconsistent number of regions across runs in {exp.dirname()}"
        )
        row["num_regions"] = regions.pop()

        # 3. Extract Stats for Algorithms
        for algo in ["polypart", "incenu", "delres"]:
            # Time
            times = [
                r[f"{algo}_time"] for r in results if r.get(f"{algo}_time") is not None
            ]
            row[f"{algo}_time_mean"] = np.mean(times) if times else None
            row[f"{algo}_time_std"] = np.std(times) if times else None

            # RAM (Handle key variations if necessary, assuming f"{algo}_peak_ram_mb")
            rams = [
                r.get(f"{algo}_peak_ram_mb")
                for r in results
                if r.get(f"{algo}_peak_ram_mb") is not None
            ]
            row[f"{algo}_ram_mean"] = np.mean(rams) if rams else None
            row[f"{algo}_ram_std"] = np.std(rams) if rams else None

        table_rows.append(row)

    # Sort by dimension 'd' then by 'm' for cleaner presentation
    table_rows.sort(key=lambda x: (x["d"], x["m"]))

    return table_rows


def results_to_latex(table_data: list[dict], baseline_ram: float = 44.0) -> str:
    """
    Generates a publication-quality LaTeX table.
    - Subtracts baseline RAM to show 'Net RAM'.
    - Uses human-readable time units (ms, s, min).
    - Hides std dev if it is 0.00 or negligible (< 1% of mean).
    """
    if not table_data:
        return "% No data available"

    def fmt_time(mean, std):
        if mean is None:
            return "N/A"

        # 1. Determine Unit
        if mean < 0.1:  # < 1s -> use milliseconds
            val = mean * 1000
            s_std = std * 1000
            unit = r"\,ms"
        elif mean < 60:  # < 1m -> use seconds
            val = mean
            s_std = std
            unit = r"\,s"
        else:  # > 1m -> use minutes
            val = mean / 60
            s_std = std / 60
            unit = r"\,min"

        # 2. Format Value
        # Hide std if it's effectively zero or super small relative to mean
        # if s_std == 0 or (s_std / val) < 0.01:
        #     return f"${val:.1f}{unit}$"

        return f"${val:.1f} \pm {s_std:.1f}{unit}$"

    def fmt_ram(mean, std):
        if mean is None:
            return "N/A"

        # Subtract Baseline
        net_mean = max(0, mean - baseline_ram)

        # If net RAM is tiny (< 1MB), just show <1
        if net_mean < 1.0:
            return "$< 1$"

        # Hide std if it's effectively zero or super small relative to mean
        if std / mean < 0.1:
            return f"${int(net_mean)}$"

        return f"${int(net_mean)} \pm {std:.2f}$"

    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    # Slightly reduce font size if needed
    latex_lines.append(r"\small")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")

    # Columns
    latex_lines.append(r"\begin{tabular}{l c c r | r r | r r | r r}")
    latex_lines.append(r"\hline")

    # Header
    latex_lines.append(
        r"\multirow{2}{*}{\textbf{Name}} & \multirow{2}{*}{$d$} & \multirow{2}{*}{$m$} & \multirow{2}{*}{$|\mathcal{C}|$} & "
        r"\multicolumn{2}{c|}{\textbf{PolyPart}} & \multicolumn{2}{c|}{\textbf{IncEnu}} & \multicolumn{2}{c}{\textbf{DelRes}} \\"
    )
    latex_lines.append(
        r" & & & & Time & RAM (MB) & Time & RAM (MB) & Time & RAM (MB) \\"
    )
    latex_lines.append(r"\hline")

    for row in table_data:
        name = row["name"].replace("_", r"\_")
        d = row["d"]
        m = int(row["m"])

        # Format Regions (Comma separated for thousands)
        regions = f"{int(row['num_regions']):,}"

        # Algorithm Stats
        ppart_t = fmt_time(row["polypart_time_mean"], row["polypart_time_std"])
        ppart_r = fmt_ram(row["polypart_ram_mean"], row["polypart_ram_std"])

        inc_t = fmt_time(row["incenu_time_mean"], row["incenu_time_std"])
        inc_r = fmt_ram(row["incenu_ram_mean"], row["incenu_ram_std"])

        del_t = fmt_time(row["delres_time_mean"], row["delres_time_std"])
        del_r = fmt_ram(row["delres_ram_mean"], row["delres_ram_std"])

        line = (
            f"{name} & {d} & {m} & {regions} & "
            f"{ppart_t} & {ppart_r} & "
            f"{inc_t} & {inc_r} & "
            f"{del_t} & {del_r} \\\\"
        )
        latex_lines.append(line)

    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}%")
    latex_lines.append(r"}")
    latex_lines.append(
        r"\caption{Comparison of execution time and net peak memory usage (baseline $\approx "
        + f"{baseline_ram:.1f}"
        + r"$ MB removed).}"
    )
    latex_lines.append(r"\label{tab:alg_comparison}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)
