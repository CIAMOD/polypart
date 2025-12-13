import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

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


# def plot_experiment_summary(experiment: Experiment, folder: str = "./data"):
#     """
#     Summary plot of all runs of an experiment, showing overall statistics
#     and moments plots per depth.
#     """
#     all_results = experiment.load(folder=folder)
#     if len(all_results) == 0:
#         print(f"No results found for experiment {experiment.dirname()}")
#         return

#     # Filter all_results without per_depth_moments_vertices
#     all_results = [
#         res
#         for res in all_results
#         if res["ppart_stats"] is not None
#         and "per_depth_moments_vertices" in res["ppart_stats"]
#     ]
#     if len(all_results) == 0:
#         print(f"No per-depth moments data found for experiment {experiment.dirname()}")
#         return

#     # Collect times per depth
#     alphas = all_results[-1]["ppart_stats"]["per_depth_moments_vertices"].keys()
#     all_per_depth_moments_candidates = {alpha: {} for alpha in alphas}
#     all_per_depth_moments_inequalities = {alpha: {} for alpha in alphas}
#     all_per_depth_moments_vertices = {alpha: {} for alpha in alphas}
#     for res in all_results:
#         ppart_stats = res["ppart_stats"]
#         moments_candidates = ppart_stats["per_depth_moments_candidates"]
#         moments_inequalities = ppart_stats["per_depth_moments_inequalities"]
#         moments_vertices = ppart_stats["per_depth_moments_vertices"]
#         for alpha in alphas:
#             if alpha in moments_candidates:
#                 for depth, moment in moments_candidates[alpha].items():
#                     if depth not in all_per_depth_moments_candidates[alpha]:
#                         all_per_depth_moments_candidates[alpha][depth] = []
#                     all_per_depth_moments_candidates[alpha][depth].append(moment)
#             else:
#                 continue
#             if alpha in moments_inequalities:
#                 for depth, moment in moments_inequalities[alpha].items():
#                     if depth not in all_per_depth_moments_inequalities[alpha]:
#                         all_per_depth_moments_inequalities[alpha][depth] = []
#                     all_per_depth_moments_inequalities[alpha][depth].append(moment)
#             else:
#                 continue
#             if alpha in moments_vertices:
#                 for depth, moment in moments_vertices[alpha].items():
#                     if depth not in all_per_depth_moments_vertices[alpha]:
#                         all_per_depth_moments_vertices[alpha][depth] = []
#                     all_per_depth_moments_vertices[alpha][depth].append(moment)
#             else:
#                 continue

#     # Data Extraction
#     # -------------------------------------------------------------------------
#     # Extract raw lists, ensuring we handle None values safely
#     regions_list = [
#         res["num_regions"] for res in all_results if res.get("num_regions") is not None
#     ]

#     # Times (PolyPart, IncEnu, and the newly requested DelRes)
#     polypart_times = [
#         res["polypart_time"]
#         for res in all_results
#         if res.get("polypart_time") is not None
#     ]
#     incenu_times = [
#         res["incenu_time"] for res in all_results if res.get("incenu_time") is not None
#     ]
#     delres_times = [
#         res["delres_time"] for res in all_results if res.get("delres_time") is not None
#     ]

#     # Calculate Region Statistics for the subtitle
#     avg_regions = np.mean(regions_list) if regions_list else 0
#     std_regions = np.std(regions_list) if regions_list else 0

#     # Plotting
#     # -------------------------------------------------------------------------
#     # 2x2 Grid: Efficient use of space
#     fig, axs = plt.subplots(
#         2, 2, figsize=(18, 12)
#     )  # Increased height slightly for better aspect ratio

#     fig.suptitle(
#         f"Experiment Summary: {experiment.dirname()} (runs={len(all_results)})",
#         fontsize=18,
#         y=0.95,
#     )

#     # --- 1. Top-Left: Time Box Plot (Sorted) ---
#     ax_time = axs[0, 0]

#     # Structure data for sorting: (Data List, Label, Color)
#     raw_time_data = [
#         (polypart_times, "PolyPart", "#1f77b4"),  # Blue
#         (incenu_times, "IncEnu", "#ff7f0e"),  # Orange
#         (delres_times, "DelRes", "#2ca02c"),  # Green
#     ]

#     # Sort by mean value (ascending) -> Fastest first
#     # Handle empty lists by assigning infinity so they drop to the bottom
#     sorted_time_data = sorted(
#         raw_time_data, key=lambda x: np.mean(x[0]) if len(x[0]) > 0 else float("inf")
#     )

#     # Unpack sorted data
#     plot_data = [item[0] for item in sorted_time_data]
#     plot_labels = [item[1] for item in sorted_time_data]
#     plot_colors = [item[2] for item in sorted_time_data]

#     # Create horizontal boxplot
#     bplot = ax_time.boxplot(
#         plot_data,
#         vert=False,
#         patch_artist=True,
#         labels=plot_labels,
#         widths=0.6,
#         # Show the Mean Line
#         showmeans=True,
#         meanline=True,
#         # Style the Mean
#         meanprops={"color": "black", "linewidth": 2.5, "linestyle": "-"},
#         # Style the Median
#         medianprops={"color": "red", "linewidth": 1, "linestyle": "--"},
#     )

#     # Apply colors to the sorted boxes
#     for patch, color in zip(bplot["boxes"], plot_colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.6)

#     # Invert Y-axis so the first item (fastest/lowest mean) is at the TOP
#     ax_time.invert_yaxis()

#     # Add legend for mean and median lines
#     ax_time.plot([], [], color="black", linewidth=2.5, label="Mean")
#     ax_time.plot([], [], color="red", linewidth=1, linestyle="--", label="Median")
#     ax_time.legend(loc="upper right", fontsize="small")

#     # Set Title with Region Stats
#     region_stats_str = f"Avg regions: ${int(avg_regions)} \pm {int(std_regions)}$"
#     ax_time.set_title(f"Execution Times ({region_stats_str})", fontsize=14)
#     ax_time.set_xlabel("Time (seconds)")
#     ax_time.grid(True, axis="x", linestyle="--", alpha=0.5)

#     # --- 2, 3, 4. The Moment Plots ---
#     # Define configuration to map data to specific subplots
#     # Order: Top-Right -> Bottom-Left -> Bottom-Right
#     moment_configs = [
#         (all_per_depth_moments_candidates, "$|A_k|$", axs[0, 1]),
#         (all_per_depth_moments_inequalities, "$|P_k|$", axs[1, 0]),
#         (all_per_depth_moments_vertices, "$|V_k|$", axs[1, 1]),
#     ]

#     for all_per_depth_moments, ylabel, ax in moment_configs:
#         for alpha in alphas:
#             # Sort depths numerically
#             depths = sorted(all_per_depth_moments[alpha].keys(), key=lambda x: int(x))

#             if not depths:
#                 continue

#             # Calculate stats
#             try:
#                 avg_moments = [
#                     np.mean(all_per_depth_moments[alpha][depth]) for depth in depths
#                 ]
#                 std_moments = [
#                     np.std(all_per_depth_moments[alpha][depth]) for depth in depths
#                 ]
#             except Exception:
#                 print(
#                     f"Warning: Failed to compute moments for alpha={alpha} in experiment {experiment.dirname()}"
#                 )
#                 continue

#             # Plot Line
#             ax.plot(
#                 depths,
#                 avg_moments,
#                 label=f"$\\alpha={alpha}$",
#                 marker="o",
#                 markersize=4,
#             )

#             # Plot Std Dev Shade
#             ax.fill_between(
#                 depths,
#                 np.array(avg_moments) - np.array(std_moments),
#                 np.array(avg_moments) + np.array(std_moments),
#                 alpha=0.15,
#             )

#         ax.set_title(f"Per-Depth Moments of {ylabel}", fontsize=14)
#         ax.set_xlabel("Depth")
#         ax.set_ylabel(ylabel)
#         ax.legend(loc="upper right", fontsize="small")
#         ax.grid(True, alpha=0.3)

#     # Final Layout Adjustment
#     # plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)
#     plt.tight_layout(pad=3.0)
#     plt.show()


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
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

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

    plt.tight_layout(pad=3.0)
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
