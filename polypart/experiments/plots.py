import matplotlib.pyplot as plt
import numpy as np

from .core import Experiment


def plot_experiment_summary(experiment: Experiment):
    """
    Summary plot of all runs of an experiment, showing overall statistics
    and moments plots per depth.
    """
    all_results = experiment.load()
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
            try:
                avg_moments = [
                    np.mean(all_per_depth_moments[alpha][depth]) for depth in depths
                ]
                std_moments = [
                    np.std(all_per_depth_moments[alpha][depth]) for depth in depths
                ]
            except Exception:
                print(
                    f"Warning: Failed to compute moments for alpha={alpha} in experiment {experiment.dirname()}"
                )
                continue

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


def plot_time_per_m_across_d(experiments: list[Experiment]):
    """
    Plot time per number of hyperplanes for each dimension in a subplot.
    """
    all_results = [exp.load() for exp in experiments]
    if not all_results:
        print("No results found for the provided experiments.")
        return
    data = [
        {
            "polypart": np.mean(
                [r["polypart_time"] for r in res if r.get("polypart_time") is not None]
            ),
            "incenu": np.mean(
                [r["incenu_time"] for r in res if r.get("incenu_time") is not None]
            ),
            "delres": np.mean(
                [r["delres_time"] for r in res if r.get("delres_time") is not None]
            ),
            "m_hyperplanes": np.mean(
                [r["m_hyperplanes"] for r in res if r.get("m_hyperplanes") is not None]
            ),
            "dim": res[0]["dim"] if res and res[0].get("dim") is not None else None,
        }
        for res in all_results
    ]
    dims = sorted(set(t["dim"] for t in data if t["dim"] is not None))
    fig, axs = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5))
    if len(dims) == 1:
        axs = [axs]
    for ax, dim in zip(axs, dims):
        dim_data = [t for t in data if t["dim"] == dim]
        m_hyperplanes = [t["m_hyperplanes"] for t in dim_data]
        polypart_times = [t["polypart"] for t in dim_data]
        incenu_times = [t["incenu"] for t in dim_data]
        delres_times = [t["delres"] for t in dim_data]

        ax.plot(
            m_hyperplanes,
            polypart_times,
            label="PolyPart",
            marker="o",
            color="#1f77b4",
        )
        ax.plot(
            m_hyperplanes,
            incenu_times,
            label="IncEnu",
            marker="o",
            color="#ff7f0e",
        )
        ax.plot(
            m_hyperplanes,
            delres_times,
            label="DelRes",
            marker="o",
            color="#2ca02c",
        )

        ax.set_title(f"Time vs. #Hyperplanes (Dim={dim})", fontsize=14)
        ax.set_xlabel("# Hyperplanes")
        ax.set_ylabel("Time (seconds)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
