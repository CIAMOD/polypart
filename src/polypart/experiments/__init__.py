"""Experiment framework for benchmarking polypart algorithms."""

import warnings

from polypart.experiments.core import ArrangementClass, Experiment, PolytopeClass
from polypart.experiments.report import (
    plot_experiment_summary,
    plot_times_per_m_across_dim,
    print_results_summary,
)
from polypart.experiments.runner import (
    ALGORITHMS,
    run_experiments,
    run_single_experiment,
)
from polypart.experiments.stats import get_ppart_stats, print_ppart_stats

warnings.filterwarnings(
    "ignore", message=".*found in sys.modules after import of package.*"
)

__all__ = [
    "ALGORITHMS",
    "ArrangementClass",
    "Experiment",
    "get_ppart_stats",
    "plot_experiment_summary",
    "plot_times_per_m_across_dim",
    "PolytopeClass",
    "print_ppart_stats",
    "print_results_summary",
    "run_experiments",
    "run_single_experiment",
]
