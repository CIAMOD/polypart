# polypart/experiments/__init__.py
import warnings

from .core import (
    ArrangementClass,
    Experiment,
    PolytopeClass,
)
from .report import (
    plot_experiment_summary,
    plot_times_per_m_across_dim,
    print_results_summary,
)
from .runner import (
    ALGORITHMS,
    run_experiments,
    run_single_experiment,
)

# ignore RuntimeWarning about sys.modules during imports
warnings.filterwarnings(
    "ignore", message=".*found in sys.modules after import of package.*"
)
