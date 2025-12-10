from polypart.experiments.core import (
    ArrangementClass,
    Experiment,
    PolytopeClass,
)

# --- DEFINE EXPERIMENTS LIST BELOW ---
experiments = [
    Experiment(
        PolytopeClass("moduli_r2"),
        ArrangementClass("moduli_r2"),
        d=d,
    )
    for d in [2, 3, 4, 5, 6, 7, 8]
]
