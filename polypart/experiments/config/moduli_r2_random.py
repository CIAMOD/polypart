from polypart.experiments.core import (
    ArrangementClass,
    Experiment,
    PolytopeClass,
)

# --- DEFINE EXPERIMENTS LIST BELOW ---
experiments = [
    Experiment(
        PolytopeClass("moduli_r2"),
        ArrangementClass("random", m, degen_ratio=0.0),
        d=d,
    )
    for d, m in zip([2, 3, 4, 5, 6, 7], [1, 4, 12, 32, 80, 192])
]
