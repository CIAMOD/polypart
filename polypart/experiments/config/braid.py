from polypart.experiments.core import (
    ArrangementClass,
    Experiment,
    PolytopeClass,
)

# --- DEFINE EXPERIMENTS LIST BELOW ---
experiments = [
    Experiment(
        PolytopeClass("cube", bbox_mode=True),
        ArrangementClass("braid"),
        d=d,
    )
    for d in [2, 3, 4, 5, 6, 7, 8, 9]
]
