from experiments import ArrangementClass, Experiment, PolytopeClass

# --- DEFINE EXPERIMENTS LIST BELOW ---
experiments = [
    Experiment(
        PolytopeClass("random"),
        ArrangementClass("random", m=m, degen_ratio=0.1),
        d=d,
    )
    for d in [2, 3]
    for m in [10, 20, 30]
]
