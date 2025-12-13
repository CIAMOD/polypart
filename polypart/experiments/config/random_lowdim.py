from polypart.experiments import (
    ArrangementClass,
    Experiment,
    PolytopeClass,
)

# --- DEFINE EXPERIMENTS LIST BELOW ---
# experiments = [
#     Experiment(
#         PolytopeClass("random"),
#         ArrangementClass("random", m=m, degen_ratio=0.0),
#         d=d,
#     )
#     # for d in [2, 3, 4, 5]
#     for d in [6, 7]
#     for m in [10, 20, 30, 40, 50]
# ]

experiments = [
    Experiment(
        PolytopeClass("random"),
        ArrangementClass("random", m=m, degen_ratio=0.0),
        d=d,
    )
    for m in [19]
    for d in [3, 4]
]
