import argparse
import json
import os
import uuid
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from polypart.arrangements import (
    get_braid_arrangement,
    get_moduli_arrangement,
    get_random_arrangement,
)
from polypart.geometry import Polytope
from polypart.polytopes import (
    get_centered_hypercube,
    get_product_of_simplices,
    sample_poisson_zero_cell_polytope,
)

from .runner import (
    run_experiments,
    run_single_experiment,
)


class PolytopeClass:
    def __init__(self, name: str, bbox_mode: bool = False):
        self.name = name
        self.bbox_mode = bbox_mode

    def get_polytope(
        self, d: int, decimals: int | None = None, seed: int | None = None
    ) -> Polytope:
        if self.name == "random":
            return sample_poisson_zero_cell_polytope(
                d, 50.0, 1.0, decimals=decimals, seed=seed
            )
        elif self.name == "cube":
            return get_centered_hypercube(d, 1.0)
        elif self.name == "moduli_n1":
            return get_product_of_simplices(1, d)
        elif self.name == "moduli_r2":
            return get_product_of_simplices(d, 1)
        else:
            raise ValueError(f"Unknown polytope class: {self.name}")

    def __str__(self):
        return self.name + ("_bb" if self.bbox_mode else "")


class ArrangementClass:
    def __init__(
        self, name: str, m: int | None = None, degen_ratio: float | None = None
    ):
        self.name = name
        self.m = m
        self.degen_ratio = degen_ratio

    def get_arrangement(
        self,
        d: int,
        P: Polytope | None,
        decimals: int | None = None,
        seed: int | None = None,
    ) -> list:
        if self.name == "random":
            assert self.degen_ratio is not None, (
                "Degeneracy ratio must be specified for random arrangement."
            )
            return get_random_arrangement(
                P, self.m, self.degen_ratio, decimals=decimals, seed=seed
            )
        elif self.name == "braid":
            return get_braid_arrangement(d)
        elif self.name == "moduli_n1":
            return get_moduli_arrangement(1, d + 1, 0)
        elif self.name == "moduli_r2":
            return get_moduli_arrangement(d, 2, 0)
        else:
            raise ValueError(f"Unknown arrangement class: {self.name}")

    def __str__(self):
        if self.name == "random":
            return f"random_m{self.m}_degen{self.degen_ratio}"
        elif self.name == "moduli":
            return f"moduli_n{self.n}_r{self.r}"
        else:
            return self.name


class Experiment:
    def __init__(self, P_class: PolytopeClass, A_class: ArrangementClass, d: int):
        self.P_class = P_class
        self.A_class = A_class
        self.d = d

        self._last_results = None

    def run(
        self,
        decimals: int = 3,
        seed: int | None = None,
        exclude_algorithms: list[str] | None = None,
        n_processes: int | None = None,
    ) -> dict:
        """
        Runs each algorithm in an isolated process to ensure clean memory state.
        Verifies geometric consistency and result consistency strictly.

        Args:
            decimals: Number of decimal places for geometry generation.
            seed: Random seed for geometry generation.
            exclude_algorithms: List of algorithm names to exclude from the run.
            n_processes: Number of parallel processes to use (None = one per algorithm).
        Returns:
            A dictionary with aggregated results from all algorithms.
        """
        results = run_single_experiment(
            P_class=self.P_class,
            A_class=self.A_class,
            d=self.d,
            decimals=decimals,
            seed=seed,
            exclude_algorithms=exclude_algorithms,
            n_processes=n_processes,
        )
        self._last_results = results
        return results

    def dirname(self) -> str:
        return f"{self.P_class}-{self.A_class}-d{self.d}"

    def save(self, results: dict, folder: str = "./data"):
        path = Path(folder) / self.dirname()
        # mkdir is thread-safe enough for creating the parent folder
        path.mkdir(parents=True, exist_ok=True)

        # 1. Generate unique components
        timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]

        # 2. Construct Filename
        seed = results.get("seed", None)
        if seed is not None:
            filename = f"experiment_{timestamp}_seed{seed}.json"
        else:
            filename = f"experiment_{timestamp}_{unique_id}.json"

        filepath = path / filename

        # 3. Write to File
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)

    def load(self, folder: str = "./data") -> list[dict]:
        path = Path(folder) / self.dirname()
        results = []
        if not path.exists():
            return results
        for filename in os.listdir(path):
            if filename.endswith(".json"):
                filepath = path / filename
                with open(filepath, "r") as f:
                    data = json.load(f)
                    results.append(data)
        return results

    def count_existing_runs(self, folder: str = "./data") -> int:
        """
        Counts how many result files already exist for this specific experiment configuration.
        """
        path = Path(folder) / self.dirname()
        if not path.exists():
            return 0
        count = 0
        for filename in os.listdir(path):
            if filename.endswith(".json"):
                count += 1
        return count

    def __str__(self):
        return f"Experiment(P={self.P_class}, A={self.A_class}, d={self.d})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run polytope partition experiments.")
    parser.add_argument(
        "-d", type=int, required=True, help="Dimension of the polytope."
    )
    parser.add_argument(
        "-p",
        type=str,
        required=True,
        help="Polytope class (e.g., 'cube', 'moduli_n1', 'moduli_r2', 'random').",
    )
    parser.add_argument(
        "-a",
        type=str,
        required=True,
        help="Arrangement class (e.g., 'braid', 'moduli_n1', 'moduli_r2', 'random').",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=None,
        help="Number of hyperplanes (required for random arrangement).",
    )
    parser.add_argument(
        "-r",
        type=float,
        default=0.0,
        help="Degeneracy ratio (required for random arrangement).",
    )
    parser.add_argument(
        "-n", type=int, default=1, help="Number of runs to perform (default: 1)."
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="List of algorithms to exclude (e.g., 'ppart', 'incenu', 'delres').",
    )

    args = parser.parse_args()

    P_class = PolytopeClass(args.p)
    A_class = ArrangementClass(args.a, m=args.m, degen_ratio=args.r)
    experiment = Experiment(P_class, A_class, d=args.d)
    run_experiments(experiment, n_runs=args.n, exclude_algorithms=args.exclude)
