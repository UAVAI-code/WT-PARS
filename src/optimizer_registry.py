"""Uniform adapters for the evolutionary-optimizer comparison.

The paper attack keeps its project-specific MOEA/D implementation as the
default.  Published comparison algorithms are loaded from the project-local
``pymoo==0.6.1.6`` installation so no global Python environment is modified.
All optimizers receive the same bounded genome, initial population, objective
function and function-evaluation budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable

import numpy as np

from .moead import MOEAD as ProposedMOEAD


PUBLISHED_OPTIMIZERS = (
    "nsga2",
    "spea2",
    "moead",
    "sms_emoa",
    "age_moea2",
)
ALL_OPTIMIZERS = (*PUBLISHED_OPTIMIZERS, "proposed_moead")


@dataclass(frozen=True)
class OptimizerResult:
    genomes: np.ndarray
    objectives: np.ndarray
    evaluations: int
    generations: int
    implementation: str


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "pymoo_runtime"


def _load_pymoo() -> dict[str, object]:
    vendor = _vendor_root()
    if not vendor.exists():
        raise RuntimeError(
            "The project-local pymoo runtime is missing. Install pymoo==0.6.1.6 "
            f"under {vendor}."
        )
    vendor_text = str(vendor)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    # pymoo's AGE-MOEA-II module uses numba only to JIT two small geometry
    # helpers.  The paper environment intentionally has no numba dependency.
    # A no-JIT compatibility module preserves the exact Python/Numpy
    # calculations and avoids changing the already frozen environment.
    temporary_numba = False
    try:
        import numba as _numba  # noqa: F401
    except ModuleNotFoundError:
        compatibility = ModuleType("numba")

        def identity_jit(*jit_args, **jit_kwargs):
            if jit_args and callable(jit_args[0]) and len(jit_args) == 1:
                return jit_args[0]

            def decorate(function):
                return function

            return decorate

        compatibility.jit = identity_jit
        compatibility.float64 = np.float64
        compatibility.intp = np.intp
        compatibility.boolean = np.bool_
        sys.modules["numba"] = compatibility
        temporary_numba = True
    from pymoo.algorithms.moo.moead import MOEAD
    try:
        from pymoo.algorithms.moo.age2 import AGEMOEA2
    finally:
        if temporary_numba:
            sys.modules.pop("numba", None)
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.algorithms.moo.spea2 import SPEA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions

    return {
        "MOEAD": MOEAD,
        "AGEMOEA2": AGEMOEA2,
        "NSGA2": NSGA2,
        "SMSEMOA": SMSEMOA,
        "SPEA2": SPEA2,
        "ElementwiseProblem": ElementwiseProblem,
        "minimize": minimize,
        "get_termination": get_termination,
        "get_reference_directions": get_reference_directions,
    }


def _prepare_initial_population(
    initial_genomes: np.ndarray | None,
    population_size: int,
    n_variables: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    population = rng.random((population_size, n_variables))
    if initial_genomes is None:
        return population
    initial = np.asarray(initial_genomes, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] != n_variables:
        raise ValueError("Initial genomes have an incompatible shape.")
    count = min(population_size, len(initial))
    population[:count] = np.clip(initial[:count], 0.0, 1.0)
    return population


def minimize_with_optimizer(
    name: str,
    evaluate: Callable[[np.ndarray], np.ndarray],
    *,
    n_variables: int,
    population_size: int,
    max_evaluations: int,
    seed: int,
    initial_genomes: np.ndarray | None = None,
    repair: Callable[[np.ndarray], np.ndarray] | None = None,
    neighbourhood_size: int = 5,
    neighbour_mating_probability: float = 0.90,
    maximum_replacements: int = 2,
    proposed_offspring_operator: Callable | None = None,
    proposed_offspring_feedback: Callable | None = None,
    callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
) -> OptimizerResult:
    """Minimize two objectives with an exact, shared evaluation budget."""
    name = str(name).lower().replace("-", "_")
    if name not in ALL_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {name!r}; choose from {ALL_OPTIMIZERS}.")
    population_size = int(population_size)
    max_evaluations = int(max_evaluations)
    n_variables = int(n_variables)
    if population_size < 4 or max_evaluations < population_size:
        raise ValueError("The budget must include an initial population of at least four.")
    initial = _prepare_initial_population(
        initial_genomes, population_size, n_variables, seed
    )

    if name == "proposed_moead":
        if (max_evaluations - population_size) % population_size:
            raise ValueError(
                "proposed_moead requires a budget equal to population_size times "
                "an integer number of initial/generation populations."
            )
        generations = (max_evaluations - population_size) // population_size
        exact_budget = population_size * (generations + 1)
        optimizer = ProposedMOEAD(
            population_size=population_size,
            generations=generations,
            n_variables=n_variables,
            seed=seed,
            neighbourhood_size=neighbourhood_size,
            neighbour_mating_probability=neighbour_mating_probability,
            maximum_replacements=maximum_replacements,
            repair=repair,
            offspring_operator=proposed_offspring_operator,
            offspring_feedback=proposed_offspring_feedback,
        )
        genomes, objectives = optimizer.minimize(
            evaluate, callback=callback, initial_genomes=initial
        )
        return OptimizerResult(
            genomes=np.asarray(genomes, dtype=np.float64),
            objectives=np.asarray(objectives, dtype=np.float64),
            evaluations=exact_budget,
            generations=generations,
            implementation="project MOEA/D with structure-specific operators",
        )

    api = _load_pymoo()
    ElementwiseProblem = api["ElementwiseProblem"]

    counter = {"value": 0}

    class TwoObjectiveProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(
                n_var=n_variables,
                n_obj=2,
                n_ieq_constr=0,
                xl=np.zeros(n_variables),
                xu=np.ones(n_variables),
            )

        def _evaluate(self, x, out, *args, **kwargs) -> None:
            # Generational pymoo algorithms can overshoot an n_eval termination
            # when duplicate elimination requests replacements for a structured
            # initial population.  The expensive unmixing oracle, rather than
            # pymoo's internal evaluator counter, defines the paper budget.
            # Extra framework proposals receive a dominated sentinel without
            # invoking the oracle and therefore cannot enter the useful front.
            if counter["value"] >= max_evaluations:
                out["F"] = np.asarray([1e12, 1e12], dtype=np.float64)
                return
            genome = np.asarray(x, dtype=np.float64)
            if repair is not None:
                genome = np.asarray(repair(genome), dtype=np.float64)
            value = np.asarray(evaluate(genome), dtype=np.float64)
            if value.shape != (2,):
                raise ValueError("The objective function must return exactly two values.")
            counter["value"] += 1
            out["F"] = value

    if name == "nsga2":
        algorithm = api["NSGA2"](pop_size=population_size, sampling=initial)
    elif name == "spea2":
        algorithm = api["SPEA2"](pop_size=population_size, sampling=initial)
    elif name == "sms_emoa":
        algorithm = api["SMSEMOA"](pop_size=population_size, sampling=initial)
    elif name == "age_moea2":
        algorithm = api["AGEMOEA2"](
            pop_size=population_size,
            sampling=initial,
        )
    else:
        reference = api["get_reference_directions"](
            "das-dennis", 2, n_partitions=population_size - 1
        )
        algorithm = api["MOEAD"](
            ref_dirs=reference,
            n_neighbors=min(max(2, neighbourhood_size), population_size),
            prob_neighbor_mating=neighbour_mating_probability,
            sampling=initial,
        )

    class ProgressCallback:
        def __init__(self) -> None:
            self.generations = 0

        def __call__(self, algorithm_value) -> None:
            self.generations = int(algorithm_value.n_gen)
            if callback is not None:
                pop = algorithm_value.pop
                callback(
                    self.generations,
                    np.asarray(pop.get("X"), dtype=np.float64),
                    np.asarray(pop.get("F"), dtype=np.float64),
                )

    progress = ProgressCallback()
    result = api["minimize"](
        TwoObjectiveProblem(),
        algorithm,
        termination=api["get_termination"]("n_eval", max_evaluations),
        seed=seed,
        callback=progress,
        save_history=False,
        verbose=False,
    )
    genomes = np.asarray(result.X, dtype=np.float64)
    objectives = np.asarray(result.F, dtype=np.float64)
    if genomes.ndim == 1:
        genomes = genomes[None]
        objectives = objectives[None]
    return OptimizerResult(
        genomes=genomes,
        objectives=objectives,
        evaluations=int(counter["value"]),
        generations=progress.generations,
        implementation=f"pymoo 0.6.1.6 {name}",
    )
