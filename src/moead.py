"""MOEA/D for bounded, low-dimensional hyperspectral attack search.

The implementation adapts the subproblem, neighbourhood, arithmetic crossover,
and external archive structure of ``mopt`` (MIT License, 2026) while preserving
the callback interface used by this project.  The original implementation is
located in ``multi-objective-optimization-main/mopt/algorithms/moead.py``.

Two corrections are applied for this use case: offspring are generated from the
current subproblem neighbourhood and replacement compares each neighbouring
incumbent with the child under that neighbour's weight vector.  Polynomial
mutation and a replacement cap are included following the standard MOEA/D
procedure so the bounded attack genome does not collapse prematurely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class _SubProblem:
    weight: np.ndarray
    neighbours: np.ndarray


def _non_dominated_indices(objectives: np.ndarray) -> np.ndarray:
    keep = np.ones(len(objectives), dtype=bool)
    for i, value in enumerate(objectives):
        if not keep[i]:
            continue
        dominated = np.all(objectives <= value, axis=1) & np.any(objectives < value, axis=1)
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    return np.flatnonzero(keep)


class MOEAD:
    """Two-objective MOEA/D with neighbourhood mating and external archive."""

    def __init__(
        self,
        population_size: int,
        generations: int,
        n_variables: int,
        seed: int = 0,
        neighbourhood_size: int = 8,
        neighbour_mating_probability: float = 0.9,
        maximum_replacements: int = 2,
        mutation_probability: float | None = None,
        mutation_distribution_index: float = 20.0,
        archive_size: int | None = None,
        repair: Callable[[np.ndarray], np.ndarray] | None = None,
        discrete_bins: dict[int, int] | None = None,
        offspring_operator: Callable[
            [np.ndarray, np.ndarray, int, int, np.random.Generator], np.ndarray
        ]
        | None = None,
        offspring_feedback: Callable[[np.ndarray, np.ndarray, int, int, int], None]
        | None = None,
    ) -> None:
        if population_size < 4:
            raise ValueError("MOEA/D needs at least four subproblems.")
        if n_variables < 1:
            raise ValueError("n_variables must be positive.")
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.n_variables = int(n_variables)
        self.neighbourhood_size = min(max(2, int(neighbourhood_size)), self.population_size)
        self.neighbour_mating_probability = float(neighbour_mating_probability)
        self.maximum_replacements = max(1, int(maximum_replacements))
        self.mutation_probability = (
            1.0 / self.n_variables if mutation_probability is None else float(mutation_probability)
        )
        self.mutation_distribution_index = float(mutation_distribution_index)
        self.archive_size = int(archive_size or max(4 * population_size, population_size))
        self.repair = repair
        self.discrete_bins = dict(discrete_bins or {})
        self.offspring_operator = offspring_operator
        self.offspring_feedback = offspring_feedback
        self.rng = np.random.default_rng(seed)
        self.archive_genomes = np.empty((0, self.n_variables), dtype=np.float64)
        self.archive_objectives = np.empty((0, 2), dtype=np.float64)

    def _build_subproblems(self) -> list[_SubProblem]:
        # Uniform two-objective weights make every requested population size
        # cover the full effectiveness-stealth trade-off.
        first = np.linspace(1e-3, 1.0 - 1e-3, self.population_size)
        weights = np.column_stack([first, 1.0 - first])
        distance = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=-1)
        order = np.argsort(distance, axis=1)[:, : self.neighbourhood_size]
        return [_SubProblem(weights[i], order[i]) for i in range(self.population_size)]

    @staticmethod
    def _decompose(objective: np.ndarray, weight: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> float:
        scale = np.maximum(nadir - ideal, 1e-8)
        normalised = np.abs(objective - ideal) / scale
        weighted = np.maximum(weight, 1e-6) * normalised
        return float(np.max(weighted) + 1e-4 * np.sum(weighted))

    def _arithmetic_crossover(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        # Adapted from mopt.core.operators.arithmetic_crossover_single.
        alpha = self.rng.uniform(-0.3, 1.3, self.n_variables)
        return np.clip(alpha * first + (1.0 - alpha) * second, 0.0, 1.0)

    def _polynomial_mutation(self, position: np.ndarray) -> np.ndarray:
        child = position.copy()
        eta = self.mutation_distribution_index
        for index in np.flatnonzero(self.rng.random(self.n_variables) < self.mutation_probability):
            if index in self.discrete_bins:
                bins = self.discrete_bins[index]
                if bins <= 1:
                    child[index] = 0.5
                    continue
                current = min(int(np.floor(child[index] * bins)), bins - 1)
                draw = int(self.rng.integers(bins - 1))
                level = draw + (draw >= current)
                child[index] = (level + 0.5) / bins
                continue
            value = child[index]
            draw = self.rng.random()
            if draw <= 0.5:
                delta = (2.0 * draw + (1.0 - 2.0 * draw) * (1.0 - value) ** (eta + 1.0)) ** (
                    1.0 / (eta + 1.0)
                ) - 1.0
            else:
                delta = 1.0 - (
                    2.0 * (1.0 - draw) + 2.0 * (draw - 0.5) * value ** (eta + 1.0)
                ) ** (1.0 / (eta + 1.0))
            child[index] = np.clip(value + delta, 0.0, 1.0)
        return child

    def _update_archive(self, genomes: np.ndarray, objectives: np.ndarray) -> None:
        all_genomes = np.vstack([self.archive_genomes, genomes])
        all_objectives = np.vstack([self.archive_objectives, objectives])
        # Remove duplicate objective vectors before dominance filtering.
        rounded = np.round(all_objectives, 10)
        _, unique = np.unique(rounded, axis=0, return_index=True)
        all_genomes, all_objectives = all_genomes[unique], all_objectives[unique]
        front = _non_dominated_indices(all_objectives)
        all_genomes, all_objectives = all_genomes[front], all_objectives[front]
        if len(all_objectives) > self.archive_size:
            order = np.argsort(all_objectives[:, 0])
            # Even spacing along the first objective retains both extremes and
            # avoids a dense cluster consuming the complete archive.
            chosen = np.linspace(0, len(order) - 1, self.archive_size, dtype=int)
            order = order[chosen]
            all_genomes, all_objectives = all_genomes[order], all_objectives[order]
        self.archive_genomes, self.archive_objectives = all_genomes, all_objectives

    def minimize(
        self,
        evaluate: Callable[[np.ndarray], np.ndarray],
        callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
        initial_genomes: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        genomes = self.rng.random((self.population_size, self.n_variables))
        if initial_genomes is not None:
            initial = np.asarray(initial_genomes, dtype=np.float64)
            if initial.ndim != 2 or initial.shape[1] != self.n_variables:
                raise ValueError("Initial genomes have an incompatible shape.")
            count = min(len(initial), self.population_size)
            genomes[:count] = np.clip(initial[:count], 0.0, 1.0)
        if self.repair is not None:
            genomes = np.asarray([self.repair(genome) for genome in genomes])

        objectives = np.asarray([evaluate(genome) for genome in genomes], dtype=np.float64)
        if objectives.shape != (self.population_size, 2):
            raise ValueError("This MOEA/D implementation expects exactly two objectives.")
        subproblems = self._build_subproblems()
        ideal = objectives.min(axis=0)
        nadir = objectives.max(axis=0)
        self._update_archive(genomes, objectives)
        if callback:
            callback(0, genomes, objectives)

        for generation in range(1, self.generations + 1):
            for index in self.rng.permutation(self.population_size):
                neighbours = subproblems[index].neighbours
                if self.rng.random() < self.neighbour_mating_probability:
                    pool = neighbours
                else:
                    pool = np.arange(self.population_size)
                parents = self.rng.choice(pool, size=2, replace=len(pool) < 2)
                if self.offspring_operator is None:
                    child = self._arithmetic_crossover(
                        genomes[parents[0]], genomes[parents[1]]
                    )
                    child = self._polynomial_mutation(child)
                else:
                    child = np.asarray(
                        self.offspring_operator(
                            genomes[parents[0]],
                            genomes[parents[1]],
                            generation,
                            int(index),
                            self.rng,
                        ),
                        dtype=np.float64,
                    )
                    if child.shape != (self.n_variables,):
                        raise ValueError(
                            "offspring_operator must return one flat genome."
                        )
                    child = np.clip(child, 0.0, 1.0)
                if self.repair is not None:
                    child = np.asarray(self.repair(child), dtype=np.float64)
                child_objective = np.asarray(evaluate(child), dtype=np.float64)
                if child_objective.shape != (2,):
                    raise ValueError("The objective function must return two values.")
                ideal = np.minimum(ideal, child_objective)
                nadir = np.maximum(nadir, child_objective)

                replaced = 0
                for neighbour in self.rng.permutation(neighbours):
                    candidate_cost = self._decompose(
                        child_objective, subproblems[neighbour].weight, ideal, nadir
                    )
                    incumbent_cost = self._decompose(
                        objectives[neighbour], subproblems[neighbour].weight, ideal, nadir
                    )
                    if candidate_cost <= incumbent_cost:
                        genomes[neighbour] = child
                        objectives[neighbour] = child_objective
                        replaced += 1
                        if replaced >= self.maximum_replacements:
                            break
                if self.offspring_feedback is not None:
                    self.offspring_feedback(
                        child.copy(),
                        child_objective.copy(),
                        replaced,
                        generation,
                        int(index),
                    )
                self._update_archive(child[None, :], child_objective[None, :])
            if callback:
                callback(generation, genomes, objectives)

        return self.archive_genomes.copy(), self.archive_objectives.copy()
