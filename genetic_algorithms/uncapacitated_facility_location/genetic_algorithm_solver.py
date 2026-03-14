
from __future__ import annotations

import argparse
import math

import numpy as np

try:
    from .common import compute_cost, load_cap_problem
except ImportError:
    from common import compute_cost, load_cap_problem


DEFAULT_POPULATION_SIZE = 70
DEFAULT_MAX_GENERATIONS = 1_000


def ensure_feasible(candidate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if not candidate.any():
        candidate[int(rng.integers(0, len(candidate)))] = 1
    return candidate


def initialize_population(
    population_size: int, dimensions: int, rng: np.random.Generator
) -> np.ndarray:
    population = rng.integers(0, 2, size=(population_size, dimensions))
    for candidate in population:
        ensure_feasible(candidate, rng)
    return population


def evaluate_population(
    population: np.ndarray, opening_costs: list[float], service_costs: list[list[float]]
) -> np.ndarray:
    return np.array(
        [compute_cost(candidate, opening_costs, service_costs) for candidate in population],
        dtype=float,
    )


def tournament_selection(
    population: np.ndarray,
    costs: np.ndarray,
    rng: np.random.Generator,
    tournament_size: int = 3,
) -> np.ndarray:
    selected = np.empty_like(population)

    for index in range(len(population)):
        competitors = rng.choice(
            len(population), size=min(tournament_size, len(population)), replace=False
        )
        winner = competitors[int(np.argmin(costs[competitors]))]
        selected[index] = population[winner]

    return selected


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < crossover_rate:
        crossover_point = int(rng.integers(1, len(parent_a)))
        child_a = np.concatenate((parent_a[:crossover_point], parent_b[crossover_point:]))
        child_b = np.concatenate((parent_b[:crossover_point], parent_a[crossover_point:]))
        return child_a, child_b

    return parent_a.copy(), parent_b.copy()


def mutate(
    individual: np.ndarray, mutation_rate: float, rng: np.random.Generator
) -> np.ndarray:
    child = individual.copy()
    mutation_mask = rng.random(len(child)) < mutation_rate
    child[mutation_mask] = 1 - child[mutation_mask]
    return ensure_feasible(child, rng)


def run_genetic_algorithm(
    problem_index: int,
    population_size: int = DEFAULT_POPULATION_SIZE,
    max_generations: int = DEFAULT_MAX_GENERATIONS,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], float]:
    rng = np.random.default_rng(seed)
    problem = load_cap_problem(problem_index)
    dimensions = len(problem.opening_costs)
    population = initialize_population(population_size, dimensions, rng)

    best_solution = population[0].copy()
    best_cost = math.inf

    for generation in range(max_generations):
        costs = evaluate_population(population, problem.opening_costs, problem.service_costs)
        generation_best_index = int(np.argmin(costs))
        generation_best_cost = float(costs[generation_best_index])

        if generation_best_cost < best_cost:
            best_cost = generation_best_cost
            best_solution = population[generation_best_index].copy()

        print(f"\rGeneration {generation:04d} | best cost: {best_cost:.4f}", end="")

        if math.isclose(best_cost, problem.known_optimal_cost, rel_tol=0.0, abs_tol=1e-6):
            break

        parents = tournament_selection(population, costs, rng)
        next_population = np.empty_like(population)

        for start in range(0, population_size, 2):
            parent_a = parents[start]
            parent_b = parents[(start + 1) % population_size]
            child_a, child_b = crossover(parent_a, parent_b, crossover_rate, rng)
            next_population[start] = mutate(child_a, mutation_rate, rng)

            if start + 1 < population_size:
                next_population[start + 1] = mutate(child_b, mutation_rate, rng)

        population = next_population

    print()
    return best_solution.tolist(), best_cost


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approximate a CAP facility-location instance with a genetic algorithm."
    )
    parser.add_argument("--problem-index", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--max-generations", type=int, default=DEFAULT_MAX_GENERATIONS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    problem = load_cap_problem(args.problem_index)
    solution, cost = run_genetic_algorithm(
        problem_index=args.problem_index,
        population_size=args.population_size,
        max_generations=args.max_generations,
        seed=args.seed,
    )
    open_facilities = [index for index, is_open in enumerate(solution) if is_open]

    print(f"Problem: {problem.name}")
    print(f"Facilities: {len(problem.opening_costs)}")
    print(f"Customers: {len(problem.service_costs[0])}")
    print(f"Opened facilities: {open_facilities}")
    print(f"Genetic algorithm cost: {cost:.4f}")
    print(f"Known optimal cost: {problem.known_optimal_cost:.4f}")
    print(f"Optimality gap: {cost - problem.known_optimal_cost:.4f}")


if __name__ == "__main__":
    main()
