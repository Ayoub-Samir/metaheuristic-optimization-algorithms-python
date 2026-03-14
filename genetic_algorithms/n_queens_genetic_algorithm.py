from __future__ import annotations

import argparse

import numpy as np


DEFAULT_POPULATION_SIZE = 100
DEFAULT_BOARD_SIZE = 8
ELITISM_RATE = 0.1


def initialize_population(population_size: int, board_size: int, rng: np.random.Generator) -> np.ndarray:
    population = np.empty((population_size, board_size), dtype=int)
    base = np.arange(board_size)

    for index in range(population_size):
        population[index] = rng.permutation(base)

    return population


def count_conflicts(individual: np.ndarray) -> int:
    conflicts = 0

    for column in range(len(individual)):
        for other_column in range(column + 1, len(individual)):
            same_diagonal = abs(individual[column] - individual[other_column]) == abs(
                column - other_column
            )
            if same_diagonal:
                conflicts += 1

    return conflicts


def calculate_fitness(population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conflicts = np.array([count_conflicts(individual) for individual in population], dtype=int)
    fitness = -conflicts
    return fitness, conflicts


def tournament_selection(
    population: np.ndarray,
    conflict_counts: np.ndarray,
    rng: np.random.Generator,
    tournament_size: int = 3,
) -> np.ndarray:
    selected = np.empty_like(population)

    for index in range(len(population)):
        competitors = rng.choice(
            len(population), size=min(tournament_size, len(population)), replace=False
        )
        winner = competitors[int(np.argmin(conflict_counts[competitors]))]
        selected[index] = population[winner]

    return selected


def ordered_crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    start, end = sorted(rng.choice(len(parent_a), size=2, replace=False))
    child = np.full(len(parent_a), -1, dtype=int)
    child[start:end] = parent_a[start:end]

    fill_values = [gene for gene in parent_b if gene not in child]
    fill_positions = [index for index, gene in enumerate(child) if gene == -1]

    for position, gene in zip(fill_positions, fill_values):
        child[position] = gene

    return child


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < crossover_rate:
        child_a = ordered_crossover(parent_a, parent_b, rng)
        child_b = ordered_crossover(parent_b, parent_a, rng)
        return child_a, child_b

    return parent_a.copy(), parent_b.copy()


def mutate(individual: np.ndarray, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
    child = individual.copy()
    if rng.random() < mutation_rate:
        swap_indices = rng.choice(len(child), size=2, replace=False)
        first_index, second_index = int(swap_indices[0]), int(swap_indices[1])
        child[first_index], child[second_index] = child[second_index], child[first_index]
    return child


def breed_population(
    population: np.ndarray,
    selected_population: np.ndarray,
    conflict_counts: np.ndarray,
    crossover_rate: float,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    population_size, board_size = selected_population.shape
    next_population = np.empty((population_size, board_size), dtype=int)
    elite_count = max(1, int(population_size * ELITISM_RATE))
    elite_indices = np.argsort(conflict_counts)[:elite_count]
    next_population[:elite_count] = population[elite_indices]

    write_index = elite_count
    while write_index < population_size:
        parent_a = selected_population[int(rng.integers(0, population_size))]
        parent_b = selected_population[int(rng.integers(0, population_size))]
        child_a, child_b = crossover(parent_a, parent_b, crossover_rate, rng)
        next_population[write_index] = mutate(child_a, mutation_rate, rng)
        write_index += 1

        if write_index < population_size:
            next_population[write_index] = mutate(child_b, mutation_rate, rng)
            write_index += 1

    return next_population


def format_board(solution: np.ndarray) -> str:
    lines = []

    for row in range(len(solution)):
        line = " ".join("Q" if solution[column] == row else "." for column in range(len(solution)))
        lines.append(line)

    return "\n".join(lines)


def run_n_queens_genetic_algorithm(
    population_size: int = DEFAULT_POPULATION_SIZE,
    board_size: int = DEFAULT_BOARD_SIZE,
    max_generations: int = 10_000,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    population = initialize_population(population_size, board_size, rng)

    best_solution = population[0].copy()
    best_conflicts = count_conflicts(best_solution)

    for generation in range(max_generations):
        _, conflict_counts = calculate_fitness(population)
        generation_best_index = int(np.argmin(conflict_counts))
        generation_best_conflicts = int(conflict_counts[generation_best_index])

        if generation_best_conflicts < best_conflicts:
            best_conflicts = generation_best_conflicts
            best_solution = population[generation_best_index].copy()

        print(f"\rGeneration {generation:05d} | best conflicts: {best_conflicts:02d}", end="")

        if generation_best_conflicts == 0:
            break

        selected_population = tournament_selection(population, conflict_counts, rng)
        population = breed_population(
            population,
            selected_population,
            conflict_counts,
            crossover_rate,
            mutation_rate,
            rng,
        )

    print()
    return best_solution, best_conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the N-Queens problem with a genetic algorithm.")
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--max-generations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    solution, conflicts = run_n_queens_genetic_algorithm(
        population_size=args.population_size,
        board_size=args.board_size,
        max_generations=args.max_generations,
        seed=args.seed,
    )

    print("Solution vector:", solution.tolist())
    print("Remaining conflicts:", conflicts)
    print("\nBoard:")
    print(format_board(solution))


if __name__ == "__main__":
    main()
