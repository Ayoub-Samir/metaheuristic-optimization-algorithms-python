
from __future__ import annotations

import argparse

import numpy as np


def sphere_function(position: np.ndarray) -> float:
    return float(np.sum(position**2))


def run_particle_swarm(
    population_size: int = 10,
    dimensions: int = 2,
    lower_bound: float = -5.0,
    upper_bound: float = 5.0,
    inertia: float = 0.7,
    cognitive_weight: float = 1.5,
    social_weight: float = 1.5,
    iterations: int = 25,
    seed: int = 1,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)

    lower_bounds = np.full(dimensions, lower_bound, dtype=float)
    upper_bounds = np.full(dimensions, upper_bound, dtype=float)
    velocity_scale = (upper_bounds - lower_bounds) * 0.1

    positions = rng.uniform(lower_bounds, upper_bounds, size=(population_size, dimensions))
    velocities = rng.uniform(-velocity_scale, velocity_scale, size=(population_size, dimensions))

    fitness = np.apply_along_axis(sphere_function, 1, positions)
    personal_best_positions = positions.copy()
    personal_best_fitness = fitness.copy()

    global_best_index = int(np.argmin(personal_best_fitness))
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_fitness = float(personal_best_fitness[global_best_index])

    print(f"Iteration 000 | best objective: {global_best_fitness:.6f}")

    for iteration in range(1, iterations + 1):
        r1 = rng.random((population_size, dimensions))
        r2 = rng.random((population_size, dimensions))

        velocities = (
            inertia * velocities
            + cognitive_weight * r1 * (personal_best_positions - positions)
            + social_weight * r2 * (global_best_position - positions)
        )
        positions = np.clip(positions + velocities, lower_bounds, upper_bounds)

        fitness = np.apply_along_axis(sphere_function, 1, positions)
        improved_mask = fitness < personal_best_fitness
        personal_best_positions[improved_mask] = positions[improved_mask]
        personal_best_fitness[improved_mask] = fitness[improved_mask]

        current_best_index = int(np.argmin(personal_best_fitness))
        current_best_fitness = float(personal_best_fitness[current_best_index])
        if current_best_fitness < global_best_fitness:
            global_best_position = personal_best_positions[current_best_index].copy()
            global_best_fitness = current_best_fitness

        print(f"Iteration {iteration:03d} | best objective: {global_best_fitness:.6f}")

    return global_best_position, global_best_fitness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimize the sphere function with Particle Swarm Optimization."
    )
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--upper-bound", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    best_position, best_value = run_particle_swarm(
        population_size=args.population_size,
        dimensions=args.dimensions,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        iterations=args.iterations,
        seed=args.seed,
    )

    print("\nBest position:", best_position)
    print(f"Best objective value: {best_value:.6f}")


if __name__ == "__main__":
    main()
