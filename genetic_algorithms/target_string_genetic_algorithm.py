from __future__ import annotations

import argparse
import random
import string
from dataclasses import dataclass


DEFAULT_POPULATION_SIZE = 100
DEFAULT_TARGET = "My Name Is AyoubSamir"
DEFAULT_GENE_POOL = string.ascii_letters + string.digits + " ,.-;:_!\"#%&/()=?@${[]}"


def calculate_fitness(chromosome: list[str], target: str) -> int:
    return sum(gene != target_gene for gene, target_gene in zip(chromosome, target))


@dataclass
class Individual:
    chromosome: list[str]
    fitness: int

    @classmethod
    def create_random(cls, target: str, gene_pool: str) -> "Individual":
        chromosome = [random.choice(gene_pool) for _ in range(len(target))]
        return cls(chromosome=chromosome, fitness=calculate_fitness(chromosome, target))

    def mate(self, partner: "Individual", target: str, gene_pool: str) -> "Individual":
        child_chromosome: list[str] = []

        for gene_a, gene_b in zip(self.chromosome, partner.chromosome):
            probability = random.random()

            if probability < 0.45:
                child_chromosome.append(gene_a)
            elif probability < 0.90:
                child_chromosome.append(gene_b)
            else:
                child_chromosome.append(random.choice(gene_pool))

        return Individual(
            chromosome=child_chromosome,
            fitness=calculate_fitness(child_chromosome, target),
        )

    def as_string(self) -> str:
        return "".join(self.chromosome)


def run_genetic_algorithm(
    target: str,
    population_size: int = DEFAULT_POPULATION_SIZE,
    max_generations: int = 1_000,
    elitism_rate: float = 0.1,
    parent_pool_rate: float = 0.5,
    gene_pool: str = DEFAULT_GENE_POOL,
    seed: int = 42,
) -> Individual:
    if not set(target).issubset(set(gene_pool)):
        raise ValueError("The target string contains characters that are not present in the gene pool.")

    random.seed(seed)

    population = [Individual.create_random(target, gene_pool) for _ in range(population_size)]
    elite_count = max(1, int(population_size * elitism_rate))
    parent_pool_size = max(2, int(population_size * parent_pool_rate))

    best_individual = min(population, key=lambda individual: individual.fitness)

    for generation in range(max_generations + 1):
        population.sort(key=lambda individual: individual.fitness)
        best_individual = population[0]
        print(
            f'Generation {generation:04d} | best string: "{best_individual.as_string()}" '
            f"| fitness: {best_individual.fitness}"
        )

        if best_individual.fitness == 0:
            return best_individual

        next_generation = population[:elite_count]
        candidate_parents = population[:parent_pool_size]

        while len(next_generation) < population_size:
            parent_a = random.choice(candidate_parents)
            parent_b = random.choice(candidate_parents)
            next_generation.append(parent_a.mate(parent_b, target, gene_pool))

        population = next_generation

    return best_individual


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use a simple genetic algorithm to evolve a target string."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--max-generations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_genetic_algorithm(
        target=args.target,
        population_size=args.population_size,
        max_generations=args.max_generations,
        seed=args.seed,
    )

    print("\nFinal result:", result.as_string())
    print("Final fitness:", result.fitness)


if __name__ == "__main__":
    main()
