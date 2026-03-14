
from __future__ import annotations

import argparse
import math
from itertools import product

try:
    from .common import compute_cost, load_cap_problem
except ImportError:
    from common import compute_cost, load_cap_problem


MAX_BRUTE_FORCE_FACILITIES = 20


def brute_force_solve(problem_index: int) -> tuple[list[int], float]:
    problem = load_cap_problem(problem_index)
    facility_count = len(problem.opening_costs)

    if facility_count > MAX_BRUTE_FORCE_FACILITIES:
        raise ValueError(
            "Brute-force search is limited to instances with at most "
            f"{MAX_BRUTE_FORCE_FACILITIES} facilities. "
            f'Problem "{problem.name}" has {facility_count}.'
        )

    best_solution: tuple[int, ...] | None = None
    best_cost = math.inf

    for candidate in product((0, 1), repeat=facility_count):
        candidate_cost = compute_cost(candidate, problem.opening_costs, problem.service_costs)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_solution = candidate

    if best_solution is None:
        raise RuntimeError("No feasible solution was found.")

    return list(best_solution), float(best_cost)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve a CAP facility-location instance with brute-force search."
    )
    parser.add_argument(
        "--problem-index",
        type=int,
        default=0,
        help="Problem index inside the bundled CAP dataset (default: 0).",
    )
    args = parser.parse_args()

    problem = load_cap_problem(args.problem_index)
    solution, cost = brute_force_solve(args.problem_index)
    open_facilities = [index for index, is_open in enumerate(solution) if is_open]

    print(f"Problem: {problem.name}")
    print(f"Facilities: {len(problem.opening_costs)}")
    print(f"Customers: {len(problem.service_costs[0])}")
    print(f"Opened facilities: {open_facilities}")
    print(f"Brute-force cost: {cost:.4f}")
    print(f"Known optimal cost: {problem.known_optimal_cost:.4f}")
    print(f"Optimality gap: {cost - problem.known_optimal_cost:.4f}")


if __name__ == "__main__":
    main()
    
