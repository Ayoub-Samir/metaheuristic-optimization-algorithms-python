from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import scipy.io as sio


DATASET_PATH = Path(__file__).resolve().parent / "data" / "cap_problems.mat"


@dataclass(frozen=True)
class FacilityLocationProblem:
    name: str
    opening_costs: list[float]
    service_costs: list[list[float]]
    known_optimal_cost: float


def compute_cost(
    open_facilities: list[int] | tuple[int, ...],
    opening_costs: list[float],
    service_costs: list[list[float]],
) -> float:
    if sum(open_facilities) == 0:
        return math.inf

    total_opening_cost = sum(
        cost * is_open for cost, is_open in zip(opening_costs, open_facilities)
    )

    total_service_cost = 0.0
    customer_count = len(service_costs[0])

    for customer_index in range(customer_count):
        best_service_cost = math.inf

        for facility_index, is_open in enumerate(open_facilities):
            if is_open:
                best_service_cost = min(
                    best_service_cost, service_costs[facility_index][customer_index]
                )

        total_service_cost += best_service_cost

    return float(total_opening_cost + total_service_cost)


def load_cap_problem(problem_index: int, dataset_path: Path = DATASET_PATH) -> FacilityLocationProblem:
    mat = sio.loadmat(dataset_path)
    problems = mat["Problems"]

    if not 0 <= problem_index < problems.shape[0]:
        raise IndexError(
            f"Problem index {problem_index} is out of range. "
            f"Expected a value between 0 and {problems.shape[0] - 1}."
        )

    return FacilityLocationProblem(
        name=str(problems[problem_index, 0][0]),
        known_optimal_cost=float(problems[problem_index, 1][0][0]),
        service_costs=problems[problem_index, 2].tolist(),
        opening_costs=problems[problem_index, 3][0].tolist(),
    )
