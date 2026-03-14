# Optimization Algorithms in Python

Small, self-contained Python examples for two optimization families:

- Genetic algorithms
- Particle swarm optimization

The repository is organized as an educational set of scripts rather than a full framework. Each example can be run on its own from the repository root.


## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── genetic_algorithms
│   ├── n_queens_genetic_algorithm.py
│   ├── target_string_genetic_algorithm.py
│   └── uncapacitated_facility_location
│       ├── brute_force_solver.py
│       ├── common.py
│       ├── genetic_algorithm_solver.py
│       └── data
│           └── cap_problems.mat
└── particle_swarm_optimization
    └── sphere_function_pso.py
```

## Included Examples

### Genetic Algorithms

- `genetic_algorithms/target_string_genetic_algorithm.py`
  Evolves a random population of strings toward a target phrase.
- `genetic_algorithms/n_queens_genetic_algorithm.py`
  Uses a genetic algorithm to search for a solution to the N-Queens problem.
- `genetic_algorithms/uncapacitated_facility_location/genetic_algorithm_solver.py`
  Uses a genetic algorithm to approximate CAP facility-location benchmark instances.
- `genetic_algorithms/uncapacitated_facility_location/brute_force_solver.py`
  Solves small CAP instances exactly by enumerating every facility-opening combination.

### Particle Swarm Optimization

- `particle_swarm_optimization/sphere_function_pso.py`
  Minimizes the sphere function in a configurable search space using PSO.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Running the Examples

```bash
python particle_swarm_optimization/sphere_function_pso.py
python genetic_algorithms/target_string_genetic_algorithm.py --target "Hello World"
python genetic_algorithms/n_queens_genetic_algorithm.py
python genetic_algorithms/uncapacitated_facility_location/brute_force_solver.py --problem-index 0
python genetic_algorithms/uncapacitated_facility_location/genetic_algorithm_solver.py --problem-index 0
```

## Notes

- The brute-force facility-location solver is intentionally limited to small instances because the search space grows exponentially.
- The bundled `cap_problems.mat` file contains 21 benchmark instances with different numbers of facilities and customers.
- These implementations are meant for learning and experimentation, not for production-scale optimization.
