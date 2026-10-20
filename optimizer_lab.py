"""Reproducible optimization-algorithm experiments for the course assignment."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Outcome:
    solution: Any
    objective: float
    evaluations: int
    iterations: int
    feasible: bool
    history: list[float]
    termination: str = "completed"


def eggholder(point: tuple[float, float]) -> float:
    x, y = point
    return -(y + 47) * math.sin(math.sqrt(abs(x / 2 + y + 47))) - x * math.sin(
        math.sqrt(abs(x - (y + 47)))
    )


def queens_conflicts(permutation: list[int]) -> int:
    return sum(
        abs(permutation[i] - permutation[j]) == j - i
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )


def solve_queens_backtracking(n: int, seed: int, node_limit: int = 1_000_000) -> Outcome:
    rng = random.Random(seed)
    placement = [-1] * n
    columns: set[int] = set()
    diag_down: set[int] = set()
    diag_up: set[int] = set()
    nodes = 0
    best = [-1] * n
    budget_exceeded = False

    def search(remaining_rows: set[int]) -> bool:
        nonlocal nodes, best, budget_exceeded
        if not remaining_rows:
            best = placement.copy()
            return True
        domains = {
            row: [
                col
                for col in range(n)
                if col not in columns
                and row - col not in diag_down
                and row + col not in diag_up
            ]
            for row in remaining_rows
        }
        if any(not domain for domain in domains.values()):
            return False
        row = min(remaining_rows, key=lambda candidate: (len(domains[candidate]), rng.random()))
        candidates = domains[row]
        rng.shuffle(candidates)

        def future_attacks(col: int) -> int:
            return sum(
                abs(other_col - col) == abs(other_row - row)
                for other_row in remaining_rows
                if other_row != row
                for other_col in domains[other_row]
            )

        candidates.sort(key=future_attacks)
        for col in candidates:
            nodes += 1
            if nodes > node_limit:
                budget_exceeded = True
                return False
            placement[row] = col
            columns.add(col)
            diag_down.add(row - col)
            diag_up.add(row + col)
            if search(remaining_rows - {row}):
                return True
            columns.remove(col)
            diag_down.remove(row - col)
            diag_up.remove(row + col)
        return False

    success = search(set(range(n)))
    objective = float(queens_conflicts(best)) if success else float("nan")
    termination = "solution_found" if success else "node_budget_exceeded" if budget_exceeded else "search_exhausted"
    return Outcome(best, objective, nodes, 0, success, [objective], termination)


def solve_knapsack_backtracking(
    weights: list[int], values: list[int], capacity: int, node_limit: int = 1_000_000
) -> Outcome:
    order = sorted(range(len(weights)), key=lambda i: values[i] / weights[i], reverse=True)
    best_value = 0
    best_bits = [0] * len(weights)
    nodes = 0
    budget_exceeded = False

    def search(depth: int, weight: int, value: int, bits: list[int]) -> None:
        nonlocal best_value, best_bits, nodes, budget_exceeded
        nodes += 1
        if nodes > node_limit:
            budget_exceeded = True
            return
        if value > best_value:
            best_value, best_bits = value, bits.copy()
        if depth == len(order):
            return
        remaining_capacity = capacity - weight
        bound = float(value)
        for idx in order[depth:]:
            if weights[idx] <= remaining_capacity:
                remaining_capacity -= weights[idx]
                bound += values[idx]
            else:
                bound += values[idx] * remaining_capacity / weights[idx]
                break
        if bound <= best_value:
            return
        idx = order[depth]
        if weight + weights[idx] <= capacity:
            bits[idx] = 1
            search(depth + 1, weight + weights[idx], value + values[idx], bits)
            bits[idx] = 0
        search(depth + 1, weight, value, bits)

    search(0, 0, 0, [0] * len(weights))
    total_weight = sum(w * bit for w, bit in zip(weights, best_bits))
    termination = "node_budget_exceeded" if budget_exceeded else "search_completed"
    return Outcome(best_bits, float(best_value), nodes, 0, total_weight <= capacity,
                   [float(best_value)], termination)


def solve_coloring_backtracking(
    adjacency: list[set[int]], colors: int, seed: int, node_limit: int = 200_000
) -> Outcome:
    rng = random.Random(seed)
    n = len(adjacency)
    assignment = [-1] * n
    # Static degree ordering is an effective fail-first heuristic for coloring.
    order = sorted(range(n), key=lambda node: (-len(adjacency[node]), rng.random()))
    nodes = 0
    completed = False
    budget_exceeded = False

    def search(depth: int) -> bool:
        nonlocal nodes, budget_exceeded
        if depth == n:
            return True
        node = order[depth]
        forbidden = {assignment[v] for v in adjacency[node] if assignment[v] >= 0}
        for color in range(colors):
            nodes += 1
            if nodes > node_limit:
                budget_exceeded = True
                return False
            if color not in forbidden:
                assignment[node] = color
                if search(depth + 1):
                    return True
                assignment[node] = -1
        return False

    previous_recursion_limit = sys.getrecursionlimit()
    try:
        if n + 100 > previous_recursion_limit:
            sys.setrecursionlimit(n + 100)
        completed = search(0)
    finally:
        sys.setrecursionlimit(previous_recursion_limit)
    conflicts = coloring_conflicts(assignment, adjacency)
    objective = float(conflicts) if completed else float("nan")
    termination = "solution_found" if completed else "node_budget_exceeded" if budget_exceeded else "search_exhausted"
    return Outcome(assignment, objective, nodes, 0, completed, [objective], termination)


def simulated_annealing(
    initial: Any,
    objective: Callable[[Any], float],
    neighbor: Callable[[Any, random.Random], Any],
    seed: int,
    iterations: int = 20_000,
    temperature: float = 10.0,
    cooling: float = 0.9995,
    minimize: bool = True,
) -> Outcome:
    rng = random.Random(seed)
    current = initial
    current_score = objective(current)
    best, best_score = current, current_score
    evaluations = 1
    history = [best_score]
    temp = temperature
    for step in range(iterations):
        candidate = neighbor(current, rng)
        score = objective(candidate)
        evaluations += 1
        delta = (score - current_score) if minimize else (current_score - score)
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
            current, current_score = candidate, score
        if (current_score < best_score if minimize else current_score > best_score):
            best, best_score = current, current_score
        history.append(best_score)
        temp *= cooling
    return Outcome(best, float(best_score), evaluations, iterations, True, history)


def genetic_algorithm(
    initial_population: list[Any],
    objective: Callable[[Any], float],
    crossover: Callable[[Any, Any, random.Random], Any],
    mutate: Callable[[Any, random.Random], Any],
    seed: int,
    generations: int = 300,
    population_size: int = 60,
    mutation_rate: float = 0.2,
    minimize: bool = True,
) -> Outcome:
    rng = random.Random(seed)
    population = initial_population[:population_size]
    while len(population) < population_size:
        population.append(mutate(population[rng.randrange(len(population))], rng))
    evaluations = len(population)
    ranked = [(individual, objective(individual)) for individual in population]
    ranked.sort(key=lambda item: item[1], reverse=not minimize)
    best, best_score = ranked[0]
    history = [float(best_score)]
    for _ in range(generations):
        next_ranked = ranked[:2]
        while len(next_ranked) < population_size:
            pool = [ranked[rng.randrange(min(5, len(ranked)))] for _ in range(3)]
            parent_a = min(pool, key=lambda item: item[1]) if minimize else max(pool, key=lambda item: item[1])
            pool = [ranked[rng.randrange(min(5, len(ranked)))] for _ in range(3)]
            parent_b = min(pool, key=lambda item: item[1]) if minimize else max(pool, key=lambda item: item[1])
            child = crossover(parent_a[0], parent_b[0], rng)
            if rng.random() < mutation_rate:
                child = mutate(child, rng)
            next_ranked.append((child, objective(child)))
            evaluations += 1
        ranked = sorted(next_ranked, key=lambda item: item[1], reverse=not minimize)
        generation_best, generation_score = ranked[0]
        improved = generation_score < best_score if minimize else generation_score > best_score
        if improved:
            best, best_score = generation_best, generation_score
        history.append(float(best_score))
    return Outcome(best, float(best_score), evaluations, generations, True, history)


def queens_ga(n: int, seed: int, generations: int, population_size: int) -> Outcome:
    rng = random.Random(seed)
    initial = [rng.sample(range(n), n) for _ in range(population_size)]

    def crossover(a: list[int], b: list[int], r: random.Random) -> list[int]:
        left, right = sorted(r.sample(range(n), 2))
        child = [-1] * n
        child[left:right] = a[left:right]
        rest = [x for x in b if x not in child]
        return rest[:left] + child[left:right] + rest[left:]

    def mutate(a: list[int], r: random.Random) -> list[int]:
        out = a.copy()
        i, j = r.sample(range(n), 2)
        out[i], out[j] = out[j], out[i]
        return out

    outcome = genetic_algorithm(initial, queens_conflicts, crossover, mutate, seed, generations, population_size)
    outcome.feasible = outcome.objective == 0
    return outcome


def knapsack_repair(bits: list[int], weights: list[int], values: list[int], capacity: int) -> list[int]:
    result = bits.copy()
    while sum(w * b for w, b in zip(weights, result)) > capacity:
        selected = [i for i, bit in enumerate(result) if bit]
        result[min(selected, key=lambda i: values[i] / weights[i])] = 0
    return result


def knapsack_heuristic(
    weights: list[int], values: list[int], capacity: int, seed: int, algorithm: str,
    iterations: int, population_size: int,
) -> Outcome:
    rng = random.Random(seed)
    n = len(weights)
    repair = lambda bits: knapsack_repair(bits, weights, values, capacity)
    objective = lambda bits: -sum(v * b for v, b in zip(values, bits))

    def mutate(bits: list[int], r: random.Random) -> list[int]:
        out = bits.copy()
        for _ in range(max(1, n // 10)):
            i = r.randrange(n)
            out[i] = 1 - out[i]
        return repair(out)

    if algorithm == "SA":
        return simulated_annealing(
            repair([rng.randrange(2) for _ in range(n)]), objective, mutate, seed,
            iterations, 50.0, 0.9995, True
        )
    initial = [[rng.randrange(2) for _ in range(n)] for _ in range(population_size)]
    initial = [repair(x) for x in initial]
    return genetic_algorithm(
        initial, objective, lambda a, b, r: repair([x if r.random() < .5 else y for x, y in zip(a, b)]),
        mutate, seed, iterations, population_size
    )


def coloring_conflicts(assignment: list[int], adjacency: list[set[int]]) -> int:
    return sum(
        assignment[u] == assignment[v]
        for u in range(len(adjacency))
        for v in adjacency[u]
        if u < v
    )


def coloring_heuristic(
    adjacency: list[set[int]], colors: int, seed: int, algorithm: str,
    iterations: int, population_size: int,
) -> Outcome:
    rng = random.Random(seed)
    n = len(adjacency)
    objective = lambda x: coloring_conflicts(x, adjacency)

    def mutate(x: list[int], r: random.Random) -> list[int]:
        out = x.copy()
        for _ in range(max(1, n // 20)):
            vertex = r.randrange(n)
            out[vertex] = r.randrange(colors)
        return out

    if algorithm == "SA":
        def neighbor(x: list[int], r: random.Random) -> list[int]:
            out = x.copy()
            v = r.randrange(n)
            out[v] = r.randrange(colors)
            return out
        outcome = simulated_annealing([rng.randrange(colors) for _ in range(n)], objective, neighbor,
                                      seed, iterations, 5.0, 0.9995)
        outcome.feasible = outcome.objective == 0
        return outcome
    initial = [[rng.randrange(colors) for _ in range(n)] for _ in range(population_size)]
    outcome = genetic_algorithm(
        initial, objective, lambda a, b, r: [x if r.random() < .5 else y for x, y in zip(a, b)],
        mutate, seed, iterations, population_size
    )
    outcome.feasible = outcome.objective == 0
    return outcome


def eggholder_heuristic(seed: int, algorithm: str, iterations: int, population_size: int) -> Outcome:
    rng = random.Random(seed)
    bound = 512.0

    def mutate(point: tuple[float, float], r: random.Random) -> tuple[float, float]:
        return tuple(max(-bound, min(bound, v + r.gauss(0, 35))) for v in point)

    if algorithm == "SA":
        return simulated_annealing(
            (rng.uniform(-bound, bound), rng.uniform(-bound, bound)), eggholder,
            lambda p, r: tuple(max(-bound, min(bound, v + r.gauss(0, 25))) for v in p),
            seed, iterations, 200.0, 0.9995
        )
    initial = [(rng.uniform(-bound, bound), rng.uniform(-bound, bound)) for _ in range(population_size)]
    crossover = lambda a, b, r: (a[0] if r.random() < .5 else b[0], a[1] if r.random() < .5 else b[1])
    return genetic_algorithm(initial, eggholder, crossover, mutate, seed, iterations, population_size)


def make_graph(n: int, seed: int, average_degree: float = 4.0) -> list[set[int]]:
    rng = random.Random(seed)
    adjacency = [set() for _ in range(n)]
    probability = min(1.0, average_degree / max(1, n - 1))
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < probability:
                adjacency[u].add(v)
                adjacency[v].add(u)
    return adjacency


def synthetic_knapsack(n: int = 30, seed: int = 2026) -> tuple[list[int], list[int], int]:
    rng = random.Random(seed)
    weights = [rng.randint(1, 30) for _ in range(n)]
    values = [rng.randint(5, 60) for _ in range(n)]
    return weights, values, int(sum(weights) * 0.35)


def run_experiments(
    runs: int = 5, iterations: int = 2000, generations: int = 200,
    population_size: int = 50, output: str = "results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []

    def record(problem: str, size: str, algorithm: str, run: int, seed: int,
               outcome: Outcome, started: float, params: dict[str, Any]) -> None:
        elapsed = time.perf_counter() - started
        results.append({
            "problem": problem, "instance": size, "algorithm": algorithm, "run": run,
            "seed": seed, "objective": outcome.objective, "evaluations": outcome.evaluations,
            "iterations": outcome.iterations, "seconds": elapsed, "feasible": outcome.feasible,
            "termination": outcome.termination, "parameters": str(params), "solution": str(outcome.solution),
        })
        for step, value in enumerate(outcome.history):
            histories.append({"problem": problem, "instance": size, "algorithm": algorithm,
                              "run": run, "seed": seed, "step": step, "best": value})

    for n in (8, 50, 100):
        for run in range(runs):
            seed = 1000 + n * 10 + run
            for algorithm in ("Backtracking", "SA", "GA"):
                params = {"seed": seed, "iterations": iterations, "population": population_size}
                started = time.perf_counter()
                if algorithm == "Backtracking":
                    outcome = solve_queens_backtracking(n, seed)
                elif algorithm == "SA":
                    rng = random.Random(seed)
                    initial = list(range(n))
                    rng.shuffle(initial)
                    outcome = simulated_annealing(
                        initial, queens_conflicts,
                        lambda x, r: _swap_neighbor(x, r), seed, iterations, 10, 0.9995
                    )
                    outcome.feasible = outcome.objective == 0
                else:
                    outcome = queens_ga(n, seed, generations, population_size)
                record("N-Queens", str(n), algorithm, run, seed, outcome, started, params)

    weights, values, capacity = synthetic_knapsack()
    for run in range(runs):
        seed = 2000 + run
        for algorithm in ("Backtracking", "SA", "GA"):
            params = {"capacity": capacity, "seed": seed, "iterations": iterations,
                      "population": population_size, "data": "synthetic; replace with task instance"}
            started = time.perf_counter()
            if algorithm == "Backtracking":
                outcome = solve_knapsack_backtracking(weights, values, capacity)
            else:
                outcome = knapsack_heuristic(weights, values, capacity, seed, algorithm,
                                             iterations if algorithm == "SA" else generations, population_size)
                outcome.objective *= -1
                outcome.history = [-value for value in outcome.history]
            record("Knapsack 0/1", str(len(weights)), algorithm, run, seed, outcome, started, params)

    for n in (50, 1000):
        adjacency = make_graph(n, 3030 + n)
        for run in range(runs):
            seed = 3000 + n + run
            for algorithm in ("Backtracking", "SA", "GA"):
                params = {"colors": 4, "average_degree_target": 4, "seed": seed,
                          "iterations": iterations, "population": population_size,
                          "backtracking_node_budget": 200000}
                started = time.perf_counter()
                if algorithm == "Backtracking":
                    outcome = solve_coloring_backtracking(adjacency, 4, seed)
                else:
                    outcome = coloring_heuristic(adjacency, 4, seed, algorithm,
                                                  iterations if algorithm == "SA" else generations,
                                                  population_size)
                record("Graph Coloring", str(n), algorithm, run, seed, outcome, started, params)

    for run in range(runs):
        seed = 4000 + run
        for algorithm in ("SA", "GA"):
            params = {"domain": "[-512, 512]^2", "seed": seed, "iterations": iterations,
                      "population": population_size}
            started = time.perf_counter()
            outcome = eggholder_heuristic(seed, algorithm,
                                          iterations if algorithm == "SA" else generations, population_size)
            record("Eggholder", "2D", algorithm, run, seed, outcome, started, params)

    results_frame = pd.DataFrame(results)
    history_frame = pd.DataFrame(histories)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path / "experimental_results.xlsx") as writer:
        results_frame.to_excel(writer, sheet_name="results", index=False)
        history_frame.to_excel(writer, sheet_name="convergence", index=False)
        results_frame.groupby(["problem", "instance", "algorithm"], as_index=False).agg(
            runs=("run", "count"), objective_mean=("objective", "mean"),
            objective_std=("objective", "std"), time_mean_s=("seconds", "mean"),
            evaluations_mean=("evaluations", "mean"), feasible_rate=("feasible", "mean")
        ).to_excel(writer, sheet_name="summary", index=False)
    _plot_results(results_frame, history_frame, output_path)
    return results_frame, history_frame


def _swap_neighbor(solution: list[int], rng: random.Random) -> list[int]:
    result = solution.copy()
    i, j = rng.sample(range(len(result)), 2)
    result[i], result[j] = result[j], result[i]
    return result


def _plot_results(results: pd.DataFrame, history: pd.DataFrame, output: Path) -> None:
    for problem, subset in history.groupby("problem"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for algorithm, curve in subset.groupby("algorithm"):
            grouped = curve.groupby("step")["best"].mean()
            ax.plot(grouped.index, grouped.values, label=algorithm)
        ax.set(title=f"Convergencia: {problem}", xlabel="Iteración / generación", ylabel="Mejor objetivo")
        ax.legend()
        fig.tight_layout()
        filename = problem.lower().replace(" ", "_").replace("/", "_")
        fig.savefig(output / f"convergence_{filename}.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    results.boxplot(column="seconds", by=["problem", "algorithm"], rot=45, ax=ax)
    ax.set_title("Tiempo por problema y algoritmo")
    ax.set_ylabel("Segundos")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output / "execution_times.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    results, _ = run_experiments(args.runs, args.iterations, args.generations,
                                 args.population, args.output)
    print(results.groupby(["problem", "instance", "algorithm"])["objective"].agg(["mean", "std"]))
    print(f"\nExcel y gráficas guardados en: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
