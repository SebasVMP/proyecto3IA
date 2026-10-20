import unittest
import math

from optimizer_lab import (
    eggholder,
    genetic_algorithm,
    make_graph,
    queens_conflicts,
    solve_coloring_backtracking,
    solve_knapsack_backtracking,
    solve_queens_backtracking,
)


class AlgorithmSmokeTests(unittest.TestCase):
    def test_queens_backtracking_returns_solution(self):
        result = solve_queens_backtracking(8, 42)
        self.assertTrue(result.feasible)
        self.assertEqual(result.objective, 0)

    def test_queens_backtracking_scales_to_50(self):
        result = solve_queens_backtracking(50, 1042)
        self.assertTrue(result.feasible)
        self.assertEqual(result.objective, 0)

    def test_knapsack_backtracking_respects_capacity(self):
        weights, values, capacity = [2, 3, 4], [3, 4, 5], 5
        result = solve_knapsack_backtracking(weights, values, capacity)
        self.assertEqual(result.objective, 7)
        self.assertLessEqual(sum(w * b for w, b in zip(weights, result.solution)), capacity)

    def test_coloring_backtracking_returns_valid_coloring(self):
        graph = make_graph(20, 7, average_degree=2)
        result = solve_coloring_backtracking(graph, 20, 7)
        self.assertTrue(result.feasible)
        self.assertEqual(result.objective, 0)

    def test_coloring_budget_is_reported_as_incomplete(self):
        graph = [{1, 2}, {0, 2}, {0, 1}]
        result = solve_coloring_backtracking(graph, 2, 7, node_limit=1)
        self.assertFalse(result.feasible)
        self.assertEqual(result.termination, "node_budget_exceeded")
        self.assertTrue(math.isnan(result.objective))

    def test_eggholder_known_global_minimum_area(self):
        self.assertAlmostEqual(eggholder((512, 404.2319)), -959.6407, delta=0.01)

    def test_genetic_algorithm_counts_objective_evaluations(self):
        calls = 0

        def objective(value):
            nonlocal calls
            calls += 1
            return value

        result = genetic_algorithm(
            [4, 3, 2, 1], objective, lambda a, b, rng: min(a, b),
            lambda value, rng: value, seed=1, generations=2, population_size=4
        )
        self.assertEqual(result.evaluations, calls)


if __name__ == "__main__":
    unittest.main()
