import numpy as np

from problem import CUAVRP


class ConstructiveHeuristic:
    """
    Class describing a constructive heuristic for the CUAVRP.
    """

    def __init__(self, problem: CUAVRP, alpha: float, w_0: float, r_d: float) -> None:
        self._problem = problem
        self._weights = w_0 * np.ones((problem.num_nodes, problem.num_nodes))
        self._alpha = alpha
        self._r_d = r_d

    def construct_feasible_solution(self) -> np.ndarray:
        solution = None
        while not self._problem.is_feasible(solution):
            print("Solution infeasible!")
            l = 0  # vehicle index
            next = 0  # next node to visit
            route = [[next]]
            remaining_nodes = set(range(1, self._problem.num_nodes))
            solution = set([])
            while len(remaining_nodes) > 0:
                rcl = self.generate_rcl(remaining_nodes)
                print("rcl", rcl)
                if rcl is not None:
                    next = self.weighted_choice(rcl)
                    route[l].append(next)
                    remaining_nodes.remove(next)
                else:
                    route[l].append(0)
                    solution.add(tuple(route[l]))
                    print("Solution with new route", solution)
                    l += 1
                    next = 0
                    route.append([next])
            solution.add(tuple(route[l]))
            self.divertisication_method(solution)
            print("SOLUTION, ", solution)
        return solution

    def generate_rcl(self, N: set) -> list:
        import random
        rcl = list(N)
        if len(rcl) > 0:
            return random.sample(rcl, max(1, int(0.7*len(rcl))))
        else:
            return None

    def weighted_choice(self, rcl: list) -> int:
        """Returns a node from the RCL."""
        import random
        return random.sample(rcl, 1)[0]

    def divertisication_method(self, solution: set) -> set:
        """Returns a new solution."""
        return
