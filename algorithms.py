import random
from typing import Any, List, Set, Tuple

import numpy as np

from problem import CUAVRP
from utils import euclidean_distance


class ConstructiveHeuristic:
    """
    Class describing a constructive heuristic for the CUAVRP.
    """

    def __init__(self, problem: CUAVRP, alpha: float, w_0: float, r_d: float) -> None:
        self._problem = problem
        self._weights = w_0 * np.ones((problem.num_nodes, problem.num_nodes))
        self._alpha = alpha
        self._r_d = r_d

    def construct_feasible_solution(self) -> Set[Tuple[int]]:
        is_infeasible = True
        while is_infeasible:
            is_infeasible = False
            next: int = 0
            route = [next]
            remaining_nodes: Set[int] = set(range(1, self._problem.num_nodes))
            solution: Set[Tuple[int]] = set([])
            while len(remaining_nodes) > 0:
                rcl = self.generate_rcl(remaining_nodes, route)
                if rcl is not None:
                    next = self.weighted_choice(rcl)
                    route.append(next)
                    remaining_nodes.remove(next)
                else:
                    route.append(0)
                    if (
                        self._compute_route_length(route)
                        > self._problem.max_travel_time
                    ):
                        is_infeasible = True
                        break
                    if self._compute_route_cost(route) > self._compute_route_cost(
                        route, reverse=True
                    ):
                        route = list(reversed(route))
                    solution.add(tuple(route))
                    if len(solution) > self._problem.num_uavs:
                        is_infeasible = True
                        break
                    next = 0
                    route = [next]
            if is_infeasible:
                continue
            if route[-1] != 0:
                route.append(0)
            if self._compute_route_length(route) > self._problem.max_travel_time:
                is_infeasible = True
                continue
            if self._compute_route_cost(route) > self._compute_route_cost(
                route, reverse=True
            ):
                route = list(reversed(route))
            solution.add(tuple(route))
            if len(solution) > self._problem.num_uavs:
                is_infeasible = True
                continue
            is_infeasible = False
            # self.divertisication_method(solution)
        return solution

    def generate_rcl(self, candidates: Set[int], route: List[int]) -> Set[int]:
        candidates = list(candidates)
        current_length = (
            sum(
                map(
                    lambda k: self._problem.graph.edges[route[k - 1], route[k]][
                        "weight"
                    ],
                    range(1, len(route)),
                )
            )
            if len(route) > 1
            else 0
        )
        for index in candidates:
            if (
                current_length
                + self._problem.graph.edges[route[-1], index]["weight"]
                + self._problem.graph.edges[index, 0]["weight"]
                > self._problem.max_travel_time
            ):
                candidates.remove(index)

        if len(candidates) == 0:
            return None
        min_distance = min(
            map(
                lambda k: self._problem.graph.edges[route[-1], k]["weight"],
                candidates,
            )
        )
        max_distance = max(
            map(
                lambda k: self._problem.graph.edges[route[-1], k]["weight"],
                candidates,
            )
        )
        candidates = list(
            filter(
                lambda k: self._problem.graph.edges[route[-1], k]["weight"]
                <= min_distance + self._alpha * max_distance,
                candidates,
            )
        )
        if len(candidates) == 0:
            return None
        else:
            return set(candidates)

    def weighted_choice(self, rcl: Set[int]) -> int:
        return random.choices(
            list(rcl), weights=list(map(lambda k: self._weights[0, k], rcl))
        )[0]

    def divertisication_method(self, solution: set) -> set:
        raise NotImplementedError

    def _compute_route_length(self, route: List[int]) -> float:
        current_length = 0
        if len(route) == 1:
            return 0
        for i in range(1, len(route)):
            current_length += self._problem.graph.edges[route[i - 1], route[i]][
                "weight"
            ]
        return current_length

    def _compute_route_cost(self, route: List[int], reverse: bool = False) -> float:
        """Compute route cost as the sum of the cumulative traveled distance at each node."""
        if len(route) == 1:
            return 0
        edge_lengths = [
            self._problem.graph.edges[route[i - 1], route[i]]["weight"]
            for i in range(1, len(route))
        ]
        if reverse:
            edge_lengths = edge_lengths[::-1]
        return sum(np.cumsum(edge_lengths))


class GRASPVND:
    def __init__(
        self,
        problem: CUAVRP,
        grasp_iters: int,
        num_constructed: int,
        alpha: float,
        w_0: float,
        r_d: float,
    ) -> None:
        self._problem = problem
        self._grasp_iters = grasp_iters
        self._num_constructed = num_constructed
        self._alpha = alpha
        self._w_0 = w_0
        self._r_d = r_d
        self._constructive_heuristic = ConstructiveHeuristic(problem, alpha, w_0, r_d)

    def run(self):
        best_solution_found = None
        best_solution_cost = None
        for i in range(self._grasp_iters):
            best_constructed = (
                self._constructive_heuristic.construct_feasible_solution()
            )
            best_constructed_cost = self._problem.evaluate(best_constructed)
            for j in range(self._num_constructed):
                S = self._constructive_heuristic.construct_feasible_solution()
                S_cost = self._problem.evaluate(S)
                if S_cost < best_constructed_cost:
                    best_constructed = S
                    best_constructed_cost = S_cost
            improved_solution = self.thread_strategy(best_constructed)
            improved_solution_cost = self._problem.evaluate(improved_solution)
            if (
                best_solution_cost is None
                or improved_solution_cost < best_solution_cost
            ):
                best_solution_found = improved_solution
                best_solution_cost = improved_solution_cost
            # self.intensification_method(best_solution_found)
        return best_solution_found

    def thread_strategy(self, solution: Set[Tuple[int]]) -> Set[Tuple[int]]:
        return solution

    def intensification_method(self, solution: Set[Tuple[int]]) -> Set[Tuple[int]]:
        raise NotImplementedError
