import copy
from enum import Enum
import multiprocessing
import random
from typing import Any, List, Set, Tuple

import numpy as np

from problem import CUAVRP
from utils import euclidean_distance
from tqdm.auto import tqdm


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
        """
        Constructs a feasible solution for the problem.

        Returns:
            A set of tuples, each tuple representing a route.
        """
        random.seed()
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


class ThreadStrategy(Enum):
    ALL_RETURN = 0
    BEST_CONTINUE = 1
    RANDOM_RETURN = 2


class LocalMoves(Enum):
    ONE_ONE_SWAP = 0
    ADJACENT_SWAP = 1
    TWO_OPT = 2
    ONE_ZERO_RELOCATION = 3
    TWO_ZERO_RELOCATION = 4
    ONE_ONE_EXCHANGE = 5
    TWO_ONE_EXCHANGE = 6
    TWO_TWO_EXCHANGE = 7
    THREE_THREE_EXCHANGE = 8


class GRASPVND:
    def __init__(
        self,
        problem: CUAVRP,
        grasp_iters: int,
        num_constructed: int,
        vnd_iters: int,
        thread_strategy: ThreadStrategy,
        num_threads: int,
        alpha: float,
        w_0: float,
        r_d: float,
    ) -> None:
        self._problem = problem
        self._grasp_iters = grasp_iters
        self._num_constructed = num_constructed
        self._vnd_iters = vnd_iters
        self._thread_strategy = thread_strategy
        self._num_threads = num_threads
        self._alpha = alpha
        self._w_0 = w_0
        self._r_d = r_d
        self._constructive_heuristic = ConstructiveHeuristic(problem, alpha, w_0, r_d)

        self._best_continue_solution = None
        self._random_return_solution = None

    def run(self):
        best_solution_found = None
        best_solution_cost = None
        for i in tqdm(range(self._grasp_iters), position=0, leave=True, unit="it"):
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
            improved_solution = self._apply_thread_strategy(best_constructed)
            improved_solution_cost = self._problem.evaluate(improved_solution)
            if (
                best_solution_cost is None
                or improved_solution_cost < best_solution_cost
            ):
                best_solution_found = improved_solution
                best_solution_cost = improved_solution_cost
                print(
                    f"Found new best solution with cost {best_solution_cost} at iteration {i}"
                )
                print(f"Best solution found so far: {best_solution_found}")
            # self.intensification_method(best_solution_found)
        return best_solution_found

    def _apply_thread_strategy(self, solution: Set[Tuple[int]]) -> Set[Tuple[int]]:
        pool = multiprocessing.Pool(self._num_threads)
        if self._thread_strategy == ThreadStrategy.ALL_RETURN:
            results = pool.map(
                self._vnd,
                map(lambda k: copy.deepcopy(solution), range(self._num_threads)),
            )
            improved_solution = min(results, key=lambda k: self._problem.evaluate(k))
        elif self._thread_strategy == ThreadStrategy.BEST_CONTINUE:
            if self._best_continue_solution is None:
                self._best_continue_solution = copy.deepcopy(solution)
            results = pool.map(
                self._vnd,
                map(
                    lambda k: copy.deepcopy(solution)
                    if k < self._num_threads - 1
                    else copy.deepcopy(self._best_continue_solution),
                    range(self._num_threads),
                ),
            )
            improved_solution = min(results, key=lambda k: self._problem.evaluate(k))
            self._best_continue_solution = copy.deepcopy(improved_solution)
        elif self._thread_strategy == ThreadStrategy.RANDOM_RETURN:
            if self._random_return_solution is None:
                self._random_return_solution = copy.deepcopy(solution)
            results = pool.map(
                self._vnd,
                map(
                    lambda k: copy.deepcopy(solution)
                    if k < self._num_threads - 1
                    else copy.deepcopy(self._random_return_solution),
                    range(self._num_threads),
                ),
            )
            improved_solution = min(results, key=lambda k: self._problem.evaluate(k))
            self._random_return_solution = copy.deepcopy(random.choice(results))
        else:
            pool.close()
            raise ValueError("Invalid thread strategy")
        pool.close()
        return improved_solution

    def intensification_method(self, solution: Set[Tuple[int]]) -> Set[Tuple[int]]:
        raise NotImplementedError

    def _vnd(self, solution: Set[Tuple[int]]) -> Set[Tuple[int]]:
        random.seed()
        for i in range(self._vnd_iters):
            route = tuple(random.sample(solution, 1)[0])
            route_cost = self._problem.evaluate_route_cost(route)
            for move in [
                LocalMoves.ONE_ONE_SWAP,
                LocalMoves.ADJACENT_SWAP,
                LocalMoves.TWO_OPT,
            ]:
                improves = True
                move_func = getattr(self, "_" + move.name.lower())
                while improves:
                    improves = False
                    new_route = move_func(route)
                    new_route_feasible = self._problem.check_route_feasibility(
                        new_route
                    )
                    if new_route_feasible:
                        new_route_cost = self._problem.evaluate_route_cost(new_route)
                        if new_route_cost < route_cost:
                            solution.remove(route)
                            solution.add(new_route)
                            route = new_route
                            route_cost = new_route_cost
                            improves = True
            if len(solution) == 1:
                continue
            routes = random.sample(solution, 2)
            route_1 = copy.deepcopy(routes[0])
            route_2 = copy.deepcopy(routes[1])
            for move in [
                LocalMoves.ONE_ZERO_RELOCATION,
                LocalMoves.TWO_ZERO_RELOCATION,
                LocalMoves.ONE_ONE_EXCHANGE,
                LocalMoves.TWO_ONE_EXCHANGE,
                LocalMoves.TWO_TWO_EXCHANGE,
                LocalMoves.THREE_THREE_EXCHANGE,
            ]:
                improves = True
                move_func = getattr(self, "_" + move.name.lower())
                while improves:
                    improves = False
                    try:
                        new_route1, new_route2 = move_func(route_1, route_2)
                        new_route1_feasible = self._problem.check_route_feasibility(
                            new_route1
                        )
                        new_route2_feasible = self._problem.check_route_feasibility(
                            new_route2
                        )
                        if new_route1_feasible and new_route2_feasible:
                            new_route1_cost = self._problem.evaluate_route_cost(
                                new_route1
                            )
                            new_route2_cost = self._problem.evaluate_route_cost(
                                new_route2
                            )
                            if (
                                new_route1_cost + new_route2_cost
                                < route_cost + route_cost
                            ):
                                solution.remove(route_1)
                                solution.remove(route_2)
                                solution.add(new_route1)
                                solution.add(new_route2)
                                route_1 = new_route1
                                route_2 = new_route2
                                route_cost = new_route1_cost + new_route2_cost
                                improves = True
                    except NotImplementedError:
                        pass
        return solution

    def _one_one_swap(self, route: Tuple[int]) -> Tuple[int]:
        if len(route) < 4:
            return route
        route = list(route)
        i, j = random.sample(range(1, len(route) - 1), 2)
        route[i], route[j] = route[j], route[i]
        return tuple(route)

    def _adjacent_swap(self, route: Tuple[int]) -> Tuple[int]:
        if len(route) < 4:
            return route
        route = list(route)
        i = random.randint(1, len(route) - 3)
        route[i], route[i + 1] = route[i + 1], route[i]
        return tuple(route)

    def _two_opt(self, route: Tuple[int]) -> Tuple[int]:
        if len(route) < 4:
            return route
        route = list(route)
        i, j = random.sample(range(1, len(route) - 2), 2)
        if i > j:
            i, j = j, i
        route[i : j + 1] = route[i : j + 1][::-1]
        return tuple(route)

    def _one_zero_relocation(
        self, route_1: Tuple[int], route_2: Tuple[int]
    ) -> Tuple[Tuple[int], Tuple[int]]:
        if len(route_1) < 3:
            return route_1, route_2
        route_1 = list(route_1)
        route_2 = list(route_2)
        i = random.randint(1, len(route_1) - 2)
        node = route_1.pop(i)
        j = random.randint(1, len(route_2) - 2)
        route_2.insert(j, node)
        return tuple(route_1), tuple(route_2)

    def _two_zero_relocation(
        self, route_1: Tuple[int], route_2: Tuple[int]
    ) -> Tuple[Tuple[int], Tuple[int]]:
        raise NotImplementedError

    def _one_one_exchange(
        self, route_1: Tuple[int], route_2: Tuple[int]
    ) -> Tuple[Tuple[int], Tuple[int], bool]:
        raise NotImplementedError

    def _two_one_exchange(
        self, route_1: Tuple[int], route_2: Tuple[int]
    ) -> Tuple[Tuple[int], Tuple[int], bool]:
        raise NotImplementedError

    def _two_two_exchange(
        self, route_1: Tuple[int], route_2: Tuple[int]
    ) -> Tuple[Tuple[int], Tuple[int], bool]:
        raise NotImplementedError

    def _three_three_exchange(
        self, route_1: Tuple[int], route_2: Tuple[int]
    ) -> Tuple[Tuple[int], Tuple[int], bool]:
        raise NotImplementedError
