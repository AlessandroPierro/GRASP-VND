from typing import Set, Tuple
import networkx as netx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import gurobipy as gp
from gurobipy import GRB

sns.set_theme(style="white")


class CUAVRP:
    """
    Class describing a Cumulative Unmanned Aerial Vehicle Routing Problem
    (CUAVRP).
    """

    def __init__(
        self, graph: netx.Graph, num_uavs: int, max_travel_time: float
    ) -> None:
        assert num_uavs > 1, "Number of UAVs must be greater than 1."
        assert max_travel_time > 0, "Maximum travel time must be positive."
        self._graph = graph
        self._num_uavs = num_uavs
        self._max_travel_time = max_travel_time

    @property
    def graph(self) -> netx.Graph:
        """Returns the graph of the problem."""
        return self._graph

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def num_uavs(self) -> int:
        """Returns the max umber of UAVs in the problem."""
        return self._num_uavs

    @property
    def max_travel_time(self) -> float:
        """Returns the maximum travel time of the UAVs."""
        return self._max_travel_time

    def plot_graph(self) -> None:
        """Plots the graph of the problem, using netx and seaborn."""
        fig, ax = plt.subplots(figsize=(10, 10))
        netx.draw_networkx(self.graph, ax=ax)
        plt.savefig("graph.png")

    def evaluate(self, solution: set) -> float:
        """Returns the cost of the solution."""
        total_cost = 0
        for route in solution:
            if len(route) == 1:
                continue
            total_cost += self.evaluate_route_cost(route)
        return total_cost

    def evaluate_route_cost(self, route: list) -> float:
        """Returns the cost of a route."""
        if len(route) < 3:
            return 0
        edge_lengths = [
            self._graph.edges[route[i - 1], route[i]]["weight"]
            for i in range(1, len(route))
        ]
        return sum(np.cumsum(edge_lengths))

    def evaluate_route_length(self, route: list) -> float:
        """Returns the length of a route."""
        if len(route) < 3:
            return 0
        edge_lengths = [
            self._graph.edges[route[i - 1], route[i]]["weight"]
            for i in range(1, len(route))
        ]
        return sum(edge_lengths)

    def check_route_feasibility(self, route: list) -> bool:
        """Checks if a route is feasible."""
        if len(route) in [1, 2] and set(route) == set([0]):
            return True
        if self.evaluate_route_cost(route) > self.max_travel_time:
            return False
        if route[0] != 0 or route[-1] != 0:
            return False
        return True

    def plot_solution(self, solution: Set[Tuple[int]]):
        fig, ax = plt.subplots(figsize=(7, 7))
        # plot just nodes
        netx.draw_networkx_nodes(
            self.graph,
            ax=ax,
            pos=netx.get_node_attributes(self.graph, "pos"),
            node_color="black",
        )
        for route in solution:
            if len(route) == 1:
                continue
            netx.draw_networkx_edges(
                self.graph,
                ax=ax,
                pos=netx.get_node_attributes(self.graph, "pos"),
                edgelist=[(route[i - 1], route[i]) for i in range(1, len(route))],
                edge_color=np.random.rand(
                    3,
                ),
            )
        plt.savefig("solution.png")

    def to_gurobi_model(self, G: float):
        """Implement model in Gurobi."""
        model = gp.Model("CUAVRP")

        N_PLUS_ONE = self.num_nodes

        V = range(N_PLUS_ONE + 1)
        V_prime = range(1, N_PLUS_ONE)
        V_without_last = range(N_PLUS_ONE)
        R = range(self.num_uavs)

        def u(problem, i, j):
            if i == j:
                return 0
            if i == N_PLUS_ONE:
                i = 0
            if j == N_PLUS_ONE:
                j = 0
            if i == j:
                return 0
            weight = problem.graph.edges[i, j]["weight"]
            assert weight >= 0
            return weight

        t = model.addVars(
            [(k, i) for k in R for i in V], vtype=GRB.CONTINUOUS, name="t"
        )
        x = model.addVars(
            [(k, i, j) for k in R for i in V_without_last for j in V if i != j],
            vtype=GRB.BINARY,
            name="x",
        )

        # objective function (1)
        model.setObjective(
            gp.quicksum(t[k, i] for k in R for i in V_prime),
            GRB.MINIMIZE,
        )

        # constraint (2)
        model.addConstrs(
            (
                gp.quicksum(x[k, i, j] for j in V if i != j)
                == gp.quicksum(x[k, j, i] for j in V_without_last if i != j)
                for k in R
                for i in V_prime
            ),
            name="constraint_2 - flow conservation",
        )

        # constraint (3)
        model.addConstrs(
            (
                gp.quicksum(x[k, i, j] for j in V if i != j) == 1
                for k in R
                for i in V_prime
            ),
            name="constraint_3 - node visit",
        )

        # constraint (4)
        model.addConstrs(
            (gp.quicksum(x[k, 0, j] for j in V if j != 0) == 1 for k in R),
            name="constraint_4 - route start",
        )

        # constraint (5)
        model.addConstrs(
            (
                gp.quicksum(x[k, j, len(V) - 1] for j in V if j != (len(V) - 1)) == 1
                for k in R
            ),
            name="constraint_5 - route end",
        )

        # constraint (6)
        model.addConstrs(
            (
                gp.quicksum(
                    x[k, i, j] * u(self, i, j)
                    for i in V_without_last
                    for j in V
                    if i != j
                )
                <= self.max_travel_time
                for k in R
            ),
            name="constraint_6 - route length",
        )

        # constraint (7)
        model.addConstrs(
            (
                -t[k, i] + t[k, j] - G * x[k, i, j] <= u(self, i, j) - G
                for i in V_without_last
                for j in V_prime
                for k in R
                if i != j
            ),
            name="constraint_7 - arrival time",
        )

        # constraint (8)
        for k in R:
            for i in V:
                t[k, i].lb = 0

        return model, t, x
