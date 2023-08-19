import networkx as netx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

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
            edge_lengths = [
                self._graph.edges[route[i - 1], route[i]]["weight"]
                for i in range(1, len(route))
            ]
            total_cost += sum(np.cumsum(edge_lengths))
        return total_cost
