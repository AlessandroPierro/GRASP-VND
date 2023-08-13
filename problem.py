import networkx as netx
import numpy as np


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

    def is_feasible(self, solution: set) -> bool:
        """Returns True if the solution is feasible, False otherwise."""
        if solution is None:
            return False
        else:
            print(f"solution {solution}")
            sol_set = set([0])
            for s in solution:
                for e in s:
                    sol_set.add(e)
                    print("adding e ", e)
            print("sol_set", sol_set)
            if len(sol_set) == self.num_nodes:
                return True
            else:
                return False
    
    def plot_graph(self) -> None:
        """Plots the graph of the problem, using netx and seaborn."""
        #plot
        import seaborn as sns
        from matplotlib import pyplot as plt
        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(10, 10))
        netx.draw_networkx(self.graph, ax=ax)
        plt.savefig("graph.png")

