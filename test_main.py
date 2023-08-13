import unittest

import networkx as netx
from typing import Callable, List

from algorithms import CUAVRP


class TestCUAVRP(unittest.TestCase):


    def setUp(self) -> None:
        # generate 2d lattice o fpoints
        points = []
        for i in range(10):
            for j in range(10):
                points.append((i, j))

        # generate graph
        self.graph = create_undirected_graph(points)
        import matplotlib.pyplot as plt
        # plot without edges
        self.graph.remove_edges_from(self.graph.edges())
        netx.draw_networkx(self.graph, with_labels=True, pos=points)
        # draw 3 paths between points with different colours
        path1 = [0, 10, 15, 75, 43, 0]
        path2 = [4, 20, 22, 35, 70, 4]
        path3 = [5, 25, 30, 55, 80, 5]
        def draw_path(path, color):
            for i in range(len(path) - 1):
                netx.draw_networkx_edges(self.graph, pos=points, edgelist=[(path[i], path[i + 1])], edge_color=color)
        draw_path(path1, "red")
        draw_path(path2, "blue")
        draw_path(path3, "green")
        plt.savefig("graphnet.png", dpi=600)
        # generate graph
        self.graph = netx.gnm_random_graph(10, 20, seed=42)
        self.num_uavs = 3
        self.max_travel_time = 10.0
        self.problem = CUAVRP(
            graph=self.graph,
            num_uavs=self.num_uavs,
            max_travel_time=self.max_travel_time,
        )

    def test_create_obj(self) -> None:
        self.assertIsInstance(self.problem, CUAVRP)

    def test_graph_prop(self) -> None:
        self.assertEqual(self.problem.graph, self.graph)

    def test_num_nodes_prop(self) -> None:
        self.assertEqual(self.problem.num_nodes, self.graph.number_of_nodes())

    def test_num_uavs_prop(self) -> None:
        self.assertEqual(self.problem.num_uavs, self.num_uavs)

    def test_max_travel_time_prop(self) -> None:
        self.assertEqual(self.problem.max_travel_time, self.max_travel_time)
    
    def test_plot_graph(self) -> None:
        self.problem.plot_graph()


if __name__ == "__main__":
    unittest.main()
