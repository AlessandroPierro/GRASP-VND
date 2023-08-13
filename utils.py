from dataclasses import dataclass
from typing import Callable, List

import networkx as netx
import random
from problem import CUAVRP


@dataclass
class Point:
    x: float
    y: float


def euclidean_distance(p1: Point, p2: Point) -> float:
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** (1 / 2)


def create_random_problem(
    m: int,
    n: int,
    density: float,
    distance_fn: Callable[[Point, Point], float],
    num_uavs: int,
    max_travel_time: float,
) -> CUAVRP:
    """
    Utility function to instantiate a random CUAVRP problem.
    """
    points = random.sample(
        list(map(lambda i: (i // m, i % m), range(m * n))), int(m * n * density)
    )

    graph = netx.Graph()
    for i in range(len(points)):
        graph.add_node(i, pos=points[i])
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            graph.add_edge(
                i, j, weight=distance_fn(Point(*points[i]), Point(*points[j]))
            )

    return CUAVRP(graph=graph, num_uavs=num_uavs, max_travel_time=max_travel_time)


create_random_problem(
    m=5,
    n=10,
    density=0.4,
    distance_fn=euclidean_distance,
    num_uavs=3,
    max_travel_time=10,
)
