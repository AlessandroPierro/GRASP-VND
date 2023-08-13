from problem import CUAVRP
from utils import create_random_problem, euclidean_distance
from algorithms import ConstructiveHeuristic

if __name__ == '__main__':
    problem: CUAVRP = create_random_problem(
        m=3,
        n=5,
        density=0.6,
        distance_fn=euclidean_distance,
        num_uavs=3,
        max_travel_time=10
    )

    heuristic = ConstructiveHeuristic(
        problem,
        alpha=0.5,
        w_0 = 0.4,
        r_d=1
    )

    print(heuristic.construct_feasible_solution())