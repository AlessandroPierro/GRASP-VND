from problem import CUAVRP
from utils import create_random_problem, euclidean_distance
from algorithms import GRASPVND, ThreadStrategy

if __name__ == "__main__":
    problem: CUAVRP = create_random_problem(
        m=3,
        n=5,
        density=0.6,
        distance_fn=euclidean_distance,
        num_uavs=3,
        max_travel_time=14,
    )

    grasp = GRASPVND(
        problem=problem,
        grasp_iters=250,
        num_constructed=5,
        vnd_iters=100,
        thread_strategy=ThreadStrategy.ALL_RETURN,
        num_threads=6,
        alpha=0.6,
        w_0=0.4,
        r_d=1,
    )

    solution = grasp.run()
    problem.plot_solution(solution)