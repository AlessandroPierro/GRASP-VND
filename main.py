from problem import CUAVRP
from utils import create_random_problem, euclidean_distance
from algorithms import GRASPVND, ThreadStrategy

if __name__ == "__main__":
    problem: CUAVRP = create_random_problem(
        m=4,
        n=4,
        density=0.5,
        distance_fn=euclidean_distance,
        num_uavs=2,
        max_travel_time=16,
    )

    grasp = GRASPVND(
        problem=problem,
        grasp_iters=problem.num_nodes * 100,
        num_constructed=10,
        vnd_iters=problem.num_nodes * 10,
        thread_strategy=ThreadStrategy.ALL_RETURN,
        num_threads=6,
        alpha=0.5,
        w_0=0.4,
        r_d=1,
    )

    solution = grasp.run()
    problem.plot_solution(solution)