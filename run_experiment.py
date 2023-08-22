import sys

sys.path.append("src")

from problem import CUAVRP
from utils import create_random_problem, euclidean_distance
from algorithms import GRASPVND, ThreadStrategy
from argparse import ArgumentParser

# add a argument parser to run different experiments


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--num_uavs", type=int, default=2)
    parser.add_argument("--max_travel_time", type=int, default=16)
    parser.add_argument("--problem_seed", type=int, default=0)
    parser.add_argument("--grasp_iters", type=int, default=100)
    parser.add_argument("--num_constructed", type=int, default=10)
    parser.add_argument("--vnd_iters", type=int, default=10)
    parser.add_argument("--thread_strategy", type=str, default="ALL_RETURN")
    parser.add_argument("--num_threads", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--w_0", type=float, default=0.4)
    parser.add_argument("--r_d", type=int, default=1)
    parser.add_argument("--algorithm_seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    problem: CUAVRP = create_random_problem(
        m=args.m,
        n=args.n,
        density=args.density,
        distance_fn=euclidean_distance,
        num_uavs=args.num_uavs,
        max_travel_time=args.max_travel_time,
        seed=args.problem_seed,
    )

    grasp = GRASPVND(
        problem=problem,
        grasp_iters=args.grasp_iters,
        num_constructed=args.num_constructed,
        vnd_iters=args.vnd_iters,
        thread_strategy=ThreadStrategy[args.thread_strategy],
        num_threads=args.num_threads,
        alpha=args.alpha,
        w_0=args.w_0,
        r_d=args.r_d,
        seed=args.algorithm_seed,
    )

    solution = grasp.run()
    problem.plot_solution(solution)
