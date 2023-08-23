import sys

sys.path.append("src")

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB

from algorithms import GRASPVND, ThreadStrategy
from problem import CUAVRP
from utils import create_random_problem, euclidean_distance

plt.rcParams.update({"font.size": 14})

if __name__ == "__main__":
    header = [
        "L",
        "min_gurobi_time",
        "mean_gurobi_time",
        "max_gurobi_time",
        "min_gurobi_gap",
        "mean_gurobi_gap",
        "max_gurobi_gap",
        "min_grasp_time",
        "mean_grasp_time",
        "max_grasp_time",
        "min_grasp_gap",
        "mean_grasp_gap",
        "max_grasp_gap",
        "gurobi_timed_out",
    ]

    data = []
    remaining_iters = 40

    for l in [3, 4, 5, 8]:
        gurobi_times = []
        grasp_times = []
        gurobi_timed_out = []
        gurobi_gap_from_opt = []
        grasp_gap_from_gurobi = []

        for seed in range(10):
            remaining_iters -= 1
            print("Remaining iters: ", remaining_iters)
            print("Estimated remaining time: ", remaining_iters * 10, " minutes = ", remaining_iters * 10 / 60, " hours")
            print("Running Gurobi for l = ", l, " and seed = ", seed)

            problem: CUAVRP = create_random_problem(
                m=l,
                n=l,
                density=0.45,
                distance_fn=euclidean_distance,
                num_uavs=2,
                max_travel_time=4*l*(2 if l == 8 else 1),
                seed=seed,
            )

            model, t, x = problem.to_gurobi_model(G=1000)
            model.setParam(GRB.Param.OutputFlag, 0)
            model.setParam(GRB.Param.TimeLimit, 5*60)
            model.optimize()

            gurobi_times.append(model.Runtime)
            gurobi_timed_out.append(model.status == GRB.TIME_LIMIT)
            gurobi_gap_from_opt.append(model.MIPGap * 100)

            # if found optimal solution
            if model.status == GRB.OPTIMAL:
                print("Optimal solution found")

            start = time.time()
            grasp = GRASPVND(
                problem=problem,
                grasp_iters=50 * problem.num_nodes,
                num_constructed=10,
                vnd_iters=5 * problem.num_nodes,
                thread_strategy=ThreadStrategy.BEST_CONTINUE,
                num_threads=6,
                alpha=0.7,
                w_0=1,
                r_d=0.1,
                r_i=0.01,
                seed=seed,
            )
            best_solution_grasp, best_solution_grasp_value = grasp.run()
            end = time.time()
            grasp_times.append(end - start)
            grasp_gap_from_gurobi.append(
                round(
                    (best_solution_grasp_value - model.objVal) / model.objVal * 100, 2
                )
            )
        data.append(
            [
                l,
                np.min(gurobi_times),
                np.mean(gurobi_times),
                np.max(gurobi_times),
                np.min(gurobi_gap_from_opt),
                np.mean(gurobi_gap_from_opt),
                np.max(gurobi_gap_from_opt),
                np.min(grasp_times),
                np.mean(grasp_times),
                np.max(grasp_times),
                np.min(grasp_gap_from_gurobi),
                np.mean(grasp_gap_from_gurobi),
                np.max(grasp_gap_from_gurobi),
                np.any(gurobi_timed_out),
            ]
        )

        df = pd.DataFrame(data, columns=header)
        df.to_csv("data.csv", index=False)
    # create plots

    # plot runtime with shaded area for min / max
    # change marker if timed out
    fig, ax = plt.subplots(figsize=(5, 3))
    for marker, timed_out in zip(["o", "x"], [False, True]):
        ax.plot(
            df[df["gurobi_timed_out"] == timed_out]["L"],
            df[df["gurobi_timed_out"] == timed_out]["mean_gurobi_time"],
            label="Gurobi optimal" if not timed_out else "Gurobi timed out",
            marker=marker,
            # increase marker size
            markersize=10,
        )
    # change marker if timed out
    ax.plot(df["L"], df["mean_grasp_time"], label="GRASP", marker="o", markersize=10)
    ax.fill_between(
        df["L"],
        df["min_gurobi_time"],
        df["max_gurobi_time"],
        alpha=0.2,
    )
    ax.fill_between(
        df["L"],
        df["min_grasp_time"],
        df["max_grasp_time"],
        alpha=0.2,
    )
    ax.set_xlabel("L")
    ax.set_ylabel("Runtime (s)")
    ax.legend()
    ax.grid(linestyle="dotted")
    plt.tight_layout()
    plt.savefig("runtime.png", dpi=600, bbox_inches="tight")

    print("Plotted runtime")

    # clear everything
    plt.clf()
    plt.cla()
    plt.close()

    # plot gap with shaded area for min / max
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["L"], df["mean_grasp_gap"], label="GRASP", marker="o", markersize=10)
    ax.fill_between(
        df["L"],
        df["min_grasp_gap"],
        df["max_grasp_gap"],
        alpha=0.2,
    )
    ax.set_xlabel("L")
    ax.set_ylabel("GRASP Gap vs Gurobi (%)")
    ax.legend()
    ax.grid(linestyle="dotted")
    plt.tight_layout()
    plt.savefig("gap.png", dpi=600, bbox_inches="tight")

    print("Plotted gap")
    
    # clear everything
    plt.clf()
    plt.cla()
    plt.close()

    # plot gap with shaded area for min / max

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(df["L"], df["mean_gurobi_gap"], label="Gurobi", marker="o", markersize=10)
    ax.fill_between(
        df["L"],
        df["min_gurobi_gap"],
        df["max_gurobi_gap"],
        alpha=0.2,
    )
    ax.set_xlabel("L")
    ax.set_ylabel("MIP Gap (%)")
    ax.legend()
    ax.grid(linestyle="dotted")
    plt.tight_layout()
    plt.savefig("mip_gap.png", dpi=600, bbox_inches="tight")

    print("Plotted gap")
    print(df)
