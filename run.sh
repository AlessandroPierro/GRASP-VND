#!/bin/bash

# Set the number of problems to run
NUM_PROBLEMS=10

# do we need to activate the venv first?
source venv/bin/activate

# Define the command to run for each problem
COMMAND="python run_experiment.py --m 4 --n 4 --density 0.5 --num_uavs 2 --max_travel_time 16 --problem_seed {} --grasp_iters 100 --num_constructed 10 --vnd_iters 10 --thread_strategy ALL_RETURN --num_threads 3 --alpha 0.5 --w_0 0.4 --r_d 1 --algorithm_seed 0"

# Generate a list of problem seeds to run
SEEDS=$(seq 0 $((NUM_PROBLEMS-1)))

# Run the commands in parallel using GNU Parallel, limiting to 4 jobs at a time, with a progress bar
parallel --bar -j 4 "$COMMAND" ::: $SEEDS
