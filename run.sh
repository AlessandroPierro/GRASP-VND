#!/bin/bash

source venv/bin/activate

seeds=( 1 2 3 4 5 )

strategies=( "ALL_RETURN" "RANDOM_RETURN" "BEST_CONTINUE" )

parallel --bar --jobs 3 --delay 5s python3 run_experiment.py --algorithm_seed {1} --thread_strategy {2} --m 4 --n 4 --density 0.5 --num_uavs 2 --max_travel_time 16  ::: "${seeds[@]}" ::: "${strategies[@]}" 
parallel --bar --jobs 3 --delay 5s python3 run_experiment.py --algorithm_seed {1} --thread_strategy {2} --m 8 --n 8 --density 0.25 --num_uavs 2 --max_travel_time 32  ::: "${seeds[@]}" ::: "${strategies[@]}"
parallel --bar --jobs 3 --delay 5s python3 run_experiment.py --algorithm_seed {1} --thread_strategy {2} --m 10 --n 10 --density 0.25 --num_uavs 4 --max_travel_time 32  ::: "${seeds[@]}" ::: "${strategies[@]}"
parallel --bar --jobs 3 --delay 5s python3 run_experiment.py --algorithm_seed {1} --thread_strategy {2} --m 12 --n 12 --density 0.25 --num_uavs 8 --max_travel_time 64  ::: "${seeds[@]}" ::: "${strategies[@]}"
parallel --bar --jobs 3 --delay 5s python3 run_experiment.py --algorithm_seed {1} --thread_strategy {2} --m 14 --n 14 --density 0.125 --num_uavs 8 --max_travel_time 64  ::: "${seeds[@]}" ::: "${strategies[@]}"