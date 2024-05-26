#!/bin/bash

# Create a file to store PIDs
#pid_file="training_pids.txt"
#> "$pid_file"  # Clear the file if it already exists

# Loop from 2 to 10, incrementing by 2
#python get_augment_stats.py task=pendulum_swingup agent=drqv2 &
# Capture the PID of the last background process and store it
#echo $! >> "$pid_file"

#for i in $(seq 2 2 10); do
for i in $(seq 21 1 32); do
#for i in 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0; do
#for i in 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0; do
    # Run the Python script with the current agent value in the background
    python get_augment_stats.py task=pendulum_swingup agent=drqv2_aug augment_type=sharp augment_param=$i
    # Capture the PID of the last background process and store it
#    echo $! >> "$pid_file"
done

# Display the PIDs stored
echo "...DONE"
#echo "Running processes' PIDs:"
#cat "$pid_file"

## Wait for all background processes to finish (optional)
#wait
#echo "All processes have completed."