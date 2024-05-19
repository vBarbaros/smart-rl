#!/bin/bash

# Create a file to store PIDs
pid_file="training_pids.txt"
> "$pid_file"  # Clear the file if it already exists

# Loop from 2 to 10, incrementing by 2
#python get_augment_stats.py task=pendulum_swingup agent=drqv2 &
# Capture the PID of the last background process and store it
#echo $! >> "$pid_file"

#for i in $(seq 2 2 10); do
for i in $(seq 2 1 15); do
    # Run the Python script with the current agent value in the background
    python get_augment_stats.py task=pendulum_swingup agent=drqv2_aug augment_type=shift augment_param=$i
    # Capture the PID of the last background process and store it
    echo $! >> "$pid_file"
done

# Display the PIDs stored
echo "Running processes' PIDs:"
cat "$pid_file"

## Wait for all background processes to finish (optional)
#wait
#echo "All processes have completed."