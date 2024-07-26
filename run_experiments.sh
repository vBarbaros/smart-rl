#!/bin/bash

# Create a file to store PIDs
#pid_file="training_pids.txt"
#> "$pid_file"  # Clear the file if it already exists

# Loop from 2 to 10, incrementing by 2
#python get_augment_stats.py task=pendulum_swingup agent=drqv2 &
# Capture the PID of the last background process and store it
#echo $! >> "$pid_file"

#for i in $(seq 2 2 10); do
#for i in $(seq 0 1 10); do
#for i in 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0; do
#for i in 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0; do

#for contrast
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9; do
#for i in 1.5; do

#for i in 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.1 2.2 2.3 2.4 2.5; do

# for shift
#for i in 0 1 2 3 4 5 6 7 8 9 10; do

# for rotate
#for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 4 5 7 9 11 13 15; do
    # Run the Python script with the current agent value in the background
#    python get_augment_stats.py task=cartpole_balance_sparse agent=drqv2_aug augment_type=rotate augment_param=$i
#    python get_augment_stats.py task=finger_turn_hard agent=drqv2_aug augment_type=rotate augment_param=$i
#    python get_augment_stats.py task=pendulum_swingup agent=drqv2_aug augment_type=contrast augment_param=$i
    python get_augment_stats.py task=cartpole_balance_sparse agent=drqv2_aug augment_type=contrast augment_param=$i
#    python get_augment_stats.py task=quadruped_run agent=drqv2_aug augment_type=rotate augment_param=$i
#    python get_augment_stats.py task=reacher_hard agent=drqv2_aug augment_type=rotate augment_param=$i
#    python get_augment_stats.py task=walker_run agent=drqv2_aug augment_type=contrast augment_param=$i
    # Capture the PID of the last background process and store it
#    echo $! >> "$pid_file"
done