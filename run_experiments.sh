#!/bin/bash

# Loop from 2 to 10, incrementing by 2
for i in $(seq 2 2 10); do
    python train.py task=pendulum_swingup agent=drqv2_pad_$i
done