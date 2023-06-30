#!/bin/bash
for ((i=5; i<6; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running $i at $current_time"
    python program-metal.py -d $i all
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished $i at $current_time"
done