#!/bin/bash
for ((i=0; i<7; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running $i at $current_time"
    python program-comp.py -d $i all
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished $i at $current_time"
done