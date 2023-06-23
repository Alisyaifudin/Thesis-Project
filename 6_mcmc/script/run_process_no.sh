#!/bin/bash
for ((i=0; i<12; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running no $i at $current_time"
    python program-no.py -d $i all
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished no $i at $current_time"
done