#!/bin/bash
for ((i=12; i<15; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-no.py -d $i -m NO all
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done