#!/bin/bash
for ((i=0; i<3; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-metal.py -d $i -m DDDM all
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done