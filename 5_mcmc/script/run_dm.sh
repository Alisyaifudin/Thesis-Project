#!/bin/bash
for ((i=0; i<4; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running n $i at $current_time"
    python program.py -d $i -m DM -t n all
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished n $i at $current_time"
done
for ((i=0; i<6; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running z $i at $current_time"
    python program.py -d $i -m DM -t z all
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished z $i at $current_time"
done
for ((i=0; i<9; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program.py -d $i -m DM -t d all
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done