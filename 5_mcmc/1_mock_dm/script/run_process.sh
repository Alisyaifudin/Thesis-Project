#!/bin/bash
for ((i=0; i<5; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running z $i at $current_time"
    python program-z.py -d $i all
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished z $i at $current_time"
done
for ((i=0; i<7; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running n $i at $current_time"
    python program-n.py n -d $i all
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished n $i at $current_time"
done