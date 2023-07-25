#!/bin/bash
for ((i=0; i<3; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-mock.py -d $i -m DM -t thin calculate_prob
    python program-mock.py -d $i -m DM -t thic calculate_prob
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done
for ((i=0; i<3; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-mock.py -d $i -m DDDM -t thin calculate_prob
    python program-mock.py -d $i -m DDDM -t thic calculate_prob
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done
for ((i=0; i<3; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-mock.py -d $i -m NO -t thin calculate_prob
    python program-mock.py -d $i -m NO -t thic calculate_prob
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done