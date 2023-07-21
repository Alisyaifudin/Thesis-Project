#!/bin/bash
for ((i=0; i<15; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-no.py -d $i -m DM plot_chain
    python program-no.py -d $i -m DM plot_corner
    python program-no.py -d $i -m DM plot_fit
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done
for ((i=0; i<15; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-no.py -d $i -m DDDM plot_chain
    python program-no.py -d $i -m DDDM plot_corner
    python program-no.py -d $i -m DDDM plot_fit
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done
for ((i=0; i<15; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-no.py -d $i -m NO plot_chain
    python program-no.py -d $i -m NO plot_corner
    python program-no.py -d $i -m NO plot_fit
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done