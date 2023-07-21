#!/bin/bash
# z
for ((i=0; i<4; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-mock.py -d $i -m DM -t z plot_chain
    python program-mock.py -d $i -m DM -t z plot_corner
    python program-mock.py -d $i -m DM -t z plot_fit
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done
for ((i=0; i<4; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-mock.py -d $i -m DDDM -t z plot_chain
    python program-mock.py -d $i -m DDDM -t z plot_corner
    python program-mock.py -d $i -m DDDM -t z plot_fit
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done
for ((i=0; i<4; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-mock.py -d $i -m NO -t z plot_chain
    python program-mock.py -d $i -m NO -t z plot_corner
    python program-mock.py -d $i -m NO -t z plot_fit
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done
# n
for ((i=0; i<4; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-mock.py -d $i -m DM -t n plot_chain
    python program-mock.py -d $i -m DM -t n plot_corner
    python program-mock.py -d $i -m DM -t n plot_fit
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done
for ((i=0; i<4; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-mock.py -d $i -m DDDM -t n plot_chain
    python program-mock.py -d $i -m DDDM -t n plot_corner
    python program-mock.py -d $i -m DDDM -t n plot_fit
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done
for ((i=0; i<4; i++))
do
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Running data $i at $current_time"
    python program-mock.py -d $i -m NO -t n plot_chain
    python program-mock.py -d $i -m NO -t n plot_corner
    python program-mock.py -d $i -m NO -t n plot_fit
    current_time=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "Finished data $i at $current_time"
done