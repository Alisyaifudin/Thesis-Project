#!/bin/bash
current_time=$(date +"%Y-%m-%dT%H:%M:%S")
echo "Running data 0 at $current_time"
python program-mock.py -d 0 -m DM -t thin all
current_time=$(date +"%Y-%m-%dT%H:%M:%S")
echo "Finished data 0 at $current_time"
current_time=$(date +"%Y-%m-%dT%H:%M:%S")
echo "Running data 0 at $current_time"
python program-mock.py -d 0 -m DDDM -t thin all
current_time=$(date +"%Y-%m-%dT%H:%M:%S")
echo "Finished data 0 at $current_time"
current_time=$(date +"%Y-%m-%dT%H:%M:%S")
echo "Running data 0 at $current_time"
python program-mock.py -d 0 -m NO -t thin all
current_time=$(date +"%Y-%m-%dT%H:%M:%S")
echo "Finished data 0 at $current_time"