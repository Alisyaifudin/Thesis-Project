#!/bin/bash
# Run the Python script in parallel
for ((i=0, j=30; i<=330; i+=30, j+=30)); do
  python -u gaia-tmass.py $i $j -90 90 | tee log/log-$i-$j.txt &
done

# Wait for all background processes to finish
wait