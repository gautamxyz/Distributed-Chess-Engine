#!/usr/bin/bash

method=$1
depth=$2

for i in {1..10}; do
    echo "Running $method against random move maker. Depth = $depth. Number of procesess = $i. Avg. Time taken:"
    logfile="tmp/${method}_${depth}_${i}.log"
    mpirun -n $i python3 main.py --method $method --depth $depth --simulate > /dev/null 2> $logfile
    echo $(awk '{ sum += $6; n++ } END { if (n > 0) print sum / n; }' $logfile)

done
