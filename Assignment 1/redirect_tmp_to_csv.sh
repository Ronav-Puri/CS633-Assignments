#!/bin/bash
echo "P,M,maxD1,maxD2,Time" > timings.csv
cat raw_results/*.tmp >> timings.csv
echo "Done! Results saved in timings.csv"
