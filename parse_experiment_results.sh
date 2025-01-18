#!/bin/bash

echo "experiment_name,recall@1,recall@5,recall@10,MR"

# Process each file
for file in ../experiments/*/job_output.txt; do
    # Extract the relevant part of the filename
    filename=$(basename "$(dirname "$file")")

    # Extract the last line with data and format the output
    awk -v fname="$filename" '
    BEGIN { FS=","; OFS="," }
    /prec@1/ { getline last }
    END {
        if (last) {
            split(last, values, ",")
            print fname, values[7], values[4], values[5], values[6]
        }
    }
    ' "$file"
done
