#!/bin/bash

output_file="experiment_results.csv"

if [ ! -f "$output_file" ]; then
	echo "experiment_name,MR,recall@1,recall@5,recall@10" > $output_file
    echo "original,psy,560.6,0.17,0.392,0.488" >> $output_file
    echo "original,semeval_verb,589.3,0.123,0.316,0.421" >> $output_file
fi

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
    ' "$file" >> $output_file
done

echo "Experiment results have been saved to $output_file"
