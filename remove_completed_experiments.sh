#!/bin/bash

# Path to the CSV file
csv_file="experiment_results.csv"

# Skip the header row and iterate through the rows
tail -n +2 "$csv_file" | while IFS=',' read -r experiment_name MR recall1 recall5 recall10; do
  # Construct the file path
  file_path="scripts/experiments/$experiment_name"
  
  # Delete the file if it exists
  if [ -f "$file_path" ]; then
    rm "$file_path"
    echo "Deleted: $file_path"
  fi

  find /path/to/target/folder -type f -exec bash -c '[[ "$(basename "{}")" == "target_filename" ]] && echo "Deleted: {}"' \;
  
  find /path/to/target/folder -type f -exec bash -c '[[ "$(basename "{}")" == "target_filename" ]] && rm -f "{}"' \;
  
done

