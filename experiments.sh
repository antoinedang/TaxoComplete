#!/bin/bash

# Check if exactly two arguments are provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 experiment_dir"
  exit 1
fi

# Second argument is the directory containing the list of folders
experiment_dir="$1"

# Verify if the name_dir exists
if [ ! -d "$experiment_dir" ]; then
  echo "Error: Directory '$experiment_dir' does not exist."
  exit 1
fi

# Get the list of directories/files in experiment_dir
experiments=$(ls "$experiment_dir")

rm job_output.txt job_error.txt sbatch_out.txt 2&> /dev/null

# Loop through all the destination directories
for experiment in $experiments; do
  cd "$HOME/experiments/$experiment" 2>/dev/null && ./cancel.sh 2>/dev/null && echo "Cancelling previous $experiment job..." && sleep 5
  cd $HOME
  echo "Copying 'TaxoComplete/' to 'experiments/$experiment'..."
  rm -rf "experiments/$experiment"
  cp -r TaxoComplete "experiments/$experiment"
  
  echo "Starting experiment '$experiment'"
  cd "experiments/$experiment"
  ./start.sh "$experiment_dir/$experiment"
done

cd $HOME/TaxoComplete

echo "Copying completed, jobs started."

