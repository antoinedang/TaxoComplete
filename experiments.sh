#!/bin/bash

# Prompt user if they want to keep the data on SCRATCH or only save the job output
echo "Do you want to run the jobs on SCRATCH (less file storage but you keep the models)? (y/n)"
read -r keep_data

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

mkdir -p "$HOME/experiments"

# Loop through all the destination directories
if [ "$keep_data" == "y" ]; then
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
else
  for experiment in $experiments; do
    cd "$HOME/experiments/$experiment" 2>/dev/null && ./cancel.sh 2>/dev/null && echo "Cancelling previous $experiment job..." && sleep 5
    cd $HOME
    echo "Copying 'TaxoComplete/' to compute node..."
    rm -rf "experiments/$experiment"
    mkdir -p "experiments/$experiment"
    rm -rf "$SLURM_TMPDIR/experiments/$experiment"
    cp -r TaxoComplete "$SLURM_TMPDIR/experiments/$experiment"
    
    echo "Starting experiment '$experiment'"
    cd "$SLURM_TMPDIR/experiments/$experiment"
    touch job_output.txt job_error.txt sbatch_out.txt
    ./start.sh "$experiment_dir/$experiment"
    ln job_output.txt "$HOME/experiments/$experiment/job_output.txt"
    ln job_error.txt "$HOME/experiments/$experiment/job_error.txt"
    ln sbatch_out.txt "$HOME/experiments/$experiment/sbatch_out.txt"
  done
fi

cd $HOME/TaxoComplete

echo "Copying completed, jobs started."

