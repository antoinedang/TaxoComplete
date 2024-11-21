scancel $(cat sbatch_out.txt | awk '{print $4}')
