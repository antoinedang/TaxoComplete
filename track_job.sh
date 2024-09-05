job_id=$(cat sbatch_out.txt | awk '{print $4}')

watch -n 0.1 "echo =======STDOUT======= && (tail job_output.txt || true) && echo '' && echo =======STDERR======= && tail job_error.txt && echo '' && echo =======STATUS======= && scontrol show job $job_id"