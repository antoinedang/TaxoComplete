watch -n 0.1 "echo =======STDOUT======= && (tail job_output.txt || true) && echo '' && echo =======STDERR======= && (tail job_error.txt || true) && echo '' && echo =======STATUS======= && scontrol show job $(cat sbatch_out.txt | awk '{print $4}')"
