experiments_dir="$HOME/experiments"

experiments=$(ls "$experiments_dir")

cd "$experiments_dir"

for experiment in $experiments; do
	cd "$experiment"

	status=$(scontrol show job $(cat sbatch_out.txt | awk '{print $4}') 2> /dev/null | grep JobState | awk -F'=' '{for(i=1;i<=NF;i++) if($i ~ /JobState/) {split($(i+1), arr, " "); print arr[1]; break}}')
	if [ -z $status ]; then
		status="DONE"
	fi
	echo "$experiment   ->   $status"
	cd ..
done
cd $HOME/TaxoComplete
