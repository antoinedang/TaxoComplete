#!/bin/bash

scancel $(squeue -u $USER | grep antoine | awk '{print $1}')
echo "All cancellations completed."

