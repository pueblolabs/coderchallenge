#!/bin/bash

# This script serves as the entry point for the reimbursement calculation.
# It now passes the case index as a fourth argument.
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount> <case_index>

set -e

# Execute the Python script with all provided arguments
# The $(dirname "$0") ensures it finds the python script even if run from another directory.
python3 "$(dirname "$0")/calculate_reimbursement.py" "$1" "$2" "$3" "$4"