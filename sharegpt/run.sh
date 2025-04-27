#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <model> <base url> <save file key> [qps_values...]"
    echo "Example: $0 meta-llama/Llama-3.1-8B-Instruct http://localhost:8000 /mnt/requests/sharegpt-run1 1.34 2.0 3.0"
    exit 1
fi

MODEL=$1
BASE_URL=$2
KEY=$3

# If QPS values are provided, use them; otherwise use default
if [[ $# -gt 3 ]]; then
    QPS_VALUES=("${@:4}")
else
    QPS_VALUES=(1.34)  # Default QPS value
fi

warm_up() {
    # $1: qps
    # $2: output file

    python3 "${SCRIPT_DIR}/sharegpt-qa.py" \
        --qps "$1" \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --output /tmp/warmup.csv \
        --log-interval 30 \
        --sharegpt-file "${SCRIPT_DIR}/warmup.json"

    sleep 10
}

run_benchmark() {
    # $1: qps
    # $2: output file

    # Real run
    python3 "${SCRIPT_DIR}/sharegpt-qa.py" \
        --qps "$1" \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --output "$2" \
        --log-interval 30 \
        --sharegpt-file "${SCRIPT_DIR}/run.json"

    sleep 10
}

# Run benchmarks for the specified QPS values
for qps in "${QPS_VALUES[@]}"; do
    output_file="${KEY}_output_${qps}.csv"
    warm_up "$qps"
    run_benchmark "$qps" "$output_file"
done
