#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <model> <base url> <save file key> [scenarios...] [qps_values...]"
    echo ""
    echo "Scenarios:"
    echo "  sharegpt        - ShareGPT benchmark"
    echo "  short-input     - Short input, short output benchmark"
    echo "  long-input      - Long input, short output benchmark"
    echo "  all            - Run all benchmarks"
    echo ""
    echo "Examples:"
    echo "  # Run all benchmarks with default QPS"
    echo "  $0 meta-llama/Llama-3.1-8B-Instruct http://localhost:8000 /mnt/requests/benchmark all"
    echo ""
    echo "  # Run specific benchmarks with custom QPS"
    echo "  $0 meta-llama/Llama-3.1-8B-Instruct http://localhost:8000 /mnt/requests/benchmark sharegpt short-input 1.34 2.0 3.0"
    exit 1
fi

MODEL=$1
BASE_URL=$2
KEY=$3

# Parse scenarios and QPS values
SCENARIOS=()
QPS_VALUES=()
found_scenarios=true

for arg in "${@:4}"; do
    if [[ "$arg" == "sharegpt" || "$arg" == "short-input" || "$arg" == "long-input" || "$arg" == "all" ]]; then
        SCENARIOS+=("$arg")
    else
        found_scenarios=false
        QPS_VALUES+=("$arg")
    fi
done

# If no scenarios specified, default to all
if [ ${#SCENARIOS[@]} -eq 0 ]; then
    SCENARIOS=("all")
fi

# If no QPS values specified, use defaults for each scenario
if [ ${#QPS_VALUES[@]} -eq 0 ]; then
    QPS_VALUES=()
fi

# Function to run ShareGPT benchmark
run_sharegpt() {
    echo "Running ShareGPT benchmark..."
    if [ ${#QPS_VALUES[@]} -eq 0 ]; then
        "${SCRIPT_DIR}/sharegpt/run.sh" "$MODEL" "$BASE_URL" "${KEY}_sharegpt"
    else
        "${SCRIPT_DIR}/sharegpt/run.sh" "$MODEL" "$BASE_URL" "${KEY}_sharegpt" "${QPS_VALUES[@]}"
    fi
}

# Function to run short input benchmark
run_short_input() {
    echo "Running short input benchmark..."
    if [ ${#QPS_VALUES[@]} -eq 0 ]; then
        "${SCRIPT_DIR}/synthetic-multi-round-qa/short_input_short_output.sh" "$MODEL" "$BASE_URL" "${KEY}_short_input"
    else
        "${SCRIPT_DIR}/synthetic-multi-round-qa/short_input_short_output.sh" "$MODEL" "$BASE_URL" "${KEY}_short_input" "${QPS_VALUES[@]}"
    fi
}

# Function to run long input benchmark
run_long_input() {
    echo "Running long input benchmark..."
    
    # Then run the actual benchmark
    if [ ${#QPS_VALUES[@]} -eq 0 ]; then
        "${SCRIPT_DIR}/synthetic-multi-round-qa/long_input_short_output_run.sh" "$MODEL" "$BASE_URL" "${KEY}_long_input"
    else
        "${SCRIPT_DIR}/synthetic-multi-round-qa/long_input_short_output_run.sh" "$MODEL" "$BASE_URL" "${KEY}_long_input" "${QPS_VALUES[@]}"
    fi
}

# Run selected scenarios
for scenario in "${SCENARIOS[@]}"; do
    case "$scenario" in
        "sharegpt")
            run_sharegpt
            ;;
        "short-input")
            run_short_input
            ;;
        "long-input")
            run_long_input
            ;;
        "all")
            run_sharegpt
            run_short_input
            run_long_input
            ;;
    esac
done 
