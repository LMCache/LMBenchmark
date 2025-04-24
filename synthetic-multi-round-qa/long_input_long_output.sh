#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <model> <base url> <save file key> [qps_values...]"
    echo "Example: $0 meta-llama/Llama-3.1-8B-Instruct http://localhost:8000 test 15 20 25"
    exit 1
fi

MODEL=$1
BASE_URL=$2
KEY=$3

# Configuration
NUM_USERS_WARMUP=750
NUM_USERS=320
NUM_ROUNDS=20
SYSTEM_PROMPT=0
CHAT_HISTORY=2000
ANSWER_LEN=1000

# If QPS values are provided, use them; otherwise use default
if [ $# -gt 3 ]; then
    QPS_VALUES=("${@:4}")
else
    QPS_VALUES=(0.7)  # Default QPS value
fi

# init-user-id starts at 1, will add 400 each iteration
INIT_USER_ID=1

warmup() {
    echo "Warming up with QPS=$((NUM_USERS_WARMUP / 2))..."
    python3 "${SCRIPT_DIR}/multi-round-qa.py" \
        --num-users 1 \
        --num-rounds 2 \
        --qps 2 \
        --shared-system-prompt "$(echo -n "$SYSTEM_PROMPT" | wc -w)" \
        --user-history-prompt "$(echo -n "$CHAT_HISTORY" | wc -w)" \
        --answer-len $ANSWER_LEN \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --init-user-id "$INIT_USER_ID" \
        --output /tmp/warmup.csv \
        --log-interval 30 \
        --time $((NUM_USERS_WARMUP / 2))
}

run_benchmark() {
    local qps=$1
    local output_file="${KEY}_qps${qps}.csv"

    # warmup with current init ID
    warmup
    
    # actual benchmark with same init ID
    echo "Running benchmark with QPS=$qps..."
    python3 "${SCRIPT_DIR}/multi-round-qa.py" \
        --num-users "$NUM_USERS" \
        --shared-system-prompt "$(echo -n "$SYSTEM_PROMPT" | wc -w)" \
        --user-history-prompt "$(echo -n "$CHAT_HISTORY" | wc -w)" \
        --answer-len "$ANSWER_LEN" \
        --num-rounds "$NUM_ROUNDS" \
        --qps "$qps" \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --init-user-id "$INIT_USER_ID" \
        --output "$output_file" \
        --time 100
    
    sleep 10

    # increment init-user-id by NUM_USERS_WARMUP
    INIT_USER_ID=$(( INIT_USER_ID + NUM_USERS_WARMUP ))
}

# Run benchmarks for each QPS value
for qps in "${QPS_VALUES[@]}"; do
    run_benchmark "$qps"
done
