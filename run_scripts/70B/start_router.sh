#!/bin/bash/
vllm-router --port 30080 \
    --service-discovery static \
    --static-backends "http://localhost:8100" \
    --static-models "meta-llama/Llama-3.1-70B-Instruct" \
    --engine-stats-interval 10 \
    --log-stats \
    --routing-logic roundrobin