# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="/root/.cargo/bin:$PATH"' >> ~/.bashrc && \
    . ~/.bashrc

# Copy the benchmark code
COPY . /app/

# Set environment variables with defaults
ENV MODEL="meta-llama/Llama-3.1-8B-Instruct" \
    BASE_URL="http://localhost:8000" \
    SAVE_FILE_KEY="benchmark_results" \
    SCENARIOS="all" \
    QPS_VALUES="1.34" \
    PYTHONPATH="/app" \
    PATH="/root/.cargo/bin:$PATH" \
    NUM_USERS_WARMUP="20" \
    NUM_USERS="15" \
    NUM_ROUNDS="20" \
    SYSTEM_PROMPT="1000" \
    CHAT_HISTORY="20000" \
    ANSWER_LEN="100" \
    INIT_USER_ID="1" \
    TEST_DURATION="100" \
    USE_CHAT_COMPLETIONS="False"

# Create a virtual environment and install dependencies
RUN . ~/.bashrc && \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Make the script executable
RUN chmod +x /app/run_benchmarks.sh

# Set the entrypoint to run the benchmark script
ENTRYPOINT ["/bin/bash", "-c", ". ~/.bashrc && . .venv/bin/activate && /app/run_benchmarks.sh \"$MODEL\" \"$BASE_URL\" \"$SAVE_FILE_KEY\" \"$SCENARIOS\" \"$QPS_VALUES\""] 