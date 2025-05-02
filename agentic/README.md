# Agentic QA Use Case

## Running the Agentic QA Benchmark

To run the agentic QA benchmark, use the following command:

```bash
python3 agentic-qa.py \
    --num-agents 3 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --user-request-interval 1 \
    --new-user-interval 5 \
    --base-url http://localhost:30080 \
    --trace-file demo.jsonl
```

The script will write each request's detailed stats to `summary.csv`.

*Note:* the above command requires there is a serving engine with the `meta-llama/Llama-3.1-8B-Instruct` model served at `http://localhost:30080/v1`.

### Arguments

#### Configuring the workload content
One way:
- `--num-rounds <int>`: The number of rounds per user.
- `--shared-system-prompt <int>`: Length of the system prompt shared across all users (in tokens).
- `--user-history-prompt <int>`: Length of the user-specific context (simulating existing chat history) (in tokens).
- `--answer-len <int>`: Length of the answer expected (in tokens).
- `--time <float>`: Total time to run the experiment (default = forever)

The other way:
- `--trace-file <str>`: The workload content file.

#### Configuring the workload trace
- `--user-request-interval <float>`: The delay (in seconds) between successive requests issued by the same user.
- `--new-user-interval <float>`: The delay (in seconds) between the arrival of new users into the simulation.
- `--num-agents <int>`: The number of agents.
- `--whole-history`: If this option is present, the script will pass the whole user chat history to the next agent; if this option is not present, the script will only pass the last agent's response to the next agent.

#### Configuring the serving engine connection
- `--model <str>`: The model name (e.g., `mistralai/Mistral-7B-Instruct-v0.2`).
- `--base-url <str>`: The URL endpoint for the language model server.

#### Configuring the experiment (Optional)
- `--log-interval <float>`: Time between each performance summary log in seconds (default = 30)
