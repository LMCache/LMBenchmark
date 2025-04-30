# Start Engines

To run the benchmark scripts, you must have a locally hosted model-serving system. For example, to launch it for the 70B model:

1. Change into the `70B` directory.
1. Execute the provided startup script.

This will start a router at `http://localhost:30080/v1` serving as the endpoint and a `Llama-3.1-70B-Instruct` model.
Repeat the same steps in the `8B` directory to serve the `Llama-3.1-8B-Instruct` model.

## vLLM v1 + LMCache

To install vLLM v1:

```bash
git clone --branch local-dev/lmcache-v1-connector-clean https://github.com/ApostaC/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

To install LMCache:

```bash
git clone https://github.com/LMCache/LMCache.git  
cd LMCache  
pip install -e .
```

Then run:

```bash
bash start_router.sh
bash start_lmcache_v1.sh
```

## vLLM v1

To install vLLM v1:

```bash
git clone --branch local-dev/lmcache-v1-connector-clean https://github.com/ApostaC/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

Then run

```bash
bash start_router.sh
bash start_vllm_v1.sh
```

## vLLM v0 + LMCache

```bash
helm repo add vllm https://vllm-project.github.io/production-stack
helm install vllm vllm/vllm-stack -f lmcache.yaml
kubectl port-forward svc/vllm-router-service 30080:80
```

## vLLM v0

```bash
helm repo add vllm https://vllm-project.github.io/production-stack
helm install vllm vllm/vllm-stack -f vllm.yaml
kubectl port-forward svc/vllm-router-service 30080:80
```

## SGLang

To install SGLang:

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.5.post2"
```

Then run:

```bash
bash start_router.sh
bash start_sglang.sh
```

## Dynamo

To set up and launch the Dynamo engine deployments for benchmarking:
1. Clone the repository and check out the specific commit:
    ```bash
    git clone https://github.com/ai-dynamo/dynamo.git
    cd dynamo

    # For vLLM 0.7.2
    git checkout 2972b7ed26231421e606658c344063d30e2e3862
    
    # For vLLM 0.8.4 (V0)
    git checkout 183941fae1250a658807eb9f88076de857d69750
    ```

2. Start the Docker environment:
    ```bash
    docker compose -f deploy/docker-compose.yml up -d
    ./container/build.sh
    ```

3. Run the container (replace `<PATH TO DYNAMO>` with the absolute path to your cloned repo):
    ```bash
    ./container/run.sh -it --image zhuohangu/dynamo:latest-vllm -v <PATH TO DYNAMO>:/workspace/dynamo
    ```

4. Outside the container, copy the appropriate model config file into the dynamo/configs/ directory:
    ```bash
    # For meta-llama/Llama-3.1-8B-Instruct
    cp LMBenchmark/run_scripts/8B/agg_router_8B_<VERSION>.yaml dynamo/configs/

    # For meta-llama/Llama-3.1-70B-Instruct
    cp LMBenchmark/run_scripts/70B/agg_router_70B_<VERSION>.yaml dynamo/configs/
    ```

    > 💡 Set `<VERSION>` to `0.7.2` or `0.8.4` depending on the commit you checked out. Use the same version string in the following steps.

5. Inside the container, launch the vLLM engine deployments:
    ```bash
    cd /workspace/dynamo/examples/llm
    
    # For meta-llama/Llama-3.1-8B-Instruct
    dynamo serve graphs.agg_router:Frontend -f ./configs/agg_router_8B_<VERSION>.yaml

    # For meta-llama/Llama-3.1-70B-Instruct
    dynamo serve graphs.agg_router:Frontend -f ./configs/agg_router_70B_<VERSION>.yaml
    ```

This will start a Dynamo-based router at http://localhost:8000/v1, serving either the Llama-3.1-8B or Llama-3.1-70B model, depending on the config you launch. You can now use this endpoint for benchmarking, just like with the other setups above.

# Model Selection

For ShareGPT and short-input-short-output traces, use the 70B model with 2*A100 (80G). For the long-input-long-output trace, use the 8B model with 1*A100 (40G).
