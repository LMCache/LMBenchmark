# Start Engines

To run the benchmark scripts, you must have a locally hosted model-serving system. For example, to launch it for the 70B model:

1. Change into the `70B` directory.
1. Execute the provided startup script.

This will start a router at `http://localhost:30080/v1` serving as the endpoint and a `Llama-3.1-70B-Instruct` model.
Repeat the same steps in the `8B` directory to serve the `Llama-3.1-8B-Instruct` model.

## vLLM v1 + LMCache

Checkout and install `https://github.com/ApostaC/vllm/tree/local-dev/lmcache-v1-connector-clean`.

```bash
bash start_router.sh
bash start_lmcache_v1.sh
```

## vLLM v1

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

```bash
bash start_router.sh
bash start_sglang.sh
```

## Dynamo
