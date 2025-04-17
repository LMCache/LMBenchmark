#!/usr/bin/env python3
"""
sharegpt_benchmark.py – streamlined ShareGPT replay tool
=======================================================

Replays prompts from a ShareGPT‑style JSON file against an OpenAI‑compatible
HTTP endpoint at a fixed QPS, records latency metrics, and writes a CSV report
**sorted by launch_time** so downstream analyses have deterministic ordering.
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import openai
import pandas as pd

from utils import AsyncLoopWrapper, init_logger

logger = init_logger(__name__, logging.INFO)

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay ShareGPT prompts against an OpenAI‑compatible "
                    "endpoint and collect latency statistics.")

    parser.add_argument("--sharegpt-file", default="round_robin_1000_5.json",
                        help="JSON file with ShareGPT prompts (default: %(default)s)")
    parser.add_argument("--base-url", required=True,
                        help="Base URL of the OpenAI‑compatible server")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. gpt-4o-mini)")
    parser.add_argument("--qps", type=float, required=True,
                        help="Target queries per second")
    parser.add_argument("--output", default="summary.csv",
                        help="Output CSV filename (default: %(default)s)")
    parser.add_argument("--log-interval", type=int, default=30,
                        help="Seconds between progress logs (default: %(default)s)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Low‑level request handling
# ---------------------------------------------------------------------------

@dataclass
class Response:
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float


class RequestExecutor:
    """Thin wrapper over OpenAI async client that measures latency."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()

    async def _async_request(self, messages, max_tokens: int) -> Response:
        start = time.time()
        first_token: Optional[float] = None
        body = ""

        stream = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                if first_token is None:
                    first_token = time.time()
                body += delta

        usage = chunk.usage  # type: ignore[attr-defined]
        return Response(
            body=body,
            ttft=(first_token or time.time()) - start,
            generation_time=time.time() - (first_token or start),
            prompt_tokens=usage.prompt_tokens,
            generation_tokens=usage.completion_tokens,
            launch_time=start,
            finish_time=time.time(),
        )

    def launch_request(self, prompt: str, max_tokens: int, on_finish) -> None:
        messages = [{"role": "user", "content": prompt}]
        fut = asyncio.run_coroutine_threadsafe(
            self._async_request(messages, max_tokens), self.loop)
        fut.add_done_callback(lambda f: on_finish(f.result()))

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Dispatch prompts at desired QPS and collect latency metrics."""

    def __init__(self, prompts: List[dict], executor: RequestExecutor, qps: float):
        self.prompts = prompts
        self.executor = executor
        self.qps = qps
        self.results: List[Response] = []
        self._next_idx = 0

    def _on_finish(self, resp: Response):
        self.results.append(resp)

    def run(self) -> pd.DataFrame:
        start = time.time()
        logger.info("Benchmark started: %d prompts at %.2f QPS", len(self.prompts), self.qps)

        while self._next_idx < len(self.prompts):
            scheduled = start + self._next_idx / self.qps
            if time.time() < scheduled:
                time.sleep(0.001)
                continue

            entry = self.prompts[self._next_idx]
            prompt = entry["input"]
            max_tokens = entry.get("output_length", 1)
            self.executor.launch_request(prompt, max_tokens, self._on_finish)
            self._next_idx += 1

        AsyncLoopWrapper.WaitLoop()  # wait for inflight requests
        logger.info("All requests completed")

        df = pd.DataFrame({
            "prompt_tokens": [r.prompt_tokens for r in self.results],
            "generation_tokens": [r.generation_tokens for r in self.results],
            "ttft": [r.ttft for r in self.results],
            "generation_time": [r.generation_time for r in self.results],
            "launch_time": [r.launch_time for r in self.results],
            "finish_time": [r.finish_time for r in self.results],
        })

        # Ensure deterministic ordering for downstream scripts/visualisation
        return df.sort_values("launch_time").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def log_summary(df: pd.DataFrame):
    duration = df["finish_time"].max() - df["launch_time"].min()
    throughput = len(df) / duration if duration > 0 else 0
    logger.info("Completed %d requests in %.2fs (%.2f QPS)", len(df), duration, throughput)
    logger.info("Average TTFT: %.3fs", df["ttft"].mean())
    logger.info("Avg generation speed per req: %.1f tokens/s", (
        df["generation_tokens"] / df["generation_time"]).mean())

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.verbose:
        init_logger(__name__, logging.DEBUG)

    with open(args.sharegpt_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    logger.info("Loaded %d ShareGPT entries", len(data))

    executor = RequestExecutor(base_url=args.base_url, api_key="EMPTY", model=args.model)
    df = BenchmarkRunner(data, executor, args.qps).run()

    df.to_csv(args.output, index=False)
    logger.info("Results written to %s", args.output)
    log_summary(df)

    AsyncLoopWrapper.StopLoop()  # ensure script terminates

if __name__ == "__main__":
    main()
