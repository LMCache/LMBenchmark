import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional
import openai
import pandas as pd
from utils import AsyncLoopWrapper, init_logger

logger = init_logger(__name__, logging.INFO)
import json

def load_mooncake_data(filepath: str) -> list[dict]:
    data = []
    with open(filepath, "r") as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                required_fields = {"hash_ids", "timestamp", "output_length"}
                if not required_fields.issubset(record):
                    logger.warning(f"Line {line_num} missing required fields.")
                    continue
                data.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
    return data

# Load the Mooncake data
mooncake_data = load_mooncake_data("conversation_trace.jsonl")

@dataclass
class WorkloadConfig:
    # Length of shared system prompt
    system_prompt_len: int
    # Length of the user-specific data
    user_info_len: int
    # Length of the answer in one round
    answer_len: int
    # Number of rounds in the conversation
    num_rounds: int
    # Overall QPS
    qps: int
    # Model name
    model: str
    # Whether to include user id in request header
    enable_user_id: bool
    # slowdown factor
    slowdown_factor: float = 1.0
    # prefill only
    prefill_only: bool = True


@dataclass
class UserConfig:
    # User id
    user_id: int
    # System prompt length
    system_prompt_len: int
    # Length of the user-specific data
    user_info_len: int
    # Answer length
    answer_len: int
    # Num rounds
    num_rounds: int
    # Whether to include user id in request header
    enable_user_id: bool
    # prefill only
    prefill_only: bool = True

    @staticmethod
    def new_user_config(user_id: int, workload_config: WorkloadConfig) -> "UserConfig":
        return UserConfig(
            user_id=user_id,
            system_prompt_len=workload_config.system_prompt_len,
            user_info_len=workload_config.user_info_len,
            answer_len=workload_config.answer_len,
            num_rounds=workload_config.num_rounds,
            enable_user_id=workload_config.enable_user_id,
            prefill_only=workload_config.prefill_only,
        )


class ChatHistory:
    def __init__(
        self,
    ):
        self.history = []

    def on_user_query(self, query: str):
        if len(self.history) == 0:
            self.history.append({"role": "user", "content": query})
        else:
            assert self.history[-1]["role"] == "assistant", "Expect system response"
            self.history.append({"role": "user", "content": query})

    def on_system_response(self, response: str):
        assert len(self.history) > 0, "Expect user query"
        assert self.history[-1]["role"] == "user", "Expect user query"
        self.history.append({"role": "assistant", "content": response})

    def get_messages_for_openai(self):
        return self.history

    def __len__(self):
        return len(self.history)


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
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        self.request_history = []

    async def _async_launch_request(self, messages, max_tokens, extra_headers=None):
        start_time = time.time()
        first_token_time = None
        words = ""
        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
            extra_headers=extra_headers,
        )
        async for tok in response:
            if not tok.choices:
                continue
            chunk_message = tok.choices[0].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != "":
                    first_token_time = time.time()
                words += chunk_message
        tokens_out = tok.usage.completion_tokens
        tokens_prefill = tok.usage.prompt_tokens
        return Response(
            body=words,
            ttft=first_token_time - start_time,
            generation_time=time.time() - first_token_time,
            prompt_tokens=tokens_prefill,
            generation_tokens=tokens_out,
            launch_time=start_time,
            finish_time=time.time(),
        )

    def launch_request(
        self,
        chat_history: ChatHistory,
        max_tokens: int,
        finish_callback,
        extra_headers=None,
    ):
        """
        finish_callback: Callable[[Response], None]
        """
        messages = chat_history.get_messages_for_openai()
        real_callback = lambda x: finish_callback(x.result())
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens, extra_headers), self.loop
        )
        future.add_done_callback(real_callback)


class UserSession:
    def __init__(
        self,
        mooncake_id,
        user_config: UserConfig,
    ):
        self.user_config = user_config
        self.mooncake_id = mooncake_id
        self.last_request_time = None
        self.chat_history = ChatHistory()
        self.question_id = 0
        self.has_unfinished_request = False
        self.last_unfinished_log = 0
        self.prompt_lengths = []
        self.generation_lengths = []
        self.ttfts = []
        self.generation_times = []
        self.launch_times = []
        self.finish_times = []
        self.question_ids = []
        self.finished = False
        self.prefill_only = user_config.prefill_only

    def _update_result(self, response: Response):
        self.prompt_lengths.append(response.prompt_tokens)
        self.generation_lengths.append(response.generation_tokens)
        self.ttfts.append(response.ttft)
        self.generation_times.append(response.generation_time)
        self.launch_times.append(response.launch_time)
        self.finish_times.append(response.finish_time)
        self.question_ids.append(self.question_id - 1)

    def _build_system_prompt(self):
        def gen_dummy_text(length):
            return " ".join(["hi"] * length)

        dummy_text_sys = gen_dummy_text(self.user_config.system_prompt_len)
        dummy_text_user = gen_dummy_text(self.user_config.user_info_len)
        system_prompt = (
            f"Hi, here's some system prompt: {dummy_text_sys}."
            + f"For user {self.user_config.user_id}, "
            + f"here are some other context: {dummy_text_user}."
        )
        return system_prompt

    def _launch_new_request(self, timestamp: float, request_executor: RequestExecutor):
        hash_ids = mooncake_data[self.mooncake_id]["hash_ids"]
        prompt = ""
        for hash_id in hash_ids:
            prompt += f"{hash_id}" + " ".join(["hi"] * 512)
        prompt += "Can you tell me a detailed story in 1000 words?"
        logger.debug(
            f"User {self.user_config.user_id} issues request {self.question_id}, "
            f"prompt: {prompt}"
        )
        self.chat_history.on_user_query(prompt)
        logger.debug(
            f"User {self.user_config.user_id} issues request {self.question_id}"
        )
        if self.prefill_only:
            max_tokens = 1 # simulate prefill only
        else:
            max_tokens = mooncake_data[self.mooncake_id]["output_length"]
        request_executor.launch_request(
            self.chat_history,
            max_tokens,
            self._on_request_finished,
            extra_headers={"x-user-id": str(self.user_config.user_id)},
        )
        self.has_unfinished_request = True
        self.last_request_time = timestamp

    def _on_request_finished(self, response: Response):
        self.chat_history.on_system_response(response.body)
        self.has_unfinished_request = False
        logger.debug(
            f"User {self.user_config.user_id} finished one request. "
            f"Prompt tokens: {response.prompt_tokens}, "
            f"generation tokens: {response.generation_tokens}"
        )
        self._update_result(response)

    def step(self, timestamp: float, request_executor: RequestExecutor):
        if self.question_id >= 1 and not self.has_unfinished_request:
            self.finished = True
            return
        if self.last_request_time is None:
            self._launch_new_request(timestamp, request_executor)
            return

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["prompt_tokens"] = self.prompt_lengths
        df["generation_tokens"] = self.generation_lengths
        df["ttft"] = self.ttfts
        df["generation_time"] = self.generation_times
        df["user_id"] = self.user_config.user_id
        df["question_id"] = self.question_ids
        df["launch_time"] = self.launch_times
        df["finish_time"] = self.finish_times
        return df


class UserSessionManager:
    def __init__(
        self,
        workload_config: WorkloadConfig,
        init_user_id=0,
        time=0,
    ):
        self.initial_time = time
        self.workload_config = workload_config
        self.sessions = []
        self.user_id = init_user_id
        self.last_user_join = 0
        self.session_summaries = []
        self.start_time = None
        self.mooncake_request_to_send = 0

    def _create_user_session(self, mooncake_id):
        self.user_id += 1
        user_config = UserConfig.new_user_config(self.user_id, self.workload_config)
        user_session = UserSession(mooncake_id, user_config)
        self.sessions.append(user_session)
        return user_session

    def _remove_finished_sessions(self):
        sessions_to_remove = [s for s in self.sessions if s.finished]
        if len(sessions_to_remove) > 0:
            logger.info(
                f"Removing {len(sessions_to_remove)} finished sessions, now "
                f"active users: {len(self.sessions) - len(sessions_to_remove)}"
            )
            for session in sessions_to_remove:
                self.session_summaries.append(session.summary())
        self.sessions = [s for s in self.sessions if not s.finished]

    def step(self, timestamp: float, executor: RequestExecutor):
        
        if self.start_time is None:
            self.start_time = timestamp
        if (len(mooncake_data) > self.mooncake_request_to_send):
            if (
                timestamp - self.initial_time
                >= (mooncake_data[self.mooncake_request_to_send]["timestamp"] / 1000)
                * self.workload_config.slowdown_factor
            ):
                self._create_user_session(self.mooncake_request_to_send)
                self.last_user_join = timestamp
                logger.info(
                    f"Joined a new user {self.user_id}, "
                    f"now active users: {len(self.sessions)}, "
                    f"Slowdown factor: {self.workload_config.slowdown_factor}"
                )
                self.mooncake_request_to_send += 1
        for session in self.sessions:
            session.step(timestamp, executor)
        self._remove_finished_sessions()

    @staticmethod
    def ProcessSummary(
        df: pd.DataFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        pending_queries: int = 0,
        qps: Optional[int] = None,
    ):
        if start_time and end_time:
            launched_queries = len(
                df.query(f"{start_time} <= launch_time <= {end_time}")
            )
            df = df.query(f"{start_time} <= finish_time <= {end_time}")
        else:
            launched_queries = len(df)
        logger.debug(
            f"Launched queries: {launched_queries}, "
            f"pending queries: {pending_queries}, "
            f"finished queries: {len(df)}"
        )
        if qps is None:
            qps = 0.0
        if start_time is None:
            start_time = df["launch_time"].min()
        if end_time is None:
            end_time = df["finish_time"].max()
        total_time = end_time - start_time
        total_requests = launched_queries + pending_queries
        _qps = total_requests / total_time
        total_finished_requests = len(df)
        finished_qps = total_finished_requests / total_time
        total_prompt_tokens = df["prompt_tokens"].sum()
        total_generation_tokens = df["generation_tokens"].sum()
        average_prefill_speed = total_prompt_tokens / total_time
        average_generation_speed = total_generation_tokens / total_time
        average_generation_speed_per_request = (
            df["generation_tokens"] / df["generation_time"]
        ).mean()
        average_ttft = df["ttft"].mean()
        logger.info("Calculating performance summary")
        print("\n")
        print("==================== Performance summary ======================")
        print(f"  \033[33mQPS: \033[32m{qps:.4f} reqs/s\033[0m\n")
        print(
            f"  \033[33mProcessing speed: "
            f"\033[32m{finished_qps:.4f} reqs/s\033[0m\n"
        )
        print(f"  \033[33mRequests on-the-fly: {pending_queries}\033[0m\n")
        print(
            "  \033[33mInput tokens per second: "
            f"\033[32m{average_prefill_speed:.4f} tokens/s\033[0m\n"
        )
        print(
            "  \033[33mOutput tokens per second: "
            f"\033[32m{average_generation_speed:.4f} tokens/s\033[0m\n"
        )
        print(
            "  \033[33mAverage generation throughput (per request): "
            f"\033[32m{average_generation_speed_per_request:.4f} "
            "tokens/req/s\033[0m\n"
        )
        print(f"  \033[33mAverage TTFT: \033[32m{average_ttft:.4f}s\033[0m\n")
        print(f"Time range: {start_time} - {end_time} ({total_time:.2f}s)")
        print("===============================================================")
        print("\n")
        return df

    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        if len(self.session_summaries) == 0 and len(self.sessions) == 0:
            return pd.DataFrame()
        df = pd.concat(
            [s for s in self.session_summaries] + [s.summary() for s in self.sessions]
        )
        pending_queries = len([s for s in self.sessions if s.has_unfinished_request])
        start_time = max(self.start_time, start_time)
        end_time = min(end_time, df["finish_time"].max())
        qps = self.workload_config.qps
        df = UserSessionManager.ProcessSummary(
            df, start_time, end_time, pending_queries, qps
        )
        return df


def warmup_engine(executor):
    logger.info("Warming up the engine")
    for i in range(10):
        chat_history = ChatHistory()
        chat_history.on_user_query(
            f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}."
        )
        executor.launch_request(chat_history, 100, lambda x: None)
    AsyncLoopWrapper.WaitLoop()


def parse_arguments() -> WorkloadConfig:
    parser = argparse.ArgumentParser(description="Parse benchmark configurations.")
    parser.add_argument(
        "--shared-system-prompt",
        type=int,
        required=True,
        help="Length of the shared system prompt (tokens)",
    )
    parser.add_argument(
        "--user-history-prompt",
        type=int,
        required=True,
        help="Length of the user-specific history prompt (tokens)",
    )
    parser.add_argument(
        "--answer-len",
        type=int,
        required=True,
        help="Length of the answer in one round",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        required=True,
        help="Number of rounds in the conversation",
    )
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL of the serving engine endpoint",
    )
    parser.add_argument(
        "--time",
        type=int,
        required=False,
        help="The time to run the simulation in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name (ended with csv or txt) "
        "for the summary csv and txt",
    )
    parser.add_argument(
        "--init-user-id", type=int, default=0, help="The initial user id to start with"
    )
    parser.add_argument(
        "--request-with-user-id",
        action="store_true",
        help="Whether to enable user id in the request headers",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
        help="The time between two summary loggings in seconds",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Whether to enable verbose logging"
    )
    parser.add_argument(
        "--slowdown-factor",
        type=float,
        default=1.0,
        help="The slowdown factor for Mooncake",
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        default=True,
        help="Whether to only prefill the request without sending it",
    )
    args = parser.parse_args()
    return args


def parse_process_summary():
    parser = argparse.ArgumentParser(
        description="Parse benchmark configurations.", add_help=False
    )
    parser.add_argument("--process-summary", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


def process_output(filename):
    logger.warning(
        f"Processing the existing summary file {filename}"
        ", ignoring all the other arguments"
    )
    UserSessionManager.ProcessSummary(pd.read_csv(filename), pending_queries=0)


def main():
    args = parse_process_summary()
    if args.process_summary:
        process_output(args.process_summary)
        return
    args = parse_arguments()
    if args.verbose:
        global logger
        logger = init_logger(__name__, level=logging.DEBUG)
    step_interval = 0.1
    executor = RequestExecutor(
        base_url=args.base_url, api_key="EMPTY", model=args.model
    )
    warmup_engine(executor)
    workload_config = WorkloadConfig(
        system_prompt_len=args.shared_system_prompt,
        user_info_len=args.user_history_prompt,
        answer_len=args.answer_len,
        num_rounds=args.num_rounds,
        qps=args.qps,
        model=args.model,
        enable_user_id=args.request_with_user_id,
        slowdown_factor=args.slowdown_factor,
        prefill_only=args.prefill_only,
    )
    start_time = time.time()
    manager = UserSessionManager(
        workload_config,
        init_user_id=args.init_user_id,
        time=start_time,
    )
    num_steps = 0
    last_summary_time = start_time
    try:
        while True:
            num_steps += 1
            manager.step(time.time(), executor)
            time.sleep(step_interval)
            if time.time() - last_summary_time > args.log_interval:
                manager.summary(last_summary_time, time.time())
                last_summary_time = time.time()
            if args.time is not None and time.time() - start_time > args.time:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")
    AsyncLoopWrapper.StopLoop()
    logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    summary = manager.summary(0, time.time())
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
