import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import openai
import pandas as pd

from utils import AsyncLoopWrapper, init_logger

logger = init_logger(__name__, logging.INFO)


@dataclass
class WorkloadConfig:
    num_agents: int
    model: List[str]
    user_request_interval: float
    new_user_interval: float
    trace_file: List[str]


@dataclass
class UserConfig:
    user_id: int
    num_agents: int
    gap_between_requests: int
    trace: Any

    @staticmethod
    def new_user_config(user_id: int, workload_config: WorkloadConfig, trace) -> "UserConfig":
        return UserConfig(
            user_id=user_id,
            num_agents=workload_config.num_agents,
            gap_between_requests=workload_config.user_request_interval,
            trace=trace,
        )


class ChatHistory:

    def __init__(
        self,
    ):
        self.history = []

    def on_user_query(self, query: str, agentID: int, roundID: int):
        self.history.append({"role": "user", "name": f"agent{agentID}-{roundID}", "content": query})

    def on_system_response(self, response: str, agentID: int, roundID: int):
        assert len(self.history) > 0, "Expect user query"
        self.history.append({"role": "assistant", "name": f"agent{agentID}-{roundID}", "content": response})

    def get_messages_for_openai(self, input_from: List, agentID: int, roundID: int):
        messages = []
        for i_f in input_from:
            name = f"agent{i_f[1]}-{i_f[0]}"
            if i_f[2] in ("input", "both"):
                messages.extend(
                    [entry for entry in self.history
                     if entry["role"] == "user" and entry["name"] == name]
                )
            if i_f[2] in ("output", "both"):
                messages.extend(
                    [entry for entry in self.history
                     if entry["role"] == "assistant" and entry["name"] == name]
                )   
        name = f"agent{agentID}-{roundID}"
        messages.extend(  
            [entry for entry in self.history
             if entry["role"] == "user" and entry["name"] == name]
        )
        return messages

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
    agentID: int


class RequestExecutor:

    def __init__(self, base_url: List[str], model: List[str]):
        self.client = []
        for bu in base_url:
            if not bu.endswith('/v1'):
                bu = bu.rstrip('/') + '/v1'
            self.client.append(openai.AsyncOpenAI(
                api_key="EMPTY",  # Dummy API key for vLLM server
                base_url=bu
            ))
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        self.request_history = []

    async def _async_launch_request(self, messages: List[Dict[str, str]],  max_tokens: int, 
                                    agentID: int, extra_headers: Optional[Dict[str, str]] = None):
        model = self.model[agentID]
        try:
            logging.info(f"Sending request to model {model} with messages: {messages}")
            
            # Initialize response tracking variables
            words = ""
            tokens_out = 0
            tokens_prefill = 0
            start_time = time.time()
            first_token_time = None

            # Make the request
            response = await self.client[agentID].chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=0.0,
                stream_options={"include_usage": True},
                extra_headers=extra_headers,
            )

            # Process the streaming response
            async for chunk in response:
                if not chunk.choices:
                    continue
                    
                # Handle content
                if chunk.choices[0].delta.content is not None:
                    if first_token_time is None and chunk.choices[0].delta.content != "":
                        first_token_time = time.time()
                    words += chunk.choices[0].delta.content
                
            # Handle token counts if available
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                tokens_out = chunk.usage.completion_tokens
                tokens_prefill = chunk.usage.prompt_tokens

            # Calculate timing metrics
            ttft = first_token_time - start_time if first_token_time else 0
            generation_time = time.time() - first_token_time if first_token_time else 0

            return Response(
                body=words,
                ttft=ttft,
                generation_time=generation_time,
                prompt_tokens=tokens_prefill,
                generation_tokens=tokens_out,
                launch_time=start_time,
                finish_time=time.time(),
                agentID=agentID,
            )

        except Exception as e:
            logging.error(f"Error in _async_launch_request: {str(e)}")
            logging.error(f"Request details - model: {model}, messages: {messages}")
            raise

    def launch_request(
        self,
        messages,
        max_tokens: int,
        finish_callback,
        agentID: int,
        roundID: int,
        input: str,
        extra_headers=None,
    ):
        """
        finish_callback: Callable[[Response, int], None]
        """
        real_callback = lambda x: finish_callback(x.result(), agentID, roundID, input)
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens, agentID, extra_headers), self.loop
        )
        future.add_done_callback(real_callback)


class UserSession:

    def __init__(self, user_config: UserConfig):
        self.user_config = user_config
        self.last_request_time = None
        self.chat_history = ChatHistory()
        self.round_id = 0

        self.has_unfinished_request = 0
        self.last_unfinished_log = 0

        self.prompt_lengths = []
        self.generation_lengths = []
        self.ttfts = []
        self.generation_times = []
        self.launch_times = []
        self.finish_times = []

        self.finished = False

        self.agentIDs = []
        self.roundIDs = []
        self.inputs = []
        self.outputs = []

    def _update_result(self, response: Response):
        self.prompt_lengths.append(response.prompt_tokens)
        self.generation_lengths.append(response.generation_tokens)
        self.ttfts.append(response.ttft)
        self.generation_times.append(response.generation_time)
        self.launch_times.append(response.launch_time)
        self.finish_times.append(response.finish_time)
        self.agentIDs.append(response.agentID)
        self.outputs.append(response.body)

    def _launch_new_request(self, timestamp: float, request_executor: RequestExecutor, request_id: int):
        agentID = self.user_config.trace[self.round_id]['agent_id'][request_id]
        max_tokens = self.user_config.trace[self.round_id]['output_tokens'][request_id]
        input_from = self.user_config.trace[self.round_id]['input_from'][request_id]

        self.chat_history.on_user_query("hihihihihi", agentID, self.round_id)
        messages = self.chat_history.get_messages_for_openai(input_from, agentID, self.round_id)
        request_executor.launch_request(
            messages,
            max_tokens,
            self._on_request_finished,
            agentID,
            self.round_id,
            messages.copy(),
            extra_headers={"x-user-id": str(self.user_config.user_id)},
        )
        self.has_unfinished_request += 1
        self.last_request_time = timestamp

    def _on_request_finished(self, response: Response, agentID: int, roundID: int, messages: str):
        self.chat_history.on_system_response(response.body, agentID, roundID)
        self.has_unfinished_request -= 1
        logger.debug(
            f"User {self.user_config.user_id} finished one request. "
            f"Prompt tokens: {response.prompt_tokens}, "
            f"generation tokens: {response.generation_tokens}"
        )
        self._update_result(response)
        self.roundIDs.append(roundID)
        self.inputs.append(messages)

    def step(self, timestamp: float, request_executor: RequestExecutor):
        num_rounds = len(self.user_config.trace)
        if (
            self.round_id >= num_rounds
            and not self.has_unfinished_request
        ):
            self.finished = True
            return

        if self.last_request_time is None:
            for request_id in range(len(self.user_config.trace[self.round_id]['agent_id'])):
                self._launch_new_request(timestamp, request_executor, request_id)
            self.round_id += 1
            return

        if timestamp - self.last_request_time > self.user_config.gap_between_requests:
            if self.has_unfinished_request:
                if timestamp - self.last_unfinished_log > 10:
                    logger.warning(
                        f"User {self.user_config.user_id} has unfinished "
                        "requests and unable to fit the QPS requirement."
                    )
                    self.last_unfinished_log = timestamp
                return

            for request_id in range(len(self.user_config.trace[self.round_id]['agent_id'])):
                self._launch_new_request(timestamp, request_executor, request_id)
            self.round_id += 1
            return

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["prompt_tokens"] = self.prompt_lengths
        df["generation_tokens"] = self.generation_lengths
        df["ttft"] = self.ttfts
        df["generation_time"] = self.generation_times
        df["user_id"] = self.user_config.user_id
        df["round_id"] = self.roundIDs
        df["launch_time"] = self.launch_times
        df["finish_time"] = self.finish_times
        df["agentID"] = self.agentIDs
        df["input"] = self.inputs
        df["output"] = self.outputs
        return df


class UserSessionManager:

    def __init__(
        self, workload_config: WorkloadConfig
    ):
        self.workload_config = workload_config
        self.sessions = []

        gap_between_requests_per_user = workload_config.user_request_interval
        self.gap_between_users = workload_config.new_user_interval

        logger.info(
            f"Gap between users: {self.gap_between_users} secs.\n"
            f"Gap between user reqs: {gap_between_requests_per_user} secs."
        )

        self.user_id = 0
        self.last_user_join = 0
        self.session_summaries = []
        self.start_time = None

        self.traces = []
        for usr_id, trace_file in enumerate(self.workload_config.trace_file):
            self.traces.append([])
            with open(trace_file, "r", encoding="utf-8") as f:
                for round_id, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        continue
                    self.traces[usr_id].append(record)

        self.continue_flag = True

    def _create_user_session(self):
        self.user_id += 1
        if self.user_id > len(self.traces):
            return None, False
        user_config = UserConfig.new_user_config(self.user_id, self.workload_config, self.traces[self.user_id - 1])
        user_session = UserSession(user_config)
        self.sessions.append(user_session)
        return user_session, True

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

        if self.continue_flag:
            if timestamp - self.last_user_join > self.gap_between_users:
                new_session, self.continue_flag = self._create_user_session()
                if new_session is not None:
                    self.last_user_join = timestamp
                    logger.info(
                        f"Joined a new user {self.user_id}, "
                        f"now active users: {len(self.sessions)}"
                    )

        for session in self.sessions:
            session.step(timestamp, executor)

        self._remove_finished_sessions()

        if not self.continue_flag and len(self.sessions) == 0:
            return False
        return True

    @staticmethod
    def ProcessSummary(
        df: pd.DataFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        pending_queries: int = 0
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

        df = UserSessionManager.ProcessSummary(
            df, start_time, end_time, pending_queries
        )
        return df


def parse_arguments() -> WorkloadConfig:
    parser = argparse.ArgumentParser(description="Parse benchmark configurations.")
    parser.add_argument("--num-agents", required=True, type=int)
    parser.add_argument(
        "--model",
        nargs="+",
        type=str,
        required=True,
        help="One or more model names, e.g. --model m1 m2 m3"
    )
    parser.add_argument("--user-request-interval", type=float, required=True)
    parser.add_argument("--new-user-interval", type=float, required=True)
    parser.add_argument(
        "--base-url",
        nargs="+",
        type=str,
        required=True,
        help="Base URL of the serving engine endpoint",
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        nargs="+",
        required=True,
        help="The trace file to load the workload from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name (ended with csv or txt) "
        "for the summary csv and txt",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
        help="The time between two summary loggings in seconds",
    )
    args = parser.parse_args()
    return args, parser


def main():

    args, parser = parse_arguments()

    step_interval = 0.1

    model = args.model
    if args.num_agents != len(args.model):
        assert len(args.model) == 1
        model = args.model * args.num_agents

    base_url = args.base_url
    if args.num_agents != len(args.base_url):
        assert len(args.base_url) == 1
        base_url = args.base_url * args.num_agents

    executor = RequestExecutor(
        base_url=base_url, model=model
    )

    workload_config = WorkloadConfig(
        num_agents=args.num_agents,
        model=model,
        user_request_interval=args.user_request_interval,
        new_user_interval=args.new_user_interval,
        trace_file=args.trace_file,
    )

    manager = UserSessionManager(
        workload_config
    )

    start_time = time.time()
    last_summary_time = start_time
    try:
        while True:
            continue_flag = manager.step(time.time(), executor)
            time.sleep(step_interval)

            if time.time() - last_summary_time > args.log_interval:
                manager.summary(last_summary_time, time.time())
                last_summary_time = time.time()

            if not continue_flag:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")

    AsyncLoopWrapper.StopLoop()

    logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    summary = manager.summary(0, time.time())
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
