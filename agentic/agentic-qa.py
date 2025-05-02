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
    # Length of shared system prompt
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Length of the answer in one round
    answer_len: int

    # Number of rounds in the conversation
    num_rounds: int

    # Model name
    model: List[str]

    user_request_interval: float
    new_user_interval: float
    num_agents: int
    whole_history: bool
    trace_file: Optional[str] = None


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

    # Gap between two requests
    gap_between_requests: int

    # Num rounds
    num_rounds: int

    num_agents: int
    whole_history: bool
    trace: Any

    @staticmethod
    def new_user_config(user_id: int, workload_config: WorkloadConfig, trace) -> "UserConfig":
        return UserConfig(
            user_id=user_id,
            system_prompt_len=workload_config.system_prompt_len,
            user_info_len=workload_config.user_info_len,
            answer_len=workload_config.answer_len,
            gap_between_requests=workload_config.user_request_interval,
            num_rounds=workload_config.num_rounds,
            num_agents=workload_config.num_agents,
            whole_history=workload_config.whole_history,
            trace=trace,
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
            self.history.append({"role": "user", "name": "user", "content": query})

    def on_system_response_whole(self, response: str, agentID: int):
        assert len(self.history) > 0, "Expect user query"
        self.history.append({"role": "assistant", "name": "agent"+f"{agentID}", "content": response})

    def on_system_response_part(self, response: str, agentID: int):
        self.history = []
        self.history.append({"role": "assistant", "name": "agent"+f"{agentID}", "content": response})

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
    agentID: int


class RequestExecutor:

    def __init__(self, base_url: str, model: List[str]):
        # Ensure base_url ends with /v1
        if not base_url.endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'
        
        # For vLLM server, we don't need an API key, but the client requires one
        self.client = openai.AsyncOpenAI(
            api_key="EMPTY",  # Dummy API key for vLLM server
            base_url=base_url
        )
        self.model = model
        logging.info(f"Initialized OpenAI client with base_url={base_url} and model={model}")
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
            response = await self.client.chat.completions.create(
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

            # If we didn't get token counts from streaming, try to get them from the final response
            if tokens_out == 0 or tokens_prefill == 0:
                print("No token counts from streaming, getting final response")
                print(f"{tokens_out}, {tokens_prefill}")
                try:
                    final_response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=False,
                    )
                    if hasattr(final_response, 'usage') and final_response.usage is not None:
                        tokens_out = final_response.usage.completion_tokens
                        tokens_prefill = final_response.usage.prompt_tokens
                except Exception as e:
                    logging.warning(f"Failed to get token counts from final response: {e}")

            # # Calculate timing metrics
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
        extra_headers=None,
    ):
        """
        finish_callback: Callable[[Response, int], None]
        """
        real_callback = lambda x: finish_callback(x.result(), agentID)
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens, agentID, extra_headers), self.loop
        )
        future.add_done_callback(real_callback)


class UserSession:

    def __init__(self, user_config: UserConfig):
        self.user_config = user_config
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

        self.finished = False

        self.agentIDs = []
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

    def _build_new_question(self):
        self.question_id += 1
        return (
            f"Here's question #{self.question_id}: can you tell me "
            + "a new long story with a happy ending?"
        )

    def _launch_new_request(self, timestamp: float, request_executor: RequestExecutor):
        agentID = self.question_id % self.user_config.num_agents
        if self.user_config.trace is None:
            prompt = self._build_new_question()
            if len(self.chat_history) == 0:
                prompt = self._build_system_prompt() + prompt
            max_tokens = self.user_config.answer_len
        else:
            round_id = self.question_id // self.user_config.num_agents + 1
            prompt = self.user_config.trace[f"round{round_id}"][f"{agentID}_input"]
            max_tokens = self.user_config.trace[f"round{round_id}"][f"{agentID}_max_tokens"]
            self.question_id += 1
        self.chat_history.on_user_query(prompt)
        logger.debug(
            f"User {self.user_config.user_id} issues request {self.question_id}"
        )
        messages = self.chat_history.get_messages_for_openai()
        self.inputs.append(messages.copy()) 
        request_executor.launch_request(
            messages,
            max_tokens,
            self._on_request_finished,
            agentID,
            extra_headers={"x-user-id": str(self.user_config.user_id)},
        )
        self.has_unfinished_request = True
        self.last_request_time = timestamp

    def _on_request_finished(self, response: Response, agentID: int):
        if self.user_config.whole_history:
            self.chat_history.on_system_response_whole(response.body, agentID)
        else:
            self.chat_history.on_system_response_part(
                response.body, agentID
            )
        self.has_unfinished_request = False
        logger.debug(
            f"User {self.user_config.user_id} finished one request. "
            f"Prompt tokens: {response.prompt_tokens}, "
            f"generation tokens: {response.generation_tokens}"
        )
        self._update_result(response)

    def set_internal_state(self, offset: float, timestamp: float):
        """Tell the session is the 'offset' seconds after the start"""
        assert len(self.chat_history) == 0, (
            "Internal state should be set " "before the first request"
        )

        num_passed_questions = int(offset / self.user_config.gap_between_requests) + 1

        passed_time = (num_passed_questions - 1) * self.user_config.gap_between_requests

        self.last_request_time = timestamp - offset + passed_time
        self.question_id = num_passed_questions
        logger.debug(
            f"Set internal state for user {self.user_config.user_id}, "
            f"question_id: {self.question_id}, "
            f"last_request_time: {self.last_request_time}"
        )

    def step(self, timestamp: float, request_executor: RequestExecutor):
        if self.user_config.trace is not None:
            num_rounds = len(self.user_config.trace)
        else:
            num_rounds = self.user_config.num_rounds
        if (
            self.question_id >= num_rounds
            and not self.has_unfinished_request
        ):
            self.finished = True
            return

        if self.last_request_time is None:
            self._launch_new_request(timestamp, request_executor)
            return

        if timestamp - self.last_request_time > self.user_config.gap_between_requests:
            if self.has_unfinished_request:
                if timestamp - self.last_unfinished_log > 10:
                    logger.warning(
                        f"User {self.user_config.user_id} has an unfinished "
                        "request and unable to fit the QPS requirement."
                    )
                    self.last_unfinished_log = timestamp
                return

            self._launch_new_request(timestamp, request_executor)
            return

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["prompt_tokens"] = self.prompt_lengths
        df["generation_tokens"] = self.generation_lengths
        df["ttft"] = self.ttfts
        df["generation_time"] = self.generation_times
        df["user_id"] = self.user_config.user_id
        df["question_id"] = range(1, len(self.prompt_lengths) + 1)
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
        if self.workload_config.trace_file is not None:
            with open(self.workload_config.trace_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Skipping invalid JSON on line {idx}: {e}")
                        continue

                    self.traces.append(record)

        self.continue_flag = True

    def _create_user_session(self):
        self.user_id += 1
        if len(self.traces) > 0:
            if self.user_id > len(self.traces):
                return None, False
            user_config = UserConfig.new_user_config(self.user_id, self.workload_config, self.traces[self.user_id - 1])
        else:
            user_config = UserConfig.new_user_config(
                self.user_id, self.workload_config, None
            )
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

    parser.add_argument(
        "--shared-system-prompt",
        type=int,
        help="Length of the shared system prompt (tokens)",
    )
    parser.add_argument(
        "--user-history-prompt",
        type=int,
        help="Length of the user-specific history prompt (tokens)",
    )
    parser.add_argument(
        "--answer-len",
        type=int,
        help="Length of the answer in one round",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        help="Number of rounds in the conversation",
    )
    parser.add_argument("--num-agents", required=True, type=int)
    parser.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="The trace file to load the workload from",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        type=str,
        required=True,
        help="One or more model names, e.g. --model m1 m2 m3"
    )
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
        "--log-interval",
        type=int,
        default=30,
        help="The time between two summary loggings in seconds",
    )
    parser.add_argument("--user-request-interval", type=float, required=True)
    parser.add_argument("--new-user-interval", type=float, required=True)
    parser.add_argument(
        "--whole-history",
        action="store_true",
        help="Include the whole history in the agentic workload"
    )
    args = parser.parse_args()
    return args, parser


def main():

    args, parser = parse_arguments()

    # 1) If they provided a trace file, they must NOT provide any manual flags
    manual_flags = {
        "--shared-system-prompt": args.shared_system_prompt,
        "--user-history-prompt": args.user_history_prompt,
        "--answer-len":           args.answer_len,
        "--num-rounds":           args.num_rounds,
        "--time": args.time,
    }

    if args.trace_file:
        conflicts = [name for name, val in manual_flags.items() if val is not None]
        if conflicts:
            parser.error(
                f"When --trace-file is used, you may not pass: {', '.join(conflicts)}"
            )

    # 2) If they did NOT provide a trace file, all manual flags become required
    else:
        missing = [name for name, val in manual_flags.items() if val is None]
        if missing:
            parser.error(
                f"When --trace-file is omitted, you MUST supply: {', '.join(missing)}"
            )

    # From here on you know you’re in exactly one mode:
    if args.trace_file:
        print("Running in trace‑mode, loading:", args.trace_file)
    else:
        print("Running manual‑mode with:",
            args.shared_system_prompt,
            args.user_history_prompt,
            args.answer_len,
            args.num_rounds,
            args.time,)

    step_interval = 0.1

    model = args.model
    if args.num_agents != len(args.model):
        assert len(args.model) == 1
        model = args.model * args.num_agents
    print(f"Using models: {model}")

    executor = RequestExecutor(
        base_url=args.base_url, model=model
    )

    workload_config = WorkloadConfig(
        system_prompt_len=args.shared_system_prompt,
        user_info_len=args.user_history_prompt,
        answer_len=args.answer_len,
        num_rounds=args.num_rounds,
        model=model,
        user_request_interval=args.user_request_interval,
        new_user_interval=args.new_user_interval,
        num_agents=args.num_agents,
        whole_history=args.whole_history,
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

            if args.time is not None and time.time() - start_time > args.time:
                break

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
