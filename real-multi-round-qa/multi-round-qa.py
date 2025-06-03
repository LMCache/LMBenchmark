import os
import random
import time
import uuid
import asyncio
import argparse
import openai
from dataclasses import dataclass, asdict
from typing import List
import json

FIRST_PROMPT = "Read and summarize this novel.\n\n{}"
FOLLOWUP_PROMPTS = [
    "Write down the author's feelings.",
    "What scene did the author most want to write?",
    "Describe the setting of the story.",
    "Describe the protagonist's development in detail.",
    "Name the climax of the story and explain why it is important.",
    "What do you think is the theme this story is trying to convey?",
    "Analyze the relationships between the characters.",
    "Name a line that made an impression on you and explain why.",
    "Tell us how you felt about the last scene.",
    "Compare this story with other works and note similarities and differences.",
    "Which character grew the most throughout the story, and how?",
    "What is the moral or lesson of the story?",
    "What role does symbolism play in this novel?",
    "How does the narrative structure influence the reader’s experience?",
    "Discuss the use of foreshadowing in the story.",
    "Explain the significance of the story’s title.",
    "What historical or cultural background is important to understand this story?",
    "Analyze how the author builds tension or suspense.",
    "Was the ending satisfying? Why or why not?",
    "Identify and analyze any use of irony.",
    "What is the role of secondary characters in the plot?",
    "Does the protagonist change their beliefs or values? Explain.",
    "How does the setting reflect the themes or mood of the story?",
    "Describe the tone and how it shifts throughout the novel.",
    "What does the author want the reader to question or reflect on?",
    "Was there an unreliable narrator? If so, what effect did that have?",
    "Identify a turning point in the plot and explain its impact.",
    "Discuss the power dynamics between characters.",
    "What philosophical or existential questions does the novel explore?",
    "Explain how time is handled (linear, non-linear, flashbacks, etc.).",
    "How is identity or self-perception explored in the novel?",
    "What role does memory play in the narrative?",
    "Are there any recurring motifs or patterns? What do they represent?",
    "Describe the emotional arc of the story.",
    "Was justice served by the end of the story? Why or why not?",
    "How does the novel depict power, control, or authority?",
    "What role does fate vs. free will play in the character's journey?",
    "How would the story change if told from another character’s perspective?",
    "Is the protagonist heroic, tragic, anti-heroic? Justify your answer.",
    "What are the ethical dilemmas faced by the characters?",
    "Identify an important internal conflict and how it is resolved.",
    "How does the author use descriptive language to evoke atmosphere?",
    "Is there any metafictional or self-referential content?",
    "How does the author use silence, ambiguity, or the unsaid?",
    "What emotions does the novel evoke, and how?",
    "Are there any scenes that are intentionally open to interpretation?",
    "What is the relationship between the personal and the political in the story?",
    "How do characters cope with loss, trauma, or regret?",
    "Does the novel challenge any societal norms or expectations?",
    "If you could ask the author one question about this novel, what would it be?"
]


@dataclass
class Result:
    session_id: str
    turn: int
    latency: float
    ttft: float
    generation_time: float
    prompt_tokens: int
    completion_tokens: int
    status: str

class ChatSession:
    def __init__(self, args):
        self.session_id = str(uuid.uuid4())
        self.turns = 0
        self.messages = []
        self.total_completion_tokens = 0
        self.model = args.model
        self.answer_len = args.answer_len
        self.src_dir = args.src_dir
        self.num_rounds = args.num_rounds
        self.first_prompt = self._load_random_file()

    def _load_random_file(self):
        files = [f for f in os.listdir(self.src_dir) if os.path.isfile(os.path.join(self.src_dir, f))]
        if not files:
            raise RuntimeError("No files found in {}".format(self.src_dir))
        with open(os.path.join(self.src_dir, random.choice(files)), encoding='utf-8') as f:
            return FIRST_PROMPT.format(f.read())

    def get_next_prompt(self):
        if self.turns == 0:
            return self.first_prompt
        return FOLLOWUP_PROMPTS[self.turns - 1]

    def is_finished(self):
        return self.turns >= min(len(FOLLOWUP_PROMPTS), self.num_rounds) + 1

    def append_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def append_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})
        self.turns += 1

async def run_turn(session: ChatSession, client: openai.AsyncOpenAI) -> Result:
    prompt = session.get_next_prompt()
    session.append_user_message(prompt)

    start_time = time.time()
    first_token_time = None
    content = ""
    completion_tokens = 0
    prompt_tokens = 0

    print(f"Session {session.session_id}, Turn {session.turns}: {prompt[:50]}...")
    response = await client.chat.completions.create(
        model=session.model,
        messages=session.messages,
        stream=True,
        max_tokens=session.answer_len,
        temperature=0,
        stream_options={"include_usage": True},
    )

    async for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.time()
            content += delta

    ttft = first_token_time - start_time if first_token_time else 0.0
    generation_time = time.time() - first_token_time if first_token_time else 0.0
    latency = time.time() - start_time

    completion_tokens = chunk.usage.completion_tokens
    prompt_tokens = chunk.usage.prompt_tokens

    result = Result(
        session_id=session.session_id,
        turn=session.turns,
        latency=latency,
        ttft=ttft,
        generation_time=generation_time,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        status="success",
    )
    
    session.append_assistant_message(content)

    return result

async def run_group(args) -> List[Result]:
    client = openai.AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    sessions = [ChatSession(args) for _ in range(args.num_users_sequential)]
    results = []

    while any(not s.is_finished() for s in sessions):
        for session in sessions:
            if session.is_finished():
                continue
            result = await run_turn(session, client)
            results.append(result)

    return results

async def run_all_concurrent(args):
    tasks = [run_group(args) for _ in range(args.num_users_concurrent)]
    all_results = await asyncio.gather(*tasks)
    return [asdict(r) for group in all_results for r in group]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-users-concurrent", type=int, required=True)
    parser.add_argument("--num-users-sequential", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--src-dir", type=str, default="gutenberg/8k")
    parser.add_argument("--answer-len", type=int, default=512)
    parser.add_argument("--output", type=str, default="summary.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    results = asyncio.run(run_all_concurrent(args))
    output_data = {
        "params": vars(args),
        "results": results,
        "src": [f for f in os.listdir(args.src_dir)
            if os.path.isfile(os.path.join(args.src_dir, f))]
    }

    output_json = os.path.splitext(args.output)[0] + ".json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results written to {output_json}")

if __name__ == "__main__":
    main()
