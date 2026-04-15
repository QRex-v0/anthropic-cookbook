"""
Session memory compaction — background pre-building for instant context swaps.
https://platform.claude.com/cookbook/misc-session-memory-compaction

Two approaches compared:
- Traditional: compact synchronously when context is full (user waits)
- Instant: build summary in background thread, swap instantly when needed

This is for long-running chat sessions (no tools). For tool-based agentic
loops, see compaction_manual.py or compaction_control.py.
"""

import argparse
import re
import threading
import time

import anthropic
from anthropic.types import MessageParam, TextBlockParam
from dotenv import load_dotenv


MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a short story writer who helps authors develop their ideas into compelling narratives.
- Help with plot, character development, and drafting
- You are the lead writer; when you disagree, say so, but defer to the author
- DO NOT ask for clarification; assume you have enough information to proceed"""

# The summary prompt is intentionally detailed — a bad summary means lost context
# after compaction, causing Claude to repeat work or forget decisions.
SESSION_MEMORY_PROMPT = """
Compress the conversation into a structured summary that preserves all information
needed to continue work seamlessly.

<analysis-instructions>
Before generating your summary, analyze the transcript in <think>...</think> tags:
1. What did the user originally request?
2. What actions succeeded? What failed and why?
3. Did the user correct or redirect the assistant?
4. What was actively being worked on at the end?
5. What tasks remain incomplete?
6. What specific details (IDs, paths, values) must survive compression?
</analysis-instructions>

<summary-format>
## User Intent
Original request and refinements. Quote key requirements.

## Completed Work
What was created/modified. Exact identifiers, values, configurations.

## Errors & Corrections
Failed approaches (so they aren't retried). User corrections verbatim.

## Active Work
What was in progress when session ended. Exact state.

## Pending Tasks
Remaining items not yet started.

## Key References
IDs, paths, URLs, values, constraints, preferences.
</summary-format>

<compression-rules>
- Weight recent messages more heavily
- Omit pleasantries and filler
- Keep each section under 500 words
- Priority: user corrections > errors > active work > completed work
</compression-rules>
"""

DEMO_MESSAGES = [
    "I want to create a story about a young detective solving a mysterious case in a small town. Generate 3 well thought out plot ideas for me to consider.",
    "I don't like those ideas, can you think of one plot something more unique and unexpected?",
    "Ok I like it. Can you help me develop the main character's backstory and motivations?",
    "Can you draft a detailed outline for the story, breaking it down into chapters and key events?",
    "Can you draft me a first chapter based on the plot and character ideas we've discussed so far? Make it around 2,000 words.",
    "Can you draft a second chapter that builds on the first one, introducing a new twist in the mystery?",
]


# --- Helpers ---

def truncate_response(text: str, max_lines: int = 15) -> str:
    lines = text.strip().split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def remove_thinking_blocks(text: str) -> tuple[str, str]:
    """Remove <think>...</think> blocks. Returns (cleaned, removed)."""
    matches = re.findall(r"<think>.*?</think>", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    return cleaned, "".join(matches)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def add_cache_control(messages: list[dict]) -> list[MessageParam]:
    """Mark the last user message with cache_control for prompt caching.

    The background summary thread sends the same message prefix as the main chat,
    so it gets a cache hit (~80% cost reduction) instead of reprocessing everything.
    """
    cached: list[MessageParam] = []
    last_user_idx = max(
        (i for i, m in enumerate(messages) if m["role"] == "user"), default=-1
    )

    for i, msg in enumerate(messages):
        content = msg["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        block: TextBlockParam = {"type": "text", "text": text}
        if i == last_user_idx:
            block["cache_control"] = {"type": "ephemeral"}
        cached.append({"role": msg["role"], "content": [block]})

    return cached


# --- Traditional (synchronous) ---

class TraditionalSession:
    """Compact synchronously when context is full. User waits."""

    def __init__(self, client: anthropic.Anthropic, context_limit: int = 10_000):
        self.client = client
        self.context_limit = context_limit
        self.messages: list[dict] = []
        self.token_count = 0

    def chat(self, user_message: str) -> tuple[str, anthropic.types.Usage]:
        if self.token_count >= self.context_limit:
            print(f"\n  Context at {self.token_count:,} tokens — compacting...")
            self._compact()

        self.messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=3500,
            system=SYSTEM_PROMPT,
            messages=add_cache_control(self.messages),
        )

        text = response.content[0].text
        self.messages.append({"role": "assistant", "content": text})

        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        total_input = response.usage.input_tokens + cache_read
        self.token_count = total_input + response.usage.output_tokens

        return text, response.usage

    def _compact(self) -> None:
        start = time.perf_counter()

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=5000,
            system=SYSTEM_PROMPT,
            messages=add_cache_control(self.messages)
            + [{"role": "user", "content": SESSION_MEMORY_PROMPT}],
        )
        elapsed = time.perf_counter() - start

        summary, removed = remove_thinking_blocks(response.content[0].text)
        approx_tokens = response.usage.output_tokens - round(len(removed) / 4)

        self.messages = [
            {"role": "user", "content": f"Continued from previous conversation. Session memory: {summary}. Continue from where we left off."}
        ]

        reduction = self.token_count - approx_tokens
        pct = (reduction / self.token_count) * 100
        print(f"  Tokens: {self.token_count:,} -> {approx_tokens} ({pct:.0f}% reduction)")
        print(f"  Compaction time: {elapsed:.2f}s (user waiting)")
        self.token_count = approx_tokens


# --- Instant (background threading) ---

class InstantSession:
    """Pre-build summary in background thread. Swap instantly when needed."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        context_limit: int = 12_000,
        init_threshold: int = 7_500,
        update_interval: int = 2_000,
    ):
        self.client = client
        self.context_limit = context_limit
        self.init_threshold = init_threshold      # Start building memory here
        self.update_interval = update_interval    # Update memory every N new tokens

        self.messages: list[dict] = []
        self.token_count = 0

        # Session memory state
        self.memory: str | None = None
        self.last_summarized_idx = 0
        self.tokens_at_last_update = 0

        # Threading
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def chat(self, user_message: str) -> tuple[str, anthropic.types.Usage, str | None]:
        if self.token_count + estimate_tokens(user_message) >= self.context_limit:
            self._compact()

        self.messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=3500,
            system=SYSTEM_PROMPT,
            messages=add_cache_control(self.messages),
        )

        text = response.content[0].text
        self.messages.append({"role": "assistant", "content": text})

        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        total_input = response.usage.input_tokens + cache_read
        self.token_count = total_input + response.usage.output_tokens

        # Proactively trigger background memory build/update
        bg_status = None
        if self._should_init() or self._should_update():
            self._trigger_background()
            bg_status = "initializing" if self.memory is None else "updating"

        return text, response.usage, bg_status

    def _should_init(self) -> bool:
        return self.memory is None and self.token_count >= self.init_threshold

    def _should_update(self) -> bool:
        if self.memory is None:
            return False
        return self.token_count - self.tokens_at_last_update >= self.update_interval

    def _trigger_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return  # One at a time

        snapshot = self.messages.copy()
        snapshot_idx = len(snapshot)
        tokens = self.token_count

        self._thread = threading.Thread(
            target=self._background_update,
            args=(snapshot, snapshot_idx, tokens),
            daemon=True,
        )
        self._thread.start()

    def _background_update(self, snapshot: list[dict], idx: int, tokens: int) -> None:
        try:
            with self._lock:
                current_memory = self.memory
                last_idx = self.last_summarized_idx

            if current_memory is None:
                new_memory = self._create_memory(snapshot)
            else:
                new_msgs = snapshot[last_idx:]
                if not new_msgs:
                    return
                new_memory = self._update_memory(new_msgs)

            with self._lock:
                self.memory = new_memory
                self.last_summarized_idx = idx
                self.tokens_at_last_update = tokens
        except Exception as e:
            print(f"  [Background] Error: {e}")

    def _create_memory(self, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=5000,
            system=SYSTEM_PROMPT,
            messages=add_cache_control(messages)
            + [{"role": "user", "content": SESSION_MEMORY_PROMPT}],
        )
        summary, _ = remove_thinking_blocks(response.content[0].text)
        cache_hit = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        print(f"  [Background] Memory created (cache hit: {cache_hit > 0})")
        return summary

    def _update_memory(self, new_messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=5000,
            system=SYSTEM_PROMPT,
            messages=new_messages + [{
                "role": "user",
                "content": SESSION_MEMORY_PROMPT
                + f"\n\nExisting session memory: {self.memory}. Return updated memory.",
            }],
        )
        summary, _ = remove_thinking_blocks(response.content[0].text)
        print("  [Background] Memory updated")
        return summary

    def _compact(self) -> None:
        prev_count = len(self.messages)

        # Wait for background if it's still running
        if self.memory is None:
            if self._thread is not None and self._thread.is_alive():
                print("  Waiting for background memory...")
                self._thread.join(timeout=30.0)

            # Fallback: create synchronously
            if self.memory is None:
                print("  No pre-built memory, creating synchronously...")
                self.memory = self._create_memory(self.messages)
                self.last_summarized_idx = len(self.messages)

        with self._lock:
            unsummarized = self.messages[self.last_summarized_idx:]
            self.messages = [
                {"role": "user", "content": f"Continued from previous conversation. Session memory: {self.memory}. Continue from where we left off."}
            ] + unsummarized
            self.last_summarized_idx = 1

        print(f"\n  INSTANT COMPACTION: {prev_count} messages -> {len(self.messages)}")


# --- Demo runner ---

def run_demo(mode: str) -> None:
    load_dotenv()
    client = anthropic.Anthropic()

    print(f"Mode: {mode}")
    print("=" * 60)

    if mode == "traditional":
        session = TraditionalSession(client)
        for i, msg in enumerate(DEMO_MESSAGES, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {msg}")
            text, usage = session.chat(msg)
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            total_in = usage.input_tokens + cache_read
            print(f"Assistant: {truncate_response(text, max_lines=3)}")
            print(f"  [tokens: {total_in:,} in / {usage.output_tokens:,} out | messages: {len(session.messages)}]")
    else:
        session = InstantSession(client)
        for i, msg in enumerate(DEMO_MESSAGES, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {msg}")
            text, usage, bg_status = session.chat(msg)
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            total_in = usage.input_tokens + cache_read
            print(f"Assistant: {truncate_response(text, max_lines=3)}")
            print(f"  [tokens: {total_in:,} in / {usage.output_tokens:,} out | messages: {len(session.messages)} | memory: {'ready' if session.memory else 'not yet'}]")
            if bg_status:
                print(f"  [background: {bg_status} memory at {session.token_count:,} tokens]")

        # One more turn to trigger compaction
        extra = "What did we just talk about? Give me one sentence."
        print(f"\n--- Turn {len(DEMO_MESSAGES) + 1} ---")
        print(f"User: {extra}")
        text, usage, bg_status = session.chat(extra)
        print(f"Assistant: {truncate_response(text, max_lines=3)}")

    print(f"\n{'=' * 60}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Session memory compaction demo")
    parser.add_argument(
        "mode",
        choices=["traditional", "instant"],
        help="Compaction strategy: traditional (sync) or instant (background)",
    )
    args = parser.parse_args()
    run_demo(args.mode)


if __name__ == "__main__":
    main()
