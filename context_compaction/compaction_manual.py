"""
Manual tool runner with optional context compaction.

A drop-in replacement for client.beta.messages.tool_runner that makes
the loop and compaction logic explicit. Use case agnostic — works with
any set of @beta_tool functions.

Usage:
    from compaction_manual import tool_runner_with_compaction

    for message in tool_runner_with_compaction(
        client=client,
        model="claude-sonnet-4-6",
        max_tokens=4096,
        tools=[my_tool_a, my_tool_b],
        messages=messages,
        compaction_threshold=5000,       # None to disable
        compaction_model="claude-haiku-4-5",
        summary_prompt="...",
    ):
        print(message.usage)
"""

from collections.abc import Iterator
from typing import Any

import anthropic
from anthropic.types import Message


DEFAULT_SUMMARY_PROMPT = """Summarize this conversation so it can be continued in a new context window.

Preserve:
1. The original task and constraints
2. What has been completed so far
3. What remains to be done
4. Key facts, IDs, and values needed to continue

Be concise but complete. Wrap in <summary></summary> tags."""


def tool_runner_with_compaction(
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    tools: list,
    messages: list[dict[str, Any]],
    max_turns: int = 100,
    compaction_threshold: int | None = None,
    compaction_model: str = "claude-haiku-4-5",
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
) -> Iterator[Message]:
    """Run an agentic tool loop, yielding each Message.

    Args:
        tools: List of @beta_tool decorated functions.
        messages: Conversation history (mutated in place, just like tool_runner).
        compaction_threshold: Compact when input_tokens exceeds this. None to disable.
        compaction_model: Model to use for generating summaries.
        summary_prompt: Prompt appended to history to generate the summary.
    """
    # Build dispatch table: name -> callable
    tool_fns = {fn.__name__: fn for fn in tools}
    # Extract JSON schemas for the stable API
    tool_schemas = [fn.to_dict() for fn in tools]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=tool_schemas,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        yield response

        if response.stop_reason == "end_turn":
            break

        # Execute tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = tool_fns[block.name](**block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})

        # Compact if over threshold
        if compaction_threshold and response.usage.input_tokens > compaction_threshold:
            prev_count = len(messages)
            summary_response = client.messages.create(
                model=compaction_model,
                max_tokens=4096,
                messages=messages + [{"role": "user", "content": summary_prompt}],
            )
            summary_text = "".join(
                block.text for block in summary_response.content if hasattr(block, "text")
            )
            messages.clear()
            messages.append({"role": "user", "content": summary_text})
            print(f"🔄 Compaction: {prev_count} messages → 1")
