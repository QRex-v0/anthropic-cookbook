# Context Compaction

Manage long-running conversations and agentic workflows by compressing conversation history when the context window fills up.

References:
- https://platform.claude.com/cookbook/tool-use-automatic-context-compaction
- https://platform.claude.com/cookbook/misc-session-memory-compaction

## Core idea

When a conversation gets too long, summarize the history and replace it with the summary. Claude continues working from the summary instead of the full history. This saves tokens and avoids hitting context limits.

## Three approaches

| Approach | File | How it works | Pros | Cons |
|---|---|---|---|---|
| API `compaction_control` | `compaction_control.py` | `client.beta.messages.tool_runner` with `compaction_control` flag | One config line, Anthropic handles everything | Beta API, black box, doesn't work with server-side tools |
| Manual tool runner | `compaction_manual.py` | Drop-in replacement for `tool_runner` with explicit compaction logic | Full control, stable API, use-case agnostic | You manage the loop yourself |
| Session memory | `session_memory.py` | Background threading pre-builds summaries before they're needed | Instant compaction (no user wait) | More complex, only for chat (no tools) |

## How compaction works

1. Track token usage via `response.usage.input_tokens`
2. When tokens exceed a threshold, generate a summary of the conversation
3. Replace the full message history with the summary
4. Continue the conversation from the summary

## `compaction_control.py` — API-level (simplest)

Uses `client.beta.messages.tool_runner` with `compaction_control` config. Runs a baseline (no compaction) then a compaction run and compares token usage.

```bash
uv run context_compaction/compaction_control.py
```

Requires `utils/` — download from: https://github.com/anthropics/claude-cookbooks/tree/main/tool_use/utils

## `compaction_manual.py` — DIY tool runner

A generic `tool_runner_with_compaction()` function that replaces `client.beta.messages.tool_runner`. Works with any `@beta_tool` functions, not coupled to any use case.

```python
from compaction_manual import tool_runner_with_compaction

for message in tool_runner_with_compaction(
    client=client,
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=[my_tool_a, my_tool_b],
    messages=messages,
    compaction_threshold=5000,
    compaction_model="claude-haiku-4-5",
):
    print(message.usage)
```

## `session_memory.py` — Background pre-building

For long-running chat sessions (no tools). Two modes:

```bash
# Synchronous — user waits during compaction
uv run context_compaction/session_memory.py traditional

# Background threading — summary pre-built, instant swap
uv run context_compaction/session_memory.py instant
```

The "instant" mode starts building the summary in a background thread before the context is full. When the limit is hit, it swaps in the pre-built summary with no wait time.

Uses `add_cache_control` to mark messages for Anthropic's prompt caching — the background summary thread shares the same message prefix as the main chat, so it gets ~80% cache hit instead of paying full price.

## Limitations

- **Server-side sampling loops** (web search, extended thinking): Cache tokens from internal loops can trigger compaction prematurely. The token count reflects server-side work, not actual conversation length.
- **Summary quality**: A bad summary = lost context. The `session_memory_prompt.py` prompt is intentionally detailed to preserve task state, decisions, and corrections.

## Files

| File | Description |
|------|-------------|
| `compaction_control.py` | API-level compaction via `tool_runner` + `compaction_control` |
| `compaction_manual.py` | Generic manual tool runner with optional compaction |
| `session_memory.py` | Chat session compaction with traditional and instant (background) modes |
| `session_memory_prompt.py` | Structured summary prompt used by session memory compaction |
| `session_memory_compaction.py` | Scratch file |
| `utils/` | Customer service tools (gitignored, download from upstream) |
