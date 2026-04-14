# `main_v3.py` — Raw SDK ReAct Loop

## Why this change

LlamaIndex's `ReActAgent` broke twice during v0.14 and hides control we need
(prompt caching, model mixing, token tracking). Claude's API now natively
supports everything the framework provided. We keep LlamaIndex for indexing
(what it's good at), drop the agent loop.

## What we gain

### Cost: ~75% cheaper per query

A "query" = one user question to the top-level router (e.g. "Give me a summary
on all the positive aspects of Chicago"), which typically triggers 2 router
LLM calls + 3–5 city sub-agent LLM calls underneath.

| | v2 (all Opus) | v3 (Opus router + Sonnet sub-agents) |
|---|---|---|
| Router model | claude-opus-4-1 ($15/$75 per 1M) | claude-opus-4-1 ($15/$75 per 1M) |
| City sub-agent model | claude-opus-4-1 ($15/$75 per 1M) | **claude-sonnet-4-5 ($3/$15 per 1M)** |
| Sub-agent input cost | 5× ~1,200 tok × $15/M = **$0.090** | 5× ~1,200 tok × $3/M = **$0.018** |
| Sub-agent output cost | 5× ~300 tok × $75/M = **$0.113** | 5× ~300 tok × $15/M = **$0.023** |
| **Estimated total** | **~$0.25/query** | **~$0.06/query** |

Sub-agents do the heavy lifting (5+ calls per query), so using Sonnet there
while keeping Opus for routing gives the best quality/cost tradeoff.

### Prompt caching: not impactful here (but the plumbing is in place)

Anthropic's prompt caching requires a minimum token threshold to activate
(1024 tokens for Sonnet, 2048 for Opus). Our system prompts and tool
definitions are ~100-200 tokens — well under the limit, so cache never kicks in.

We *could* stuff the full source document into the system prompt to exceed the
threshold, but that defeats the purpose of tool-based retrieval (fetching only
relevant chunks). We'd pay more total tokens just to show cache savings — a net
cost increase.

The `cache_control` annotations are left in the code as scaffolding. They become
valuable in patterns with larger system prompts — e.g., long multi-turn
conversations, large tool schemas, or RAG with pre-loaded context.

### Index persistence: seconds vs minutes on repeat runs

| | v2 | v3 |
|---|---|---|
| First run | Build indexes in memory (~60s) | Build + persist to disk (~60s) |
| Subsequent runs | Build indexes in memory (~60s) | **Load from disk (~2s)** |

SHA-256 hash of source files detects changes — only rebuilds when data updates.

### Other gains

- **Token visibility**: per-model breakdown with cache hit rates, no hidden LLM calls from framework internals
- **Streaming**: reasoning text appears live as the model generates it
- **No asyncio**: simpler to debug, no event loop complexity
- **Stability**: no more breakage from LlamaIndex agent API changes

## Files changed

- **Create**: `multi_doc_agents/main_v3.py`
- **Modify**: `.gitignore` — add `multi_doc_agents/storage/`

## Structure of `main_v3.py`

Fully synchronous. ~290 lines.

### Configuration

```python
ROUTER_MODEL = "claude-opus-4-1"     # Top-level routing
CITY_MODEL   = "claude-sonnet-4-5"   # City sub-agents (cheaper)
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE   = 512
STORAGE_DIR  = Path(__file__).parent / "storage"
MAX_STEPS    = 10  # safety limit per agent
```

### TokenTracker

Track per-model usage (input, output, cache_creation, cache_read). Use
`response.usage` attributes directly from the SDK. Print per-model cost
breakdown at the end.

> **Note:** LlamaIndex query engines internally call the LLM for response
> synthesis — those tokens are NOT tracked. This is documented in a comment.

### Functions

| Function | What it does | Changes from v2 |
|---|---|---|
| `setup_models()` | Init `anthropic.Anthropic()` client + HuggingFace embeddings + `Settings` | Returns SDK client instead of LlamaIndex LLM. Still sets `Settings.llm` to LlamaIndex Anthropic wrapper (CITY_MODEL) for query engine synthesis. |
| `fetch_wikipedia_data()` | Download Wikipedia articles | Same as v2, plus skips already-downloaded files |
| `load_documents()` | Load text files into Documents | Same as v2 |
| `build_or_load_indexes()` | Per-city: check SHA-256 hash of source file. If match, load from `storage/{city}/vector/` and `storage/{city}/summary/`. Otherwise build + persist. | **New.** Replaces in-memory-only indexing |
| `run_city_agent()` | Raw SDK ReAct loop with streaming. System prompt + vector_tool + summary_tool. Uses `CITY_MODEL`. | **New.** Replaces LlamaIndex `ReActAgent` |
| `run_top_agent()` | Raw SDK ReAct loop with streaming. System prompt + one `query_{city}` tool per city. Uses `ROUTER_MODEL`. Dispatches to `run_city_agent()`. | **New.** Replaces LlamaIndex top-level `ReActAgent` |
| `main()` | Orchestrate everything, print token summary | Synchronous (no `asyncio.run`) |

### ReAct loop pattern (both agents)

```python
messages = [{"role": "user", "content": question}]
for step in range(MAX_STEPS):
    with client.messages.stream(
        model=..., max_tokens=4096, system=system, tools=tools, messages=messages
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)  # live streaming
    response = stream.get_final_message()
    tracker.record(response.usage, model=...)
    if response.stop_reason == "end_turn":
        break
    # append assistant content + tool_results, continue
```

### Prompt caching (scaffolding only)

- System prompt: `cache_control: {"type": "ephemeral"}` on the text block
- Tool definitions: `cache_control: {"type": "ephemeral"}` on the **last** tool
- These annotations are present but don't activate in this demo — system prompts
  and tool definitions are below the minimum token threshold (1024 for Sonnet,
  2048 for Opus). See "What we gain" section for details.

### Index persistence

- `storage/{city}/vector/` and `storage/{city}/summary/` — LlamaIndex's `storage_context.persist()`
- `storage/{city}/source_hash.txt` — SHA-256 of the source `.txt` file
- If hash matches → load from disk (no re-embedding). Otherwise rebuild + persist.

### Output format

```
============================================================
QUESTION: Give me a summary on all the positive aspects of Chicago

--- Step 1 ---
REASONING: I need to look up information about Chicago...  ← streams live
ACTION: query_chicago({"question": "..."})
    [Chicago step 1] → summary_tool
    [Chicago result] Chicago is known for...
RESULT: Chicago is known for...

============================================================
FINAL ANSWER:
...                                                         ← streams live
============================================================

TOKEN USAGE
  claude-opus-4-1:    2 calls |  2,306 in |   923 out
  claude-sonnet-4-5:  5 calls |  5,836 in | 1,595 out
  Total: $0.0582
```

## Verification

1. Run `python multi_doc_agents/main_v3.py`
2. First run: prints "Building indexes for {city}..." for each city
3. Second run: prints "Loading cached indexes for {city}..." (fast, no re-embedding)
4. Output shows ReAct loop clearly with streaming text
5. Token summary shows per-model breakdown
6. Compare final answer quality with v2 output
