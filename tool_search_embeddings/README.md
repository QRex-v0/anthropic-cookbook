# Tool Search with Embeddings

Use semantic embeddings to find the most relevant tools for a user query, instead of passing all tools to the LLM on every call.

Reference: https://platform.claude.com/cookbook/tool-use-tool-search-with-embeddings

## How it works

1. Convert each tool definition (name + description + parameters) into a text string
2. Embed all tool texts using `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors)
3. When a user query comes in, embed it and compute cosine similarity against all tool vectors
4. Pass only the top-k most relevant tools to Claude

## When is this actually useful?

Embedding-based tool search is an **optimization for latency and extreme scale**. For most practical tool counts, simpler approaches work fine or better.

### Alternative: LLM manifest approach

Instead of embeddings, give the LLM a compact manifest — one line per tool (name + description). The LLM reads the manifest, picks which tools it needs, then you load the full schemas for just those tools.

- 200 tools x ~20 tokens each = ~4,000 tokens — not expensive
- The LLM is **better at understanding intent** than cosine similarity (handles paraphrasing, indirect references, multi-step reasoning about which tool fits)
- No embedding infrastructure needed
- This is essentially what Claude Code and similar systems do (load all skill descriptions, pick on demand)

### Comparison

| | Pass all tools | LLM manifest | Embedding search |
|---|---|---|---|
| **Complexity** | None | Low | Medium (needs embedding model) |
| **Tool selection quality** | Best (LLM sees everything) | Good (LLM picks from summaries) | Decent (cosine similarity, misses nuance) |
| **Latency** | One LLM call | Extra LLM round-trip | ~1ms vector lookup |
| **Token cost per call** | Scales with tool count | Manifest + selected schemas | Only selected schemas |
| **Sweet spot** | < 20 tools | 20-200 tools | 200+ tools, or latency-critical |

## Deep dive: how `defer_loading` and `tool_reference` work

The cookbook (Nov 2025) passes all tools to the API without `defer_loading`, which defeats the purpose of tool search. Here's how it's actually supposed to work:

### The `defer_loading` mechanism

All tool definitions must be in the `tools` array (the API needs them for schema validation). The key is the `defer_loading: true` flag:

- **Without defer**: All 200 tool schemas are rendered in the **system prompt prefix** — Claude reads all 200 on every turn
- **With defer**: Schemas are stripped from the system prompt. They only enter Claude's context when a `tool_reference` expands them **inline in the conversation**

The 200 deferred tool definitions exist only in the API request payload for server-side lookup — they never enter Claude's context window.

### `tool_reference` expansion is cumulative

Discovered tools expand inline in conversation history, not re-read from the full catalog:

- **Turn 1**: Claude sees only `tool_search` (~500 tokens). Calls tool_search.
- **Turn 2**: `tool_reference` expands `get_weather` inline (~150 tokens). Claude calls it.
- **Turn 3**: History already contains `get_weather` schema. Claude can reuse it or search for more.
- **Turn 4**: New search finds `get_forecast`, expands inline. Claude now sees both.

Token comparison for a 3-tool query across 200 available tools:

| | No defer (all tools loaded) | With defer |
|---|---|---|
| Turn 1 | 30k (200 schemas) + query | ~500 + query |
| Turn 2 | 30k + history | ~500 + ~150 (1 schema) + history |
| Turn 3 | 30k + history | ~500 + ~300 (2 schemas) + history |
| **Total tool token overhead** | **~90k tokens** | **~2k tokens** |

### Why the API needs all tools in the array

`tool_reference` is not "load from nowhere" — it's "unhide this deferred tool." The API must have the schema to:
- Generate valid `tool_use` blocks with proper input structure
- Validate Claude's tool call arguments against the schema
- Return a 400 error if `tool_reference` points to an unknown tool

### The fully client-side alternative

You can skip `defer_loading` and `tool_reference` entirely:

1. Claude has only a `tool_search` tool + a generic `execute_tool` tool
2. Search returns tool descriptions as **plain text** (including input schemas)
3. Claude reads the schema in the text, calls `execute_tool(name="get_weather", args={...})`
4. You dispatch client-side

Trade-off: no API-level schema validation, but simpler — no need to pass 200 tool definitions in every request.

### Anthropic's built-in server-side search

Anthropic also provides built-in tool search (`tool_search_tool_regex` and `tool_search_tool_bm25`) that runs server-side. The cookbook's custom embedding approach is useful for learning but may be redundant if the built-in search works well enough.

### The cookbook's bug

Line 167 passes `tools=TOOL_LIBRARY + [TOOL_SEARCH_DEFINITION]` without `defer_loading: true` on library tools. Fix:

```python
deferred_tools = [{**tool, "defer_loading": True} for tool in TOOL_LIBRARY]
tools = deferred_tools + [TOOL_SEARCH_DEFINITION]
```

## Open questions for experimentation

1. **At what scale does embedding search actually beat the manifest approach?** Is it 100 tools? 500? 1000?
2. **How does cosine similarity accuracy compare to LLM selection?** Embedding might miss tools when the user's phrasing doesn't match the tool description (e.g., "how much is a dollar worth in euros?" vs `convert_currency`)
3. **Hybrid approach** — use embeddings as a first-pass filter (top 20), then LLM picks from those. Does the two-stage pipeline justify the added complexity?
4. **Does the embedding model matter?** `all-MiniLM-L6-v2` is small/fast but 384-dim. Would a larger model (e.g., `bge-base-en-v1.5` at 768-dim) improve tool selection accuracy enough to matter?
5. **`defer_loading` token verification** — run the same query with and without `defer_loading`, compare `response.usage.input_tokens` to confirm the savings are real
6. **Client-side vs `defer_loading`** — compare the generic `execute_tool` approach against proper `defer_loading` + `tool_reference`. What's the quality trade-off from losing schema validation?
7. **Built-in vs custom search** — compare Anthropic's `tool_search_tool_bm25` against custom embedding search for accuracy and latency

## Files

| File | Description |
|------|-------------|
| `main.py` | Original cookbook implementation (has `defer_loading` bug — all tools loaded into context) |
| `main_v2.py` | Fixed version with `defer_loading` and other improvements |
| `tool_lib.py` | Tool library — 8 tools across weather and finance domains |

## Usage

```bash
export ANTHROPIC_API_KEY=sk-...
python tool_search_embeddings/main.py
```
