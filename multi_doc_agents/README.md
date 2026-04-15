# Multi-Document Agents with LlamaIndex and Claude

Build a multi-document agent system that can query and summarize information across multiple Wikipedia articles using LlamaIndex and Claude.

Reference: https://platform.claude.com/cookbook/third-party-llamaindex-multi-document-agents

## Architecture

1. **Per-city agents** — Each city gets a `ReActAgent` with two tools: a vector search tool (for specific fact retrieval) and a summary tool (for summarization).
2. **Top-level router agent** — A `ReActAgent` that wraps each city agent as a `FunctionTool` and routes queries to the appropriate city.

## Bug Fixes (Step 1)

The original notebook code was broken due to upstream API and library changes. Here's what was fixed:

### Wikipedia API requires User-Agent header
Wikipedia's API now returns **403 Forbidden** without a `User-Agent` header. Added:
```python
headers={"User-Agent": "MultiDocAgents/1.0 (educational project)"}
```

### LlamaIndex v0.14: `ReActAgent` constructor change
`ReActAgent.from_tools()` no longer exists. The new API uses the constructor directly:
```python
# Old (broken)
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)

# New
agent = ReActAgent(tools=query_engine_tools, llm=llm, verbose=True)
```

### LlamaIndex v0.14: `IndexNode(obj=agent)` no longer works
The workflow-based `ReActAgent` is no longer a `BaseQueryEngine`, so `IndexNode(obj=agent)` fails. The fix replaces the old `ObjectIndex`-based routing with a top-level `ReActAgent` that routes via `FunctionTool` wrappers around each city agent:
```python
def _make_query_fn(agent):
    async def query_city(question: str) -> str:
        response = await agent.run(question)
        return str(response)
    return query_city

tool = FunctionTool.from_defaults(async_fn=_make_query_fn(city_agent), ...)
```

### Data path fix
Changed from `data/` (relative to CWD) to `Path(__file__).parent / "data"` (relative to script location), so the script works regardless of where it's invoked from.

## Step 2: Reorganize & Instrument (`main_v2.py`)

`main_v2.py` is a clean rewrite of `main.py` — same logic, modular structure:

| Function | Responsibility |
|----------|---------------|
| `setup_settings()` | LLM, embeddings, global LlamaIndex config |
| `fetch_wikipedia_data()` | Download Wikipedia articles |
| `load_documents()` | Load text files into LlamaIndex Documents |
| `build_city_agent()` | Per-city ReActAgent with vector + summary tools |
| `build_top_agent()` | Top-level router wrapping city agents as FunctionTools |
| `query_with_events()` | Run query with ReAct loop logging |

### Event logging

Replaces LlamaIndex's `verbose=True` with structured output showing the ReAct loop:

```
--- Step 1 ---
REASONING: I need to look up information about Chicago...
ACTION:    query_chicago({'question': '...'})
    [Chicago step 1] -> summary_tool
    [Chicago step 2] -> vector_tool
RESULT:    Chicago is known for...

--- Step 2 ---
...

============================================================
FINAL ANSWER:
...
```

### Token tracking

A shared `TokenTracker` accumulates usage across **all** agents (top-level + sub-agents) and prints a cost summary:

```
============================================================
TOKEN USAGE — 7 LLM calls
  Input:    8,142 tokens
  Output:   2,518 tokens
  Total:   10,660 tokens
  Cost:   $0.3111 (in: $0.1221 + out: $0.1889)
============================================================
```

Pricing table is configurable via the `PRICING` dict at the top of the file.

## Step 3: Modernize — Raw SDK ReAct Loop (`main_v3.py`)

### Why move away from LlamaIndex's ReActAgent?

LlamaIndex ReActAgent was state of the art in 2023-2024, when Claude's tool use API was new and you needed a framework to manage the ReAct loop. **The API has since caught up**, making the framework abstraction more costly than helpful:

- **Fragile across versions** — LlamaIndex v0.14 broke our code twice (`from_tools()` removed, `IndexNode(obj=agent)` removed). Every major release risks the same.
- **Hides control you need** — prompt caching, model mixing (Sonnet for sub-agents, Opus for routing), and token tracking all require fighting the framework or scraping internal events.
- **The ReAct loop is now trivial** — with native `tools` parameter, structured `tool_use` blocks, and `stop_reason` checking, the core loop is ~30 lines of Python.

**What LlamaIndex is still good at**: indexing and retrieval (`VectorStoreIndex`, `SummaryIndex`, `SimpleDirectoryReader`). We keep that, drop the agent loop.

### Plan

1. **Raw SDK ReAct loop** — Replace LlamaIndex `ReActAgent` with a manual `while` loop over `client.messages.create()`. Check `stop_reason == "tool_use"`, dispatch tools, append results, repeat.
2. **Mix models** — Sonnet for city sub-agents (simple retrieval + summarization), Opus for top-level router (complex routing decisions). Biggest cost saver.
3. **Prompt caching** — Use `cache_control` breakpoints on system prompt and tool definitions so we stop re-paying for them on every ReAct step.
4. **Streaming** — Native SSE streaming to show Think/Act/Observe in real-time instead of waiting for full responses.
5. **Built-in token counting** — `response.usage.input_tokens` / `output_tokens` directly from the API, no event scraping.
6. **Persist vector indexes to disk** — Currently `VectorStoreIndex.from_documents()` re-embeds all chunks on every run. Persist to disk, only rebuild when source data changes.

## Step 4: Automated Evaluation (`eval.py`)

`eval.py` runs all three versions as subprocesses (avoids LlamaIndex Settings conflicts) and uses Claude as an automated judge to score and compare answers.

### How it works

1. Each version accepts a question via `sys.argv[1]` and prints the answer between `===EVAL_ANSWER_START===` / `===EVAL_ANSWER_END===` delimiters
2. 4 questions × 3 versions = 12 subprocess runs (sequential)
3. Claude Sonnet judges each question with **shuffled, blind labels** (A/B/C) across 3 rounds to mitigate position bias
4. Scores averaged across rounds; winner by majority vote

### Results

| Question | v1 | v2 | v3 | Winner |
|----------|-----|-----|-----|--------|
| Chicago positive aspects | avg=4.9 | avg=4.5 | avg=4.6 | **v1** |
| Houston population | avg=5.0 | avg=5.0 | avg=5.0 | **v3** (tiebreak: added context) |
| Toronto vs Boston transit | avg=4.2 | avg=3.9 | avg=4.9 | **v3** |
| Strongest economy (5 cities) | avg=1.0 | avg=1.0 | avg=4.9 | **v3** |

**Aggregate: v1=1, v2=0, v3=3 wins**

### Key findings

- **v3 is 2-3× faster** — Sonnet sub-agents respond faster than Opus, and it compounds across 5-15+ LLM round-trips per query
- **v1 & v2 timed out (300s) on multi-city comparison** — the LlamaIndex ReAct loop couldn't finish routing to all 5 city agents in time. v3 completed in 171s.
- **v2 answers truncated** — on longer answers, v2's output cut off mid-sentence (likely a framework buffering issue)
- **v1 won single-city queries** — when the task is simple enough that Opus-everywhere doesn't hit the timeout, its stronger reasoning helps. But this is the minority case.
- **Model mixing works** — using Sonnet for sub-agent retrieval/summarization (a simple task) doesn't hurt quality vs Opus, while cutting cost ~75%

### Speed comparison

| Question | v1 | v2 | v3 |
|----------|-----|-----|-----|
| Chicago summary | 139s | 125s | **43s** |
| Houston population | 35s | 37s | **15s** |
| Transit comparison | 184s | 187s | **93s** |
| Economy comparison | 300s (timeout) | 300s (timeout) | **171s** |

## Design critique: why per-city agents don't scale

The cookbook's architecture (one agent per city, routing layer on top) is a poor design for anything beyond a demo:

- **Linear cost scaling** — adding a document means adding an agent. At 50 documents, you have 50 sub-agents with 50 system prompts.
- **Sequential fanout** — a multi-city comparison query triggers 5+ sub-agents one by one, each running its own ReAct loop. v1/v2 timed out (300s) on a 5-city question.
- **Prompt caching doesn't help** — each sub-agent call is a short, independent conversation. There's not enough repeated prefix for caching to matter.
- **The main v3 win was model mixing** — Sonnet for sub-agents instead of Opus everywhere (~75% cost reduction, 2-3x faster). Not architectural cleverness.

## Step 5 (future): v4 — single agent, merged index, model pipeline

The right architecture for a production multi-document system:

```
User query
    |
Master agent (Sonnet) — decides WHAT to search, formulates queries
    |
Tool: search(query, filters?) — embedding cosine similarity against ONE index
    |
Returns top-k chunks (from any/all cities)
    |
Cheap model (Haiku) — reads chunks, filters noise, summarizes
    |
Clean summary back to master agent
    |
Master reasons over summary, maybe searches again, then answers
```

### Why this is better

| | Per-city agents (v1-v3) | Single agent + search (v4) |
|---|---|---|
| Adding documents | Add a new agent | Add chunks to the index |
| Multi-doc queries | Sequential sub-agent calls | One search returns chunks from all docs |
| Token cost | All models see raw chunks | Only Haiku sees raw chunks |
| Complexity | Routing layer + N agents | One agent + search tool + Haiku filter |

### Key techniques

**Query formulation** — the master model needs to write good search queries, not just forward the user's question. Solutions:
- Query expansion: generate 2-3 search queries per intent, run all, deduplicate
- HyDE (Hypothetical Document Embeddings): ask a cheap model to write a hypothetical answer, embed *that* instead of the question
- Metadata filters: chunks tagged with city/category, master specifies `search(query="transit", city="Chicago")`

**Haiku as a filter** — raw chunks are noisy. Haiku reads them cheaply and returns:
- A summary of relevant content
- Confidence rating (high/medium/low) per chunk
- 2-3 key verbatim sentences as evidence
- Chunk count: "found 25, 3 relevant" so the master knows whether to search again

**Scaling the index:**

| Scale | Solution |
|---|---|
| < 100k chunks | In-memory numpy (like tool_search_embeddings) |
| 100k - 10M | Vector database (Pinecone, pgvector) with ANN indexing |
| 10M+ | Hybrid: BM25 keyword + vector similarity, re-rank combined results |

The pattern: **cheap and broad first, expensive and focused last.** Embeddings filter millions to thousands, metadata/BM25 narrows to hundreds, Haiku filters to tens, master reasons over a few clean summaries.

## Files

| File | Description |
|------|-------------|
| `main.py` | Original fixed script (flat, notebook-style) |
| `main_v2.py` | Reorganized with modular functions, event logging, token tracking |
| `main_v3.py` | Raw SDK ReAct loop, model mixing, prompt caching, streaming |
| `eval.py` | LLM-as-Judge evaluation across all three versions |

## Usage

```bash
# Install dependencies
pip install llama-index llama-index-llms-anthropic llama-index-embeddings-huggingface python-dotenv anthropic

# Set your API key
export ANTHROPIC_API_KEY=sk-...

# Run
python multi_doc_agents/main_v3.py

# Run evaluation
python multi_doc_agents/eval.py
```
