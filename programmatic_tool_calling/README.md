# Programmatic Tool Calling (PTC)

Let Claude write Python scripts that call your tools, instead of round-tripping each tool call through the model's context.

Reference: https://platform.claude.com/cookbook/tool-use-programmatic-tool-calling-ptc

## The problem PTC solves

Traditional tool calling: every tool result goes into Claude's context window.

```
Turn 1: Claude calls get_team_members() → 15 members enter context
Turn 2: Claude calls get_expenses("alice") → 50 records enter context
Turn 3: Claude calls get_expenses("bob") → 40 more records enter context
...
Turn N: Claude has 450 expense records in context just to sum travel costs
```

With PTC: Claude writes a script, the script calls tools, raw data stays in the script's memory, only `print()` output enters Claude's context. Result: **~85% fewer tokens** for the same answer.

## How it actually works

```
1. Claude writes a Python script (e.g., fetch members, loop expenses, aggregate)
2. Anthropic's container starts running it
3. Script hits `await get_team_members()` → PAUSES
4. API sends the tool call back to YOUR code
5. YOUR machine executes get_team_members() (hits your DB, your API, etc.)
6. You send the result back via the API
7. Container RESUMES — result becomes the return value in Python memory
8. Script continues (loops, filters, aggregates)
9. Only print() output goes back to Claude's context
```

**Your local machine always runs the tools.** The container never touches your database or APIs. It can only ask you (via the API) to run tools on its behalf.

The container is a real CPython process hosted by Anthropic. It persists Python variables between pause/resume cycles (same `container_id` across turns). It's sandboxed — no network access, can't harm your system.

## Key API concepts

### `allowed_callers`

Grants the sandbox permission to call a tool:

```python
{"name": "get_expenses", "allowed_callers": ["code_execution_20250825"], ...}
```

Without this, only Claude can call tools directly (normal tool calling).

### `code_execution_20250825`

The sandbox tool type. Add it to your tools array to enable PTC:

```python
tools = [
    {"type": "code_execution_20250825", "name": "code_execution"},
    {"name": "get_expenses", "allowed_callers": ["code_execution_20250825"], ...},
]
```

### `container_id`

The sandbox is stateful. Pass the same `container_id` across turns to preserve Python variables. Containers expire after ~4.5 minutes of inactivity.

### `caller` field

On each `tool_use` block, check who invoked it:

- `block.caller.type == "code_execution_20250825"` → called from the script (PTC)
- `block.caller.type == "direct"` → called by Claude directly (normal)

In both cases, your code executes the tool and returns results the same way. The API routes them to the right place.

## Trust model

- **Tools** — you wrote them, you control the schema, you trust them. Claude can only call them with parameters you defined.
- **Model-generated code** — Claude writes arbitrary Python. The container sandboxes it so it can't do damage. The only things the code can do: call your pre-approved tools and print output.

## Token cost

You pay tokens for Claude writing the script and reading the `print()` output. Tool results flowing between you and the container do NOT count as Claude's input tokens — they go into Python memory, not Claude's context.

## When to use PTC

- Large tool results (expense records, database rows, API responses with metadata)
- Many sequential/parallel tool calls (fetch expenses for 15 people)
- Aggregation logic (sum, filter, group-by) that Python handles better than the model

When NOT to use:

- Small tool results — normal tool calling is simpler
- Sensitive data — results transit through Anthropic's container (see below)

## Sensitive data considerations

PTC's container is hosted by Anthropic — your tool results pass through their infrastructure. For sensitive data, the options are:

1. **Self-hosted sandbox** — Run the same pattern on your infrastructure (Docker, Firecracker, E2B). Data never leaves your network. You build the pause/resume protocol.
2. **Restricted local interpreter** — Run model-generated code locally but strip dangerous modules (`os`, `subprocess`, `sys`). Only expose your tool functions.

Both are "build your own PTC container." The hard part is the pause/resume protocol (intercept tool calls in the script, route to your handlers, resume with results).

Normal tool calling (no PTC) avoids this entirely but puts all raw data into the model's context — fine for small results, expensive for large ones.

## Comparison with OpenAI's approach

| | Anthropic PTC | OpenAI shell tool |
|---|---|---|
| Where code runs | Anthropic's hosted sandbox | Your local machine |
| Safety | Sandboxed by default | You implement sandboxing |
| Data locality | Transits through Anthropic | Stays local |
| Flexibility | Python only, sandboxed | Full shell, any language |
| Setup | One tool type in the array | You build the executor |

OpenAI's shell tool runs locally by default — model proposes commands, your code runs them via `subprocess`. More flexible, but you own the security surface.

## Files

| File | Description |
|------|-------------|
| `main.py` | Traditional tool calling vs PTC comparison |
| `tool_def.py` | Tool schemas + `allowed_callers` config for PTC |
| `team_expense_api.py` | Mock expense API (downloaded from upstream) |

## Usage

```bash
uv run programmatic_tool_calling/main.py
```
