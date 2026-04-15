# Anthropic Cookbook

Implementations and notes from the [Anthropic cookbook](https://platform.claude.com/cookbook), restructured with bug fixes and deeper analysis.

## Projects

| Folder | Topic | Reference |
|--------|-------|-----------|
| `multi_doc_agents/` | Multi-document agents with LlamaIndex — query routing across multiple doc indexes | [cookbook](https://platform.claude.com/cookbook/agent-multi-document-agents-with-llamaindex) |
| `tool_search_embeddings/` | Semantic tool search with embeddings — find relevant tools via cosine similarity instead of passing all tools to the model | [cookbook](https://platform.claude.com/cookbook/tool-use-tool-search-with-embeddings) |
| `context_compaction/` | Context compaction — manage long conversations by summarizing history when context fills up (API-level, manual, and background threading approaches) | [cookbook 1](https://platform.claude.com/cookbook/tool-use-automatic-context-compaction), [cookbook 2](https://platform.claude.com/cookbook/misc-session-memory-compaction) |
| `programmatic_tool_calling/` | Programmatic Tool Calling (PTC) — Claude writes Python scripts that call tools in a sandbox, keeping raw data out of context | [cookbook](https://platform.claude.com/cookbook/tool-use-programmatic-tool-calling-ptc) |

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY=sk-...
```

Each folder has its own README with usage instructions.
