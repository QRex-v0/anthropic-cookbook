"""
Tool search with embeddings — v2 (restructured, defer_loading bug fixed).
https://platform.claude.com/cookbook/tool-use-tool-search-with-embeddings

Changes from v1 (main.py):
- Add defer_loading: True to all TOOL_LIBRARY tools so they aren't in context until discovered
- Wrap everything in functions (no module-level side effects)
- Track and print token usage per turn
- Add __main__ guard
"""
import argparse
import json
from typing import Any

import anthropic
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from tool_lib import TOOL_LIBRARY, mock_tool_execution


MODEL = "claude-sonnet-4-6"

TOOL_SEARCH_DEFINITION = {
    "name": "tool_search",
    "description": "Search for available tools that can help with a task. Returns tool definitions for matching tools. Use this when you need a tool but don't have it available yet.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language description of what kind of tool you need (e.g., 'weather information', 'currency conversion', 'stock prices')",
            },
            "top_k": {
                "type": "number",
                "description": "Number of tools to return (default: 5)",
            },
        },
        "required": ["query"],
    },
}


def setup_models() -> tuple[anthropic.Anthropic, SentenceTransformer]:
    """Initialize the Anthropic client and embedding model."""
    load_dotenv()
    client = anthropic.Anthropic()

    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✓ Clients initialized successfully")

    return client, embedding_model


def tool_to_text(tool: dict[str, Any]) -> str:
    """Convert a tool definition into a text representation for embedding."""
    text_parts = [
        f"Tool: {tool['name']}",
        f"Description: {tool['description']}",
    ]

    if "input_schema" in tool and "properties" in tool["input_schema"]:
        params = tool["input_schema"]["properties"]
        param_descriptions = []
        for param_name, param_info in params.items():
            param_desc = param_info.get("description", "")
            param_type = param_info.get("type", "")
            param_descriptions.append(f"{param_name} ({param_type}): {param_desc}")

        if param_descriptions:
            text_parts.append("Parameters: " + ", ".join(param_descriptions))

    return "\n".join(text_parts)


def build_tool_embeddings(model: SentenceTransformer, tools: list[dict[str, Any]]) -> np.ndarray:
    """Create embeddings for all tool definitions."""
    print("Creating embeddings for all tools...")
    tool_texts = [tool_to_text(tool) for tool in tools]
    embeddings = model.encode(tool_texts, convert_to_numpy=True)
    print(f"✓ Created embeddings: {embeddings.shape[0]} tools, {embeddings.shape[1]} dimensions")
    return embeddings


def search_tools(
    query: str,
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search for tools using semantic similarity."""
    query_embedding = model.encode(query, convert_to_numpy=True)
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [
        {"tool": TOOL_LIBRARY[idx], "similarity_score": float(similarities[idx])}
        for idx in top_indices
    ]


def handle_tool_search(
    query: str,
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Handle a tool_search invocation and return tool_reference content blocks."""
    results = search_tools(query, embeddings, model, top_k)

    print(f"\n🔍 Tool search: '{query}'")
    print(f"   Found {len(results)} tools:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['tool']['name']} (similarity: {result['similarity_score']:.3f})")

    return [{"type": "tool_reference", "tool_name": r["tool"]["name"]} for r in results]


def run_conversation(
    client: anthropic.Anthropic,
    embedding_model: SentenceTransformer,
    embeddings: np.ndarray,
    user_message: str,
    max_turns: int = 5,
    defer_loading: bool = False,
) -> None:
    """Run a conversation with Claude using the tool search pattern.

    Args:
        defer_loading: If True, add defer_loading to library tools so their schemas
            only enter context when discovered via tool_reference. Set to False to
            compare token usage without deferral.
    """
    print(f"\n{'=' * 80}")
    print(f"USER: {user_message}")
    print(f"{'=' * 80}\n")

    if defer_loading:
        library_tools = [{**tool, "defer_loading": True} for tool in TOOL_LIBRARY]
    else:
        library_tools = list(TOOL_LIBRARY)
    tools = library_tools + [TOOL_SEARCH_DEFINITION]

    messages = [{"role": "user", "content": user_message}]
    total_input_tokens = 0
    total_output_tokens = 0

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=tools,
            messages=messages,
            extra_headers={"anthropic-beta": "advanced-tool-use-2025-11-20"},
        )

        # Track tokens
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        print(f"  [tokens: {input_tokens:,} in / {output_tokens:,} out]")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            print("\n✓ Conversation complete\n")
            for block in response.content:
                if block.type == "text":
                    print(f"ASSISTANT: {block.text}")
            break

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "text":
                    print(f"\nASSISTANT: {block.text}")

                elif block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    print(f"\n🔧 Tool invocation: {tool_name}")
                    print(f"   Input: {json.dumps(tool_input, indent=2)}")

                    if tool_name == "tool_search":
                        query = tool_input["query"]
                        top_k = tool_input.get("top_k", 5)
                        tool_references = handle_tool_search(query, embeddings, embedding_model, top_k)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_references,
                        })
                    else:
                        mock_result = mock_tool_execution(tool_name, tool_input)
                        preview = mock_result[:150] + "..." if len(mock_result) > 150 else mock_result
                        print(f"   ✅ Mock result: {preview}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": mock_result,
                        })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        else:
            print(f"\nUnexpected stop reason: {response.stop_reason}")
            break

    print(f"\nTOTAL: {total_input_tokens:,} input / {total_output_tokens:,} output tokens")
    print(f"{'=' * 80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool search with embeddings")
    parser.add_argument(
        "--defer-loading",
        action="store_true",
        default=False,
        help="Enable defer_loading on library tools to reduce input tokens",
    )
    args = parser.parse_args()

    client, embedding_model = setup_models()
    embeddings = build_tool_embeddings(embedding_model, TOOL_LIBRARY)
    run_conversation(
        client,
        embedding_model,
        embeddings,
        "If I invest $10,000 at 5% annual interest for 10 years with monthly compounding, how much will I have?",
        defer_loading=args.defer_loading,
    )


if __name__ == "__main__":
    main()
