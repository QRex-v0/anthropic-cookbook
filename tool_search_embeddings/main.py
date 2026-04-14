"""
Tool search with embeddings.
https://platform.claude.com/cookbook/tool-use-tool-search-with-embeddings
"""
import json
from typing import Any

import anthropic
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from tool_lib import TOOL_LIBRARY, mock_tool_execution


load_dotenv()

MODEL = "claude-sonnet-4-6"
claude_client = anthropic.Anthropic()

print("Loading SentenceTransformer model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
 
print("✓ Clients initialized successfully")

print(f"✓ Defined {len(TOOL_LIBRARY)} tools in the library")
print("✓ Mock tool execution function created")


def tool_to_text(tool: dict[str, Any]) -> str:
    """
    Convert a tool definition into a text representation from embedding.
    Combine a tool name, description, and parameter information.
    """
    text_parts = [
        f"Tool: {tool['name']}",
        f"Description: {tool['description']}",
    ]

    # Add parameter information if available
    if 'input_schema' in tool and "properties" in tool['input_schema']:
        params = tool["input_schema"]["properties"]
        param_descriptions = []
        for param_name, param_info in params.items():
            param_desc = param_info.get("description", "")
            param_type = param_info.get("type", "")
            param_descriptions.append(f"{param_name} ({param_type}): {param_desc}")
        
        if param_descriptions:
            text_parts.append("Parameters: " + ", ".join(param_descriptions))
    
    return "\n".join(text_parts)

sample_text = tool_to_text(TOOL_LIBRARY[0])
print(f"\n\nSample tool text:\n{sample_text}")

print("Creating embeddings for all tools...")


tool_texts = [tool_to_text(tool) for tool in TOOL_LIBRARY]
tool_embeddings = embedding_model.encode(tool_texts, convert_to_numpy=True)
print(f"✓ Created embeddings with shape: {tool_embeddings.shape}")
print(f"  - {tool_embeddings.shape[0]} tools")
print(f"  - {tool_embeddings.shape[1]} dimensions per embedding")

def search_tools(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Search for tools using semantic similarity.
 
    Args:
        query: Natural language description of what tool is needed
        top_k: Number of top tools to return
 
    Returns:
        List of tool definitions most relevant to the query
    """
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    similarities = np.dot(tool_embeddings, query_embedding)

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({"tool": TOOL_LIBRARY[idx], "similarity_score": float(similarities[idx])})
    return results

# Test the search function
test_query = "I need to check the weather"
test_results = search_tools(test_query, top_k=3)
 
print(f"Search query: '{test_query}'\n")
print("Top 3 matching tools:")
for i, result in enumerate(test_results, 1):
    tool_name = result["tool"]["name"]
    score = result["similarity_score"]
    print(f"{i}. {tool_name} (similarity: {score:.3f})")

# The tool_search tool definition
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
 
print("✓ Tool search definition created")

def handle_tool_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Handle a tool_search invocation and return tool references.
 
    Returns a list of tool_reference content blocks for discovered tools.
    """
    results = search_tools(query, top_k)
    tool_references = [{
        "type": "tool_reference",
        "tool_name": result["tool"]["name"],
    } for result in results]
    
    print(f"\n🔍 Tool search: '{query}'")
    print(f"   Found {len(tool_references)} tools:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['tool']['name']} (similarity: {result['similarity_score']:.3f})")
 
    return tool_references

# Test the handler
test_result = handle_tool_search("stock market data", top_k=3)
print(f"\nReturned {len(test_result)} tool references:")
for ref in test_result:
    print(f"  {ref}")


def run_tool_search_conversation(user_message: str, max_turns: int = 5) -> None:
    """
    Run a conversation with Claude using the tool search pattern.
 
    Args:
        user_message: The initial user message
        max_turns: Maximum number of conversation turns
    """
    print(f"\n{'=' * 80}")
    print(f"USER: {user_message}")
    print(f"{'=' * 80}\n")
 
    # Initialize conversation with only tool_search available
    messages = [{"role": "user", "content": user_message}]
 
    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")
 
        # Call Claude with current message history
        response = claude_client.messages.create(
            model=MODEL,
            max_tokens=1024,
            # BUG: should use defer_loading=True on TOOL_LIBRARY tools
            # Without it, all tools are in context from the start, defeating tool search.
            # See compare_defer_loading() results: 1,464 vs 654 input tokens with 8 tools.
            # Fixed in main_v2.py.
            tools=TOOL_LIBRARY + [TOOL_SEARCH_DEFINITION],
            messages=messages,
            # IMPORTANT: This beta header enables tool definitions in tool results
            extra_headers={"anthropic-beta": "advanced-tool-use-2025-11-20"},
        )
 
        # Add assistant's response to messages
        messages.append({"role": "assistant", "content": response.content})
 
        # Check if we're done
        if response.stop_reason == "end_turn":
            print("\n✓ Conversation complete\n")
            # Print final response
            for block in response.content:
                if block.type == "text":
                    print(f"ASSISTANT: {block.text}")
            break
 
        # Handle tool uses
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
                        # Handle tool search
                        query = tool_input["query"]
                        top_k = tool_input.get("top_k", 5)
 
                        # Get tool references
                        tool_references = handle_tool_search(query, top_k)
 
                        # Create tool result with tool_reference content blocks
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": tool_references,
                            }
                        )
                    else:
                        # Execute the discovered tool with mock data
                        mock_result = mock_tool_execution(tool_name, tool_input)
 
                        # Print a preview of the result
                        if len(mock_result) > 150:
                            print(f"   ✅ Mock result: {mock_result[:150]}...")
                        else:
                            print(f"   ✅ Mock result: {mock_result}")
 
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": mock_result,
                            }
                        )
 
            # Add tool results to messages
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        else:
            print(f"\nUnexpected stop reason: {response.stop_reason}")
            break
 
    print(f"\n{'=' * 80}\n")
 
 
print("✓ Conversation loop implemented")


run_tool_search_conversation(
    "If I invest $10,000 at 5% annual interest for 10 years with monthly compounding, how much will I have?"
)
