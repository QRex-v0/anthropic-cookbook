"""
Programmatic Tool Calling (PTC).
https://platform.claude.com/cookbook/tool-use-programmatic-tool-calling-ptc

Claude writes and executes code to call tools programmatically in a sandbox,
instead of requiring a round-trip API call for each tool invocation.
Key API feature: `allowed_callers` on tool definitions.
"""

from dotenv import load_dotenv
import anthropic

from tool_def import tool_functions, tools, ptc_tools

load_dotenv()

MODEL = "claude-sonnet-4-6"



client = anthropic.Anthropic()


import json
import time
 
from anthropic.types import TextBlock, ToolUseBlock
from anthropic.types.beta import (
    BetaMessageParam as MessageParam,
)
from anthropic.types.beta import (
    BetaTextBlock,
    BetaToolUseBlock,
)
 
messages: list[MessageParam] = []
 
 
def run_agent_without_ptc(user_message):
    """Run agent using traditional tool calling"""
    messages.append({"role": "user", "content": user_message})
    total_tokens = 0
    start_time = time.time()
    api_counter = 0
 
    while True:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            tools=tools,
            messages=messages,
            betas=["advanced-tool-use-2025-11-20"],
        )
 
        api_counter += 1
 
        # Track token usage
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        print(f"  Turn {api_counter}: {response.usage.input_tokens:,} in / {response.usage.output_tokens:,} out | stop: {response.stop_reason}")
        if response.stop_reason == "end_turn":
            # Extract the first text block from the response
            final_response = next(
                (
                    block.text
                    for block in response.content
                    if isinstance(block, (BetaTextBlock, TextBlock))
                ),
                None,
            )
            elapsed_time = time.time() - start_time
            return final_response, messages, total_tokens, elapsed_time, api_counter
 
        # Process tool calls
        if response.stop_reason == "tool_use":
            # First, add the assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})
 
            # Collect all tool results
            tool_results = []
 
            for block in response.content:
                if isinstance(block, (BetaToolUseBlock, ToolUseBlock)):
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id
 
                    result = tool_functions[tool_name](**tool_input)
 
                    content = str(result)
 
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    }
                    tool_results.append(tool_result)
 
            # Append all tool results at once after collecting them
            messages.append({"role": "user", "content": tool_results})
 
        else:
            print(f"\nUnexpected stop reason: {response.stop_reason}")
            elapsed_time = time.time() - start_time
 
            final_response = next(
                (
                    block.text
                    for block in response.content
                    if isinstance(block, (BetaTextBlock, TextBlock))
                ),
                f"Stopped with reason: {response.stop_reason}",
            )
            return final_response, messages, total_tokens, elapsed_time, api_counter


query = "Which engineering team members exceeded their Q3 travel budget? Standard quarterly travel budget is $5,000. However, some employees have custom budget limits. For anyone who exceeded the $5,000 standard budget, check if they have a custom budget exception. If they do, use that custom limit instead to determine if they truly exceeded their budget."

# # Run the agent
# result, conversation, total_tokens, elapsed_time, api_count_without_ptc = run_agent_without_ptc(
#     query
# )
 
# print(f"Result: {result}")
# print(f"API calls made: {api_count_without_ptc}")
# print(f"Total tokens used: {total_tokens:,}")
# print(f"Total time taken: {elapsed_time:.2f}s")


messages = []

def run_agent_with_ptc(user_message):
    """Run agent using programmatic tool calling"""
    messages.append({"role": "user", "content": user_message})
    total_tokens = 0
    start_time = time.time()
    container_id = None
    api_counter = 0

    while True:
        request_params = {
            "model": MODEL,
            "max_tokens": 4000,
            "tools": ptc_tools,
            "messages": messages,
        }
        response = client.beta.messages.create(
            **request_params,
            betas=["advanced-tool-use-2025-11-20"],
            extra_body={"container": container_id} if container_id else None,
        )
        api_counter += 1

        # Track container for stateful execution
        if hasattr(response, "container") and response.container:
            container_id = response.container.id
            print(f"\n[Container] ID: {container_id}")
            if hasattr(response.container, "expires_at"):
                # If the container has expired, we would need to restart our workflow. In our case, it completes before expiration.
                print(f"[Container] Expires at: {response.container.expires_at}")
        
        # Track token usage
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        print(f"  Turn {api_counter}: {response.usage.input_tokens:,} in / {response.usage.output_tokens:,} out | stop: {response.stop_reason}")

        if response.stop_reason == "end_turn":
            # Extract the first text block from the response
            final_response = next(
                (block.text for block in response.content if isinstance(block, BetaTextBlock)),
                None,
            )
            elapsed_time = time.time() - start_time
            return final_response, messages, total_tokens, elapsed_time, api_counter
        
        # As before, we process tool calls
        if response.stop_reason == "tool_use":
            # First, add the assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})
 
            # Collect all tool results
            tool_results = []
 
            for block in response.content:
                if isinstance(block, BetaToolUseBlock):
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id
 
                    # We can use caller type to understand how the tool was invoked
                    caller_type = block.caller.type
 
                    if caller_type == "code_execution_20250825":
                        print(f"[PTC] Tool called from code execution environment: {tool_name}")
 
                    elif caller_type == "direct":
                        print(f"[Direct] Tool called by model: {tool_name}")
 
                    result = tool_functions[tool_name](**tool_input)
 
                    # Format result as proper content for the API
                    if isinstance(result, list) and result and isinstance(result[0], str):
                        content = "\n".join(result)
                    elif isinstance(result, (dict, list)):
                        content = json.dumps(result)
                    else:
                        content = str(result)
 
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": content,
                        }
                    )
 
            messages.append({"role": "user", "content": tool_results})
 
        else:
            print(f"\nUnexpected stop reason: {response.stop_reason}")
            elapsed_time = time.time() - start_time
 
            final_response = next(
                (block.text for block in response.content if isinstance(block, BetaTextBlock)),
                f"Stopped with reason: {response.stop_reason}",
            )
            return final_response, messages, total_tokens, elapsed_time, api_counter

# Run the PTC agent
result_ptc, conversation_ptc, total_tokens_ptc, elapsed_time_ptc, api_count_with_ptc = (
    run_agent_with_ptc(query)
)
print(f"\n{'=' * 60}")
print(f"Result: {result_ptc}")
print(f"\n{'=' * 60}")
print("Performance Metrics:")
print(
    f"  Total API calls to Claude: {len([m for m in conversation_ptc if m['role'] == 'assistant'])}"
)
print(f"  Total tokens used: {total_tokens_ptc:,}")
print(f"  Total time taken: {elapsed_time_ptc:.2f}s")
