"""Multi-document agents with raw Claude SDK ReAct loop.

Keeps LlamaIndex for indexing/retrieval, replaces the agent loop with direct
Anthropic SDK calls. Gives us prompt caching, model mixing, token tracking,
and streaming — things the framework hides or breaks between versions.

Reference: https://platform.claude.com/cookbook/third-party-llamaindex-multi-document-agents
"""

import hashlib
import json
from pathlib import Path

import anthropic
import dotenv
import requests
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic as LlamaIndexAnthropic

dotenv.load_dotenv()

# --- Configuration ---
WIKI_TITLES = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]
ROUTER_MODEL = "claude-opus-4-1"      # Top-level routing
CITY_MODEL = "claude-sonnet-4-5"      # City sub-agents (cheaper)
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 512
DATA_DIR = Path(__file__).parent / "data"
STORAGE_DIR = Path(__file__).parent / "storage"
MAX_STEPS = 10  # safety limit per agent

# --- Pricing per million tokens (USD) ---
PRICING = {
    "claude-opus-4-1": {
        "input": 15.00,
        "output": 75.00,
        "cache_read": 1.50,
        "cache_creation": 18.75,
    },
    "claude-sonnet-4-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_creation": 3.75,
    },
}


# --- Token Tracker ---
class TokenTracker:
    """Accumulates per-model token usage across all agents."""

    def __init__(self):
        self.usage = {}  # model -> {input, output, cache_creation, cache_read, calls}

    def record(self, usage, model: str) -> None:
        if model not in self.usage:
            self.usage[model] = {
                "input": 0, "output": 0,
                "cache_creation": 0, "cache_read": 0, "calls": 0,
            }
        s = self.usage[model]
        s["input"] += getattr(usage, "input_tokens", 0)
        s["output"] += getattr(usage, "output_tokens", 0)
        s["cache_creation"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        s["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        s["calls"] += 1

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print("TOKEN USAGE")
        total_cost = 0.0
        total_cache_savings = 0.0
        for model, s in self.usage.items():
            p = PRICING.get(model, {})
            cost = (
                s["input"] / 1e6 * p.get("input", 0)
                + s["output"] / 1e6 * p.get("output", 0)
                + s["cache_creation"] / 1e6 * p.get("cache_creation", 0)
                + s["cache_read"] / 1e6 * p.get("cache_read", 0)
            )
            # Cache savings: difference between full-price input and discounted cache_read
            saved = s["cache_read"] / 1e6 * (p.get("input", 0) - p.get("cache_read", 0))
            total_cost += cost
            total_cache_savings += saved
            parts = [
                f"{s['calls']:>3} calls",
                f"{s['input']:>7,} in",
                f"{s['output']:>7,} out",
                f"{cost:.4f} cost",
            ]
            if s["cache_read"]:
                parts.append(f"cache: {s['cache_read']:,} read")
            if s["cache_creation"]:
                parts.append(f"{s['cache_creation']:,} created")
            print(f"  {model}: {' | '.join(parts)}")
        print(f"  Total: ${total_cost:.4f} (saved ${total_cache_savings:.4f} from caching)")
        # NOTE: LlamaIndex query engines internally call the LLM for response
        # synthesis — those tokens are NOT tracked here.
        print(f"{'='*60}")


# --- Setup ---
def setup_models():
    """Initialize Anthropic SDK client, HuggingFace embeddings, and LlamaIndex settings."""
    client = anthropic.Anthropic()
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # LlamaIndex still needs its own LLM wrapper for query engine synthesis
    Settings.llm = LlamaIndexAnthropic(temperature=0.0, model=CITY_MODEL)
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE

    return client, embed_model


# --- Data fetching ---
def fetch_wikipedia_data(titles: list[str], data_dir: Path) -> None:
    """Download Wikipedia articles to local text files."""
    """Unchanged from v2.py"""
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    for title in titles:
        filepath = data_dir / f"{title}.txt"
        if filepath.exists():
            continue
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            },
            headers={"User-Agent": "MultiDocAgents/1.0 (educational project)"},
            timeout=30,
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        with open(filepath, "w") as fp:
            fp.write(page["extract"])
        print(f"  Downloaded {title}")


# --- Document loading ---
def load_documents(titles: list[str], data_dir: Path) -> dict[str, list]:
    """Load text files into LlamaIndex Document objects."""
    """Unchanged from v2.py"""
    city_docs = {}
    for title in titles:
        city_docs[title] = SimpleDirectoryReader(
            input_files=[str(data_dir / f"{title}.txt")]
        ).load_data()
    return city_docs


# --- Index persistence ---
def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_or_load_indexes(
    titles: list[str], city_docs: dict[str, list], data_dir: Path
) -> dict[str, dict[str, QueryEngineTool]]:
    """Build or load persisted vector + summary indexes per city."""
    """For demo purposes, we just use JSON files; for advanced use, we would use a database."""
    city_tools = {}

    for title in titles:
        source_path = data_dir / f"{title}.txt"
        current_hash = _file_hash(source_path)
        city_storage = STORAGE_DIR / title.lower()
        hash_file = city_storage / "source_hash.txt"

        cached = (
            hash_file.exists()
            and hash_file.read_text().strip() == current_hash
            and (city_storage / "vector").exists()
            and (city_storage / "summary").exists()
        )

        if cached:
            print(f"  Loading cached indexes for {title}...")
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(city_storage / "vector"))
            )
            summary_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(city_storage / "summary"))
            )
        else:
            print(f"  Building indexes for {title}...")
            docs = city_docs[title]
            vector_index = VectorStoreIndex.from_documents(docs)
            summary_index = SummaryIndex.from_documents(docs)

            # Persist
            city_storage.mkdir(parents=True, exist_ok=True)
            vector_index.storage_context.persist(persist_dir=str(city_storage / "vector"))
            summary_index.storage_context.persist(persist_dir=str(city_storage / "summary"))
            hash_file.write_text(current_hash)

        city_tools[title] = {
            "vector": QueryEngineTool(
                query_engine=vector_index.as_query_engine(),
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=f"Useful for retrieving specific context from {title}",
                ),
            ),
            "summary": QueryEngineTool(
                query_engine=summary_index.as_query_engine(),
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=f"Useful for summarization questions related to {title}",
                ),
            ),
        }

    return city_tools


# --- ReAct loop helpers ---
def _extract_text(content_blocks) -> str:
    """Extract concatenated text from response content blocks."""
    return "".join(b.text for b in content_blocks if b.type == "text")


def _extract_tool_calls(content_blocks) -> list:
    """Extract tool_use blocks from response content."""
    return [b for b in content_blocks if b.type == "tool_use"]


# --- City agent ---
def run_city_agent(
    client: anthropic.Anthropic,
    tracker: TokenTracker,
    title: str,
    city_tools: dict[str, QueryEngineTool],
    question: str,
) -> str:
    """Run a raw SDK ReAct loop for a single city."""
    system = [
        {
            "type": "text",
            "text": (
                f"You are an expert research assistant for {title}. "
                f"Use the provided tools to answer questions about {title}. "
                "Think step-by-step. If the question asks for a summary, prefer the summary_tool. "
                "For specific facts, prefer the vector_tool."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]

    tools = [
        {
            "name": "vector_tool",
            "description": f"Useful for retrieving specific context from {title}",
            "input_schema": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The query to search for"}},
                "required": ["question"],
            },
        },
        {
            "name": "summary_tool",
            "description": f"Useful for summarization questions related to {title}",
            "input_schema": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The query to summarize"}},
                "required": ["question"],
            },
            "cache_control": {"type": "ephemeral"},  # last tool gets cache_control
        },
    ]

    messages = [{"role": "user", "content": question}]

    for step in range(1, MAX_STEPS + 1):
        with client.messages.stream(
            model=CITY_MODEL, max_tokens=4096, system=system, tools=tools, messages=messages
        ) as stream:
            for text in stream.text_stream:
                pass  # city agent text is not streamed to console
        response = stream.get_final_message()
        tracker.record(response.usage, model=CITY_MODEL)

        tool_calls = _extract_tool_calls(response.content)
        if response.stop_reason == "end_turn" or not tool_calls:
            return _extract_text(response.content)

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            tool_name = tc.name
            tool_input = tc.input
            print(f"    [{title} step {step}] -> {tool_name}")

            qe_tool = city_tools.get(tool_name.replace("_tool", ""))
            if not qe_tool:
                qe_tool = city_tools.get("vector")  # fallback

            result = str(qe_tool.query_engine.query(tool_input.get("question", question)))
            preview = (result[:150] + "...") if len(result) > 150 else result
            print(f"    [{title} result] {preview}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    return _extract_text(response.content)


# --- Top-level agent ---
def run_top_agent(
    client: anthropic.Anthropic,
    tracker: TokenTracker,
    city_tools_map: dict[str, dict[str, QueryEngineTool]],
    question: str,
) -> str:
    """Run the top-level router agent with raw SDK ReAct loop + streaming."""
    city_names = list(city_tools_map.keys())

    system = [
        {
            "type": "text",
            "text": (
                "You are a helpful research assistant that can look up information about cities. "
                "You have access to specialized tools for each city. "
                "Use the appropriate city tool to answer questions. "
                "For questions about a specific city, use that city's tool. "
                "For comparative questions, query each relevant city separately then synthesize."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]

    tools = []
    for i, title in enumerate(city_names):
        tool_def = {
            "name": f"query_{title.lower()}",
            "description": (
                f"Use this tool to look up specific facts about {title}. "
                f"Do not use this tool if you want to analyze multiple cities."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": f"The question to ask about {title}"}},
                "required": ["question"],
            },
        }
        # Cache control on the last tool only
        if i == len(city_names) - 1:
            tool_def["cache_control"] = {"type": "ephemeral"}
        tools.append(tool_def)

    messages = [{"role": "user", "content": question}]

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Step {step} ---")

        with client.messages.stream(
            model=ROUTER_MODEL, max_tokens=4096, system=system, tools=tools, messages=messages
        ) as stream:
            print("REASONING: ", end="")
            for text in stream.text_stream:
                print(text, end="", flush=True)
            print()
        response = stream.get_final_message()
        tracker.record(response.usage, model=ROUTER_MODEL)

        tool_calls = _extract_tool_calls(response.content)
        if response.stop_reason == "end_turn" or not tool_calls:
            final_text = _extract_text(response.content)
            print(f"\n{'='*60}")
            print(f"FINAL ANSWER:\n{final_text}")
            print(f"{'='*60}")
            return final_text

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            tool_name = tc.name
            tool_input = tc.input
            city_title = next(
                (t for t in city_names if f"query_{t.lower()}" == tool_name), None
            )
            print(f"ACTION: {tool_name}({json.dumps(tool_input)})")

            if city_title:
                result = run_city_agent(
                    client, tracker, city_title, city_tools_map[city_title], tool_input["question"]
                )
            else:
                result = f"Unknown tool: {tool_name}"

            preview = (result[:200] + "...") if len(result) > 200 else result
            print(f"RESULT: {preview}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    final_text = _extract_text(response.content)
    return final_text


# --- Main ---
def main():
    print("Setting up models...")
    client, _ = setup_models()

    print("Fetching Wikipedia data...")
    fetch_wikipedia_data(WIKI_TITLES, DATA_DIR)

    print("Loading documents...")
    city_docs = load_documents(WIKI_TITLES, DATA_DIR)

    print("Building/loading indexes...")
    city_tools_map = build_or_load_indexes(WIKI_TITLES, city_docs, DATA_DIR)

    tracker = TokenTracker()

    run_top_agent(
        client, tracker, city_tools_map,
        "Give me a summary on all the positive aspects of Chicago",
    )

    tracker.print_summary()


if __name__ == "__main__":
    main()
