"""Multi-document agents with LlamaIndex and Claude.

Reorganized version of main.py with modular functions for each pipeline step.
Reference: https://platform.claude.com/cookbook/third-party-llamaindex-multi-document-agents
"""

import asyncio
from pathlib import Path

import dotenv
import requests
from llama_index.core import Settings, SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

dotenv.load_dotenv()

# --- Configuration ---
WIKI_TITLES = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]
MODEL = "claude-opus-4-1"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 512
DATA_DIR = Path(__file__).parent / "data"

# --- Pricing per million tokens (USD) ---
PRICING = {
    "claude-opus-4-1":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5": {"input":  3.00, "output": 15.00},
    "claude-haiku-3-5":  {"input":  0.80, "output":  4.00},
}


# --- Token Tracker ---
class TokenTracker:
    """Accumulates token usage across all agents (top-level + sub-agents)."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0  # number of LLM round-trips

    def record(self, usage: dict) -> None:
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.calls += 1

    def print_summary(self, model: str) -> None:
        print(f"\n{'='*60}")
        print(f"TOKEN USAGE — {self.calls} LLM calls")
        print(f"  Input:  {self.input_tokens:>8,} tokens")
        print(f"  Output: {self.output_tokens:>8,} tokens")
        print(f"  Total:  {self.input_tokens + self.output_tokens:>8,} tokens")
        prices = PRICING.get(model)
        if prices:
            cost_in = self.input_tokens / 1_000_000 * prices["input"]
            cost_out = self.output_tokens / 1_000_000 * prices["output"]
            print(f"  Cost:   ${cost_in + cost_out:.4f} "
                  f"(in: ${cost_in:.4f} + out: ${cost_out:.4f})")
        print(f"{'='*60}")


# --- Step 1: Setup LLM & Embeddings ---
def setup_settings() -> tuple[Anthropic, HuggingFaceEmbedding]:
    """Initialize LLM, embedding model, and global LlamaIndex settings."""
    llm = Anthropic(temperature=0.0, model=MODEL)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE

    return llm, embed_model


# --- Step 2: Fetch Data ---
def fetch_wikipedia_data(titles: list[str], data_dir: Path) -> None:
    """Download Wikipedia articles to local text files."""
    if not data_dir.exists():
        data_dir.mkdir()

    for title in titles:
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
        wiki_text = page["extract"]

        with open(data_dir / f"{title}.txt", "w") as fp:
            fp.write(wiki_text)


# --- Step 3: Load Documents ---
def load_documents(titles: list[str], data_dir: Path) -> dict[str, list]:
    """Load text files into LlamaIndex Document objects."""
    city_docs = {}
    for title in titles:
        city_docs[title] = SimpleDirectoryReader(
            input_files=[str(data_dir / f"{title}.txt")]
        ).load_data()
    return city_docs


# --- Step 4: Build Per-City Agents ---
def build_city_agent(title: str, documents: list, llm: Anthropic) -> ReActAgent:
    """Build a ReActAgent with vector + summary tools for one city."""
    vector_index = VectorStoreIndex.from_documents(documents)
    summary_index = SummaryIndex.from_documents(documents)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_index.as_query_engine(),
            metadata=ToolMetadata(
                name="vector_tool",
                description=f"Useful for retrieving specific context from {title}",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_index.as_query_engine(),
            metadata=ToolMetadata(
                name="summary_tool",
                description=f"Useful for summarization questions related to {title}",
            ),
        ),
    ]

    return ReActAgent(tools=query_engine_tools, llm=llm, verbose=False)


# --- Step 5: Build Top-Level Router ---
def build_top_agent(
    agents: dict[str, ReActAgent], llm: Anthropic, tracker: TokenTracker
) -> ReActAgent:
    """Wrap city agents as FunctionTools and create a routing agent."""
    city_tools = []
    for title, city_agent in agents.items():

        def _make_query_fn(agent, city_name):
            async def query_city(question: str) -> str:
                """Query this city's knowledge base."""
                handler = agent.run(question)
                sub_step = 1
                async for event in handler.stream_events():
                    if isinstance(event, AgentOutput):
                        usage = event.response.additional_kwargs.get("usage", {})
                        tracker.record(usage)
                        if event.tool_calls:
                            names = ", ".join(tc.tool_name for tc in event.tool_calls)
                            print(f"    [{city_name} step {sub_step}] -> {names}")
                            sub_step += 1
                return str(await handler)
            return query_city

        tool = FunctionTool.from_defaults(
            async_fn=_make_query_fn(city_agent, title),
            name=f"query_{title.lower()}",
            description=(
                f"Use this tool to look up specific facts about {title}. "
                f"Do not use this tool if you want to analyze multiple cities."
            ),
        )
        city_tools.append(tool)

    return ReActAgent(tools=city_tools, llm=llm, verbose=False)


# --- Step 6: Query with Event Streaming ---
async def query_with_events(
    agent: ReActAgent, question: str, tracker: TokenTracker
) -> str:
    """Run a query and print the Reasoning -> Action -> Observation loop."""
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    handler = agent.run(question)
    step = 1

    async for event in handler.stream_events():
        if isinstance(event, AgentOutput):
            usage = event.response.additional_kwargs.get("usage", {})
            tracker.record(usage)

            reasoning = event.response.content
            if event.tool_calls:
                print(f"\n--- Step {step} ---")
                print(f"REASONING: {reasoning}")
            else:
                print(f"\n{'='*60}")
                print(f"FINAL ANSWER:\n{reasoning}")
                print(f"{'='*60}")

        elif isinstance(event, ToolCall):
            print(f"ACTION:    {event.tool_name}({event.tool_kwargs})")

        elif isinstance(event, ToolCallResult):
            output = event.tool_output.content
            preview = (output[:200] + "...") if len(output) > 200 else output
            print(f"RESULT:    {preview}")
            step += 1

    return str(await handler)


# --- Main ---
async def main():
    llm, embed_model = setup_settings()
    fetch_wikipedia_data(WIKI_TITLES, DATA_DIR)
    city_docs = load_documents(WIKI_TITLES, DATA_DIR)

    tracker = TokenTracker()
    agents = {t: build_city_agent(t, city_docs[t], llm) for t in WIKI_TITLES}
    top_agent = build_top_agent(agents, llm, tracker)

    await query_with_events(
        top_agent, "Give me a summary on all the positive aspects of Chicago", tracker
    )
    tracker.print_summary(MODEL)


if __name__ == "__main__":
    asyncio.run(main())
