"""
Reference: https://platform.claude.com/cookbook/third-party-llamaindex-multi-document-agents

An updated version of the multi-document agents recipe.
"""
import sys

import dotenv
import os

dotenv.load_dotenv()

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

llm = Anthropic(temperature=0.0, model="claude-opus-4-1")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

from llama_index.core import Settings
 
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]
 
from pathlib import Path
 
import requests
 
for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
        headers={"User-Agent": "MultiDocAgents/1.0 (educational project)"},
        timeout=30,
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]
 
    data_path = Path(__file__).parent / "data"
    if not data_path.exists():
        Path.mkdir(data_path)
 
    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

# Load all wiki documents
 
from llama_index.core import SimpleDirectoryReader
 
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[str(Path(__file__).parent / "data" / f"{wiki_title}.txt")]
    ).load_data()

from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
 
# Build agents dictionary
agents = {}
 
for wiki_title in wiki_titles:
    # build vector index
    vector_index = VectorStoreIndex.from_documents(
        city_docs[wiki_title],
    )
    # build summary index
    summary_index = SummaryIndex.from_documents(
        city_docs[wiki_title],
    )
    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine()

    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(f"Useful for retrieving specific context from {wiki_title}"),
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(f"Useful for summarization questions related to {wiki_title}"),
            ),
        ),
    ]

    agent = ReActAgent(
        tools=query_engine_tools,
        llm=llm,
        verbose=True,
    )

    agents[wiki_title] = agent

import asyncio
from llama_index.core.tools import FunctionTool

# Wrap each city agent as a tool for the top-level agent
city_tools = []
for wiki_title in wiki_titles:
    city_agent = agents[wiki_title]

    def _make_query_fn(agent):
        async def query_city(question: str) -> str:
            """Query this city's knowledge base."""
            response = await agent.run(question)
            return str(response)
        return query_city

    tool = FunctionTool.from_defaults(
        async_fn=_make_query_fn(city_agent),
        name=f"query_{wiki_title.lower()}",
        description=(
            f"Use this tool to look up specific facts about {wiki_title}. "
            f"Do not use this tool if you want to analyze multiple cities."
        ),
    )
    city_tools.append(tool)

# Top-level agent that routes to city-specific agents
top_agent = ReActAgent(
    tools=city_tools,
    llm=llm,
    verbose=True,
)

async def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "Give me a summary on all the positive aspects of Chicago"
    response = await top_agent.run(question)
    print("===EVAL_ANSWER_START===")
    print(response)
    print("===EVAL_ANSWER_END===")

asyncio.run(main())
