import json
import random
from datetime import datetime, timedelta
from typing import Any

import anthropic
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from tool_lib import TOOL_LIBRARY


load_dotenv()

MODEL = "claude-sonnet-4-6"
claude_client = anthropic.Anthropic()

print("Loading SentenceTransformer model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
 
print("✓ Clients initialized successfully")

print(f"✓ Defined {len(TOOL_LIBRARY)} tools in the library")