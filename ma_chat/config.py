import importlib.resources.abc
import importlib.abc
import sys

# google-generativeai expects importlib.abc.Traversable, which only exists
# under importlib.resources.abc on this Python version.
sys.modules['importlib.abc'].Traversable = importlib.resources.abc.Traversable

import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
import google.generativeai as genai

from tavily import TavilyClient
from github import Github

MATH_SERVER_PATH = os.path.abspath("math_server.py")

load_dotenv()

# API Keys Initialization
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

print("--- API Key Status ---")
print("NVIDIA_API_KEY:", "Found" if NVIDIA_API_KEY else "Not Found")
print("GROQ_API_KEY:", "Found" if GROQ_API_KEY else "Not Found")
print("GEMINI_API_KEY:", "Found" if GEMINI_API_KEY else "Not Found")
print("TAVILY_API_KEY:", "Found" if TAVILY_API_KEY else "Not Found")
print("GITHUB_TOKEN:", "Found" if GITHUB_TOKEN else "Not Found")

# Gemini Configurations
GEMINI_MODEL = "gemini-2.5-flash"
genai.configure(api_key=GEMINI_API_KEY, transport="rest")
gemini = genai.GenerativeModel(GEMINI_MODEL)

# Database Parameters
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ragdb",  # psycopg3 prefers dbname over database
    "user": "postgres",
    "password": "12345"
}

TOP_K = 5
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"

# NVIDIA serves the embeddings (Groq has no embeddings API).
nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Chat/tool-calling LLM used by every agent. Groq when GROQ_API_KEY is set, otherwise NVIDIA.
if GROQ_API_KEY:
    LLM_MODEL = "openai/gpt-oss-120b"
    llm_client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
else:
    LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"
    llm_client = nvidia_client

# Keep standard clients for threading offload wrappers
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
github_client = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

# Mutable process-wide state, shared across modules via `config.<name>`
# (import the module, not these names, so writes elsewhere are visible).
conn = None
mcp_client = None
app = None
