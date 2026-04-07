import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# Paths — DATA_DIR can be overridden for Railway volume mounts
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
RAW_PAGES_DIR = DATA_DIR / "raw_pages"
CHROMADB_DIR = DATA_DIR / "chromadb"

# Scraper
SITEMAP_URL = "https://docs.dynatrace.com/docs/sitemap.xml"
CRAWL_DELAY = 1.5  # seconds between page loads
CRAWL_CONCURRENCY = 3

# Chunking
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 100  # tokens
MAX_CHUNK_SIZE = 1200  # tokens (allow overflow for code blocks)

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB
COLLECTION_NAME = "dynatrace_docs"

# RAG — model can be overridden via env var
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-haiku-20240307")
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "claude-3-haiku-20240307")
MAX_RETRIEVAL_RESULTS = 8
TOP_K_CHUNKS = 5
MAX_ANSWER_TOKENS = 2048

# Conversation
MAX_HISTORY_TURNS = 6
REWRITE_HISTORY_TURNS = 3
SESSION_TTL_MINUTES = 30
MAX_TURNS_PER_SESSION = 20

# Web — PORT from Railway, HOST always 0.0.0.0
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))
