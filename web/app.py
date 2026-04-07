import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import HOST, MAX_TURNS_PER_SESSION, PORT, SESSION_TTL_MINUTES
from agent.rag import DynatraceRAGAgent

app = FastAPI(title="Dynatrace Docs Agent")

# Session store: {session_id: {history: [...], last_access: timestamp}}
_sessions: Dict[str, dict] = {}

# Lazy-init agent (loads embedding model on first request)
_agent: Optional[DynatraceRAGAgent] = None


def _get_agent() -> DynatraceRAGAgent:
    global _agent
    if _agent is None:
        _agent = DynatraceRAGAgent()
    return _agent


def _get_session(session_id: str) -> dict:
    now = time.time()
    # Clean expired sessions
    expired = [
        sid for sid, s in _sessions.items()
        if now - s["last_access"] > SESSION_TTL_MINUTES * 60
    ]
    for sid in expired:
        del _sessions[sid]

    if session_id not in _sessions:
        _sessions[session_id] = {"history": [], "last_access": now}

    session = _sessions[session_id]
    session["last_access"] = now
    return session


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str
    rewritten_query: Optional[str] = None


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = _get_session(session_id)

    agent = _get_agent()
    result = agent.answer(req.message, session["history"])

    # Store turn in history
    session["history"].append({"role": "user", "content": req.message})
    session["history"].append({"role": "assistant", "content": result["answer"]})

    # Trim old turns
    if len(session["history"]) > MAX_TURNS_PER_SESSION * 2:
        session["history"] = session["history"][-MAX_TURNS_PER_SESSION * 2 :]

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=session_id,
        rewritten_query=result.get("rewritten_query"),
    )


@app.get("/api/health")
async def health():
    agent = _get_agent()
    count = agent._retriever._store.count()
    return {"status": "ok", "index_size": count}


# --- Scrape & Index API ---

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

# Track background scrape jobs
_scrape_status: Dict[str, dict] = {}


def _check_admin(token: str):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")


class ScrapeRequest(BaseModel):
    sources: List[str] = ["docs", "community", "blog", "github"]
    max_pages: Optional[int] = 50
    token: Optional[str] = ""


def _run_scrape_and_index(job_id: str, sources: List[str], max_pages: int):
    """Run scrape + index in a background thread."""
    import asyncio
    from scraper.sitemap import fetch_sitemap_urls
    from scraper.crawler import crawl_urls
    from scraper.community import fetch_community_urls, crawl_community
    from scraper.blog import fetch_blog_urls, crawl_blog
    from scraper.github import fetch_github_docs, crawl_github
    from indexer.chunker import chunk_page
    from indexer.embeddings import EmbeddingService
    from indexer.vectorstore import VectorStore
    from config import SITEMAP_URL, RAW_PAGES_DIR

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    status = _scrape_status[job_id]
    all_stats = {}

    try:
        # Scrape each source
        for source in sources:
            status["current"] = f"scraping {source}"
            try:
                if source == "docs":
                    urls = loop.run_until_complete(fetch_sitemap_urls(SITEMAP_URL))
                    stats = loop.run_until_complete(crawl_urls(urls, max_pages=max_pages))
                elif source == "community":
                    urls = loop.run_until_complete(fetch_community_urls(max_urls=max_pages))
                    stats = loop.run_until_complete(crawl_community(urls, max_pages=max_pages))
                elif source == "blog":
                    urls = loop.run_until_complete(fetch_blog_urls(max_urls=max_pages))
                    stats = loop.run_until_complete(crawl_blog(urls, max_pages=max_pages))
                elif source == "github":
                    files = loop.run_until_complete(fetch_github_docs(max_files=max_pages))
                    stats = loop.run_until_complete(crawl_github(files, max_pages=max_pages))
                else:
                    continue
                all_stats[source] = stats
            except Exception as e:
                all_stats[source] = {"error": str(e)}

        # Index all scraped pages
        status["current"] = "indexing"
        page_files = sorted(RAW_PAGES_DIR.glob("*.json"))
        if page_files:
            embedder = EmbeddingService()
            store = VectorStore(embedding_service=embedder)
            store.clear()

            all_chunks = []
            for page_file in page_files:
                page = json.loads(page_file.read_text())
                chunks = chunk_page(page)
                all_chunks.extend(chunks)

            store.add_chunks(all_chunks)
            status["index_size"] = store.count()

        status["status"] = "completed"
        status["stats"] = all_stats

    except Exception as e:
        status["status"] = "failed"
        status["error"] = str(e)
    finally:
        loop.close()


@app.post("/api/scrape")
async def scrape(req: ScrapeRequest, background_tasks: BackgroundTasks):
    _check_admin(req.token or "")

    valid_sources = {"docs", "community", "blog", "github"}
    sources = [s for s in req.sources if s in valid_sources]
    if not sources:
        raise HTTPException(status_code=400, detail=f"No valid sources. Choose from: {valid_sources}")

    job_id = str(uuid.uuid4())[:8]
    _scrape_status[job_id] = {
        "status": "running",
        "current": "starting",
        "sources": sources,
        "max_pages": req.max_pages,
    }

    import threading
    thread = threading.Thread(
        target=_run_scrape_and_index,
        args=(job_id, sources, req.max_pages or 50),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "started", "sources": sources, "max_pages": req.max_pages}


@app.get("/api/scrape/{job_id}")
async def scrape_status(job_id: str):
    if job_id not in _scrape_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return _scrape_status[job_id]


# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host=HOST, port=PORT, reload=True)
