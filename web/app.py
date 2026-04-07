import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
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


# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host=HOST, port=PORT, reload=True)
