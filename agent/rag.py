from __future__ import annotations

import anthropic

from config import CLAUDE_MODEL, MAX_ANSWER_TOKENS, MAX_HISTORY_TURNS
from agent.rewriter import rewrite_query
from agent.retriever import Retriever


SYSTEM_PROMPT = """You are a Dynatrace technical expert assistant. You answer questions about
Dynatrace using context from multiple official sources:

- **Dynatrace Docs** (docs.dynatrace.com) — official product documentation
- **Dynatrace Community** (community.dynatrace.com) — community Q&A and discussions
- **Dynatrace Blog** (dynatrace.com/news/blog) — engineering blog posts and announcements
- **Dynatrace GitHub** (github.com/dynatrace) — open-source repos, SDKs, and tools

Rules:
- Answer ONLY based on the provided context. Do not use prior knowledge about Dynatrace.
- If the context does not contain enough information to fully answer the question, say so explicitly.
- Cite your sources by including the relevant URLs at the end of your answer.
- Indicate which type of source each citation comes from (Docs, Community, Blog, or GitHub).
- Be precise and technical. Include configuration examples, API calls, or CLI commands when relevant.
- Format your answers with clear headings and bullet points for readability.
- If the user asks about something outside of Dynatrace, politely redirect them.
- When community discussions provide practical tips or workarounds, highlight those alongside the official docs.

When citing sources, use this format at the end of your response:
**Sources:**
- [Page Title](URL) — Source Type
"""


class DynatraceRAGAgent:
    def __init__(self, retriever: Retriever | None = None):
        self._client = anthropic.Anthropic()
        self._retriever = retriever or Retriever()

    def answer(self, message: str, history: list[dict] | None = None) -> dict:
        """
        Answer a user question using RAG.

        Returns: {answer: str, sources: list[{url, title}], rewritten_query: str | None}
        """
        history = history or []

        # Step 1: Rewrite follow-ups as standalone queries
        rewritten = None
        if history:
            rewritten = rewrite_query(message, history)
            search_query = rewritten
        else:
            search_query = message

        # Step 2: Retrieve relevant chunks
        chunks = self._retriever.retrieve(search_query)

        # Step 3: Build context block
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"--- Context {i} ---\n{chunk['text']}")
        context_block = "\n\n".join(context_parts)

        # Step 4: Build messages for Claude
        messages = []

        # Add conversation history (last N turns)
        recent_history = history[-MAX_HISTORY_TURNS * 2 :]
        for turn in recent_history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Add current message with context
        user_message = f"""Here is relevant documentation context:

{context_block}

---

User question: {message}"""

        messages.append({"role": "user", "content": user_message})

        # Step 5: Call Claude
        response = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_ANSWER_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        answer_text = response.content[0].text

        # Extract source URLs from chunks
        sources = []
        seen_urls = set()
        for chunk in chunks:
            url = chunk["metadata"]["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "url": url,
                    "title": chunk["metadata"].get("title", ""),
                })

        return {
            "answer": answer_text,
            "sources": sources,
            "rewritten_query": rewritten,
        }
