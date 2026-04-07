from __future__ import annotations

import anthropic

from config import REWRITE_MODEL, REWRITE_HISTORY_TURNS

_client = anthropic.Anthropic()

REWRITE_SYSTEM = """You rewrite follow-up questions into standalone, self-contained search queries.
Given a conversation history and a follow-up question, produce a single search query that
captures the full intent without needing any prior context.
Output ONLY the rewritten query, nothing else."""


def rewrite_query(message: str, history: list[dict]) -> str:
    """
    If the message appears to be a follow-up, rewrite it as a standalone query.
    Returns the original message if no history or it's already standalone.
    """
    if not history:
        return message

    # Use only recent history
    recent = history[-REWRITE_HISTORY_TURNS * 2 :]

    # Build conversation context
    conv_lines = []
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Assistant"
        # Truncate long assistant responses
        content = turn["content"][:500]
        conv_lines.append(f"{role}: {content}")

    conv_text = "\n".join(conv_lines)

    response = _client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=200,
        system=REWRITE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Conversation so far:\n{conv_text}\n\nFollow-up question: {message}\n\nRewritten standalone query:",
            }
        ],
    )

    rewritten = response.content[0].text.strip()
    return rewritten if rewritten else message
