import re

import tiktoken

from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CHUNK_SIZE

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _split_by_separators(text: str, separators: list[str]) -> list[str]:
    """Recursively split text by a list of separators in priority order."""
    if not separators:
        return [text]

    sep = separators[0]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split_by_separators(text, separators[1:])

    # Re-attach separator to the beginning of each part (except first)
    result = [parts[0]]
    for part in parts[1:]:
        result.append(sep + part)
    return [p for p in result if p.strip()]


def _merge_small_chunks(chunks: list[str], max_tokens: int) -> list[str]:
    """Merge consecutive small chunks that fit within max_tokens."""
    merged = []
    current = ""
    for chunk in chunks:
        combined = (current + "\n" + chunk).strip() if current else chunk
        if count_tokens(combined) <= max_tokens:
            current = combined
        else:
            if current:
                merged.append(current)
            current = chunk
    if current:
        merged.append(current)
    return merged


def _add_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """Add token overlap between consecutive chunks."""
    if len(chunks) <= 1 or overlap_tokens <= 0:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = _enc.encode(chunks[i - 1])
        overlap_text = _enc.decode(prev_tokens[-overlap_tokens:]) if len(prev_tokens) > overlap_tokens else ""
        if overlap_text.strip():
            result.append(overlap_text.strip() + "\n" + chunks[i])
        else:
            result.append(chunks[i])
    return result


def _is_code_or_table_block(text: str) -> bool:
    """Check if text is primarily a code block or table."""
    stripped = text.strip()
    return stripped.startswith("```") or stripped.startswith("|") or stripped.startswith("+-")


def chunk_page(page: dict) -> list[dict]:
    """
    Split a scraped page into chunks with metadata.

    Returns list of {text, metadata} dicts where metadata includes
    url, title, section heading, and chunk index.
    """
    text = page["text"]
    url = page["url"]
    title = page["title"]
    source = page.get("source", "docs")

    # Split into sections by H2 headings
    # Look for lines that match heading patterns from the extracted text
    h2_pattern = re.compile(r"^(.{3,80})$", re.MULTILINE)

    # Use the page's heading data to find section boundaries
    headings = page.get("headings", [])
    h2_headings = [h["text"] for h in headings if h["level"] == 2]

    if h2_headings:
        sections = _split_into_sections(text, h2_headings)
    else:
        sections = [("", text)]

    chunks = []
    chunk_idx = 0

    for section_heading, section_text in sections:
        if not section_text.strip():
            continue

        section_tokens = count_tokens(section_text)

        if section_tokens <= CHUNK_SIZE:
            # Section fits in one chunk
            chunks.append(_make_chunk(section_text, url, title, section_heading, chunk_idx, source))
            chunk_idx += 1
        else:
            # Need to split further
            sub_chunks = _split_section(section_text, page.get("headings", []))
            sub_chunks = _merge_small_chunks(sub_chunks, CHUNK_SIZE)
            sub_chunks = _add_overlap(sub_chunks, CHUNK_OVERLAP)

            for sub in sub_chunks:
                chunks.append(_make_chunk(sub, url, title, section_heading, chunk_idx, source))
                chunk_idx += 1

    return chunks


def _split_into_sections(text: str, h2_headings: list[str]) -> list[tuple[str, str]]:
    """Split text into sections based on H2 heading text."""
    sections = []
    remaining = text

    for i, heading in enumerate(h2_headings):
        idx = remaining.find(heading)
        if idx == -1:
            continue

        # Text before this heading belongs to previous section
        before = remaining[:idx].strip()
        if before and not sections:
            sections.append(("", before))
        elif before and sections:
            # Append to previous section
            prev_heading, prev_text = sections[-1]
            sections[-1] = (prev_heading, prev_text + "\n" + before)

        remaining = remaining[idx:]

        # Find next heading to determine section boundary
        next_idx = len(remaining)
        if i + 1 < len(h2_headings):
            ni = remaining.find(h2_headings[i + 1], len(heading))
            if ni != -1:
                next_idx = ni

        section_text = remaining[:next_idx].strip()
        sections.append((heading, section_text))
        remaining = remaining[next_idx:]

    # Any remaining text
    if remaining.strip():
        if sections:
            prev_heading, prev_text = sections[-1]
            sections[-1] = (prev_heading, prev_text + "\n" + remaining.strip())
        else:
            sections.append(("", remaining.strip()))

    if not sections:
        sections = [("", text)]

    return sections


def _split_section(text: str, headings: list[dict]) -> list[str]:
    """Split a large section into smaller chunks, preserving code blocks."""
    # First try splitting by H3 headings
    h3_headings = [h["text"] for h in headings if h["level"] == 3]
    parts = []

    if h3_headings:
        current = text
        for h3 in h3_headings:
            idx = current.find(h3)
            if idx > 0:
                parts.append(current[:idx].strip())
                current = current[idx:]
        if current.strip():
            parts.append(current.strip())
    else:
        parts = [text]

    # Further split any parts that are still too large
    final_parts = []
    for part in parts:
        if count_tokens(part) <= MAX_CHUNK_SIZE:
            final_parts.append(part)
        else:
            # Recursive split by paragraph, sentence, then space
            sub = _split_by_separators(part, ["\n\n", "\n", ". ", " "])
            sub = _merge_small_chunks(sub, CHUNK_SIZE)
            final_parts.extend(sub)

    return final_parts


def _make_chunk(text: str, url: str, title: str, section: str, idx: int, source: str = "docs") -> dict:
    """Create a chunk dict with metadata header prepended."""
    header = f"Source: {source} | Page: {title}"
    if section:
        header += f" | Section: {section}"
    header += f" | URL: {url}"

    full_text = f"{header}\n\n{text}"

    return {
        "text": full_text,
        "metadata": {
            "url": url,
            "title": title,
            "section": section,
            "chunk_index": idx,
            "source": source,
        },
        "id": f"{url}#chunk-{idx}",
    }
