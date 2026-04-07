from __future__ import annotations

from config import MAX_RETRIEVAL_RESULTS, TOP_K_CHUNKS
from indexer.vectorstore import VectorStore


class Retriever:
    def __init__(self, store: VectorStore | None = None):
        self._store = store or VectorStore()

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve and deduplicate relevant chunks for a query.
        Returns top-K chunks after merging adjacent chunks from the same page.
        """
        raw_results = self._store.query(query, n_results=MAX_RETRIEVAL_RESULTS)

        # Deduplicate: merge adjacent chunks from same URL
        merged = self._merge_adjacent(raw_results)

        # Return top K
        return merged[:TOP_K_CHUNKS]

    def _merge_adjacent(self, chunks: list[dict]) -> list[dict]:
        """Merge chunks from the same page that are adjacent (consecutive chunk_index)."""
        if not chunks:
            return []

        # Group by URL
        by_url: dict[str, list[dict]] = {}
        for chunk in chunks:
            url = chunk["metadata"]["url"]
            by_url.setdefault(url, []).append(chunk)

        merged = []
        for url, url_chunks in by_url.items():
            url_chunks.sort(key=lambda c: c["metadata"]["chunk_index"])

            groups = []
            current_group = [url_chunks[0]]

            for i in range(1, len(url_chunks)):
                prev_idx = current_group[-1]["metadata"]["chunk_index"]
                curr_idx = url_chunks[i]["metadata"]["chunk_index"]
                if curr_idx - prev_idx == 1:
                    current_group.append(url_chunks[i])
                else:
                    groups.append(current_group)
                    current_group = [url_chunks[i]]
            groups.append(current_group)

            for group in groups:
                if len(group) == 1:
                    merged.append(group[0])
                else:
                    # Merge into one chunk
                    combined_text = "\n\n".join(c["text"] for c in group)
                    best_distance = min(c["distance"] for c in group)
                    merged.append({
                        "id": group[0]["id"],
                        "text": combined_text,
                        "metadata": group[0]["metadata"],
                        "distance": best_distance,
                    })

        # Sort by distance (lower = more similar)
        merged.sort(key=lambda c: c["distance"])
        return merged
