from __future__ import annotations

import chromadb
from chromadb.config import Settings

from config import CHROMADB_DIR, COLLECTION_NAME
from indexer.embeddings import EmbeddingService


class VectorStore:
    def __init__(self, embedding_service: EmbeddingService | None = None):
        CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(CHROMADB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedding_service or EmbeddingService()

    def add_chunks(self, chunks: list[dict], batch_size: int = 100) -> int:
        """Add chunks to the vector store. Uses upsert for idempotency."""
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c["id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]
            embeddings = self._embedder.embed_documents(texts)

            self._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            total += len(batch)
            print(f"  Indexed {total}/{len(chunks)} chunks")
        return total

    def query(self, text: str, n_results: int = 8, where_filter: dict | None = None) -> list[dict]:
        """Query the vector store for similar chunks."""
        embedding = self._embedder.embed_query(text)
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": n_results,
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self._collection.query(**kwargs)

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return chunks

    def delete_by_source(self, url: str):
        """Delete all chunks from a given source URL."""
        self._collection.delete(where={"url": url})

    def count(self) -> int:
        return self._collection.count()

    def clear(self):
        """Delete and recreate the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
