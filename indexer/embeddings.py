from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts."""
        embeddings = self._model.encode(texts, show_progress_bar=True, batch_size=64)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        embedding = self._model.encode(text)
        return embedding.tolist()
