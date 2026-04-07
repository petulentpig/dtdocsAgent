import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_PAGES_DIR
from indexer.chunker import chunk_page
from indexer.embeddings import EmbeddingService
from indexer.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Index scraped Dynatrace docs into ChromaDB")
    parser.add_argument("--clear", action="store_true", help="Clear existing index before indexing")
    parser.add_argument("--max", type=int, help="Maximum number of pages to index")
    args = parser.parse_args()

    if not RAW_PAGES_DIR.exists():
        print(f"No scraped pages found at {RAW_PAGES_DIR}. Run the scraper first.")
        return

    page_files = sorted(RAW_PAGES_DIR.glob("*.json"))
    if args.max:
        page_files = page_files[: args.max]

    print(f"Found {len(page_files)} scraped pages")

    # Initialize
    embedder = EmbeddingService()
    store = VectorStore(embedding_service=embedder)

    if args.clear:
        print("Clearing existing index...")
        store.clear()

    # Chunk all pages
    all_chunks = []
    for page_file in page_files:
        page = json.loads(page_file.read_text())
        chunks = chunk_page(page)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from {len(page_files)} pages")
    print(f"Average chunks per page: {len(all_chunks) / len(page_files):.1f}")

    # Index
    print("Indexing chunks...")
    store.add_chunks(all_chunks)
    print(f"Index now contains {store.count()} chunks")


if __name__ == "__main__":
    main()
