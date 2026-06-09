"""
Retrieval module: given a query, find the most relevant lecture chunks
from ChromaDB using semantic similarity.

This module is designed to be both imported by app.py AND run standalone
for testing. Run it as:
    python src/retrieve.py

Then type queries in any language. It will print the top-K matching chunks.
"""

import os

# Silence Windows symlinks warning (must be set before importing HF libs)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# === Configuration ===
CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "lecture_transcript"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 4

# Module-level caches: load these once, reuse across queries
_collection = None
_embedder = None


def _get_collection():
    """Lazily load the ChromaDB collection."""
    global _collection
    if _collection is None:
        if not Path(CHROMA_PERSIST_DIR).exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_PERSIST_DIR}. "
                f"Run 'python src/ingest.py' first."
            )
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_embedder():
    """Lazily load the embedding model (heavy initialisation, do it once)."""
    global _embedder
    if _embedder is None:
        print("Loading embedding model (BGE-M3)...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def retrieve_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Find the top_k most relevant chunks for the given query.

    Args:
        query: The user's question, in any language supported by BGE-M3
        top_k: Number of chunks to return (default 4)

    Returns:
        List of dicts: [{"text": "...", "distance": 0.23, "chunk_id": "chunk_0003"}, ...]
        Sorted by similarity (lowest distance = most similar = first)
    """
    if not query or not query.strip():
        return []

    collection = _get_collection()
    embedder = _get_embedder()

    # Embed the query using the same model used during ingestion
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,  # Must match ingestion settings
    ).tolist()

    # Query ChromaDB for similar chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    # Reshape into a clean list of dicts
    chunks = []
    for chunk_id, doc, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        chunks.append({
            "chunk_id": chunk_id,
            "text": doc,
            "distance": distance,
        })

    return chunks


# === Standalone testing mode ===
def main():
    print("=" * 60)
    print("RETRIEVAL TEST MODE")
    print("Type a question. Type 'quit' or 'exit' to exit.")
    print("=" * 60)

    # Pre-load to avoid first-query lag
    _get_collection()
    _get_embedder()

    print("\nReady. Total chunks indexed:", _get_collection().count())

    while True:
        print()
        query = input("Query> ").strip()
        if query.lower() in ("quit", "exit", "q", ""):
            break

        chunks = retrieve_chunks(query, top_k=DEFAULT_TOP_K)

        print(f"\nTop {len(chunks)} chunks:")
        print("-" * 60)
        for i, chunk in enumerate(chunks, 1):
            # Distance 0 = identical; lower distance = more similar
            similarity = 1 - chunk["distance"]
            print(f"\n[{i}] {chunk['chunk_id']} "
                  f"(distance={chunk['distance']:.4f}, similarity={similarity:.4f})")
            print(f"    {chunk['text'][:250]}...")


if __name__ == "__main__":
    main()