"""
Chunk the transcript, embed each chunk, and store in ChromaDB.

This is the offline indexing step. Run once after transcription;
subsequent retrieval queries use the persisted vector database.

Usage:
    python src/ingest.py
"""
import os
os.environ["TRANSFORMERS_PREFER_SAFETENSORS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# === Configuration ===
TRANSCRIPT_PATH = "data/transcript.json"
CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "lecture_transcript"

# Multilingual embedding model with strong Indic support
# ~2GB download on first run, cached after
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Chunking parameters
TARGET_CHUNK_CHARS = 900        # Roughly 250-300 tokens for Hindi
OVERLAP_CHARS = 180             # ~50 tokens of overlap between chunks
MIN_CHUNK_CHARS = 200           # Skip chunks shorter than this


def load_transcript(path: str) -> str:
    """Load just the transcript text from the Saaras JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcript = data.get("transcript", "")
    if not transcript:
        raise ValueError(f"No 'transcript' field found in {path}")

    print(f"Loaded transcript: {len(transcript)} characters")
    return transcript


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences, handling both Devanagari danda (।)
    and Latin punctuation (. ! ?). Whitespace around punctuation is normalized.
    """
    # Split on sentence-ending punctuation while keeping the punctuation
    sentences = re.split(r"(?<=[।.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    target_chars: int = TARGET_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """
    Build chunks by accumulating sentences until target_chars is reached.
    Adds overlap from the tail of each chunk to the head of the next.
    Skips chunks that are too short to be useful.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        current.append(sentence)
        current_len += len(sentence) + 1  # +1 for the joining space

        if current_len >= target_chars:
            chunks.append(" ".join(current))

            # Build the overlap for the next chunk by walking backwards
            overlap_buffer = []
            overlap_len = 0
            for s in reversed(current):
                overlap_buffer.insert(0, s)
                overlap_len += len(s) + 1
                if overlap_len >= overlap_chars:
                    break

            current = overlap_buffer
            current_len = overlap_len

    # Add the final accumulator if it has content
    if current:
        final_chunk = " ".join(current)
        # Only add if it's not just a duplicate of the previous chunk's tail
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    # Filter out tiny chunks
    chunks = [c for c in chunks if len(c) >= min_chars]

    return chunks


def main():
    print("=" * 60)
    print("CHUNKING + EMBEDDING + INDEXING")
    print("=" * 60)

    # --- Step 1: Load transcript ---
    print("\n[1/4] Loading transcript...")
    transcript = load_transcript(TRANSCRIPT_PATH)

    # --- Step 2: Chunk ---
    print("\n[2/4] Chunking...")
    chunks = chunk_text(transcript)
    print(f"      Created {len(chunks)} chunks")
    print(f"      Avg chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")
    print(f"      Min chunk size: {min(len(c) for c in chunks)} chars")
    print(f"      Max chunk size: {max(len(c) for c in chunks)} chars")
    print(f"\n      First chunk preview:\n      {chunks[0][:200]}...")
    print(f"\n      Last chunk preview:\n      {chunks[-1][:200]}...")

    # --- Step 3: Embed ---
    print(f"\n[3/4] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print("      (First run downloads ~2GB. Cached after.)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("      Computing embeddings...")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True,  # Required for cosine similarity
    )
    print(f"      Embedding shape: {embeddings.shape}")

    # --- Step 4: Store in ChromaDB ---
    print(f"\n[4/4] Storing in ChromaDB at {CHROMA_PERSIST_DIR}")
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Wipe any existing collection for clean re-runs during development
    try:
        client.delete_collection(COLLECTION_NAME)
        print("      (Deleted existing collection)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Cosine similarity for normalized vectors
    )

    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=[f"chunk_{i:04d}" for i in range(len(chunks))],
    )

    print(f"\n      Indexed {len(chunks)} chunks successfully.")

    # --- Verification ---
    print("\n[Verify] Re-querying with the first chunk as a sanity check...")
    test_results = collection.query(
        query_embeddings=embeddings[0:1].tolist(),
        n_results=3,
    )
    print(f"      Top-3 similar chunks (should include chunk_0000 itself):")
    for chunk_id, distance in zip(test_results["ids"][0], test_results["distances"][0]):
        print(f"        {chunk_id}: distance={distance:.4f}")

    print("\n" + "=" * 60)
    print("Done. Vector DB ready at:", CHROMA_PERSIST_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()