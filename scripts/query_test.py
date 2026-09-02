"""
Embed a text query with Gemini Embedding 2 and rank stored page images
using ChromaDB vector similarity.

Usage:
    python query_test.py "how do I calibrate the sensor array"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
import config
from embedders import gemini_multimodal_embedder as embedder


def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python query_test.py "your question here"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    collection = get_chroma_collection()
    total_docs = collection.count()
    if total_docs == 0:
        print("ChromaDB collection is empty. Run 'python run_embedding_pipeline.py' first.")
        sys.exit(1)

    print(f"Connected to ChromaDB '{config.CHROMA_COLLECTION_NAME}' ({total_docs} pages indexed)")

    client = embedder.get_client()
    print(f"Embedding query: {query!r} ...")
    result = embedder.embed_query_text(client, query)
    qvec = result["vector"]

    # Query ChromaDB (filtering front-matter)
    res = collection.query(
        query_embeddings=[qvec],
        n_results=10,
        where={"is_front_matter": False},
        include=["metadatas", "distances", "documents"]
    )

    ids = res["ids"][0] if res["ids"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    distances = res["distances"][0] if res["distances"] else []

    print(f"\nTop matches for: {query!r}\n")
    print(f"{'Rank':<5}{'Similarity':<12}{'Page Image':<45}{'Chapter':<25}{'Section'}")
    print("-" * 110)

    for i, (meta, dist) in enumerate(zip(metas, distances), start=1):
        sim = 1.0 - dist
        page_img = meta.get("page_image", "")
        chapter = str(meta.get("chapter", "Unknown"))[:24]
        section = str(meta.get("section", "Unknown"))[:25]
        print(f"{i:<5}{sim:<12.4f}{page_img:<45}{chapter:<25}{section}")

    if metas:
        top_page = metas[0].get("page_image", "")
        top_path = config.IMAGE_CACHE_DIR / top_page
        print(f"\nTop match image: {top_path}")
        print("Open it and check whether it actually answers the query.")


if __name__ == "__main__":
    main()
