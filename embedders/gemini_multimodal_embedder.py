"""
Multimodal embedding via Gemini Embedding 2.

Embeds page IMAGES directly -- no separate captioning/extraction step.
Also embeds text queries into the same vector space, so a plain-text
question can be compared directly against page-image vectors.
"""

from __future__ import annotations
import os 
import time
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

import config

def get_client() -> genai.Client:
    api_key = config.get_gemini_api_key()
    return genai.Client(api_key=api_key)


def embed_page_image(client: genai.Client, image_path: Path) -> dict:
    """Embed one rendered page image directly (no text captioning step)."""
    image_bytes = image_path.read_bytes()

    start = time.monotonic()
    response = client.models.embed_content(
        model=config.GEMINI_EMBED_MODEL,
        contents=[
            config.EMBED_INSTRUCTION_DOCUMENT,
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        ],
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBED_OUTPUT_DIMENSIONALITY,
        ),
    )
    elapsed = time.monotonic() - start
    vector = response.embeddings[0].values

    return {
        "page_image": image_path.name,
        "model": config.GEMINI_EMBED_MODEL,
        "dimensions": len(vector),
        "elapsed_seconds": round(elapsed, 2),
        "vector": vector,
        "error": None,
    }


def embed_query_text(client: genai.Client, query: str) -> dict:
    """Embed a plain-text query into the same vector space as page images."""
    start = time.monotonic()
    response = client.models.embed_content(
        model=config.GEMINI_EMBED_MODEL,
        contents=[config.EMBED_INSTRUCTION_QUERY, query],
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBED_OUTPUT_DIMENSIONALITY,
        ),
    )
    elapsed = time.monotonic() - start
    vector = response.embeddings[0].values

    return {
        "query": query,
        "model": config.GEMINI_EMBED_MODEL,
        "dimensions": len(vector),
        "elapsed_seconds": round(elapsed, 2),
        "vector": vector,
        "error": None,
    }


def embed_all_pages(image_paths: list[Path]) -> list[dict]:
    client = get_client()
    results = []
    for i, image_path in enumerate(image_paths, start=1):
        print(f"  [gemini-embedding-2] page {i}/{len(image_paths)}: "
              f"{image_path.name} ...")
        try:
            result = embed_page_image(client, image_path)
            print(f"    done in {result['elapsed_seconds']}s, "
                  f"dim={result['dimensions']}")
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR: {e}")
            result = {
                "page_image": image_path.name,
                "model": config.GEMINI_EMBED_MODEL,
                "dimensions": None,
                "elapsed_seconds": None,
                "vector": None,
                "error": str(e),
            }
        results.append(result)
    return results
