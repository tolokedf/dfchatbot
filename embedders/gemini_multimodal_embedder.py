"""
Multimodal embedding via Gemini Embedding 2.

Embeds page IMAGES directly -- no separate captioning/extraction step.
Also embeds text queries into the same vector space, so a plain-text
question can be compared directly against page-image vectors.
Includes resilient automatic retry with exponential backoff for API rate limits.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Callable, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

import config

logger = logging.getLogger(__name__)


def get_client() -> genai.Client:
    api_key = config.get_gemini_api_key()
    return genai.Client(api_key=api_key)


def is_fatal_client_error(e: Exception) -> bool:
    """Returns True if the error is a permanent client configuration error (e.g. invalid API key)."""
    err_str = str(e).lower()
    if any(code in err_str for code in ["401", "403", "permission_denied", "api_key_invalid", "invalid api key", "unregistered caller"]):
        return True
    return False


def is_retryable_error(e: Exception) -> bool:
    """Determines whether an error is transient (rate limit 429, 503, connection drops) and should be retried."""
    if is_fatal_client_error(e):
        return False

    err_str = str(e).lower()
    retryable_signatures = [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "rate_limit",
        "too many requests",
        "resource has been exhausted",
        "500",
        "502",
        "503",
        "504",
        "unavailable",
        "deadline_exceeded",
        "timed out",
        "timeout",
        "connection reset",
        "connection refused",
        "remotedisconnected",
        "readtimeout",
        "connecttimeout",
        "temporary failure",
        "server disconnected",
        "try again later",
    ]

    return any(sig in err_str for sig in retryable_signatures)


def embed_page_image(
    client: genai.Client,
    image_path: Path,
    max_retries: int = config.EMBED_MAX_RETRIES,
    log_fn: Optional[Callable[[str], None]] = None
) -> dict:
    """
    Embed one rendered page image directly.
    If a rate limit (HTTP 429 / RESOURCE_EXHAUSTED) or transient network issue occurs,
    automatically retries the exact page with exponential backoff until it succeeds.
    """
    image_bytes = image_path.read_bytes()
    delay = config.EMBED_INITIAL_BACKOFF
    attempt = 0

    while True:
        attempt += 1
        try:
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

            if attempt > 1:
                success_msg = f"  ✅ Page '{image_path.name}' successfully embedded on attempt {attempt} after rate limit."
                logger.info(success_msg)
                if log_fn:
                    log_fn(success_msg)

            return {
                "page_image": image_path.name,
                "model": config.GEMINI_EMBED_MODEL,
                "dimensions": len(vector),
                "elapsed_seconds": round(elapsed, 2),
                "vector": vector,
                "attempts": attempt,
                "error": None,
            }

        except Exception as e:
            if is_fatal_client_error(e):
                error_msg = f"Fatal Gemini API error (Permission/Auth): {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            if not is_retryable_error(e) or attempt >= max_retries:
                error_msg = (
                    f"Failed to embed page '{image_path.name}' after {attempt} attempts: {e}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            # Retryable rate limit or transient error encountered
            jitter = random.uniform(0.5, 2.0)
            sleep_duration = min(config.EMBED_MAX_BACKOFF, delay + jitter)
            
            retry_msg = (
                f"  ⏳ [Rate Limit / API Quota] Page '{image_path.name}' hit limit. "
                f"Pausing {sleep_duration:.1f}s before re-embedding (Attempt {attempt}/{max_retries})..."
            )
            logger.warning(retry_msg)
            if log_fn:
                log_fn(retry_msg)

            time.sleep(sleep_duration)
            delay = min(config.EMBED_MAX_BACKOFF, delay * config.EMBED_BACKOFF_FACTOR)


def embed_query_text(
    client: genai.Client,
    query: str,
    max_retries: int = 5,
    log_fn: Optional[Callable[[str], None]] = None
) -> dict:
    """Embed a plain-text query into the same vector space with automatic retry on transient rate limits."""
    delay = 2.0
    attempt = 0

    while True:
        attempt += 1
        try:
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
                "attempts": attempt,
                "error": None,
            }
        except Exception as e:
            if is_fatal_client_error(e) or not is_retryable_error(e) or attempt >= max_retries:
                raise
            time.sleep(delay)
            delay = min(15.0, delay * 1.5)


def embed_all_pages(
    image_paths: list[Path],
    log_fn: Optional[Callable[[str], None]] = None
) -> list[dict]:
    """Embeds all pages sequentially, ensuring every single page is re-embedded until success."""
    client = get_client()
    results = []
    total = len(image_paths)

    for i, image_path in enumerate(image_paths, start=1):
        msg = f"  [gemini-embedding-2] page {i}/{total}: {image_path.name}..."
        print(msg)
        if log_fn:
            log_fn(msg)

        result = embed_page_image(client, image_path, log_fn=log_fn)
        done_msg = f"    done in {result['elapsed_seconds']}s, dim={result['dimensions']}"
        print(done_msg)
        if log_fn:
            log_fn(done_msg)

        results.append(result)
        time.sleep(config.EMBED_INTER_PAGE_DELAY)

    return results

