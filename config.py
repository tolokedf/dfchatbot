"""
Shared configuration for the Gemini Embedding 2 pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOURCE_DIR = PROJECT_ROOT / "source"
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGE_CACHE_DIR = OUTPUT_DIR / "rendered_pages"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "pdf_pages"

SOURCE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Page rendering (PDF -> image)
# ---------------------------------------------------------------------------
RENDER_DPI = 200
MAX_IMAGE_DIMENSION = 2000

# ---------------------------------------------------------------------------
# Models & Prompts
# ---------------------------------------------------------------------------
GEMINI_EMBED_MODEL = "gemini-embedding-2"
GEMINI_QA_MODEL = os.environ.get("GEMINI_QA_MODEL", "gemini-3.5-flash-lite")
EMBED_OUTPUT_DIMENSIONALITY = 3072

def get_gemini_api_key() -> str:
    """
    Dynamically loads and returns GEMINI_API_KEY from .env, overriding any stale shell exports.
    """
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(
            f"GEMINI_API_KEY is not set or empty in {ENV_FILE}.\n"
            "Please open .env and set your API key:\n"
            "  GEMINI_API_KEY=your_gemini_api_key_here"
        )
    return key

EMBED_INSTRUCTION_DOCUMENT = (
    "This is a page from a technical robotics software user manual. It may "
    "contain prose, tables, UI screenshots, or diagrams with embedded text. "
    "Represent its full content for retrieval by a user's question about "
    "the manual."
)
EMBED_INSTRUCTION_QUERY = (
    "Represent this question for retrieving the most relevant manual page."
)

REQUEST_TIMEOUT_SECONDS = 120