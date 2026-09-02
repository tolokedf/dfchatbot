"""
Shared configuration for the Gemini Embedding 2 pipeline.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

# Ensure src/ and PROJECT_ROOT are on sys.path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

# ---------------------------------------------------------------------------
# Paths (Supports unified data/ storage or root folders)
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

# Source documents directory (PDF manuals)
if (DATA_DIR / "source_docs").exists():
    SOURCE_DIR = DATA_DIR / "source_docs"
else:
    SOURCE_DIR = PROJECT_ROOT / "source"

# Pipeline output directory (ChromaDB, rendered images, metadata)
if (DATA_DIR / "output").exists():
    OUTPUT_DIR = DATA_DIR / "output"
else:
    OUTPUT_DIR = PROJECT_ROOT / "output"

IMAGE_CACHE_DIR = OUTPUT_DIR / "rendered_pages"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "pdf_pages"

# User database and profile pictures folder
if (DATA_DIR / "user_storage").exists():
    USER_DB_DIR = DATA_DIR / "user_storage"
else:
    USER_DB_DIR = PROJECT_ROOT / "User database"

USER_DB_PATH = USER_DB_DIR / "users_and_chats.db"
USER_AVATAR_DIR = USER_DB_DIR / "profile_pictures"
USER_UPLOADS_DIR = USER_DB_DIR / "uploaded_attachments"

SOURCE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
USER_DB_DIR.mkdir(parents=True, exist_ok=True)
USER_AVATAR_DIR.mkdir(parents=True, exist_ok=True)
USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

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

# ---------------------------------------------------------------------------
# Security & Authentication
# ---------------------------------------------------------------------------
ADMIN_ID = os.environ.get("ADMIN_ID", "df")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "df")
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "df-rag-multimodal-secret-key-2026")

def verify_admin_password(password: str) -> bool:
    """Verifies if the provided password matches the configured ADMIN_PASSWORD."""
    if not password:
        return False
    current_pass = os.environ.get("ADMIN_PASSWORD", "df").strip()
    return password.strip() == current_pass

def verify_admin_credentials(admin_id: str, admin_password: str) -> bool:
    """Verifies if the provided ID and password match the admin credentials (df / df)."""
    if not admin_id or not admin_password:
        return False
    current_id = os.environ.get("ADMIN_ID", "df").strip()
    current_pass = os.environ.get("ADMIN_PASSWORD", "df").strip()
    return admin_id.strip() == current_id and admin_password.strip() == current_pass

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

# ---------------------------------------------------------------------------
# Rate Limiting & Resilient Retry Settings
# ---------------------------------------------------------------------------
EMBED_MAX_RETRIES = int(os.environ.get("EMBED_MAX_RETRIES", 20))
EMBED_INITIAL_BACKOFF = float(os.environ.get("EMBED_INITIAL_BACKOFF", 5.0))
EMBED_MAX_BACKOFF = float(os.environ.get("EMBED_MAX_BACKOFF", 60.0))
EMBED_BACKOFF_FACTOR = float(os.environ.get("EMBED_BACKOFF_FACTOR", 1.5))
EMBED_INTER_PAGE_DELAY = float(os.environ.get("EMBED_INTER_PAGE_DELAY", 0.3))