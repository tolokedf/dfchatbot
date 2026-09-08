"""
Flask Web Application for Multimodal Technical PDF RAG.
Includes:
- Main Chatbot Interface (/): Outline-aware retrieval with chapter-bounded expansion.
- Admin Management Console (/admin): PDF file manager, manual embedding triggers, API key configuration.
- REST APIs for chat, file uploads/deletions, embedding pipeline, and diagnostics.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_from_directory, session, Response, stream_with_context, make_response, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
import chromadb

import config
import pipeline_service
import auth_and_chat_db
import report_exporter
try:
    from embedders import gemini_multimodal_embedder as embedder
except ImportError:
    try:
        from src.embedders import gemini_multimodal_embedder as embedder
    except ImportError:
        src_dir = Path(__file__).resolve().parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from embedders import gemini_multimodal_embedder as embedder

try:
    import pymupdf as fitz
except ImportError:
    import fitz

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(config.TEMPLATES_DIR), static_folder=str(config.STATIC_DIR))
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload limit


def user_required(f):
    """Decorator to enforce user authentication for chat tabs and personal histories."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({
                "status": "error",
                "error": "Authentication required. Please log in to continue.",
                "auth_required": True
            }), 401
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to enforce admin password authentication on admin endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_authenticated"):
            # Check for Bearer token authorization header
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1].strip()
                if config.verify_admin_password(token):
                    return f(*args, **kwargs)
            return jsonify({
                "status": "error",
                "error": "Admin authentication required. Please unlock with password.",
                "auth_required": True
            }), 401
        return f(*args, **kwargs)
    return decorated_function


def parse_page_filename(filename: str) -> tuple[str, int]:
    """Extracts (pdf_stem, page_number) from page image filename."""
    match = re.search(r"^(.*?)(?:_?page_|\bpage_)(\d+)\.png$", filename, re.IGNORECASE)
    if match:
        pfx = match.group(1).rstrip("_")
        p_num = int(match.group(2))
        return (pfx, p_num)
    return (filename, 0)


def sort_key(filename: str):
    return parse_page_filename(filename)


# ============================================================================
# Front-End Web Page Routes
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin")
def admin_page():
    return render_template("admin.html")


def find_best_matching_pdf(target_name: str) -> Path | None:
    """
    Fuzzy and semantic matcher that accurately maps any manual citation, query string,
    or variation (e.g., 'NavWiz 3.0 User Manual', 'NavWiz 4.0', 'NavWiz', 'DFleet', 
    'Field Deployment Handbook', 'Lanxin') to the correct underlying PDF Path in config.SOURCE_DIR.
    """
    if not config.SOURCE_DIR.exists():
        return None

    all_pdfs = sorted(list(config.SOURCE_DIR.glob("*.pdf")) + list(config.SOURCE_DIR.glob("*.xlsx")))
    if not all_pdfs:
        return None

    if not target_name or not target_name.strip():
        return all_pdfs[0]

    clean_target = target_name.strip()
    # Strip brackets, quotes, and .pdf/.xlsx extension
    clean_target = re.sub(r"^[\[\"']|[\]\"']$", "", clean_target).strip()
    clean_target_no_ext = re.sub(r"\.(pdf|xlsx)$", "", clean_target, flags=re.IGNORECASE)

    # 1. Exact match (case-insensitive) on filename or stem
    for f in all_pdfs:
        if (
            f.name.lower() == clean_target.lower()
            or f.stem.lower() == clean_target.lower()
            or f.stem.lower() == clean_target_no_ext.lower()
        ):
            return f

    # Helper for alphanumeric normalization
    def normalize_str(s: str) -> str:
        s_low = s.lower().replace("copy of", "").replace(".pdf", "").replace(".xlsx", "")
        return re.sub(r"[^a-z0-9]", "", s_low)

    target_norm = normalize_str(clean_target)
    if target_norm:
        for f in all_pdfs:
            if normalize_str(f.stem) == target_norm:
                return f

    # 2. Key brand/product keyword matching
    target_lower = clean_target.lower()
    if "muar" in target_lower or "arv" in target_lower.split():
        for f in all_pdfs:
            if "muar" in f.name.lower():
                return f

    if "navwiz" in target_lower or "nav wiz" in target_lower or "nav" in target_lower.split():
        for f in all_pdfs:
            if "navwiz" in f.name.lower():
                return f

    if "dfleet" in target_lower or "d fleet" in target_lower or "df" in target_lower.split():
        for f in all_pdfs:
            if "dfleet" in f.name.lower():
                return f

    if "lanxin" in target_lower or "lx" in target_lower.split():
        for f in all_pdfs:
            if "lanxin" in f.name.lower():
                return f

    if "deployment" in target_lower or "handbook" in target_lower or "field" in target_lower:
        for f in all_pdfs:
            if "deployment" in f.name.lower() or "handbook" in f.name.lower() or "field" in f.name.lower():
                return f

    # 3. Token overlap scoring
    target_tokens = set(re.findall(r"\w+", target_lower))
    stop_words = {"copy", "of", "the", "manual", "user", "pdf", "v", "ver", "version"}
    meaningful_target = target_tokens - stop_words
    if not meaningful_target:
        meaningful_target = target_tokens

    best_pdf = None
    best_score = -1.0

    for f in all_pdfs:
        f_tokens = set(re.findall(r"\w+", f.stem.lower()))
        meaningful_f = f_tokens - stop_words
        if not meaningful_f:
            meaningful_f = f_tokens

        intersection = meaningful_target & meaningful_f
        union = meaningful_target | meaningful_f
        score = (len(intersection) / len(union)) if union else 0.0

        if normalize_str(f.stem) in target_norm or target_norm in normalize_str(f.stem):
            score += 0.5

        if score > best_score:
            best_score = score
            best_pdf = f

    if best_pdf and best_score > 0.1:
        return best_pdf

    return all_pdfs[0]


@app.route("/pdf-viewer")
def pdf_viewer_page():
    """Renders a dedicated page citation viewer jumping directly to the requested manual page."""
    target_file = request.args.get("file", "").strip()
    try:
        page_num = max(1, int(request.args.get("page", 1)))
    except (ValueError, TypeError):
        page_num = 1

    # Match against source PDF files using fuzzy & keyword matching
    found_pdf = find_best_matching_pdf(target_file)
    if not found_pdf:
        return "No PDF manuals available in source directory.", 404

    filename = found_pdf.name
    stem = found_pdf.stem
    doc = fitz.open(found_pdf)
    total_pages = len(doc)
    doc.close()

    page_num = max(1, min(page_num, total_pages))
    padded_num = f"{page_num:03d}"
    image_name = f"{stem}_page_{padded_num}.png"
    image_url = f"/rendered_pages/{image_name}"

    return render_template(
        "pdf_viewer.html",
        filename=filename,
        stem=stem,
        page_number=page_num,
        total_pages=total_pages,
        image_url=image_url
    )


@app.route("/api/pdf/raw/<path:filename>")
def serve_raw_pdf(filename: str):
    """Serves the raw PDF file with inline disposition so browsers natively open the PDF at #page=X."""
    found_pdf = find_best_matching_pdf(filename)
    if not found_pdf or not found_pdf.exists():
        return jsonify({"status": "error", "error": f"PDF '{filename}' not found."}), 404

    return send_from_directory(
        config.SOURCE_DIR,
        found_pdf.name,
        mimetype="application/pdf",
        as_attachment=False
    )


@app.route("/rendered_pages/<path:filename>")
def serve_rendered_page(filename: str):
    return send_from_directory(config.IMAGE_CACHE_DIR, filename)


# ============================================================================
# User Authentication APIs
# ============================================================================

@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    """Registers a new user account with no-space validation and password confirmation."""
    data = request.get_json(force=True, silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    confirm_password = data.get("confirm_password", "")

    try:
        user_data = auth_and_chat_db.register_user(username, password, confirm_password)
        return jsonify({
            "status": "ok",
            "pending_approval": True,
            "message": "Account created! An administrator must approve your account before you can log in.",
            "user": {
                "id": user_data["id"],
                "username": user_data["username"],
                "role": user_data["role"],
                "status": user_data["status"]
            }
        })
    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in register: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    """Authenticates user with username (Name) and password (no spaces)."""
    data = request.get_json(force=True, silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    try:
        user_data = auth_and_chat_db.authenticate_user(username, password)
        if not user_data:
            return jsonify({"status": "error", "error": "Invalid username or password. Please try again."}), 401

        session["user_id"] = user_data["id"]
        session["username"] = user_data["username"]
        session["role"] = user_data["role"]
        if user_data["role"] == "admin":
            session["admin_authenticated"] = True

        tabs = auth_and_chat_db.list_user_tabs(user_data["id"])
        return jsonify({
            "status": "ok",
            "message": "Logged in successfully!",
            "user": {
                "id": user_data["id"],
                "username": user_data["username"],
                "role": user_data["role"]
            },
            "tabs": tabs
        })
    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in login: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/auth/guest", methods=["POST"])
def auth_guest():
    """Logs in the session as a guest user."""
    session["user_id"] = "guest"
    session["username"] = "Guest"
    session["role"] = "guest"
    return jsonify({
        "status": "ok",
        "message": "Logged in as guest.",
        "user": {
            "id": "guest",
            "username": "Guest",
            "role": "guest",
            "profile_pic": ""
        },
        "default_tab_id": "guest-tab"
    })


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("user_id", None)
    session.pop("username", None)
    session.pop("role", None)
    session.pop("admin_authenticated", None)
    return jsonify({"status": "ok", "message": "Logged out successfully."})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"authenticated": False, "user": None})

    if user_id == "guest":
        return jsonify({
            "authenticated": True,
            "is_guest": True,
            "user": {
                "id": "guest",
                "username": session.get("username", "Guest"),
                "role": "guest",
                "profile_pic": ""
            }
        })

    user = auth_and_chat_db.get_user_by_id(user_id)
    if not user:
        session.clear()
        return jsonify({"authenticated": False, "user": None})

    return jsonify({
        "authenticated": True,
        "is_guest": False,
        "user": user
    })


@app.route("/api/user/profile-picture", methods=["POST"])
@user_required
def upload_profile_picture():
    """Uploads and saves user profile picture inside the 'data/user_storage/profile_pictures' directory."""
    user_id = session.get("user_id")
    if user_id == "guest":
        return jsonify({"status": "error", "error": "Guest accounts cannot update profile pictures. Please register or log in."}), 403

    if "file" not in request.files:
        return jsonify({"status": "error", "error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"status": "error", "error": "No file selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
        return jsonify({"status": "error", "error": "Unsupported image format. Allowed: PNG, JPG, JPEG, WEBP, GIF."}), 400

    try:
        # Validate image content
        img = Image.open(file.stream)
        img.verify()
        file.stream.seek(0)
    except Exception as e:
        return jsonify({"status": "error", "error": f"Invalid image file: {e}"}), 400

    # Save to data/user_storage/profile_pictures/
    saved_filename = f"avatar_u{user_id}_{int(time.time())}{ext}"
    save_path = config.USER_AVATAR_DIR / saved_filename
    file.save(save_path)

    # Update database record
    updated_user = auth_and_chat_db.update_user_profile_picture(user_id, saved_filename)
    logger.info(f"User {user_id} updated profile picture: {saved_filename}")

    return jsonify({
        "status": "ok",
        "message": "Profile picture updated successfully!",
        "profile_pic": saved_filename,
        "profile_pic_url": f"/api/user/profile-picture/{saved_filename}",
        "user": updated_user
    })


@app.route("/api/user/profile-picture/<path:filename>", methods=["GET"])
def serve_profile_picture(filename: str):
    """Serves user profile picture from 'data/user_storage/profile_pictures'."""
    return send_from_directory(config.USER_AVATAR_DIR, filename)


@app.route("/api/user/uploads/<path:filename>", methods=["GET"])
def serve_user_upload(filename: str):
    """Serves user-uploaded chat attachments (photos/PDFs) from 'data/user_storage/uploaded_attachments'."""
    return send_from_directory(config.USER_UPLOADS_DIR, filename)


# ============================================================================
# Conversational / Direct Intent Classifier (Zero-Vector Latency Optimization)
# ============================================================================

def is_conversational_or_meta_query(text: str) -> bool:
    """
    Detects if the query is a greeting, small talk, identity, or capability question.
    Allows responding instantly without triggering expensive vector search and image loading.
    """
    clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    if not clean:
        return True

    # Technical keywords related to robotics manuals & procedures
    technical_keywords = {
        "navwiz", "dfleet", "agv", "amr", "sensor", "sensors", "dock", "docking", "wheel", "wheels",
        "motor", "motors", "lidar", "obstacle", "safety", "zone", "zones", "calibration", "calibrate",
        "battery", "charge", "charging", "map", "mapping", "slam", "laser", "pin", "pins", "wiring",
        "port", "ip", "ethernet", "wifi", "plc", "relay", "fuse", "alarm", "alarms", "error", "errors",
        "warning", "fault", "manual", "manuals", "page", "chapter", "parameter", "parameters", "baud",
        "firmware", "install", "deploy", "deployment", "handbook", "pallet", "chassis", "emergency",
        "stop", "reset", "reboot", "canbus", "modbus", "ros", "hardware", "troubleshoot",
        "troubleshooting", "spec", "specs", "specification", "voltage", "amp", "connector"
    }

    # If any specific technical keyword or error code is present, treat as technical query
    words = set(clean.split())
    if words & technical_keywords:
        return False
    if re.search(r'\b[eew]\d{2,}\b', clean):  # e.g. E-9921 or E01
        return False

    # Common conversational phrases
    conversational_phrases = [
        "hi", "hello", "hey", "hola", "good morning", "good afternoon", "good evening",
        "how are you", "how r u", "whats up", "what is up", "sup",
        "who are you", "what is your name", "what can you do", "what do you know",
        "how can you help", "what manuals", "help", "i have a problem", "i have problem",
        "what can i do", "what should i do", "what to do", "can you help",
        "how do you work", "what questions", "what is this", "tell me about yourself",
        "thanks", "thank you", "bye", "goodbye", "see you", "nice to meet you", "ok", "okay"
    ]
    for phrase in conversational_phrases:
        if phrase in clean:
            return True

    # If short non-technical text (<= 5 words) and no technical keywords
    if len(words) <= 5:
        return True

    return False


# ---------------------------------------------------------------------------
# Rate Limiting & Abuse Prevention Helpers (Strategies 2 & 3)
# ---------------------------------------------------------------------------
RATE_LIMIT_STORE = defaultdict(list)

def check_rate_limit(client_id: str) -> bool:
    """
    Sliding-window rate limiter per client ID or IP address (Strategy 3).
    Returns True if allowed, False if limit exceeded.
    """
    now = time.time()
    window_seconds = 60.0
    timestamps = [t for t in RATE_LIMIT_STORE[client_id] if now - t < window_seconds]
    RATE_LIMIT_STORE[client_id] = timestamps
    if len(timestamps) >= config.MAX_REQUESTS_PER_MINUTE:
        return False
    RATE_LIMIT_STORE[client_id].append(now)
    return True


def get_static_conversational_reply(text: str) -> str | None:
    """
    Checks if query matches common greetings or small talk and returns an instant
    canned response without calling Gemini (Strategy 2 - 0 tokens cost).
    """
    clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    if not clean:
        return "Hello! I am DF Chatbot. How can I assist you with NavWiz, DFleet, or AGV procedures today?"

    greetings = {"hi", "hello", "hey", "hola", "good morning", "good afternoon", "good evening", "greetings"}
    if clean in greetings or clean.startswith("hello ") or clean.startswith("hi ") or clean.startswith("hey "):
        return "Hello! I am DF Chatbot, the Multimodal Technical Assistant by DF Automation. How can I assist you with your robotics manuals today?"

    identity_queries = {"who are you", "what is your name", "what are you", "who r u", "tell me about yourself"}
    if clean in identity_queries:
        return "I am DF Chatbot, an expert technical assistant developed by DF Automation. I can help you navigate technical manuals, calibrate sensors, troubleshoot error codes, and review AGV schematics."

    capabilities_queries = {"what can you do", "what do you know", "help", "how can you help", "what manuals"}
    if clean in capabilities_queries:
        return (
            "I can assist you with:\n"
            "• Looking up procedures in NavWiz & DFleet manuals\n"
            "• AGV navigation, docking, and safety zone configuration\n"
            "• Sensor calibration (LiDAR, optical, sonar)\n"
            "• Battery charging and electrical troubleshooting\n"
            "• Error code diagnoses (e.g. E01, E04)\n\n"
            "You can also attach screenshots, photos, or PDF schematics for visual analysis!"
        )

    gratitude_queries = {"thank you", "thanks", "thank u", "thx", "many thanks", "appreciate it"}
    if clean in gratitude_queries:
        return "You're very welcome! Feel free to ask if you have any more questions about DF robotics or technical procedures."

    farewell_queries = {"bye", "goodbye", "see you", "cya", "have a good day"}
    if clean in farewell_queries:
        return "Goodbye! Have a safe and productive day."

    acknowledgement_queries = {"ok", "okay", "got it", "understood", "alright", "sure"}
    if clean in acknowledgement_queries:
        return "Understood. Let me know what you would like to explore or troubleshoot next."

    return None


def is_obvious_gibberish(text: str) -> bool:
    """Detects obvious character mashing or nonsensical repeated sequences."""
    clean = text.strip().lower()
    if len(clean) >= 5 and re.search(r'(.)\1{4,}', clean):
        return True
    # Common keyboard walk sequences
    qwerty_patterns = ['asdfgh', 'sdfghj', 'dfghjk', 'qwerty', 'wertyu', 'zxcvbn']
    if any(p in clean for p in qwerty_patterns):
        return True
    words = clean.split()
    for w in words:
        if len(w) >= 7 and not any(c in "aeiou" for c in w) and w != "rhythms":
            return True
    return False


# ============================================================================
# Chat Tabs API (Per-User Isolated Sessions & Memory)
# ============================================================================

@app.route("/api/chat/tabs", methods=["GET"])
def get_user_tabs():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"status": "ok", "tabs": []})
    if user_id == "guest":
        return jsonify({
            "status": "ok",
            "tabs": [{"id": "guest-tab", "title": "Guest Session", "message_count": 0}]
        })
    tabs = auth_and_chat_db.list_user_tabs(user_id)
    return jsonify({"status": "ok", "tabs": tabs})


@app.route("/api/chat/tabs", methods=["POST"])
@user_required
def create_user_tab():
    user_id = session.get("user_id")
    if user_id == "guest":
        return jsonify({
            "status": "error",
            "error": "Guest session does not support multiple persistent tabs. Please register or log in."
        }), 403
    data = request.get_json(force=True, silent=True) or {}
    title = data.get("title", "New Chat")
    new_tab = auth_and_chat_db.create_tab(user_id, title)
    return jsonify({"status": "ok", "tab": new_tab})


@app.route("/api/chat/tabs/<tab_id>", methods=["DELETE"])
@user_required
def delete_user_tab(tab_id: str):
    user_id = session.get("user_id")
    if user_id == "guest":
        return jsonify({
            "status": "ok",
            "message": "Guest tab reset.",
            "tabs": [{"id": "guest-tab", "title": "Guest Session", "message_count": 0}]
        })
    deleted = auth_and_chat_db.delete_tab(tab_id, user_id)
    if not deleted:
        return jsonify({"status": "error", "error": "Tab not found or unauthorized."}), 404
    remaining = auth_and_chat_db.list_user_tabs(user_id)
    return jsonify({"status": "ok", "message": "Tab deleted successfully.", "tabs": remaining})


@app.route("/api/chat/tabs/<tab_id>/messages", methods=["GET"])
@user_required
def get_tab_message_history(tab_id: str):
    user_id = session.get("user_id")
    if user_id == "guest" or tab_id == "guest-tab":
        return jsonify({"status": "ok", "messages": []})
    messages = auth_and_chat_db.get_tab_messages(tab_id, user_id)
    return jsonify({"status": "ok", "messages": messages})


# ============================================================================
# Core Chat & Conversational Retrieval API (with Multimodal File Uploads & Fast Routing)
# ============================================================================

@app.route("/api/status", methods=["GET"])
def get_status():
    try:
        collection = pipeline_service.get_chroma_collection()
        total_docs = collection.count()
        pdfs_info = pipeline_service.get_all_pdfs_status()
        
        # Collect all manuals available in source/ directory or indexed in ChromaDB
        all_stems = set()
        for p in pdfs_info:
            all_stems.add(p["stem"])

        if total_docs > 0:
            try:
                metas = collection.get(include=["metadatas"])["metadatas"]
                for m in metas:
                    if m and m.get("pdf_stem"):
                        all_stems.add(m.get("pdf_stem"))
            except Exception:
                pass

        return jsonify({
            "status": "ok",
            "collection": config.CHROMA_COLLECTION_NAME,
            "total_indexed_pages": total_docs,
            "embed_model": config.GEMINI_EMBED_MODEL,
            "qa_model": config.GEMINI_QA_MODEL,
            "sources": sorted(list(all_stems))
        })
    except Exception as e:
        logger.error(f"Error in /api/status: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    # Support both JSON payload and multipart/form-data with file attachments
    if request.is_json:
        data = request.get_json(force=True, silent=True) or {}
    # Extract multi-manual filters
    raw_filters = []
    if request.is_json:
        data = request.get_json(force=True, silent=True) or {}
        user_prompt = str(data.get("message") or data.get("prompt") or data.get("query") or "").strip()
        tab_id = str(data.get("tab_id", "")).strip()
        f_val = data.get("pdf_filters") or data.get("pdf_filter")
        if isinstance(f_val, list):
            raw_filters = [str(x).strip() for x in f_val if str(x).strip()]
        elif isinstance(f_val, str) and f_val.strip():
            raw_filters = [x.strip() for x in f_val.split(",") if x.strip()]
        try:
            top_k = max(1, min(int(data.get("top_k", 5)), 25))
        except (ValueError, TypeError):
            top_k = 5
        visual_mode = str(data.get("visual_mode") or "strict").strip().lower()
    else:
        user_prompt = str(request.form.get("message") or request.form.get("prompt") or request.form.get("query") or "").strip()
        tab_id = str(request.form.get("tab_id", "")).strip()
        f_list = request.form.getlist("pdf_filters") or request.form.getlist("pdf_filter")
        if f_list:
            for item in f_list:
                for part in str(item).split(","):
                    if part.strip():
                        raw_filters.append(part.strip())
        else:
            f_single = request.form.get("pdf_filter", "").strip()
            if f_single:
                raw_filters = [x.strip() for x in f_single.split(",") if x.strip()]
        try:
            top_k = max(1, min(int(request.form.get("top_k", 5)), 25))
        except (ValueError, TypeError):
            top_k = 5
        visual_mode = str(request.form.get("visual_mode") or "strict").strip().lower()

    if visual_mode not in ["strict", "nearest", "off"]:
        visual_mode = "strict"

    # Process up to 5 user-uploaded files (Images & PDFs)
    uploaded_files = request.files.getlist("files") or request.files.getlist("attachments")
    uploaded_files = [f for f in uploaded_files if f and f.filename][:5]

    if not user_prompt and not uploaded_files:
        return jsonify({"error": "Please provide a question or attach an image/PDF."}), 400

    # Strategy 3: Sliding-Window Rate Limiting (per IP or User ID)
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1").split(",")[0].strip()
    client_id = session.get("user_id") or client_ip
    if not check_rate_limit(client_id):
        logger.warning(f"Rate limit exceeded for client '{client_id}'")
        return jsonify({
            "error": f"You are sending requests too quickly. Please wait before asking another question (limit: {config.MAX_REQUESTS_PER_MINUTE} req/min)."
        }), 429

    # Strategy 4: Strict Question Length Cap
    if len(user_prompt) > config.MAX_PROMPT_LENGTH:
        return jsonify({
            "error": f"Question is too long ({len(user_prompt)} characters). Maximum allowed is {config.MAX_PROMPT_LENGTH} characters."
        }), 400

    # Strategy 4: Strict Guest Query Limit
    user_id = session.get("user_id")
    if not user_id or user_id == "guest":
        guest_count = session.get("guest_query_count", 0)
        if guest_count >= config.GUEST_MAX_QUERIES:
            logger.warning(f"Guest quota reached ({guest_count}/{config.GUEST_MAX_QUERIES}) for client '{client_ip}'")
            return jsonify({
                "error": f"Guest query limit reached ({config.GUEST_MAX_QUERIES} questions). Please sign up or log in with an authorized account to continue."
            }), 403
        session["guest_query_count"] = guest_count + 1

    if not user_prompt and uploaded_files:
        user_prompt = "Please analyze the attached image(s) or document(s) and explain any findings, error messages, or instructions."

    # Validate manual selection: If explicitly empty or "none", block query
    if "none" in [x.lower() for x in raw_filters] or (len(raw_filters) == 1 and raw_filters[0].lower() in ["none", "empty"]):
        return jsonify({"error": "You should choose at least one source to ask a question."}), 400

    # Resolve manual filter stems
    selected_stems = []
    is_search_all = False
    for rf in raw_filters:
        if rf.lower() == "all":
            is_search_all = True
            continue
        matched_pdf = find_best_matching_pdf(rf)
        if matched_pdf:
            selected_stems.append(matched_pdf.stem)
        else:
            selected_stems.append(rf)

    # Deduplicate stems
    selected_stems = list(dict.fromkeys(selected_stems))

    # If raw_filters was provided non-empty but not 'all' and no stems resolved
    if raw_filters and not is_search_all and not selected_stems:
        return jsonify({"error": "You should choose at least one source to ask a question."}), 400

    try:
        api_key = config.get_gemini_api_key()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        user_id = session.get("user_id", "guest")
        user_pil_images = []
        saved_attachments_meta = []

        # 1. Process & Save User-Uploaded Attachments
        for f in uploaded_files:
            ext = Path(f.filename).suffix.lower()
            if ext not in [".png", ".jpg", ".jpeg", ".webp", ".pdf"]:
                continue

            safe_filename = f"att_u{user_id}_{int(time.time())}_{secure_filename(f.filename)}"
            dest_path = config.USER_UPLOADS_DIR / safe_filename
            f.save(dest_path)

            if ext == ".pdf":
                saved_attachments_meta.append({
                    "name": f.filename,
                    "type": "pdf",
                    "url": f"/api/user/uploads/{safe_filename}",
                    "filename": safe_filename
                })
                # Render PDF pages to PIL images for Gemini Multimodal input
                try:
                    doc = fitz.open(dest_path)
                    for pno in range(min(5, len(doc))):
                        pix = doc[pno].get_pixmap(dpi=150)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        user_pil_images.append(img)
                    doc.close()
                except Exception as e:
                    logger.error(f"Error rendering user-uploaded PDF: {e}")
            else:
                saved_attachments_meta.append({
                    "name": f.filename,
                    "type": "image",
                    "url": f"/api/user/uploads/{safe_filename}",
                    "filename": safe_filename
                })
                try:
                    with Image.open(dest_path) as img:
                        user_pil_images.append(img.copy())
                except Exception as e:
                    logger.error(f"Error reading user-uploaded image: {e}")

        # 2. Load conversational memory for this tab
        memory_turns = []
        if tab_id:
            memory_turns = auth_and_chat_db.get_tab_conversation_memory(tab_id, max_turns=5)

        memory_context_str = ""
        if memory_turns:
            memory_lines = []
            for t in memory_turns:
                role_label = "User" if t["role"] == "user" else "Assistant"
                memory_lines.append(f"{role_label}: {t['content']}")
            memory_context_str = "Prior Conversation History in this Tab:\n" + "\n".join(memory_lines) + "\n\n"

        # 3. Abuse Defense & Fast Path (Strategies 1, 2, 4)
        if not user_pil_images:
            # Drop obvious character mashing immediately (0 token cost)
            if is_obvious_gibberish(user_prompt):
                logger.info(f"Dropped obvious gibberish prompt: '{user_prompt}'")
                return jsonify({
                    "answer": "I am not sure about that. Please ask a specific question regarding DF robotics manuals, software (NavWiz, DFleet), or hardware.",
                    "seeds": [],
                    "seed_count": 0,
                    "expanded_count": 0,
                    "citations": [],
                    "attachments": saved_attachments_meta,
                    "visual_mode": visual_mode,
                    "visual_previews": [],
                    "is_conversational": True
                })

            # Strategy 2: Instant Static Canned Answers for Greetings & Small Talk (0 token cost)
            static_reply = get_static_conversational_reply(user_prompt)
            if static_reply:
                logger.info(f"Serving 0-token static reply for: '{user_prompt}'")
                uid = session.get("user_id")
                if tab_id and tab_id != "guest-tab" and uid:
                    try:
                        auth_and_chat_db.add_chat_message(tab_id=tab_id, role="user", content=user_prompt, attachments=saved_attachments_meta, user_id=uid)
                        auth_and_chat_db.add_chat_message(tab_id=tab_id, role="assistant", content=static_reply, user_id=uid)
                    except Exception as db_err:
                        logger.warning(f"Failed to persist static greeting in DB: {db_err}")

                return jsonify({
                    "answer": static_reply,
                    "seeds": [],
                    "seed_count": 0,
                    "expanded_count": 0,
                    "citations": [],
                    "attachments": saved_attachments_meta,
                    "visual_mode": visual_mode,
                    "visual_previews": [],
                    "is_conversational": True
                })

        # Conversational / Meta-Query Fallback with Max Output Tokens Cap
        if not user_pil_images and is_conversational_or_meta_query(user_prompt):
            logger.info(f"Fast-pathing conversational query without vector search: '{user_prompt}'")
            genai_client = embedder.get_client()
            
            # Fetch active manuals list for dynamic help response
            pdfs_info = pipeline_service.get_all_pdfs_status()
            manual_stems = [p["stem"] for p in pdfs_info if p["status"] == "embedded"]
            manual_list_str = ", ".join(manual_stems) if manual_stems else "NavWiz 4.0 User Manual 1.0, DFleet 4.0 User Manual, Copy of Field Deployment Handbook"

            meta_prompt = (
                "You are the friendly, expert DF Chatbot, the Multimodal Technical Assistant by DF Automation.\n"
                f"You have access to high-resolution technical manuals including: {manual_list_str}.\n"
                "Respond in a friendly, helpful, and concise manner.\n"
                "- If the user greets you or asks how you are doing, greet them back warmly and explain what you can help with.\n"
                "- If the user asks what you know or what questions to ask, summarize key capabilities (e.g., AGV navigation parameters, sensor calibration, pallet docking, wiring diagrams, battery charging, error code troubleshooting).\n"
                "- Mention that the user can also upload photos (e.g., error screens, equipment wiring) or PDF documents for visual troubleshooting.\n\n"
                f"{memory_context_str}Current User Message: {user_prompt}"
            )

            # Strategy 4: Enforce max_output_tokens cap on fast-path response
            fast_path_cfg = types.GenerateContentConfig(
                max_output_tokens=config.MAX_OUTPUT_TOKENS,
                temperature=0.2
            )
            response = genai_client.models.generate_content(
                model=config.GEMINI_QA_MODEL,
                contents=meta_prompt,
                config=fast_path_cfg
            )
            answer_text = response.text.strip() if response.text else "Hello! How can I assist you with your DF technical manuals or robotics questions today?"

            # Persist in DB if logged in and not a guest tab
            uid = session.get("user_id")
            if tab_id and tab_id != "guest-tab" and uid:
                try:
                    auth_and_chat_db.add_chat_message(
                        tab_id=tab_id,
                        role="user",
                        content=user_prompt,
                        attachments=saved_attachments_meta,
                        user_id=uid
                    )
                    auth_and_chat_db.add_chat_message(
                        tab_id=tab_id,
                        role="assistant",
                        content=answer_text,
                        user_id=uid
                    )
                except Exception as db_err:
                    logger.warning(f"Failed to persist greeting message in DB: {db_err}")

            return jsonify({
                "answer": answer_text,
                "seeds": [],
                "seed_count": 0,
                "expanded_count": 0,
                "citations": [],
                "attachments": saved_attachments_meta,
                "visual_mode": visual_mode,
                "visual_previews": [],
                "is_conversational": True
            })

        # 4. Multimodal Technical Retrieval Path (ChromaDB + Gemini Vision)
        collection = pipeline_service.get_chroma_collection()
        total_indexed = collection.count()
        if total_indexed == 0 and not user_pil_images:
            return jsonify({
                "error": "ChromaDB collection is empty. Please open the Admin page to upload and embed source PDFs."
            }), 400

        retrieved_seed_info = []
        sorted_pages = []

        if total_indexed > 0:
            # Embed query with Gemini Embedding 2
            embedder_client = embedder.get_client()
            embed_res = embedder.embed_query_text(embedder_client, user_prompt)
            qvec = embed_res["vector"]

            # Build ChromaDB filter with Multi-Manual support
            if selected_stems and not is_search_all:
                if len(selected_stems) == 1:
                    where_clause = {
                        "$and": [
                            {"is_front_matter": False},
                            {"pdf_stem": selected_stems[0]}
                        ]
                    }
                else:
                    where_clause = {
                        "$and": [
                            {"is_front_matter": False},
                            {"pdf_stem": {"$in": selected_stems}}
                        ]
                    }
            else:
                where_clause = {"is_front_matter": False}

            logger.info(f"Querying ChromaDB with where_clause: {where_clause}")

            # Query ChromaDB
            query_res = collection.query(
                query_embeddings=[qvec],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances", "documents"]
            )

            metas = query_res["metadatas"][0] if query_res.get("metadatas") else []
            distances = query_res["distances"][0] if query_res.get("distances") else []
            docs = query_res["documents"][0] if query_res.get("documents") else []

            NEIGHBOR_RADIUS = 3
            pages_to_load = set()
            xlsx_chunks_to_load = []

            for idx, (meta, dist) in enumerate(zip(metas, distances)):
                sim = float(1.0 - dist)
                doc_type = meta.get("doc_type", "pdf")
                doc_text = docs[idx] if idx < len(docs) else ""

                if doc_type == "xlsx":
                    sheet_name = meta.get("sheet_name", "Spreadsheet")
                    row_start = int(meta.get("row_start", 1))
                    row_end = int(meta.get("row_end", 1))
                    source_file = meta.get("source_file", meta.get("pdf_stem", "Spreadsheet"))
                    retrieved_seed_info.append({
                        "page_image": "",
                        "image_url": "",
                        "similarity": sim,
                        "chapter": sheet_name,
                        "section": str(meta.get("section", sheet_name)).strip(),
                        "page_number": row_start,
                        "row_start": row_start,
                        "row_end": row_end,
                        "pdf_stem": meta.get("pdf_stem", source_file),
                        "source_file": source_file,
                        "doc_type": "xlsx",
                        "text": doc_text
                    })
                    xlsx_chunks_to_load.append({
                        "source_file": source_file,
                        "sheet_name": sheet_name,
                        "row_start": row_start,
                        "row_end": row_end,
                        "text": doc_text
                    })
                else:
                    image_name = meta.get("page_image", "")
                    pdf_stem = meta.get("pdf_stem")
                    page_num = meta.get("page_number")
                    if not pdf_stem or not page_num:
                        inferred_stem, inferred_num = parse_page_filename(image_name)
                        pdf_stem = pdf_stem or inferred_stem
                        page_num = page_num or inferred_num

                    page_num = int(page_num)
                    seed_chapter = str(meta.get("chapter", "Unknown")).strip()

                    retrieved_seed_info.append({
                        "page_image": image_name,
                        "image_url": f"/rendered_pages/{image_name}",
                        "similarity": sim,
                        "chapter": seed_chapter,
                        "section": str(meta.get("section", "Unknown")).strip(),
                        "page_number": page_num,
                        "pdf_stem": pdf_stem,
                        "doc_type": "pdf"
                    })

                    if image_name:
                        pages_to_load.add(image_name)

            # Strategy 1: ChromaDB Relevance / Similarity Gating (Drop off-topic / nonsense before image loading & Gemini call)
            top_similarity = max([s["similarity"] for s in retrieved_seed_info]) if retrieved_seed_info else 0.0
            if top_similarity < config.RELEVANCE_SIMILARITY_THRESHOLD and not user_pil_images:
                logger.info(
                    f"Query rejected due to low relevance: '{user_prompt}' "
                    f"(top similarity: {top_similarity:.4f} < threshold: {config.RELEVANCE_SIMILARITY_THRESHOLD})"
                )
                relevance_refusal = (
                    "I couldn't find any relevant procedures or sections in the DF robotics manuals (NavWiz / DFleet) "
                    "matching your question. Please rephrase or ask about AGV navigation, sensors, or maintenance."
                )
                uid = session.get("user_id")
                if tab_id and tab_id != "guest-tab" and uid:
                    try:
                        auth_and_chat_db.add_chat_message(
                            tab_id=tab_id,
                            role="user",
                            content=user_prompt,
                            attachments=saved_attachments_meta,
                            user_id=uid
                        )
                        auth_and_chat_db.add_chat_message(
                            tab_id=tab_id,
                            role="assistant",
                            content=relevance_refusal,
                            user_id=uid
                        )
                    except Exception as db_err:
                        logger.warning(f"Failed to persist low-relevance refusal in DB: {db_err}")

                return jsonify({
                    "answer": relevance_refusal,
                    "seeds": [],
                    "seed_count": 0,
                    "expanded_count": 0,
                    "citations": [],
                    "attachments": saved_attachments_meta,
                    "visual_mode": visual_mode,
                    "visual_previews": [],
                    "is_conversational": True
                })

            for meta, dist in zip(metas, distances):
                if meta.get("doc_type") == "xlsx":
                    continue
                image_name = meta.get("page_image", "")
                pdf_stem = meta.get("pdf_stem")
                page_num = meta.get("page_number")
                if not pdf_stem or not page_num:
                    inferred_stem, inferred_num = parse_page_filename(image_name)
                    pdf_stem = pdf_stem or inferred_stem
                    page_num = page_num or inferred_num
                page_num = int(page_num)
                seed_chapter = str(meta.get("chapter", "Unknown")).strip()

                if pdf_stem and page_num > 0:
                    # Expand strictly within the same native chapter
                    for direction in [-1, 1]:
                        for offset in range(1, NEIGHBOR_RADIUS + 1):
                            p = page_num + (offset * direction)
                            if p < 1:
                                break

                            neighbor_id = f"{pdf_stem}_page_{p:03d}"
                            neighbor_res = collection.get(ids=[neighbor_id], include=["metadatas"])

                            if not neighbor_res["ids"]:
                                break

                            n_meta = neighbor_res["metadatas"][0]
                            n_chapter = str(n_meta.get("chapter", "")).strip()
                            n_is_front = bool(n_meta.get("is_front_matter", False))

                            if n_chapter != seed_chapter or n_is_front:
                                break

                            neighbor_file = n_meta.get("page_image", f"{neighbor_id}.png")
                            neighbor_path = config.IMAGE_CACHE_DIR / neighbor_file
                            if neighbor_path.exists():
                                pages_to_load.add(neighbor_file)

            sorted_pages = sorted(list(pages_to_load), key=sort_key)

        # 5. Build Interleaved Multimodal Input with Explicit PDF Page Labels and Structured Spreadsheets
        multimodal_contents = []

        if user_pil_images:
            multimodal_contents.append("=== USER ATTACHED IMAGES / SCREENSHOTS ===")
            for idx, u_img in enumerate(user_pil_images, 1):
                multimodal_contents.append(f"[User Uploaded Image #{idx}]")
                multimodal_contents.append(u_img)

        if sorted_pages:
            multimodal_contents.append("=== GROUNDING TECHNICAL MANUAL PAGES ===")
            for page_file in sorted_pages:
                img_path = config.IMAGE_CACHE_DIR / page_file
                if img_path.exists():
                    stem, p_num = parse_page_filename(page_file)
                    multimodal_contents.append(
                        f"[DOCUMENT SOURCE: \"{stem}\" | EXACT PDF PAGE NUMBER: {p_num} (File: {page_file})]"
                    )
                    with Image.open(img_path) as img:
                        multimodal_contents.append(img.copy())

        if xlsx_chunks_to_load:
            multimodal_contents.append("=== GROUNDING TECHNICAL SPREADSHEETS & SPECIFICATION TABLES ===")
            for xchunk in xlsx_chunks_to_load:
                multimodal_contents.append(
                    f"[SPREADSHEET SOURCE: \"{xchunk['source_file']}\" | SHEET: \"{xchunk['sheet_name']}\" | ROWS: {xchunk['row_start']}-{xchunk['row_end']}]\n{xchunk['text']}"
                )

        # 6. Build System Prompt with Exact Manual Titles & Citations Guidance
        attachment_notice = ""
        if user_pil_images:
            attachment_notice = (
                f"- USER UPLOADS: The user has attached {len(user_pil_images)} image(s)/document page(s). "
                "Carefully inspect the user's uploaded images to detect error messages, identify components, "
                "verify configurations, and relate them to the manual instructions.\n"
            )

        # Dynamic exact list of indexed source manuals and spreadsheets
        active_source_docs = sorted(list(config.SOURCE_DIR.glob("*.pdf")) + list(config.SOURCE_DIR.glob("*.xlsx")))
        manual_names_bullet_list = "\n".join([f'- "{p.stem}"' for p in active_source_docs]) if active_source_docs else '- "DFleet 4.0 User Manual"\n- "NavWiz 4.0 User Manual 1.0"'

        system_prompt = (
            "You are DF Chatbot, the expert technical assistant for NavWiz, DFleet, Field Deployment, and Project Site Engineering documentation by DF Automation.\n"
            "Answer the user's question accurately, thoroughly, and concisely using the provided manual page images, spreadsheet tables, and user uploads.\n\n"
            f"{attachment_notice}"
            "CRITICAL CITATION RULES:\n"
            "- For technical manual pages: cite as `[Exact Manual Title, p.N]` using the EXACT PDF PAGE NUMBER provided in the document label.\n"
            "- For spreadsheet/Excel documentation: cite as `[Exact Document Title, Sheet: SheetName, Rows: X-Y]` (or `Row: X`).\n"
            "- In spreadsheets, note cell status annotations: `[🟢 Active/Tested/Recoverable]`, `[🔴 Cannot Recover/Critical]`, `[🟠 Discrepancy/Notice/Teaching]`, `[🟡 Pending]`, etc.\n"
            "- Available source documents:\n"
            f"{manual_names_bullet_list}\n\n"
            "CONVERSATION MEMORY:\n"
            "- Use the prior conversation history in this tab for context.\n\n"
            "STRICT GUARDRAIL: If the user's question is gibberish, meaningless text, or completely unrelated to "
            "robotics/manual software/site documentation, and cannot be answered by the provided documents, respond EXACTLY with:\n"
            "\"I am not sure about that.\""
        )

        genai_client = embedder.get_client()
        full_text_prompt = f"{system_prompt}\n\n{memory_context_str}Current User Question: {user_prompt}"
        contents = multimodal_contents + [full_text_prompt]

        # Strategy 4: Enforce strict max_output_tokens cap
        qa_cfg = types.GenerateContentConfig(
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
            temperature=0.2
        )
        response = genai_client.models.generate_content(
            model=config.GEMINI_QA_MODEL,
            contents=contents,
            config=qa_cfg
        )

        answer_text = response.text.strip() if response.text else "I am not sure about that."

        # 7. Extract structured citations from text and resolve to canonical stems
        citation_matches = re.findall(r"\[(.*?),\s*p\.?\s*(\d+)\]", answer_text, re.IGNORECASE)
        structured_citations = []
        for manual_title, p_str in citation_matches:
            try:
                p_num = int(p_str)
                matched_pdf = find_best_matching_pdf(manual_title)
                canonical_stem = matched_pdf.stem if matched_pdf else manual_title.strip()
                structured_citations.append({
                    "manual": canonical_stem,
                    "page_number": p_num,
                    "type": "pdf",
                    "url": f"/pdf-viewer?file={encodeURIComponent(canonical_stem)}&page={p_num}"
                })
            except Exception:
                pass

        xlsx_matches = re.findall(r"\[([^\[\]]+?),\s*Sheet:\s*([^,]+?),\s*Row(?:s)?:\s*([^\]]+?)\]", answer_text, re.IGNORECASE)
        for wb_title, sheet_title, row_str in xlsx_matches:
            structured_citations.append({
                "manual": wb_title.strip(),
                "sheet": sheet_title.strip(),
                "rows": row_str.strip(),
                "type": "xlsx"
            })

        if not structured_citations and retrieved_seed_info:
            for s in retrieved_seed_info[:3]:
                if s.get("doc_type") == "xlsx":
                    structured_citations.append({
                        "manual": s.get("source_file", s["pdf_stem"]),
                        "sheet": s.get("chapter", "Spreadsheet"),
                        "rows": f"{s.get('row_start', 1)}-{s.get('row_end', 1)}",
                        "type": "xlsx"
                    })
                else:
                    structured_citations.append({
                        "manual": s["pdf_stem"],
                        "page_number": s["page_number"],
                        "type": "pdf",
                        "url": f"/pdf-viewer?file={s['pdf_stem']}&page={s['page_number']}"
                    })

        # 8. Build Visual Preview Cards based on User's Visual Mode
        visual_previews = []
        if visual_mode == "strict":
            # Show photo if available: preview pages directly cited in the answer (PDF only)
            seen_previews = set()
            for cit in structured_citations:
                if cit.get("type") == "xlsx":
                    continue
                c_manual = cit.get("manual", "")
                c_page = cit.get("page_number", 1)
                key = (c_manual, c_page)
                if key not in seen_previews:
                    seen_previews.add(key)
                    padded = f"{c_page:03d}"
                    img_name = f"{c_manual}_page_{padded}.png"
                    img_path = config.IMAGE_CACHE_DIR / img_name
                    if img_path.exists():
                        visual_previews.append({
                            "manual": c_manual,
                            "page_number": c_page,
                            "page_image": img_name,
                            "image_url": f"/rendered_pages/{img_name}",
                            "caption": f"Cited: {c_manual}, Page {c_page}",
                            "is_direct_citation": True
                        })
            # Fallback to top seed if no direct citation image exists
            if not visual_previews and retrieved_seed_info:
                top_s = retrieved_seed_info[0]
                if top_s.get("doc_type") != "xlsx" and top_s.get("page_image"):
                    visual_previews.append({
                        "manual": top_s["pdf_stem"],
                        "page_number": top_s["page_number"],
                        "page_image": top_s["page_image"],
                        "image_url": top_s["image_url"],
                        "caption": f"Top Match: {top_s['pdf_stem']}, Page {top_s['page_number']}",
                        "similarity": top_s.get("similarity", 0.0),
                        "is_direct_citation": False
                    })
        elif visual_mode == "nearest":
            # Show photo as near as possible (might have hallucination): include all candidate seeds (PDF only)
            seen_previews = set()
            for s in retrieved_seed_info:
                if s.get("doc_type") == "xlsx" or not s.get("page_image"):
                    continue
                key = (s["pdf_stem"], s["page_number"])
                if key not in seen_previews:
                    seen_previews.add(key)
                    visual_previews.append({
                        "manual": s["pdf_stem"],
                        "page_number": s["page_number"],
                        "page_image": s["page_image"],
                        "image_url": s["image_url"],
                        "caption": f"{s['pdf_stem']}, Page {s['page_number']}",
                        "similarity": s.get("similarity", 0.0),
                        "is_direct_citation": False
                    })

        # 9. Persist messages in database if tab_id & user session exists
        uid = session.get("user_id")
        if tab_id and tab_id != "guest-tab" and uid:
            try:
                auth_and_chat_db.add_chat_message(
                    tab_id=tab_id,
                    role="user",
                    content=user_prompt,
                    attachments=saved_attachments_meta,
                    user_id=uid
                )
                auth_and_chat_db.add_chat_message(
                    tab_id=tab_id,
                    role="assistant",
                    content=answer_text,
                    citations=structured_citations,
                    top_k=retrieved_seed_info,
                    expanded_count=len(sorted_pages),
                    user_id=uid
                )

                # If tab has default title "New Chat", auto-update title with prompt topic
                tabs = auth_and_chat_db.list_user_tabs(uid)
                current_tab = next((t for t in tabs if t["id"] == tab_id), None)
                if current_tab and current_tab["title"] in ["New Chat", ""]:
                    suggested_title = user_prompt[:35].strip()
                    if len(user_prompt) > 35:
                        suggested_title += "..."
                    auth_and_chat_db.update_tab_title(tab_id, suggested_title)
            except Exception as db_err:
                logger.warning(f"Failed to persist QA message in DB: {db_err}")

        return jsonify({
            "answer": answer_text,
            "seeds": retrieved_seed_info,
            "seed_count": len(retrieved_seed_info),
            "expanded_count": len(sorted_pages),
            "citations": structured_citations,
            "attachments": saved_attachments_meta,
            "visual_mode": visual_mode,
            "visual_previews": visual_previews
        })

    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Admin Authentication APIs
# ============================================================================

@app.route("/api/admin/auth-status", methods=["GET"])
def admin_auth_status():
    is_auth = bool(session.get("admin_authenticated"))
    return jsonify({
        "status": "ok",
        "authenticated": is_auth
    })


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(force=True, silent=True) or {}
    admin_id = str(data.get("id") or data.get("username") or data.get("admin_id") or "").strip()
    password = str(data.get("password") or data.get("admin_password") or "").strip()

    if admin_id.lower() == "df" and (password == "df" or config.verify_admin_password(password)):
        session["admin_authenticated"] = True
        session.permanent = True
        
        # Link user session as df
        admin_user = auth_and_chat_db.authenticate_user("df", password)
        if admin_user:
            session["user_id"] = admin_user["id"]
            session["username"] = admin_user["username"]
            session["role"] = "admin"

        logger.info("Admin authentication successful for ID: df.")
        return jsonify({
            "status": "ok",
            "message": "Admin authentication successful."
        })

    logger.warning(f"Failed admin login attempt for ID: {admin_id}.")
    return jsonify({
        "status": "error",
        "error": "Invalid admin ID or password. (Admin ID: df, Password: df)"
    }), 401


@app.route("/api/admin/logout", methods=["POST"])
def admin_logout():
    session.pop("admin_authenticated", None)
    logger.info("Admin session logged out.")
    return jsonify({
        "status": "ok",
        "message": "Admin console locked / logged out successfully."
    })


# ============================================================================
# Admin User Analytics & Chat History Inspection APIs
# ============================================================================

@app.route("/api/admin/users", methods=["GET"])
@admin_required
def get_admin_users():
    try:
        users = auth_and_chat_db.list_all_users_with_stats()
        return jsonify({"status": "ok", "users": users})
    except Exception as e:
        logger.error(f"Error fetching admin users: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>/chats", methods=["GET"])
@admin_required
def get_admin_user_chats(user_id: int):
    try:
        user_data = auth_and_chat_db.get_user_full_chat_history(user_id)
        if not user_data:
            return jsonify({"status": "error", "error": "User not found"}), 404
        return jsonify({"status": "ok", "user": user_data})
    except Exception as e:
        logger.error(f"Error fetching admin user chats: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
@admin_required
def delete_admin_user(user_id: int):
    try:
        deleted = auth_and_chat_db.delete_user(user_id)
        if not deleted:
            return jsonify({"status": "error", "error": "User not found."}), 404
        remaining = auth_and_chat_db.list_all_users_with_stats()
        return jsonify({
            "status": "ok",
            "message": "User and associated chat history deleted successfully.",
            "users": remaining
        })
    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>/approve", methods=["POST"])
@admin_required
def approve_admin_user(user_id: int):
    """Approves a pending user account, allowing them to log in."""
    try:
        updated = auth_and_chat_db.update_user_status(user_id, "approved")
        if not updated:
            return jsonify({"status": "error", "error": "User not found."}), 404
        remaining = auth_and_chat_db.list_all_users_with_stats()
        return jsonify({
            "status": "ok",
            "message": "User account approved successfully.",
            "users": remaining
        })
    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error approving user {user_id}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>/decline", methods=["POST"])
@admin_required
def decline_admin_user(user_id: int):
    """Declines a pending user account registration."""
    try:
        updated = auth_and_chat_db.update_user_status(user_id, "declined")
        if not updated:
            return jsonify({"status": "error", "error": "User not found."}), 404
        remaining = auth_and_chat_db.list_all_users_with_stats()
        return jsonify({
            "status": "ok",
            "message": "User account registration declined.",
            "users": remaining
        })
    except ValueError as e:
        return jsonify({"status": "error", "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error declining user {user_id}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>/export/html", methods=["GET"])
@admin_required
def export_admin_user_chats_html(user_id: int):
    try:
        user_data = auth_and_chat_db.get_user_full_chat_history(user_id)
        if not user_data:
            return jsonify({"status": "error", "error": "User not found"}), 404
        
        tab_id = request.args.get("tab_id")
        html_content = report_exporter.generate_html_report(user_data, tab_id=tab_id)
        
        username_safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', user_data.get("username", "user"))
        scope_suffix = f"tab_{tab_id[:8]}" if tab_id else "all_tabs"
        filename = f"df_chatbot_report_{username_safe}_{scope_suffix}.html"
        
        as_attachment = request.args.get("download", "0") == "1"
        disposition = f'attachment; filename="{filename}"' if as_attachment else f'inline; filename="{filename}"'
        
        response = make_response(html_content)
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        response.headers["Content-Disposition"] = disposition
        return response
    except Exception as e:
        logger.error(f"Error exporting user chats HTML (user {user_id}): {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>/export/pdf", methods=["GET"])
@admin_required
def export_admin_user_chats_pdf(user_id: int):
    try:
        user_data = auth_and_chat_db.get_user_full_chat_history(user_id)
        if not user_data:
            return jsonify({"status": "error", "error": "User not found"}), 404
        
        tab_id = request.args.get("tab_id")
        pdf_bytes = report_exporter.generate_pdf_report(user_data, tab_id=tab_id)
        
        username_safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', user_data.get("username", "user"))
        scope_suffix = f"tab_{tab_id[:8]}" if tab_id else "all_tabs"
        filename = f"df_chatbot_report_{username_safe}_{scope_suffix}.pdf"
        
        as_attachment = request.args.get("download", "1") != "0"
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=as_attachment,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error exporting user chats PDF (user {user_id}): {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


def _get_requested_user_ids() -> Optional[List[int]]:
    user_ids = []
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        user_ids = data.get("user_ids", [])
    if not user_ids:
        ids_param = request.args.get("user_ids", "")
        if ids_param:
            user_ids = [int(x.strip()) for x in ids_param.split(",") if x.strip().isdigit()]
    return user_ids if user_ids else None


@app.route("/api/admin/users/export/html", methods=["GET", "POST"])
@admin_required
def export_admin_multi_users_html():
    try:
        user_ids = _get_requested_user_ids()
        users_data = auth_and_chat_db.get_multiple_users_full_chat_history(user_ids)
        if not users_data:
            return jsonify({"status": "error", "error": "No user accounts found"}), 404
        
        html_content = report_exporter.generate_multi_user_html_report(users_data)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"df_chatbot_multi_user_report_{len(users_data)}users_{timestamp_str}.html"
        
        as_attachment = request.args.get("download", "0") == "1"
        disposition = f'attachment; filename="{filename}"' if as_attachment else f'inline; filename="{filename}"'
        
        response = make_response(html_content)
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        response.headers["Content-Disposition"] = disposition
        return response
    except Exception as e:
        logger.error(f"Error exporting multi-user HTML: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/users/export/pdf", methods=["GET", "POST"])
@admin_required
def export_admin_multi_users_pdf():
    try:
        user_ids = _get_requested_user_ids()
        users_data = auth_and_chat_db.get_multiple_users_full_chat_history(user_ids)
        if not users_data:
            return jsonify({"status": "error", "error": "No user accounts found"}), 404
        
        pdf_bytes = report_exporter.generate_multi_user_pdf_report(users_data)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"df_chatbot_multi_user_report_{len(users_data)}users_{timestamp_str}.pdf"
        
        as_attachment = request.args.get("download", "1") != "0"
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=as_attachment,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error exporting multi-user PDF: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/users/export/zip", methods=["GET", "POST"])
@admin_required
def export_admin_multi_users_zip():
    try:
        user_ids = _get_requested_user_ids()
        users_data = auth_and_chat_db.get_multiple_users_full_chat_history(user_ids)
        if not users_data:
            return jsonify({"status": "error", "error": "No user accounts found"}), 404
        
        doc_format = request.args.get("format", "pdf").lower()
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            doc_format = data.get("format", doc_format).lower()
            
        zip_bytes = report_exporter.generate_multi_user_zip(users_data, format=doc_format)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"df_chatbot_user_reports_archive_{len(users_data)}users_{timestamp_str}.zip"
        
        return send_file(
            io.BytesIO(zip_bytes),
            mimetype="application/zip",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error exporting multi-user ZIP: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500



# ============================================================================
# Admin & Source File Management APIs
# ============================================================================

@app.route("/api/admin/files", methods=["GET"])
@admin_required
def list_admin_files():
    try:
        files = pipeline_service.get_all_pdfs_status()
        return jsonify({"status": "ok", "files": files})
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/files/upload", methods=["POST"])
@admin_required
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"status": "error", "error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"status": "error", "error": "No file selected."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"status": "error", "error": "Only PDF files are supported."}), 400

    filename = secure_filename(file.filename)
    save_path = config.SOURCE_DIR / filename

    try:
        file.save(save_path)
        logger.info(f"Uploaded new PDF: {filename}")
        return jsonify({
            "status": "ok",
            "message": f"Successfully uploaded '{filename}'. Ready to embed.",
            "filename": filename
        })
    except Exception as e:
        logger.error(f"Error saving upload: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/files/<path:filename>", methods=["DELETE"])
@admin_required
def delete_pdf(filename: str):
    try:
        res = pipeline_service.delete_pdf_and_cleanup(filename)
        return jsonify({
            "status": "ok",
            "message": f"Removed '{filename}' and deleted {res['deleted_vectors_count']} vectors from ChromaDB.",
            "details": res
        })
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/files/download/<path:filename>", methods=["GET"])
def download_pdf(filename: str):
    if not session.get("admin_authenticated"):
        # If accessing directly from browser without auth, redirect or 401
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            if config.verify_admin_password(token):
                return send_from_directory(config.SOURCE_DIR, filename, as_attachment=False)
        return jsonify({"status": "error", "error": "Admin authentication required."}), 401
    return send_from_directory(config.SOURCE_DIR, filename, as_attachment=False)


@app.route("/api/admin/files/page-image/<path:image_name>", methods=["GET"])
def serve_page_image(image_name: str):
    """Serves rendered PNG page images with admin authentication check or active session."""
    if not session.get("admin_authenticated"):
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            if not config.verify_admin_password(token):
                return jsonify({"status": "error", "error": "Admin authentication required."}), 401
        else:
            return jsonify({"status": "error", "error": "Admin authentication required."}), 401

    image_path = config.IMAGE_CACHE_DIR / image_name
    if not image_path.exists():
        return jsonify({"status": "error", "error": f"Image '{image_name}' not found."}), 404

    return send_from_directory(config.IMAGE_CACHE_DIR, image_name)


@app.route("/api/admin/files/<path:filename>/pages", methods=["GET"])
@admin_required
def get_file_pages(filename: str):
    """Returns all pages for a PDF with image URLs, indexed state, metadata tags, and document text."""
    try:
        data = pipeline_service.get_pdf_pages_detail(filename)
        return jsonify({"status": "ok", **data})
    except FileNotFoundError as e:
        return jsonify({"status": "error", "error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error fetching pages for {filename}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/files/<path:filename>/pages/<int:page_number>/metadata", methods=["POST"])
@admin_required
def update_file_page_metadata(filename: str, page_number: int):
    """Updates/adds/deletes metadata tags and document summary for a specific page."""
    data = request.get_json(force=True, silent=True) or {}
    new_metadata = data.get("metadata", {})
    document_text = data.get("document")

    try:
        res = pipeline_service.update_page_metadata(filename, page_number, new_metadata, document_text)
        return jsonify(res)
    except FileNotFoundError as e:
        return jsonify({"status": "error", "error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error updating metadata for {filename} page {page_number}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/files/<path:filename>/pages/<int:page_number>", methods=["DELETE"])
@admin_required
def delete_file_page(filename: str, page_number: int):
    """Deletes an unneeded or irrelevant page from ChromaDB vector collection."""
    try:
        res = pipeline_service.delete_page_from_chroma(filename, page_number)
        return jsonify(res)
    except Exception as e:
        logger.error(f"Error deleting page {page_number} for {filename}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/files/<path:filename>/pages/<int:page_number>/embed", methods=["POST"])
@admin_required
def embed_single_file_page(filename: str, page_number: int):
    """Renders, embeds, and syncs a single page into ChromaDB without having to embed the whole document."""
    data = request.get_json(force=True, silent=True) or {}
    custom_metadata = data.get("metadata")

    try:
        res = pipeline_service.embed_single_page_and_sync(filename, page_number, custom_metadata)
        return jsonify(res)
    except RuntimeError as e:
        return jsonify({"status": "error", "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error embedding single page {page_number} for {filename}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/files/<path:filename>/pages/batch", methods=["POST"])
@admin_required
def batch_pages_operation(filename: str):
    """Performs batch operations (embed or delete) on selected pages."""
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")  # 'embed' or 'delete'
    page_numbers = data.get("page_numbers", [])

    if not action or action not in ["embed", "delete"]:
        return jsonify({"status": "error", "error": "Invalid action. Must be 'embed' or 'delete'."}), 400

    try:
        res = pipeline_service.batch_pages_action(filename, action, page_numbers)
        return jsonify(res)
    except Exception as e:
        logger.error(f"Error in batch action {action} on {filename}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


# ============================================================================
# Admin Pipeline Embedding & Config APIs
# ============================================================================

@app.route("/api/admin/embed", methods=["POST"])
@admin_required
def trigger_embedding():
    data = request.get_json(force=True, silent=True) or {}
    target_filename = data.get("filename")  # None = embed all pending/modified
    mode = data.get("mode", "required").lower()
    force_flag = bool(data.get("force", False)) or (mode == "all")

    try:
        config.get_gemini_api_key()
    except RuntimeError as err:
        return jsonify({
            "status": "error",
            "error": f"API Key Error: {err}. Please enter your GEMINI_API_KEY in the Settings tab."
        }), 400

    pdfs_info = pipeline_service.get_all_pdfs_status()
    if target_filename:
        targets = [p for p in pdfs_info if p["filename"] == target_filename]
        if not targets:
            return jsonify({"status": "error", "error": f"File '{target_filename}' not found."}), 404
    else:
        if force_flag:
            targets = pdfs_info
        else:
            targets = [p for p in pdfs_info if p["status"] != "embedded" or p.get("missing_pages_count", 0) > 0 or p.get("is_modified", False)]

    if not targets:
        return jsonify({
            "status": "ok",
            "message": "All PDF files are already up-to-date and 100% indexed in ChromaDB.",
            "embedded_files": []
        })

    processed = []
    for t in targets:
        pdf_path = config.SOURCE_DIR / t["filename"]
        res = pipeline_service.process_and_embed_pdf(pdf_path, mode=mode, force=force_flag)
        processed.append(res)

    total_pages = sum(p["pages_embedded"] for p in processed)
    newly_pages = sum(p.get("newly_embedded", 0) for p in processed)
    return jsonify({
        "status": "ok",
        "message": f"Successfully processed {len(processed)} PDF(s) ({newly_pages} newly embedded, {total_pages} total pages indexed in ChromaDB).",
        "processed": processed
    })


@app.route("/api/admin/embed/stream", methods=["GET", "POST"])
@admin_required
def trigger_embedding_stream():
    """Streams real-time embedding progress, page counts, remaining pages, % and ETA via Server-Sent Events."""
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
    else:
        data = request.args.to_dict()

    target_filename = data.get("filename")
    mode = data.get("mode", "required").lower()
    force_flag = str(data.get("force", "false")).lower() in ["true", "1", "yes"] or (mode == "all")

    try:
        config.get_gemini_api_key()
    except RuntimeError as err:
        return jsonify({
            "status": "error",
            "error": f"API Key Error: {err}. Please enter your GEMINI_API_KEY in Settings."
        }), 400

    pdfs_info = pipeline_service.get_all_pdfs_status()
    if target_filename:
        targets = [p for p in pdfs_info if p["filename"] == target_filename]
        if not targets:
            return jsonify({"status": "error", "error": f"File '{target_filename}' not found."}), 404
    else:
        if force_flag:
            targets = pdfs_info
        else:
            targets = [p for p in pdfs_info if p["status"] != "embedded" or p.get("missing_pages_count", 0) > 0 or p.get("is_modified", False)]

    def event_stream():
        for event in pipeline_service.generate_embedding_progress(targets, mode=mode, force=force_flag):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )



@app.route("/api/admin/config", methods=["GET"])
@admin_required
def get_admin_config():
    try:
        try:
            raw_key = config.get_gemini_api_key()
            if len(raw_key) > 8:
                masked_key = raw_key[:4] + "•" * (len(raw_key) - 8) + raw_key[-4:]
            else:
                masked_key = "••••••••"
            has_key = True
        except Exception:
            masked_key = ""
            has_key = False

        collection = pipeline_service.get_chroma_collection()

        return jsonify({
            "status": "ok",
            "has_key": has_key,
            "masked_key": masked_key,
            "qa_model": config.GEMINI_QA_MODEL,
            "embed_model": config.GEMINI_EMBED_MODEL,
            "total_indexed_pages": collection.count(),
            "render_dpi": config.RENDER_DPI
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/config", methods=["POST"])
@admin_required
def update_admin_config():
    data = request.get_json(force=True, silent=True) or {}
    new_key = data.get("api_key")
    new_model = data.get("qa_model")
    new_password = data.get("admin_password")

    try:
        res = pipeline_service.update_env_config(
            api_key=new_key,
            qa_model=new_model,
            admin_password=new_password
        )
        return jsonify({
            "status": "ok",
            "message": "Configuration updated and saved to .env successfully.",
            "config": res
        })
    except Exception as e:
        logger.error(f"Error saving config: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/config/test", methods=["POST"])
@admin_required
def test_api_key_route():
    data = request.get_json(force=True, silent=True) or {}
    test_key = data.get("api_key")
    res = pipeline_service.test_gemini_api(api_key=test_key)
    if res.get("success"):
        return jsonify(res)
    else:
        return jsonify(res), 400


@app.route("/api/admin/db/reset", methods=["POST"])
@admin_required
def reset_database():
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
        # Recreate empty
        client.create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        pipeline_service.save_pipeline_state({})
        return jsonify({
            "status": "ok",
            "message": "ChromaDB vector database and pipeline state cleared."
        })
    except Exception as e:
        logger.error(f"Error resetting database: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)