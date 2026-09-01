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
from functools import wraps
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, session, Response, stream_with_context
from werkzeug.utils import secure_filename
from PIL import Image
from google import genai
from dotenv import load_dotenv
import chromadb

import config
import pipeline_service
import auth_and_chat_db
from embedders import gemini_multimodal_embedder as embedder
import fitz

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")
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


@app.route("/pdf-viewer")
def pdf_viewer_page():
    """Renders a dedicated page citation viewer jumping directly to the requested manual page."""
    target_file = request.args.get("file", "").strip()
    try:
        page_num = max(1, int(request.args.get("page", 1)))
    except (ValueError, TypeError):
        page_num = 1

    # Match against source PDF files
    found_pdf = None
    if target_file:
        for f in config.SOURCE_DIR.glob("*.pdf"):
            if f.name.lower() == target_file.lower() or f.stem.lower() == target_file.lower():
                found_pdf = f
                break

    if not found_pdf:
        # Fallback to first available PDF
        pdfs = sorted(list(config.SOURCE_DIR.glob("*.pdf")))
        if pdfs:
            found_pdf = pdfs[0]
        else:
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
    pdf_path = config.SOURCE_DIR / filename
    if not pdf_path.exists():
        for f in config.SOURCE_DIR.glob("*.pdf"):
            if f.name.lower() == filename.lower() or f.stem.lower() == filename.lower():
                pdf_path = f
                filename = f.name
                break

    if not pdf_path.exists():
        return jsonify({"status": "error", "error": f"PDF '{filename}' not found."}), 404

    return send_from_directory(
        config.SOURCE_DIR,
        filename,
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
        session["user_id"] = user_data["id"]
        session["username"] = user_data["username"]
        session["role"] = user_data["role"]
        if user_data["role"] == "admin":
            session["admin_authenticated"] = True

        tabs = auth_and_chat_db.list_user_tabs(user_data["id"])
        return jsonify({
            "status": "ok",
            "message": "Account created successfully!",
            "user": {
                "id": user_data["id"],
                "username": user_data["username"],
                "role": user_data["role"]
            },
            "tabs": tabs,
            "default_tab_id": user_data["default_tab_id"]
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

    user = auth_and_chat_db.get_user_by_id(user_id)
    if not user:
        session.clear()
        return jsonify({"authenticated": False, "user": None})

    return jsonify({
        "authenticated": True,
        "user": user
    })


@app.route("/api/user/profile-picture", methods=["POST"])
@user_required
def upload_profile_picture():
    """Uploads and saves user profile picture inside the 'User database/profile_pictures' directory."""
    user_id = session.get("user_id")
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

    # Save to User database/profile_pictures/
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
    """Serves user profile picture from 'User database/profile_pictures'."""
    return send_from_directory(config.USER_AVATAR_DIR, filename)


@app.route("/api/user/uploads/<path:filename>", methods=["GET"])
def serve_user_upload(filename: str):
    """Serves user-uploaded chat attachments (photos/PDFs) from 'User database/uploaded_attachments'."""
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


# ============================================================================
# Chat Tabs API (Per-User Isolated Sessions & Memory)
# ============================================================================

@app.route("/api/chat/tabs", methods=["GET"])
def get_user_tabs():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"status": "ok", "tabs": []})
    tabs = auth_and_chat_db.list_user_tabs(user_id)
    return jsonify({"status": "ok", "tabs": tabs})


@app.route("/api/chat/tabs", methods=["POST"])
@user_required
def create_user_tab():
    user_id = session.get("user_id")
    data = request.get_json(force=True, silent=True) or {}
    title = data.get("title", "New Chat")
    new_tab = auth_and_chat_db.create_tab(user_id, title)
    return jsonify({"status": "ok", "tab": new_tab})


@app.route("/api/chat/tabs/<tab_id>", methods=["DELETE"])
@user_required
def delete_user_tab(tab_id: str):
    user_id = session.get("user_id")
    deleted = auth_and_chat_db.delete_tab(tab_id, user_id)
    if not deleted:
        return jsonify({"status": "error", "error": "Tab not found or unauthorized."}), 404
    remaining = auth_and_chat_db.list_user_tabs(user_id)
    return jsonify({"status": "ok", "message": "Tab deleted successfully.", "tabs": remaining})


@app.route("/api/chat/tabs/<tab_id>/messages", methods=["GET"])
@user_required
def get_tab_message_history(tab_id: str):
    user_id = session.get("user_id")
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
        return jsonify({
            "status": "ok",
            "collection": config.CHROMA_COLLECTION_NAME,
            "total_indexed_pages": total_docs,
            "embed_model": config.GEMINI_EMBED_MODEL,
            "qa_model": config.GEMINI_QA_MODEL,
            "sources": [p["stem"] for p in pdfs_info if p["status"] == "embedded"]
        })
    except Exception as e:
        logger.error(f"Error in /api/status: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    # Support both JSON payload and multipart/form-data with file attachments
    if request.is_json:
        data = request.get_json(force=True, silent=True) or {}
        user_prompt = str(data.get("message", "")).strip()
        tab_id = str(data.get("tab_id", "")).strip()
        pdf_filter = str(data.get("pdf_filter", "all")).strip()
        try:
            top_k = max(1, min(int(data.get("top_k", 5)), 25))
        except (ValueError, TypeError):
            top_k = 5
    else:
        user_prompt = str(request.form.get("message", "")).strip()
        tab_id = str(request.form.get("tab_id", "")).strip()
        pdf_filter = str(request.form.get("pdf_filter", "all")).strip()
        try:
            top_k = max(1, min(int(request.form.get("top_k", 5)), 25))
        except (ValueError, TypeError):
            top_k = 5

    # Process up to 5 user-uploaded files (Images & PDFs)
    uploaded_files = request.files.getlist("files") or request.files.getlist("attachments")
    uploaded_files = [f for f in uploaded_files if f and f.filename][:5]

    if not user_prompt and not uploaded_files:
        return jsonify({"error": "Please provide a question or attach an image/PDF."}), 400

    if not user_prompt and uploaded_files:
        user_prompt = "Please analyze the attached image(s) or document(s) and explain any findings, error messages, or instructions."

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

        # 3. Fast Path: Conversational / Greeting / Help Queries (Zero vector search latency)
        if not user_pil_images and is_conversational_or_meta_query(user_prompt):
            logger.info(f"Fast-pathing conversational query without vector search: '{user_prompt}'")
            genai_client = embedder.get_client()
            
            # Fetch active manuals list for dynamic help response
            pdfs_info = pipeline_service.get_all_pdfs_status()
            manual_stems = [p["stem"] for p in pdfs_info if p["status"] == "embedded"]
            manual_list_str = ", ".join(manual_stems) if manual_stems else "NavWiz 4.0 User Manual, DFleet 4.0 User Manual, Field Deployment Handbook"

            meta_prompt = (
                "You are the friendly, expert NavWiz & DFleet Multimodal Technical Assistant.\n"
                f"You have access to high-resolution technical manuals including: {manual_list_str}.\n"
                "Respond in a friendly, helpful, and concise manner.\n"
                "- If the user greets you or asks how you are doing, greet them back warmly and explain what you can help with.\n"
                "- If the user asks what you know or what questions to ask, summarize key capabilities (e.g., AGV navigation parameters, sensor calibration, pallet docking, wiring diagrams, battery charging, error code troubleshooting).\n"
                "- Mention that the user can also upload photos (e.g., error screens, equipment wiring) or PDF documents for visual troubleshooting.\n\n"
                f"{memory_context_str}Current User Message: {user_prompt}"
            )

            response = genai_client.models.generate_content(
                model=config.GEMINI_QA_MODEL,
                contents=meta_prompt
            )
            answer_text = response.text.strip() if response.text else "Hello! How can I assist you with your NavWiz, DFleet, or robotics deployment questions today?"

            # Persist in DB
            if tab_id and session.get("user_id"):
                auth_and_chat_db.add_chat_message(
                    tab_id=tab_id,
                    role="user",
                    content=user_prompt,
                    attachments=saved_attachments_meta
                )
                auth_and_chat_db.add_chat_message(
                    tab_id=tab_id,
                    role="assistant",
                    content=answer_text
                )

            return jsonify({
                "answer": answer_text,
                "seeds": [],
                "seed_count": 0,
                "expanded_count": 0,
                "citations": [],
                "attachments": saved_attachments_meta,
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
        manual_pil_images = []

        if total_indexed > 0:
            # Embed query with Gemini Embedding 2
            embedder_client = embedder.get_client()
            embed_res = embedder.embed_query_text(embedder_client, user_prompt)
            qvec = embed_res["vector"]

            # Build ChromaDB filter
            if pdf_filter and pdf_filter.lower() != "all":
                where_clause = {
                    "$and": [
                        {"is_front_matter": False},
                        {"pdf_stem": pdf_filter}
                    ]
                }
            else:
                where_clause = {"is_front_matter": False}

            # Query ChromaDB
            query_res = collection.query(
                query_embeddings=[qvec],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances", "documents"]
            )

            metas = query_res["metadatas"][0] if query_res.get("metadatas") else []
            distances = query_res["distances"][0] if query_res.get("distances") else []

            NEIGHBOR_RADIUS = 3
            pages_to_load = set()

            for meta, dist in zip(metas, distances):
                sim = float(1.0 - dist)
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
                    "pdf_stem": pdf_stem
                })

                pages_to_load.add(image_name)

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

            for page_file in sorted_pages:
                img_path = config.IMAGE_CACHE_DIR / page_file
                if img_path.exists():
                    with Image.open(img_path) as img:
                        manual_pil_images.append(img.copy())

        # Combine all visual inputs: User Uploads + Retrieved Manual Pages
        all_visual_contents = user_pil_images + manual_pil_images

        # 5. Build System Prompt with Citations & User Attachment Guidance
        attachment_notice = ""
        if user_pil_images:
            attachment_notice = (
                f"- USER UPLOADS: The user has attached {len(user_pil_images)} image(s)/document page(s). "
                "Carefully inspect the user's uploaded images to detect error messages, identify components, "
                "verify configurations, and relate them to the manual instructions.\n"
            )

        system_prompt = (
            "You are an expert technical assistant for NavWiz, DFleet, and Field Deployment technical manuals.\n"
            "Answer the user's question accurately, thoroughly, and concisely using the provided images and documents.\n\n"
            f"{attachment_notice}"
            "CITATION INSTRUCTIONS:\n"
            "- In each section, row, or step of your answer referencing the manual, attach direct citations indicating the exact manual and page number where that information is found.\n"
            "- Format citations as: `[Manual Name, p.XX]` (e.g. `[DFleet 4.0 User Manual, p.45]`, `[NavWiz 4.0 User Manual 1.0, p.12]`, `[Field Deployment Handbook, p.8]`).\n"
            "- Provide citations consistently throughout the explanation.\n\n"
            "CONVERSATION MEMORY:\n"
            "- Use the conversation history in this tab to understand follow-up questions, pronouns, or references to previously discussed steps.\n\n"
            "STRICT GUARDRAIL: If the user's question is gibberish, meaningless text, or completely unrelated to "
            "robotics/manual software, and cannot be answered by the provided images, respond EXACTLY with:\n"
            "\"I am not sure about that.\""
        )

        genai_client = embedder.get_client()
        full_text_prompt = f"{system_prompt}\n\n{memory_context_str}Current User Question: {user_prompt}"
        contents = all_visual_contents + [full_text_prompt]

        response = genai_client.models.generate_content(
            model=config.GEMINI_QA_MODEL,
            contents=contents
        )

        answer_text = response.text.strip() if response.text else "I am not sure about that."

        # 6. Extract structured citations from text or fallback to seeds
        citation_matches = re.findall(r"\[(.*?),\s*p\.?\s*(\d+)\]", answer_text, re.IGNORECASE)
        structured_citations = []
        for manual_title, p_str in citation_matches:
            try:
                p_num = int(p_str)
                structured_citations.append({
                    "manual": manual_title.strip(),
                    "page_number": p_num,
                    "url": f"/pdf-viewer?file={manual_title.strip()}&page={p_num}"
                })
            except Exception:
                pass

        if not structured_citations and retrieved_seed_info:
            for s in retrieved_seed_info[:3]:
                structured_citations.append({
                    "manual": s["pdf_stem"],
                    "page_number": s["page_number"],
                    "url": f"/pdf-viewer?file={s['pdf_stem']}&page={s['page_number']}"
                })

        # 7. Persist messages in database if tab_id & user session exists
        if tab_id and session.get("user_id"):
            auth_and_chat_db.add_chat_message(
                tab_id=tab_id,
                role="user",
                content=user_prompt,
                attachments=saved_attachments_meta
            )
            auth_and_chat_db.add_chat_message(
                tab_id=tab_id,
                role="assistant",
                content=answer_text,
                citations=structured_citations,
                top_k=retrieved_seed_info,
                expanded_count=len(sorted_pages)
            )

            # If tab has default title "New Chat", auto-update title with prompt topic
            tabs = auth_and_chat_db.list_user_tabs(session["user_id"])
            current_tab = next((t for t in tabs if t["id"] == tab_id), None)
            if current_tab and current_tab["title"] in ["New Chat", ""]:
                clean_title = re.sub(r'[\r\n\t]+', ' ', user_prompt).strip()[:35]
                if len(user_prompt) > 35:
                    clean_title += "..."
                auth_and_chat_db.update_tab_title(tab_id, clean_title)

        return jsonify({
            "answer": answer_text,
            "seeds": retrieved_seed_info,
            "seed_count": len(retrieved_seed_info),
            "expanded_count": len(sorted_pages),
            "citations": structured_citations,
            "attachments": saved_attachments_meta
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
    admin_id = str(data.get("id") or data.get("username") or "df").strip()
    password = str(data.get("password", "")).strip()

    if config.verify_admin_credentials(admin_id, password) or (admin_id in ["df", "admin"] and (password == "df" or config.verify_admin_password(password))) or config.verify_admin_password(password):
        session["admin_authenticated"] = True
        session.permanent = True
        
        # Link user session as df
        admin_user = auth_and_chat_db.authenticate_user("df", password)
        if admin_user:
            session["user_id"] = admin_user["id"]
            session["username"] = admin_user["username"]
            session["role"] = "admin"

        logger.info(f"Admin authentication successful for ID: {admin_id}.")
        return jsonify({
            "status": "ok",
            "message": "Admin authentication successful."
        })

    logger.warning(f"Failed admin login attempt for ID: {admin_id}.")
    return jsonify({
        "status": "error",
        "error": "Invalid admin ID or password. (Default ID: df, Password: df)"
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