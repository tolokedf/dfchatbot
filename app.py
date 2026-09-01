"""
Flask Web Application for Multimodal Technical PDF RAG.
Includes:
- Main Chatbot Interface (/): Outline-aware retrieval with chapter-bounded expansion.
- Admin Management Console (/admin): PDF file manager, manual embedding triggers, API key configuration.
- REST APIs for chat, file uploads/deletions, embedding pipeline, and diagnostics.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from google import genai
from dotenv import load_dotenv
import chromadb

import config
import pipeline_service
from embedders import gemini_multimodal_embedder as embedder

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload limit


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


@app.route("/rendered_pages/<path:filename>")
def serve_rendered_page(filename: str):
    return send_from_directory(config.IMAGE_CACHE_DIR, filename)


# ============================================================================
# Core Chat & Retrieval API
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
    data = request.get_json(force=True, silent=True) or {}
    user_prompt = str(data.get("message", "")).strip()
    pdf_filter = str(data.get("pdf_filter", "all")).strip()

    try:
        top_k = max(1, min(int(data.get("top_k", 5)), 25))
    except (ValueError, TypeError):
        top_k = 5

    if not user_prompt:
        return jsonify({"error": "Message cannot be empty."}), 400

    try:
        api_key = config.get_gemini_api_key()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        collection = pipeline_service.get_chroma_collection()
        total_indexed = collection.count()
        if total_indexed == 0:
            return jsonify({
                "error": "ChromaDB collection is empty. Please open the Admin page to upload and embed source PDFs."
            }), 400

        # 1. Embed query with Gemini Embedding 2
        embedder_client = embedder.get_client()
        embed_res = embedder.embed_query_text(embedder_client, user_prompt)
        qvec = embed_res["vector"]

        # 2. Build ChromaDB filter
        if pdf_filter and pdf_filter.lower() != "all":
            where_clause = {
                "$and": [
                    {"is_front_matter": False},
                    {"pdf_stem": pdf_filter}
                ]
            }
        else:
            where_clause = {"is_front_matter": False}

        # 3. Query ChromaDB
        query_res = collection.query(
            query_embeddings=[qvec],
            n_results=top_k,
            where=where_clause,
            include=["metadatas", "distances", "documents"]
        )

        metas = query_res["metadatas"][0] if query_res.get("metadatas") else []
        distances = query_res["distances"][0] if query_res.get("distances") else []

        if not metas:
            return jsonify({
                "answer": "I am not sure about that. (No matching content found for the selected manual)",
                "seeds": [],
                "seed_count": 0,
                "expanded_count": 0
            })

        NEIGHBOR_RADIUS = 3
        pages_to_load = set()
        retrieved_seed_info = []

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
                            break  # Stop expanding on chapter shift or front matter

                        neighbor_file = n_meta.get("page_image", f"{neighbor_id}.png")
                        neighbor_path = config.IMAGE_CACHE_DIR / neighbor_file
                        if neighbor_path.exists():
                            pages_to_load.add(neighbor_file)

        sorted_pages = sorted(list(pages_to_load), key=sort_key)

        pil_images = []
        for page_file in sorted_pages:
            img_path = config.IMAGE_CACHE_DIR / page_file
            if img_path.exists():
                with Image.open(img_path) as img:
                    pil_images.append(img.copy())

        if not pil_images:
            return jsonify({
                "answer": "I am not sure about that.",
                "seeds": retrieved_seed_info,
                "seed_count": len(retrieved_seed_info),
                "expanded_count": 0
            })

        system_prompt = (
            "You are an expert technical assistant for technical robotics software and deployment manuals (NavWiz, DFleet, Field Deployment). "
            "Answer the user's question accurately and concisely using only the provided image pages as context.\n\n"
            "STRICT RULE: If the user's question is gibberish, meaningless text, completely unrelated to "
            "robotics/manual software, or cannot be answered by the provided manual pages, respond EXACTLY with:\n"
            "\"I am not sure about that.\""
        )

        genai_client = embedder.get_client()
        contents = pil_images + [f"{system_prompt}\n\nUser Question: {user_prompt}"]

        response = genai_client.models.generate_content(
            model=config.GEMINI_QA_MODEL,
            contents=contents
        )

        answer_text = response.text.strip() if response.text else "I am not sure about that."

        return jsonify({
            "answer": answer_text,
            "seeds": retrieved_seed_info,
            "seed_count": len(retrieved_seed_info),
            "expanded_count": len(sorted_pages)
        })

    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Admin & Source File Management APIs
# ============================================================================

@app.route("/api/admin/files", methods=["GET"])
def list_admin_files():
    try:
        files = pipeline_service.get_all_pdfs_status()
        return jsonify({"status": "ok", "files": files})
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/files/upload", methods=["POST"])
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
    return send_from_directory(config.SOURCE_DIR, filename, as_attachment=False)


# ============================================================================
# Admin Pipeline Embedding & Config APIs
# ============================================================================

@app.route("/api/admin/embed", methods=["POST"])
def trigger_embedding():
    data = request.get_json(force=True, silent=True) or {}
    target_filename = data.get("filename")  # None = embed all pending

    try:
        # Check API key first
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
            targets = [p for p in pdfs_info if p["status"] != "embedded" or data.get("force", False)]

        if not targets:
            return jsonify({
                "status": "ok",
                "message": "All PDF files are already up-to-date and embedded in ChromaDB.",
                "embedded_files": []
            })

        processed = []
        for t in targets:
            pdf_path = config.SOURCE_DIR / t["filename"]
            res = pipeline_service.process_and_embed_pdf(pdf_path)
            processed.append(res)

        total_pages = sum(p["pages_embedded"] for p in processed)
        return jsonify({
            "status": "ok",
            "message": f"Successfully embedded {len(processed)} PDF(s) ({total_pages} total pages indexed).",
            "processed": processed
        })

    except Exception as e:
        logger.error(f"Embedding execution error: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/config", methods=["GET"])
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
def update_admin_config():
    data = request.get_json(force=True, silent=True) or {}
    new_key = data.get("api_key")
    new_model = data.get("qa_model")

    try:
        res = pipeline_service.update_env_config(api_key=new_key, qa_model=new_model)
        return jsonify({
            "status": "ok",
            "message": "Configuration updated and saved to .env successfully.",
            "config": res
        })
    except Exception as e:
        logger.error(f"Error saving config: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/admin/config/test", methods=["POST"])
def test_api_key_route():
    data = request.get_json(force=True, silent=True) or {}
    test_key = data.get("api_key")
    res = pipeline_service.test_gemini_api(api_key=test_key)
    if res.get("success"):
        return jsonify(res)
    else:
        return jsonify(res), 400


@app.route("/api/admin/db/reset", methods=["POST"])
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