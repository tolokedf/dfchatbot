"""
Flask Web Application for Multimodal Technical PDF RAG.
Features ChromaDB vector retrieval, native outline-aware chapter expansion,
and multimodal generation with Gemini 3.5 Flash.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from google import genai
from dotenv import load_dotenv
import chromadb

import config
from embedders import gemini_multimodal_embedder as embedder

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

_chroma_collection = None


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_collection


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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    try:
        collection = get_chroma_collection()
        total_docs = collection.count()
        return jsonify({
            "status": "ok",
            "collection": config.CHROMA_COLLECTION_NAME,
            "total_indexed_pages": total_docs,
            "embed_model": config.GEMINI_EMBED_MODEL,
            "qa_model": config.GEMINI_QA_MODEL
        })
    except Exception as e:
        logger.error(f"Error in /api/status: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/rendered_pages/<path:filename>")
def serve_rendered_page(filename: str):
    return send_from_directory(config.IMAGE_CACHE_DIR, filename)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_prompt = str(data.get("message", "")).strip()

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
        collection = get_chroma_collection()
        total_indexed = collection.count()
        if total_indexed == 0:
            return jsonify({"error": "ChromaDB collection is empty. Run 'python run_embedding_pipeline.py' first."}), 400

        # 1. Embed query with Gemini Embedding 2
        embedder_client = embedder.get_client()
        embed_res = embedder.embed_query_text(embedder_client, user_prompt)
        qvec = embed_res["vector"]

        # 2. Query ChromaDB (filtering out front-matter)
        query_res = collection.query(
            query_embeddings=[qvec],
            n_results=top_k,
            where={"is_front_matter": False},
            include=["metadatas", "distances", "documents"]
        )

        metas = query_res["metadatas"][0] if query_res.get("metadatas") else []
        distances = query_res["distances"][0] if query_res.get("distances") else []

        if not metas:
            return jsonify({
                "answer": "I am not sure about that.",
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
            
            # Robust PDF stem and page number lookup
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
                "page_number": page_num
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
                            break  # Stop expanding immediately on chapter shift or front matter

                        neighbor_file = n_meta.get("page_image", f"{neighbor_id}.png")
                        neighbor_path = config.IMAGE_CACHE_DIR / neighbor_file
                        if neighbor_path.exists():
                            pages_to_load.add(neighbor_file)

        sorted_pages = sorted(list(pages_to_load), key=sort_key)
        
        # Load and verify images into memory safely closing file handles
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)