"""
Service module for PDF ingestion, status tracking, ChromaDB synchronization,
and configuration management.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import fitz  # PyMuPDF
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

import config
from embedders import gemini_multimodal_embedder as embedder

logger = logging.getLogger(__name__)

STATE_FILE = config.OUTPUT_DIR / "pipeline_state.json"


def load_pipeline_state() -> dict:
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pipeline state: {e}")
            return {}
    return {}


def save_pipeline_state(state: dict) -> None:
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)


def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_all_pdfs_status() -> list[dict]:
    """Returns detailed status of all PDFs in source/."""
    state = load_pipeline_state()
    collection = get_chroma_collection()
    pdf_files = sorted(config.SOURCE_DIR.glob("*.pdf"))

    results = []
    for pdf_path in pdf_files:
        filename = pdf_path.name
        stem = pdf_path.stem
        file_size = pdf_path.stat().st_size
        modified_ts = pdf_path.stat().st_mtime
        modified_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified_ts))

        # Total pages from PDF document
        try:
            doc = fitz.open(pdf_path)
            total_pdf_pages = len(doc)
            has_toc = bool(doc.get_toc())
            doc.close()
        except Exception:
            total_pdf_pages = 0
            has_toc = False

        # Count records in ChromaDB for this PDF
        try:
            db_records = collection.get(where={"pdf_stem": stem}, include=[])
            db_page_count = len(db_records["ids"]) if db_records else 0
        except Exception:
            db_page_count = 0

        is_recorded_in_state = (filename in state and state[filename].get("size") == file_size)

        if db_page_count > 0 and is_recorded_in_state:
            status = "embedded"
        elif db_page_count > 0 and not is_recorded_in_state:
            status = "outdated"  # File was modified after embedding
        else:
            status = "pending"   # Not yet embedded

        results.append({
            "filename": filename,
            "stem": stem,
            "size_bytes": file_size,
            "size_formatted": format_bytes(file_size),
            "modified_at": modified_str,
            "total_pdf_pages": total_pdf_pages,
            "embedded_pages_count": db_page_count,
            "has_toc": has_toc,
            "status": status
        })

    return results


def build_pdf_metadata_map(doc: fitz.Document) -> dict:
    """Parses PDF TOC hierarchy and marks front-matter pages."""
    toc = doc.get_toc()
    total_pages = len(doc)
    page_meta = {}

    for i in range(1, total_pages + 1):
        page_meta[i] = {
            "chapter": "Unknown",
            "section": "Unknown",
            "subsection": "Unknown",
            "is_front_matter": False
        }

    if not toc:
        return page_meta

    active_hier = {1: "Unknown", 2: "Unknown", 3: "Unknown"}
    toc_by_page = {}

    for item in toc:
        lvl, title, p_num = item[0], str(item[1]).strip(), item[2]
        if p_num not in toc_by_page:
            toc_by_page[p_num] = []
        toc_by_page[p_num].append((lvl, title))

    for p in range(1, total_pages + 1):
        if p in toc_by_page:
            for lvl, title in toc_by_page[p]:
                active_hier[lvl] = title
                for l in range(lvl + 1, 10):
                    active_hier[l] = "Unknown"

        current_ch = active_hier[1].strip()
        is_front = (current_ch in ["Copyright Notice", "Table of Contents", "Unknown", "Cover"])

        page_meta[p] = {
            "chapter": active_hier[1].strip(),
            "section": active_hier[2].strip(),
            "subsection": active_hier[3].strip(),
            "is_front_matter": is_front
        }

    return page_meta


def render_pdf_to_images(pdf_path: Path, log_fn: Optional[Callable[[str], None]] = None) -> list[Path]:
    doc = fitz.open(pdf_path)
    zoom = config.RENDER_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    image_paths: list[Path] = []
    for page_index in range(len(doc)):
        out_path = config.IMAGE_CACHE_DIR / f"{pdf_path.stem}_page_{page_index + 1:03d}.png"
        needs_render = (
            not out_path.exists()
            or out_path.stat().st_mtime < pdf_path.stat().st_mtime
        )

        if needs_render:
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix)
            if max(pix.width, pix.height) > config.MAX_IMAGE_DIMENSION:
                scale = config.MAX_IMAGE_DIMENSION / max(pix.width, pix.height)
                new_matrix = fitz.Matrix(zoom * scale, zoom * scale)
                pix = page.get_pixmap(matrix=new_matrix)
            pix.save(out_path)
            if log_fn:
                log_fn(f"Rendered {out_path.name}")

        image_paths.append(out_path)

    doc.close()
    return image_paths


def process_and_embed_pdf(
    pdf_path: Path,
    log_fn: Optional[Callable[[str], None]] = None
) -> dict:
    """Renders, embeds, and updates ChromaDB for a single PDF."""
    def log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    # 1. Verify API key
    api_key = config.get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set or empty in .env.")

    log(f"Processing '{pdf_path.name}' ({format_bytes(pdf_path.stat().st_size)})...")

    # 2. Extract outline
    log("Extracting native PDF outline metadata...")
    doc = fitz.open(pdf_path)
    metadata_map = build_pdf_metadata_map(doc)
    doc.close()

    # 3. Render pages
    log(f"Rendering pages at {config.RENDER_DPI} DPI...")
    image_paths = render_pdf_to_images(pdf_path, log_fn=log)

    # 4. Embed pages with Gemini
    log(f"Embedding {len(image_paths)} pages with {config.GEMINI_EMBED_MODEL}...")
    client = embedder.get_client()

    results = []
    for i, image_path in enumerate(image_paths, start=1):
        log(f"  Embedding page {i}/{len(image_paths)}: {image_path.name}...")
        try:
            res = embedder.embed_page_image(client, image_path)
            results.append(res)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                raise RuntimeError(
                    f"Gemini API rate limit exceeded (HTTP 429) on page {i}/{len(image_paths)}. "
                    "Please wait a minute or use a key with higher quota."
                ) from e
            elif "403" in err_msg or "PERMISSION_DENIED" in err_msg:
                raise RuntimeError(f"Gemini API key is invalid or lacks permissions: {err_msg}") from e
            else:
                raise RuntimeError(f"Error embedding page {image_path.name}: {err_msg}") from e

    # 5. Upsert into ChromaDB
    log(f"Upserting {len(results)} embeddings into ChromaDB collection '{config.CHROMA_COLLECTION_NAME}'...")
    collection = get_chroma_collection()

    ids, embeddings, metadatas, documents = [], [], [], []
    for idx, r in enumerate(results):
        if r.get("vector") is None:
            continue
        page_num = idx + 1
        meta = metadata_map.get(page_num, {})
        page_id = f"{pdf_path.stem}_page_{page_num:03d}"

        ids.append(page_id)
        embeddings.append(r["vector"])
        metadatas.append({
            "pdf_stem": pdf_path.stem,
            "page_image": r["page_image"],
            "page_number": page_num,
            "chapter": str(meta.get("chapter", "Unknown")).strip(),
            "section": str(meta.get("section", "Unknown")).strip(),
            "subsection": str(meta.get("subsection", "Unknown")).strip(),
            "is_front_matter": bool(meta.get("is_front_matter", False)),
            "model": str(r.get("model", config.GEMINI_EMBED_MODEL)),
            "dimensions": int(r.get("dimensions", config.EMBED_OUTPUT_DIMENSIONALITY) or config.EMBED_OUTPUT_DIMENSIONALITY),
            "elapsed_seconds": float(r.get("elapsed_seconds", 0.0) or 0.0),
        })
        documents.append(
            f"{pdf_path.stem} - Page {page_num:03d} | Chapter: {meta.get('chapter', 'Unknown')} | Section: {meta.get('section', 'Unknown')}"
        )

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    # 6. Update pipeline state
    state = load_pipeline_state()
    state[pdf_path.name] = {"size": pdf_path.stat().st_size, "embedded_at": time.time()}
    save_pipeline_state(state)

    log(f"✅ Finished embedding '{pdf_path.name}' successfully ({len(ids)} pages indexed).")
    return {
        "filename": pdf_path.name,
        "stem": pdf_path.stem,
        "pages_embedded": len(ids)
    }


def delete_pdf_and_cleanup(filename: str) -> dict:
    """Removes a PDF file, its rendered images, ChromaDB vectors, and state."""
    pdf_path = config.SOURCE_DIR / filename
    stem = pdf_path.stem

    # 1. Delete physical PDF
    deleted_pdf = False
    if pdf_path.exists():
        pdf_path.unlink()
        deleted_pdf = True

    # 2. Delete from ChromaDB
    collection = get_chroma_collection()
    deleted_vectors_count = 0
    try:
        existing = collection.get(where={"pdf_stem": stem}, include=[])
        if existing and existing["ids"]:
            deleted_vectors_count = len(existing["ids"])
            collection.delete(where={"pdf_stem": stem})
    except Exception as e:
        logger.error(f"Error removing records from ChromaDB for {stem}: {e}")

    # 3. Delete rendered PNG images
    deleted_images_count = 0
    for img_path in config.IMAGE_CACHE_DIR.glob(f"{stem}_page_*.png"):
        try:
            img_path.unlink()
            deleted_images_count += 1
        except Exception as e:
            logger.error(f"Error removing image {img_path}: {e}")

    # 4. Remove from pipeline state
    state = load_pipeline_state()
    if filename in state:
        del state[filename]
        save_pipeline_state(state)

    return {
        "filename": filename,
        "deleted_pdf": deleted_pdf,
        "deleted_vectors_count": deleted_vectors_count,
        "deleted_images_count": deleted_images_count
    }


def update_env_config(api_key: Optional[str] = None, qa_model: Optional[str] = None) -> dict:
    """Updates .env file with new API key and/or QA model."""
    env_path = config.PROJECT_ROOT / ".env"
    current_lines = []
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            current_lines = f.readlines()

    env_dict = {}
    for line in current_lines:
        line_str = line.strip()
        if "=" in line_str and not line_str.startswith("#"):
            k, v = line_str.split("=", 1)
            env_dict[k.strip()] = v.strip()

    if api_key is not None and api_key.strip():
        env_dict["GEMINI_API_KEY"] = api_key.strip()
    if qa_model is not None and qa_model.strip():
        env_dict["GEMINI_QA_MODEL"] = qa_model.strip()

    with env_path.open("w", encoding="utf-8") as f:
        for k, v in env_dict.items():
            f.write(f"{k}={v}\n")

    # Reload environment
    load_dotenv(dotenv_path=env_path, override=True)
    if "GEMINI_QA_MODEL" in env_dict:
        config.GEMINI_QA_MODEL = env_dict["GEMINI_QA_MODEL"]

    return {
        "status": "ok",
        "api_key_set": bool(env_dict.get("GEMINI_API_KEY")),
        "qa_model": config.GEMINI_QA_MODEL
    }


def test_gemini_api(api_key: Optional[str] = None) -> dict:
    """Tests Gemini API key validity by making a lightweight test call."""
    key = api_key or config.get_gemini_api_key()
    if not key:
        return {"success": False, "error": "API Key is empty or not set."}

    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=["Ping test. Reply with 'pong'."]
        )
        return {
            "success": True,
            "message": "Gemini API connected successfully!",
            "reply": response.text.strip() if response.text else "pong"
        }
    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg or "PERMISSION_DENIED" in err_msg:
            return {"success": False, "error": "Invalid API Key or Permission Denied (HTTP 403)."}
        elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            return {"success": False, "error": "Gemini API Rate Limit hit (HTTP 429). Please wait a moment."}
        return {"success": False, "error": f"API Connection failed: {err_msg}"}
