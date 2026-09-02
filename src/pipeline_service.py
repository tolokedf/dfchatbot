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
    """Returns detailed status of all PDFs in source/, including missing pages and modification state."""
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

        # Count records and existing IDs in ChromaDB for this PDF
        try:
            db_records = collection.get(where={"pdf_stem": stem}, include=[])
            db_ids = set(db_records["ids"]) if db_records and db_records["ids"] else set()
            db_page_count = len(db_ids)
        except Exception:
            db_ids = set()
            db_page_count = 0

        # Calculate exact missing page numbers
        missing_pages = []
        for p in range(1, total_pdf_pages + 1):
            if f"{stem}_page_{p:03d}" not in db_ids:
                missing_pages.append(p)

        last_embedded_at = state.get(filename, {}).get("embedded_at", 0)
        last_size = state.get(filename, {}).get("size")
        
        is_modified = bool(last_embedded_at and modified_ts > last_embedded_at + 1.0) or (filename in state and last_size != file_size)
        is_recorded_in_state = (filename in state and last_size == file_size and not is_modified)

        if db_page_count > 0 and db_page_count >= total_pdf_pages and is_recorded_in_state:
            status = "embedded"
        elif is_modified:
            status = "outdated"  # File was modified after embedding
        elif db_page_count > 0 and db_page_count < total_pdf_pages:
            status = "pending"   # Missing pages
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
            "missing_pages_count": len(missing_pages),
            "missing_page_numbers": missing_pages,
            "is_modified": is_modified,
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
    """Renders all PDF pages to PNG images at RENDER_DPI."""
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


def format_time_remaining(seconds: float) -> str:
    """Formats ETA seconds into human-readable string like '2m 15s' or '45s'."""
    sec = max(0, int(round(seconds)))
    if sec <= 0:
        return "0s"
    if sec < 60:
        return f"{sec}s"
    minutes = sec // 60
    rem_sec = sec % 60
    if minutes < 60:
        return f"{minutes}m {rem_sec:02d}s"
    hours = minutes // 60
    rem_min = minutes % 60
    return f"{hours}h {rem_min:02d}m"


def generate_embedding_progress(targets: list[dict], mode: str = "required", force: bool = False):
    """
    Generator that yields real-time progress events as dicts during the embedding pipeline.
    
    Supports 2 Embedding Options:
      Option 1 (mode='all' / force=True): Completely re-embeds all pages from scratch.
      Option 2 (mode='required' / force=False): Smart scan: embeds only missing or modified pages.
    """
    # 1. Verify API key
    try:
        config.get_gemini_api_key()
    except Exception as e:
        yield {"type": "error", "error": f"API Key Error: {e}. Please enter your GEMINI_API_KEY in Settings."}
        return

    is_full_reembed = (mode == "all" or force)
    state = load_pipeline_state()
    collection = get_chroma_collection()

    # 2. Pre-scan all targets to calculate total pages needed
    pdf_plans = []
    total_pages_to_embed = 0

    mode_label = "Full Re-embed (All Pages)" if is_full_reembed else "Embed Required (Missing & Changed Pages)"
    yield {
        "type": "status",
        "message": f"Scanning PDF documents for [{mode_label}]..."
    }

    for t in targets:
        filename = t["filename"]
        pdf_path = config.SOURCE_DIR / filename
        if not pdf_path.exists():
            continue

        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()
        except Exception:
            num_pages = 0

        try:
            existing = collection.get(where={"pdf_stem": pdf_path.stem}, include=[])
            existing_ids = set(existing["ids"]) if existing and existing["ids"] else set()
        except Exception:
            existing_ids = set()

        last_embedded_at = state.get(filename, {}).get("embedded_at", 0)
        last_size = state.get(filename, {}).get("size")
        is_modified = bool(last_embedded_at and pdf_path.stat().st_mtime > last_embedded_at + 1.0) or (filename in state and last_size != pdf_path.stat().st_size)

        pages_to_process = []
        if is_full_reembed:
            # Mode 1: Re-embed all pages
            pages_to_process = list(range(1, num_pages + 1))
        else:
            # Mode 2: Embed required (missing or modified)
            if is_modified:
                # If modified after embedding, re-embed all pages
                pages_to_process = list(range(1, num_pages + 1))
            else:
                for p in range(1, num_pages + 1):
                    pid = f"{pdf_path.stem}_page_{p:03d}"
                    if pid not in existing_ids:
                        pages_to_process.append(p)

        pdf_plans.append({
            "filename": filename,
            "path": pdf_path,
            "stem": pdf_path.stem,
            "total_pages": num_pages,
            "existing_ids": existing_ids,
            "pages_to_process": set(pages_to_process),
            "pages_to_embed": len(pages_to_process),
            "is_modified": is_modified
        })
        total_pages_to_embed += len(pages_to_process)

    if total_pages_to_embed == 0:
        yield {
            "type": "complete",
            "message": "All pages are already up-to-date and 100% indexed in ChromaDB. No embedding required.",
            "overall_total": 0,
            "overall_done": 0,
            "overall_remaining": 0,
            "percentage": 100.0,
            "eta_formatted": "0s",
            "processed": []
        }
        return

    # 3. Emit start event
    overall_done = 0
    overall_start_time = time.monotonic()
    recent_page_durations = []

    yield {
        "type": "start",
        "mode": "all" if is_full_reembed else "required",
        "total_files": len(pdf_plans),
        "overall_total": total_pages_to_embed,
        "overall_done": 0,
        "overall_remaining": total_pages_to_embed,
        "percentage": 0.0,
        "eta_formatted": "Calculating...",
        "message": f"Starting {mode_label} for {len(pdf_plans)} PDF(s) ({total_pages_to_embed} pages total)..."
    }

    client = embedder.get_client()
    processed_summary = []

    for file_idx, plan in enumerate(pdf_plans, start=1):
        pdf_path = plan["path"]
        stem = plan["stem"]
        filename = plan["filename"]
        pages_to_process = plan["pages_to_process"]

        if not pages_to_process:
            continue

        yield {
            "type": "file_start",
            "filename": filename,
            "file_index": file_idx,
            "total_files": len(pdf_plans),
            "file_total_pages": plan["total_pages"],
            "pages_to_embed_in_file": len(pages_to_process),
            "message": f"[{file_idx}/{len(pdf_plans)}] Processing '{filename}' ({len(pages_to_process)} pages to embed)..."
        }

        # Extract TOC metadata
        doc = fitz.open(pdf_path)
        metadata_map = build_pdf_metadata_map(doc)
        doc.close()

        # Render pages (re-renders any modified pages)
        image_paths = render_pdf_to_images(pdf_path)

        existing_ids = plan["existing_ids"]
        file_embedded_this_run = 0

        for page_idx, image_path in enumerate(image_paths, start=1):
            page_num = page_idx
            page_id = f"{stem}_page_{page_num:03d}"

            if page_num not in pages_to_process:
                continue

            page_start_time = time.monotonic()

            remaining = max(0, total_pages_to_embed - overall_done)
            pct = round((overall_done / total_pages_to_embed) * 100, 1)

            if recent_page_durations:
                avg_duration = sum(recent_page_durations[-10:]) / len(recent_page_durations[-10:])
                eta_sec = avg_duration * remaining
                eta_str = format_time_remaining(eta_sec)
            else:
                eta_str = "Calculating..."

            yield {
                "type": "progress",
                "filename": filename,
                "file_index": file_idx,
                "total_files": len(pdf_plans),
                "current_page": page_num,
                "total_file_pages": len(image_paths),
                "overall_total": total_pages_to_embed,
                "overall_done": overall_done,
                "overall_remaining": remaining,
                "percentage": pct,
                "eta_formatted": eta_str,
                "current_action": f"Embedding page {page_num}/{len(image_paths)}: {image_path.name}",
                "message": f"Embedding page {page_num}/{len(image_paths)} ({image_path.name})..."
            }

            # Embed with retry logger
            retry_logs = []
            def retry_logger(msg: str):
                retry_logs.append(msg)

            try:
                res = embedder.embed_page_image(client, image_path, log_fn=retry_logger)
            except Exception as e:
                yield {
                    "type": "error",
                    "filename": filename,
                    "page_number": page_num,
                    "error": str(e)
                }
                raise

            # If retries occurred, yield those logs to frontend
            for r_log in retry_logs:
                yield {
                    "type": "log",
                    "filename": filename,
                    "page_number": page_num,
                    "log": r_log
                }

            # Immediate upsert to ChromaDB
            meta = metadata_map.get(page_num, {})
            doc_summary = (
                f"{stem} - Page {page_num:03d} | "
                f"Chapter: {meta.get('chapter', 'Unknown')} | "
                f"Section: {meta.get('section', 'Unknown')}"
            )
            collection.upsert(
                ids=[page_id],
                embeddings=[res["vector"]],
                metadatas=[{
                    "pdf_stem": stem,
                    "page_image": res["page_image"],
                    "page_number": page_num,
                    "chapter": str(meta.get("chapter", "Unknown")).strip(),
                    "section": str(meta.get("section", "Unknown")).strip(),
                    "subsection": str(meta.get("subsection", "Unknown")).strip(),
                    "is_front_matter": bool(meta.get("is_front_matter", False)),
                    "model": str(res.get("model", config.GEMINI_EMBED_MODEL)),
                    "dimensions": int(res.get("dimensions", config.EMBED_OUTPUT_DIMENSIONALITY) or config.EMBED_OUTPUT_DIMENSIONALITY),
                    "elapsed_seconds": float(res.get("elapsed_seconds", 0.0) or 0.0),
                }],
                documents=[doc_summary]
            )
            existing_ids.add(page_id)
            file_embedded_this_run += 1
            overall_done += 1

            page_duration = time.monotonic() - page_start_time
            recent_page_durations.append(page_duration)

            remaining = max(0, total_pages_to_embed - overall_done)
            pct = round((overall_done / total_pages_to_embed) * 100, 1)
            avg_duration = sum(recent_page_durations[-10:]) / len(recent_page_durations[-10:])
            eta_sec = avg_duration * remaining
            eta_str = format_time_remaining(eta_sec)

            yield {
                "type": "page_complete",
                "filename": filename,
                "current_page": page_num,
                "total_file_pages": len(image_paths),
                "overall_total": total_pages_to_embed,
                "overall_done": overall_done,
                "overall_remaining": remaining,
                "percentage": pct,
                "eta_formatted": eta_str if remaining > 0 else "Almost done...",
                "page_duration": round(page_duration, 2),
                "message": f"✅ Indexed page {page_num}/{len(image_paths)} ({image_path.name}) in {page_duration:.2f}s"
            }

            time.sleep(config.EMBED_INTER_PAGE_DELAY)

        # Update pipeline state for this PDF if all pages are now indexed
        final_records = collection.get(where={"pdf_stem": stem}, include=[])
        final_count = len(final_records["ids"]) if final_records and final_records["ids"] else 0
        if final_count >= len(image_paths):
            state = load_pipeline_state()
            state[filename] = {"size": pdf_path.stat().st_size, "embedded_at": time.time()}
            save_pipeline_state(state)

        processed_summary.append({
            "filename": filename,
            "stem": stem,
            "total_pages": final_count,
            "newly_embedded": file_embedded_this_run
        })

    # 4. Final Completion event
    yield {
        "type": "complete",
        "message": f"Successfully processed {len(pdf_plans)} PDF(s) ({total_pages_to_embed} pages indexed).",
        "overall_total": total_pages_to_embed,
        "overall_done": total_pages_to_embed,
        "overall_remaining": 0,
        "percentage": 100.0,
        "eta_formatted": "Complete",
        "processed": processed_summary
    }


def process_and_embed_pdf(
    pdf_path: Path,
    log_fn: Optional[Callable[[str], None]] = None,
    mode: str = "required",
    force: bool = False
) -> dict:
    """
    Renders, embeds, and updates ChromaDB for a single PDF.
    Supports:
      - Option 1 (mode='all' / force=True): Re-embeds all pages from scratch.
      - Option 2 (mode='required' / force=False): Embeds only missing or modified pages.
    """
    def log(msg: str):
        logger.info(msg)
        if log_fn:
            log_fn(msg)

    # 1. Verify API key
    api_key = config.get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set or empty in .env.")

    is_full = (mode == "all" or force)
    mode_text = "Full Re-embed (All Pages)" if is_full else "Embed Required (Missing & Changed)"
    log(f"Processing '{pdf_path.name}' [{mode_text}] ({format_bytes(pdf_path.stat().st_size)})...")

    # 2. Extract outline
    log("Extracting native PDF outline metadata...")
    doc = fitz.open(pdf_path)
    metadata_map = build_pdf_metadata_map(doc)
    doc.close()

    # 3. Render pages
    log(f"Rendering pages at {config.RENDER_DPI} DPI...")
    image_paths = render_pdf_to_images(pdf_path, log_fn=log)

    # 4. Check existing indexed pages in ChromaDB for incremental sync
    collection = get_chroma_collection()
    existing_records = collection.get(where={"pdf_stem": pdf_path.stem}, include=["metadatas"])
    existing_ids = set(existing_records["ids"]) if existing_records and existing_records["ids"] else set()

    state = load_pipeline_state()
    last_embedded_at = state.get(pdf_path.name, {}).get("embedded_at", 0)
    last_size = state.get(pdf_path.name, {}).get("size")
    is_modified = bool(last_embedded_at and pdf_path.stat().st_mtime > last_embedded_at + 1.0) or (pdf_path.name in state and last_size != pdf_path.stat().st_size)

    pages_to_process = set()
    if is_full or is_modified:
        pages_to_process = set(range(1, len(image_paths) + 1))
    else:
        for p in range(1, len(image_paths) + 1):
            if f"{pdf_path.stem}_page_{p:03d}" not in existing_ids:
                pages_to_process.add(p)

    if not pages_to_process:
        log(f"  All {len(image_paths)} pages for '{pdf_path.name}' are already up-to-date in ChromaDB.")
        state[pdf_path.name] = {"size": pdf_path.stat().st_size, "embedded_at": time.time()}
        save_pipeline_state(state)
        return {
            "filename": pdf_path.name,
            "stem": pdf_path.stem,
            "pages_embedded": len(existing_ids),
            "newly_embedded": 0
        }

    # 5. Embed each page with automatic retry until success and immediate upsert
    log(f"Embedding {len(pages_to_process)} pages with {config.GEMINI_EMBED_MODEL} (resilient retry enabled)...")
    client = embedder.get_client()

    newly_embedded_count = 0
    total_pages = len(image_paths)

    for i, image_path in enumerate(image_paths, start=1):
        page_num = i
        page_id = f"{pdf_path.stem}_page_{page_num:03d}"

        if page_num not in pages_to_process:
            log(f"  Page {i}/{total_pages} ({image_path.name}) already indexed. Skipping.")
            continue

        log(f"  Embedding page {i}/{total_pages}: {image_path.name}...")
        # embed_page_image will automatically retry with exponential backoff on 429/quota limits
        res = embedder.embed_page_image(client, image_path, log_fn=log)

        # Immediate upsert into ChromaDB
        meta = metadata_map.get(page_num, {})
        doc_summary = (
            f"{pdf_path.stem} - Page {page_num:03d} | "
            f"Chapter: {meta.get('chapter', 'Unknown')} | "
            f"Section: {meta.get('section', 'Unknown')}"
        )
        collection.upsert(
            ids=[page_id],
            embeddings=[res["vector"]],
            metadatas=[{
                "pdf_stem": pdf_path.stem,
                "page_image": res["page_image"],
                "page_number": page_num,
                "chapter": str(meta.get("chapter", "Unknown")).strip(),
                "section": str(meta.get("section", "Unknown")).strip(),
                "subsection": str(meta.get("subsection", "Unknown")).strip(),
                "is_front_matter": bool(meta.get("is_front_matter", False)),
                "model": str(res.get("model", config.GEMINI_EMBED_MODEL)),
                "dimensions": int(res.get("dimensions", config.EMBED_OUTPUT_DIMENSIONALITY) or config.EMBED_OUTPUT_DIMENSIONALITY),
                "elapsed_seconds": float(res.get("elapsed_seconds", 0.0) or 0.0),
            }],
            documents=[doc_summary]
        )
        existing_ids.add(page_id)
        newly_embedded_count += 1

        # Pacing delay to reduce rate limit pressure
        time.sleep(config.EMBED_INTER_PAGE_DELAY)

    # 6. Verify total pages in ChromaDB
    final_records = collection.get(where={"pdf_stem": pdf_path.stem}, include=[])
    final_count = len(final_records["ids"]) if final_records and final_records["ids"] else 0

    # 7. Update pipeline state only if all pages are successfully stored
    if final_count >= len(image_paths):
        state = load_pipeline_state()
        state[pdf_path.name] = {"size": pdf_path.stat().st_size, "embedded_at": time.time()}
        save_pipeline_state(state)

    log(f"✅ Finished '{pdf_path.name}': {final_count}/{len(image_paths)} pages verified in ChromaDB ({newly_embedded_count} embedded this run).")
    return {
        "filename": pdf_path.name,
        "stem": pdf_path.stem,
        "pages_embedded": final_count,
        "newly_embedded": newly_embedded_count
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


# ============================================================================
# Page & Metadata Inspector, Tag Editor & Single-Page Sync
# ============================================================================

CUSTOM_METADATA_FILE = config.OUTPUT_DIR / "custom_page_metadata.json"


def load_custom_metadata() -> dict:
    if not CUSTOM_METADATA_FILE.exists():
        return {}
    try:
        with CUSTOM_METADATA_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading custom metadata: {e}")
        return {}


def save_custom_metadata(data: dict) -> None:
    try:
        CUSTOM_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with CUSTOM_METADATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving custom metadata: {e}")


def get_pdf_pages_detail(filename: str) -> dict:
    """
    Returns detailed inspection data for all pages in a given PDF manual.
    Includes PNG image URLs, indexed status in ChromaDB, current metadata tags, and document summaries.
    """
    pdf_path = config.SOURCE_DIR / filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF manual '{filename}' not found in source directory.")

    stem = pdf_path.stem
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    outline_map = build_pdf_metadata_map(doc)
    doc.close()

    # Ensure page images are rendered for preview
    image_paths = render_pdf_to_images(pdf_path)

    # Query existing records from ChromaDB
    collection = get_chroma_collection()
    db_records_map = {}
    try:
        records = collection.get(where={"pdf_stem": stem}, include=["metadatas", "documents"])
        if records and records["ids"]:
            for i, pid in enumerate(records["ids"]):
                meta = records["metadatas"][i] if records["metadatas"] else {}
                doc_text = records["documents"][i] if records["documents"] else ""
                db_records_map[pid] = {
                    "metadata": meta,
                    "document": doc_text
                }
    except Exception as e:
        logger.error(f"Error fetching ChromaDB records for {stem}: {e}")

    custom_meta_store = load_custom_metadata().get(filename, {})

    pages_list = []
    for p in range(1, total_pages + 1):
        page_id = f"{stem}_page_{p:03d}"
        image_name = f"{stem}_page_{p:03d}.png"
        img_file = config.IMAGE_CACHE_DIR / image_name

        is_indexed = page_id in db_records_map
        db_item = db_records_map.get(page_id, {})
        
        # Merge metadata: ChromaDB meta (if indexed) OR Outline TOC meta + Custom overrides
        default_meta = outline_map.get(p, {})
        page_meta = {}

        if is_indexed:
            page_meta = dict(db_item.get("metadata", {}))
        else:
            page_meta = {
                "pdf_stem": stem,
                "page_image": image_name,
                "page_number": p,
                "chapter": str(default_meta.get("chapter", "Unknown")).strip(),
                "section": str(default_meta.get("section", "Unknown")).strip(),
                "subsection": str(default_meta.get("subsection", "Unknown")).strip(),
                "is_front_matter": bool(default_meta.get("is_front_matter", False)),
            }

        # Apply any saved custom metadata overrides
        if str(p) in custom_meta_store:
            page_meta.update(custom_meta_store[str(p)])

        doc_summary = db_item.get("document") or (
            f"{stem} - Page {p:03d} | "
            f"Chapter: {page_meta.get('chapter', 'Unknown')} | "
            f"Section: {page_meta.get('section', 'Unknown')}"
        )

        pages_list.append({
            "page_number": p,
            "page_id": page_id,
            "image_name": image_name,
            "image_url": f"/api/admin/files/page-image/{image_name}",
            "image_exists": img_file.exists(),
            "is_indexed": is_indexed,
            "metadata": page_meta,
            "document": doc_summary,
            "model": page_meta.get("model", config.GEMINI_EMBED_MODEL),
            "dimensions": page_meta.get("dimensions", config.EMBED_OUTPUT_DIMENSIONALITY)
        })

    indexed_count = sum(1 for pg in pages_list if pg["is_indexed"])

    return {
        "filename": filename,
        "stem": stem,
        "total_pages": total_pages,
        "indexed_count": indexed_count,
        "missing_count": total_pages - indexed_count,
        "pages": pages_list
    }


def update_page_metadata(filename: str, page_number: int, new_metadata: dict, document_text: Optional[str] = None) -> dict:
    """
    Updates or deletes metatags for a specific page in ChromaDB and persists custom overrides.
    """
    pdf_path = config.SOURCE_DIR / filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF manual '{filename}' not found.")

    stem = pdf_path.stem
    page_id = f"{stem}_page_{page_number:03d}"
    image_name = f"{stem}_page_{page_number:03d}.png"

    # Sanitize metadata types for ChromaDB (str, int, float, bool)
    cleaned_meta = {}
    for k, v in new_metadata.items():
        if v is None:
            continue
        k_clean = str(k).strip()
        if isinstance(v, bool):
            cleaned_meta[k_clean] = v
        elif isinstance(v, (int, float)):
            cleaned_meta[k_clean] = v
        else:
            cleaned_meta[k_clean] = str(v).strip()

    cleaned_meta["pdf_stem"] = stem
    cleaned_meta["page_number"] = page_number
    cleaned_meta["page_image"] = image_name

    # 1. Update in ChromaDB if currently indexed
    collection = get_chroma_collection()
    is_indexed = False
    try:
        existing = collection.get(ids=[page_id], include=["metadatas"])
        if existing and existing["ids"]:
            is_indexed = True
            doc_update = [str(document_text).strip()] if document_text else None
            collection.update(
                ids=[page_id],
                metadatas=[cleaned_meta],
                documents=doc_update
            )
            logger.info(f"Updated ChromaDB metadata for {page_id}")
    except Exception as e:
        logger.error(f"Error updating ChromaDB for {page_id}: {e}")

    # 2. Persist in custom metadata store
    store = load_custom_metadata()
    if filename not in store:
        store[filename] = {}
    store[filename][str(page_number)] = cleaned_meta
    save_custom_metadata(store)

    return {
        "status": "ok",
        "page_id": page_id,
        "page_number": page_number,
        "is_indexed": is_indexed,
        "metadata": cleaned_meta,
        "message": f"Successfully updated metadata for Page {page_number}."
    }


def delete_page_from_chroma(filename: str, page_number: int) -> dict:
    """
    Deletes an unneeded or irrelevant page from ChromaDB vector collection.
    """
    pdf_path = config.SOURCE_DIR / filename
    stem = pdf_path.stem
    page_id = f"{stem}_page_{page_number:03d}"

    collection = get_chroma_collection()
    deleted = False
    try:
        existing = collection.get(ids=[page_id], include=[])
        if existing and existing["ids"]:
            collection.delete(ids=[page_id])
            deleted = True
            logger.info(f"Deleted {page_id} from ChromaDB.")
    except Exception as e:
        logger.error(f"Error deleting {page_id} from ChromaDB: {e}")
        raise

    # Also update pipeline state if indexed count decreases below total
    state = load_pipeline_state()
    if filename in state:
        # Check remaining count
        remaining = collection.get(where={"pdf_stem": stem}, include=[])
        doc = fitz.open(pdf_path)
        total = len(doc)
        doc.close()
        if len(remaining["ids"]) < total:
            # Not fully embedded anymore
            pass

    return {
        "status": "ok",
        "deleted": deleted,
        "page_id": page_id,
        "page_number": page_number,
        "message": f"Removed Page {page_number} from vector database."
    }


def embed_single_page_and_sync(filename: str, page_number: int, custom_metadata: Optional[dict] = None) -> dict:
    """
    Renders, embeds, and syncs a single page into ChromaDB without having to embed the whole document.
    """
    pdf_path = config.SOURCE_DIR / filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF manual '{filename}' not found.")

    stem = pdf_path.stem
    page_id = f"{stem}_page_{page_number:03d}"
    image_name = f"{stem}_page_{page_number:03d}.png"
    image_path = config.IMAGE_CACHE_DIR / image_name

    # 1. Verify API Key
    api_key = config.get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    # 2. Render page if needed
    doc = fitz.open(pdf_path)
    if page_number > len(doc) or page_number < 1:
        doc.close()
        raise ValueError(f"Page number {page_number} is out of bounds (1-{len(doc)}).")

    zoom = config.RENDER_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)
    page = doc[page_number - 1]
    pix = page.get_pixmap(matrix=matrix)
    if max(pix.width, pix.height) > config.MAX_IMAGE_DIMENSION:
        scale = config.MAX_IMAGE_DIMENSION / max(pix.width, pix.height)
        new_matrix = fitz.Matrix(zoom * scale, zoom * scale)
        pix = page.get_pixmap(matrix=new_matrix)
    pix.save(image_path)

    # 3. Extract outline metadata
    outline_map = build_pdf_metadata_map(doc)
    doc.close()
    default_meta = outline_map.get(page_number, {})

    # 4. Merge metadata
    final_meta = {
        "pdf_stem": stem,
        "page_image": image_name,
        "page_number": page_number,
        "chapter": str(default_meta.get("chapter", "Unknown")).strip(),
        "section": str(default_meta.get("section", "Unknown")).strip(),
        "subsection": str(default_meta.get("subsection", "Unknown")).strip(),
        "is_front_matter": bool(default_meta.get("is_front_matter", False)),
    }

    # Load custom overrides from store or argument
    store = load_custom_metadata().get(filename, {})
    if str(page_number) in store:
        final_meta.update(store[str(page_number)])
    if custom_metadata:
        for k, v in custom_metadata.items():
            if v is not None:
                final_meta[str(k).strip()] = v

    # 5. Generate Multimodal Embedding with resilient backoff
    client = embedder.get_client()
    res = embedder.embed_page_image(client, image_path)

    final_meta["model"] = str(res.get("model", config.GEMINI_EMBED_MODEL))
    final_meta["dimensions"] = int(res.get("dimensions", config.EMBED_OUTPUT_DIMENSIONALITY) or config.EMBED_OUTPUT_DIMENSIONALITY)
    final_meta["elapsed_seconds"] = float(res.get("elapsed_seconds", 0.0) or 0.0)

    # 6. Upsert into ChromaDB
    doc_summary = (
        f"{stem} - Page {page_number:03d} | "
        f"Chapter: {final_meta.get('chapter', 'Unknown')} | "
        f"Section: {final_meta.get('section', 'Unknown')}"
    )

    collection = get_chroma_collection()
    collection.upsert(
        ids=[page_id],
        embeddings=[res["vector"]],
        metadatas=[final_meta],
        documents=[doc_summary]
    )

    # 7. Check if all pages are now embedded to update pipeline state
    final_records = collection.get(where={"pdf_stem": stem}, include=[])
    final_count = len(final_records["ids"]) if final_records and final_records["ids"] else 0

    doc_full = fitz.open(pdf_path)
    total_doc_pages = len(doc_full)
    doc_full.close()

    if final_count >= total_doc_pages:
        state = load_pipeline_state()
        state[filename] = {"size": pdf_path.stat().st_size, "embedded_at": time.time()}
        save_pipeline_state(state)

    return {
        "status": "ok",
        "page_id": page_id,
        "page_number": page_number,
        "image_url": f"/api/admin/files/page-image/{image_name}",
        "metadata": final_meta,
        "document": doc_summary,
        "elapsed_seconds": res.get("elapsed_seconds", 0.0),
        "message": f"✅ Page {page_number} successfully embedded and synced into ChromaDB!"
    }


def batch_pages_action(filename: str, action: str, page_numbers: list[int]) -> dict:
    """
    Performs batch operations ('embed' or 'delete') on selected pages.
    """
    if not page_numbers:
        return {"status": "ok", "processed": 0, "message": "No pages selected."}

    results = []
    errors = []

    for p in page_numbers:
        try:
            if action == "delete":
                r = delete_page_from_chroma(filename, p)
                results.append(r)
            elif action == "embed":
                r = embed_single_page_and_sync(filename, p)
                results.append(r)
            else:
                raise ValueError(f"Unknown batch action: {action}")
        except Exception as e:
            errors.append({"page_number": p, "error": str(e)})

    return {
        "status": "ok" if not errors else "partial",
        "action": action,
        "total_selected": len(page_numbers),
        "successful_count": len(results),
        "errors": errors,
        "message": f"Batch {action} completed: {len(results)}/{len(page_numbers)} pages processed."
    }


def update_env_config(api_key: Optional[str] = None, qa_model: Optional[str] = None, admin_password: Optional[str] = None) -> dict:
    """Updates .env file with new API key, QA model, and/or Admin Password."""
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
    if admin_password is not None and admin_password.strip():
        env_dict["ADMIN_PASSWORD"] = admin_password.strip()

    with env_path.open("w", encoding="utf-8") as f:
        for k, v in env_dict.items():
            f.write(f"{k}={v}\n")

    # Reload environment
    load_dotenv(dotenv_path=env_path, override=True)
    if "GEMINI_QA_MODEL" in env_dict:
        config.GEMINI_QA_MODEL = env_dict["GEMINI_QA_MODEL"]
    if "ADMIN_PASSWORD" in env_dict:
        config.ADMIN_PASSWORD = env_dict["ADMIN_PASSWORD"]

    return {
        "status": "ok",
        "api_key_set": bool(env_dict.get("GEMINI_API_KEY")),
        "qa_model": config.GEMINI_QA_MODEL,
        "admin_password_set": bool(env_dict.get("ADMIN_PASSWORD"))
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
