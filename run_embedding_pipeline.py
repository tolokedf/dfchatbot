"""
Renders pages, extracts native PDF outline metadata, and embeds all PDFs.
"""
from __future__ import annotations

import json
from pathlib import Path
import fitz
import chromadb

import config
from embedders import gemini_multimodal_embedder as embedder

STATE_FILE = config.OUTPUT_DIR / "pipeline_state.json"

def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def load_state() -> dict:
    if STATE_FILE.exists():
        with STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(state: dict) -> None:
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

def build_pdf_metadata_map(doc: fitz.Document) -> dict:
    """Parses the native PDF outline (TOC) to map pages to chapters and sections."""
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
                    
        # Identify non-informative setup pages to filter them out of seed retrieval
        current_ch = active_hier[1].strip()
        is_front = (current_ch in ["Copyright Notice", "Table of Contents", "Unknown", "Cover"])
        
        page_meta[p] = {
            "chapter": active_hier[1].strip(),
            "section": active_hier[2].strip(),
            "subsection": active_hier[3].strip(),
            "is_front_matter": is_front
        }
        
    return page_meta

def render_pdf_to_images_namespaced(pdf_path: Path) -> list[Path]:
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
            print(f"    rendered {out_path.name}")
            
        image_paths.append(out_path)

    doc.close()
    return image_paths

def save_outputs_for_pdf(pdf_stem: str, results: list[dict], metadata_map: dict) -> None:
    collection = get_chroma_collection()
    
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for idx, r in enumerate(results):
        if r.get("vector") is None:
            continue
        page_num = idx + 1
        meta = metadata_map.get(page_num, {})
        page_id = f"{pdf_stem}_page_{page_num:03d}"
        
        ids.append(page_id)
        embeddings.append(r["vector"])
        metadatas.append({
            "pdf_stem": pdf_stem,
            "page_image": r["page_image"],
            "page_number": page_num,
            "chapter": str(meta.get("chapter", "Unknown")),
            "section": str(meta.get("section", "Unknown")),
            "subsection": str(meta.get("subsection", "Unknown")),
            "is_front_matter": bool(meta.get("is_front_matter", False)),
            "model": str(r.get("model", config.GEMINI_EMBED_MODEL)),
            "dimensions": int(r.get("dimensions", config.EMBED_OUTPUT_DIMENSIONALITY) or config.EMBED_OUTPUT_DIMENSIONALITY),
            "elapsed_seconds": float(r.get("elapsed_seconds", 0.0) or 0.0),
        })
        documents.append(
            f"{pdf_stem} - Page {page_num:03d} | Chapter: {meta.get('chapter', 'Unknown')} | Section: {meta.get('section', 'Unknown')}"
        )

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        print(f"    upserted {len(ids)} pages into ChromaDB collection '{config.CHROMA_COLLECTION_NAME}'")

def main() -> None:
    state = load_state()
    pdfs = sorted(config.SOURCE_DIR.glob("*.pdf"))
    
    if not pdfs:
        print(f"No PDFs found in {config.SOURCE_DIR}")
        return
        
    print(f"Found {len(pdfs)} PDF(s) in {config.SOURCE_DIR}\n")

    for pdf_path in pdfs:
        file_size = pdf_path.stat().st_size
        
        if pdf_path.name in state and state[pdf_path.name]["size"] == file_size:
            print(f"⏭️  Skipping '{pdf_path.name}' (Size unchanged)")
            continue
            
        print(f"⚙️  Processing '{pdf_path.name}'...")
        
        print("  Extracting native PDF outline metadata...")
        doc = fitz.open(pdf_path)
        metadata_map = build_pdf_metadata_map(doc)
        doc.close()
        
        print("  Rendering pages to images...")
        image_paths = render_pdf_to_images_namespaced(pdf_path)
        
        print(f"  Embedding {len(image_paths)} page(s) with {config.GEMINI_EMBED_MODEL}...")
        results = embedder.embed_all_pages(image_paths)
        
        print("  Saving outputs...")
        save_outputs_for_pdf(pdf_path.stem, results, metadata_map)
        
        state[pdf_path.name] = {"size": file_size}
        save_state(state)
        print()

    print("Pipeline complete.")

if __name__ == "__main__":
    main()