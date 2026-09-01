# RAG Extraction/Embedding Pipeline — Gemini Embedding 2 + ChromaDB

Uses **Gemini Embedding 2**, Google's multimodal embedding model, to embed
PDF page images directly into a shared vector space with text, storing vectors
and outline metadata in **ChromaDB**.

## 1. Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your Gemini API key to `.env`:
```bash
# In .env:
GEMINI_API_KEY=your-gemini-api-key-here
```
All scripts and the Flask application automatically load and reload the key directly from `.env`. No manual `export` required!

## 2. Add your source PDFs

Drop PDF manuals into `source/`. Multiple PDFs are supported simultaneously.

## 3. Run the pipeline

```bash
python run_embedding_pipeline.py
```

This:
1. Parses native PDF Table of Contents (TOC) to map chapters/sections and flag front-matter.
2. Renders pages to high-resolution images in `output/rendered_pages/`.
3. Embeds each page image with `gemini-embedding-2` (3072 dimensions).
4. Stores/upserts vectors, metadata, and page IDs directly into **ChromaDB** (`output/chroma_db/`).

## 4. Run Retrieval & Chatbot

### Interactive Web Chatbot (Flask)
```bash
python app.py
```
Open your browser at `http://localhost:5000`. Features:
- Modern Tailwind CSS responsive interface with Markdown parsing and code blocks.
- Real-time ChromaDB status inspection.
- Native outline-aware retrieval with chapter-bounded neighbor expansion ($\pm 3$ pages).
- Visual seed citation thumbnails with full-screen image inspection Lightbox.
- Grounded generation with `gemini-3.5-flash`.

### CLI Query Test
```bash
python query_test.py "how do I calibrate the sensor array"
```
Queries ChromaDB, ranks stored pages by cosine similarity, and prints top matches with chapter/section metadata.

## File Overview

```
config.py                          paths, model names, instructions, ChromaDB config
embedders/gemini_multimodal_embedder.py   embeds page images + text queries via Google GenAI
run_embedding_pipeline.py          entry point: render + embed all PDFs into ChromaDB
query_test.py                      CLI diagnostic tool for ChromaDB retrieval
app.py                             Flask web application & REST API
templates/index.html               Flask web UI template (Tailwind, Marked.js, Lightbox)
source/                            drop source PDF manuals here
output/
├── chroma_db/                     persistent ChromaDB vector index and metadata
├── rendered_pages/                rendered PNG page images
└── pipeline_state.json            incremental state tracking for PDF processing
```
