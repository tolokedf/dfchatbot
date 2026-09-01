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

## 4. Run Web Application & Admin Console

### Launch the Flask App
```bash
python app.py
```
Open your browser at `http://localhost:5000`:

- **💬 Chat Assistant (`/`):**
  - Modern Tailwind CSS responsive interface with Markdown parsing and code blocks.
  - **Manual Filtering:** Select a specific manual or search across all manuals simultaneously.
  - **Chapter-Bounded Expansion:** Automatically expands context up to $\pm 3$ pages within logical chapter boundaries.
  - **Visual Seed Cards & Lightbox:** Click any citation card to open high-resolution page inspection.
  - **Conversation Export:** Download your chat session directly as a formatted Markdown transcript (`.md`).
  - **Grounded Generation:** Strict anti-hallucination guardrails powered by `gemini-3.5-flash-lite`.

- **🛠️ Admin Console (`/admin`):**
  - **PDF Source Management:** View, upload new PDFs, preview, or remove source manuals in `source/`.
  - **Live Embedding Triggers:** Run embedding directly from the UI with real-time status and error reporting (detects rate limits, missing keys, invalid files).
  - **API Key & Model Manager:** Update `GEMINI_API_KEY` and switch QA models (`gemini-3.5-flash-lite`, `gemini-3.5-flash`) securely into `.env` without seeing code.
  - **API Connection Tester:** Instant one-click ping to verify Google Gemini API connectivity.
  - **Vector DB Maintenance:** Inspect topology and reset ChromaDB collection if needed.

### CLI Tools
```bash
# Diagnostic query test
python query_test.py "how do I calibrate the sensor array"

# Multi-threaded concurrent stress test
python stress_test.py --requests 10 --concurrency 3
```

## File Overview

```
config.py                          paths, model names, instructions, ChromaDB config
pipeline_service.py                service layer for PDF ingestion, ChromaDB sync & .env config
embedders/gemini_multimodal_embedder.py   embeds page images + text queries via Google GenAI
run_embedding_pipeline.py          CLI entry point: render + embed all PDFs into ChromaDB
query_test.py                      CLI diagnostic tool for ChromaDB retrieval
stress_test.py                     automated multi-threaded load and sanity testing suite
app.py                             Flask web application & REST APIs (/ and /admin)
templates/index.html               Chat Assistant web UI template (Tailwind, Marked.js, Lightbox)
templates/admin.html               Admin Management Console web UI template
source/                            source PDF manuals
output/chroma_db/                  persistent ChromaDB vector database
output/rendered_pages/             cached PNG page renderings
```
output/
├── chroma_db/                     persistent ChromaDB vector index and metadata
├── rendered_pages/                rendered PNG page images
└── pipeline_state.json            incremental state tracking for PDF processing
```
