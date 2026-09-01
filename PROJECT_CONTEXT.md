# Project Context & Technical Reference: Multimodal PDF RAG

> **Purpose:** This document serves as the persistent cross-session architectural and operational reference for the `df_rag_project` codebase.

---

## 1. Executive Summary

`df_rag_project` is a **Vision-First Multimodal Retrieval-Augmented Generation (RAG)** system designed specifically for complex technical documentation (robotics manuals, hardware deployment handbooks, and software user guides such as NavWiz and DFleet).

### Core Paradigm: Direct Multimodal Embeddings + ChromaDB
Instead of the traditional two-stage text RAG pipeline (PDF → OCR/Text Extraction → Text Embedding), this system renders PDF pages directly into high-resolution images and embeds them into a shared vector space using Google's **`gemini-embedding-2`** model, persisting embeddings and document topology into **ChromaDB**.

```
                    ┌─────────────────────────┐
                    │    Source PDF Manuals   │
                    └───────────┬─────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
   ┌───────────────────────────┐ ┌───────────────────────────┐
   │  PDF Outline / TOC Parse  │ │  High-Res Page Rendering  │
   │  (Chapter/Section Meta)   │ │    (PyMuPDF 200 DPI PNG)  │
   └─────────────┬─────────────┘ └─────────────┬─────────────┘
                 │                             │
                 │               ┌─────────────┘
                 │               ▼
                 │ ┌───────────────────────────┐
                 │ │  gemini-embedding-2 (Doc) │
                 │ │ (Direct Image -> 3072-dim)│
                 │ └─────────────┬─────────────┘
                 ▼               ▼
   ┌───────────────────────────────────────────┐
   │        ChromaDB Vector Collection         │
   │   (output/chroma_db/ - 3072 dims, cosine) │
   └─────────────────────┬─────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌───────────────────────┐     ┌───────────────────────┐
│   Flask App (app.py)  │     │  CLI Diagnostic Tool  │
│ • Tailwind UI / API   │     │  (query_test.py)      │
│ • ChromaDB Retrieval  │     │  • ChromaDB Query     │
│ • Chapter Expansion   │     │  • Cosine Similarity  │
│ • gemini-3.5-flash QA │     └───────────────────────┘
└───────────────────────┘
```

### Key Advantages
1. **Preserves Visual Semantics:** Accurately indexes complex layouts, schematics, UI screenshots, tables, wiring diagrams, and flowcharts that break typical OCR extractors.
2. **Persistent Vector Database (ChromaDB):** Replaces flat `.npy`/`.jsonl` arrays with indexed HNSW vector storage, native metadata querying, and unified multi-PDF aggregation.
3. **Native Outline-Aware Filtering & Expansion:** Uses document outline hierarchy (TOC) to filter out non-informative front-matter (covers, copyright, TOCs) and expands retrieved seed pages across neighboring pages strictly bounded within the same chapter.
4. **Single-Hop Embedding:** Eliminates text extraction artifacts, caption hallucination, and multi-model pipeline latency.

---

## 2. Codebase Architecture & File Map

```
/home/tinonn/df_rag_project/
├── config.py                       # Global configuration, hyperparameters, instructions, ChromaDB paths
├── run_embedding_pipeline.py       # Batch pipeline: TOC parsing, page rendering, ChromaDB upsert
├── query_test.py                   # CLI diagnostic tool to query ChromaDB and rank pages
├── app.py                          # Flask web application & REST API
├── templates/
│   └── index.html                  # Responsive frontend UI (Tailwind CSS, Marked.js, Lightbox)
├── requirements.txt                # Python package dependencies (google-genai, PyMuPDF, chromadb, flask, etc.)
├── .env                            # Environment secrets (GEMINI_API_KEY)
├── embedders/
│   ├── __init__.py
│   └── gemini_multimodal_embedder.py # Google GenAI SDK wrapper for image & query embeddings
├── source/                         # Input directory for source PDF documents
│   ├── Copy of Field Deployment Handbook.pdf
│   ├── DFleet 4.0 User Manual.pdf
│   └── NavWiz 4.0 User Manual 1.0.pdf
└── output/                         # Generated artifacts and vector index
    ├── chroma_db/                  # Persistent ChromaDB vector database (SQLite + HNSW index)
    ├── pipeline_state.json         # State tracking for incremental processing
    └── rendered_pages/             # Cached PNG renderings of all PDF pages
        └── <pdf_stem>_page_<num>.png
```

### Detailed Component Roles

| File | Primary Role & Responsibilities |
|---|---|
| [`config.py`](file:///home/tinonn/df_rag_project/config.py) | Defines all system paths (`SOURCE_DIR`, `OUTPUT_DIR`, `IMAGE_CACHE_DIR`, `CHROMA_DIR`), collection name (`CHROMA_COLLECTION_NAME="pdf_pages"`), rendering specs (`RENDER_DPI=200`, `MAX_IMAGE_DIMENSION=2000`), model parameters (`GEMINI_EMBED_MODEL="gemini-embedding-2"`, `EMBED_OUTPUT_DIMENSIONALITY=3072`), and asymmetric embedding instructions. |
| [`run_embedding_pipeline.py`](file:///home/tinonn/df_rag_project/run_embedding_pipeline.py) | Iterates over `source/*.pdf`, checks `pipeline_state.json` to skip unchanged files, extracts native document outline (TOC) with PyMuPDF, renders pages to PNG, calls the Gemini embedder, and upserts vectors + metadata into ChromaDB. |
| [`embedders/gemini_multimodal_embedder.py`](file:///home/tinonn/df_rag_project/embedders/gemini_multimodal_embedder.py) | Interacts with `google.genai.Client`. Provides `embed_page_image()` for PNG byte payloads and `embed_query_text()` for plain-text search queries, handling latency measurement and error reporting. |
| [`app.py`](file:///home/tinonn/df_rag_project/app.py) | Flask web application serving both the HTML interface and REST API endpoints (`/api/status`, `/api/chat`, `/rendered_pages/<filename>`). Connects to ChromaDB, executes outline-aware retrieval with chapter-bounded expansion, and generates answers using `gemini-3.5-flash`. |
| [`templates/index.html`](file:///home/tinonn/df_rag_project/templates/index.html) | Interactive single-page web UI featuring real-time index status, chat history, Markdown formatting, citation cards with similarity scores, and a high-resolution image modal lightbox. |
| [`query_test.py`](file:///home/tinonn/df_rag_project/query_test.py) | Standalone CLI utility for validating retrieval quality against ChromaDB and inspecting top-ranked page image filenames and chapters. |

---

## 3. ChromaDB Schema & Data Structures

### 3.1 Collection Configuration
- **Collection Name:** `pdf_pages`
- **Metric:** Cosine similarity (`metadata={"hnsw:space": "cosine"}`)
- **Persistence Path:** `output/chroma_db/`

### 3.2 Record Structure in ChromaDB
| Field | Type | Description / Example |
|---|---|---|
| **ID** | `str` | Format: `{pdf_stem}_page_{page_num:03d}` (e.g. `DFleet 4.0 User Manual_page_008`) |
| **Embedding** | `list[float]` | 3072-dimensional vector from `gemini-embedding-2` |
| **Document** | `str` | Text summary: `DFleet 4.0 User Manual - Page 008 \| Chapter: Preface \| Section: Section 1: Introduction` |
| **Metadata** | `dict` | Key/Value attributes: <br>• `pdf_stem`: `"DFleet 4.0 User Manual"`<br>• `page_image`: `"DFleet 4.0 User Manual_page_008.png"`<br>• `page_number`: `8`<br>• `chapter`: `"Preface"`<br>• `section`: `"Section 1: Introduction"`<br>• `subsection`: `"Unknown"`<br>• `is_front_matter`: `false`<br>• `model`: `"gemini-embedding-2"`<br>• `dimensions`: `3072`<br>• `elapsed_seconds`: `1.24` |

---

## 4. Key Retrieval & Generation Mechanics

### Asymmetric Retrieval Instructions
`gemini-embedding-2` uses instruction strings to bridge query and document representation spaces:
- **Document instruction:** `"This is a page from a technical robotics software user manual. It may contain prose, tables, UI screenshots, or diagrams with embedded text. Represent its full content for retrieval by a user's question about the manual."`
- **Query instruction:** `"Represent this question for retrieving the most relevant manual page."`

### Front-Matter Filtering in ChromaDB
Pages classified under `"Copyright Notice"`, `"Table of Contents"`, or `"Unknown"` before the first real chapter are marked with `"is_front_matter": true`. ChromaDB's native metadata query filters these out:
```python
results = collection.query(
    query_embeddings=[qvec],
    n_results=top_k,
    where={"is_front_matter": False},
    include=["metadatas", "distances", "documents"]
)
```

### Chapter-Bounded Neighbor Expansion
A single page often contains only part of a procedure. `app.py` employs a window expansion strategy:
1. Identify Top-$K$ seed pages via ChromaDB query.
2. For each seed page, look outward up to radius $R = 3$ ($[-3, +3]$ pages).
3. Query neighbor metadata via `collection.get(ids=[neighbor_id], include=["metadatas"])`.
4. If a neighbor page belongs to a different chapter or is front-matter, **halt expansion in that direction immediately**.
5. Pass the deduplicated, ordered sequence of page images into the generation context.

### Grounding & Hallucination Guardrail Prompt
`app.py` enforces strict context adherence on `gemini-3.5-flash`:
```text
You are an expert technical assistant for the NavWiz AMR software manual.
Answer the user's question accurately using only the provided image pages as context.

STRICT RULE: If the user's question is gibberish, meaningless text, completely unrelated to
robotics/NavWiz software, or cannot be answered by the provided manual pages, respond EXACTLY with:
"I am not sure about that."
```

---

## 5. Environment & Operational Runbook

### 5.1 Environment Prerequisites
- Python 3.10+
- Installed packages: `google-genai`, `PyMuPDF`, `chromadb`, `numpy`, `flask`, `Pillow`, `python-dotenv`
- `GEMINI_API_KEY` configured in `.env` (automatically loaded and refreshed by `config.get_gemini_api_key()` with `override=True`)

### 5.2 Common Workflows

#### A. Ingesting New or Updated PDFs
1. Place PDF file(s) in `source/`.
2. Run the embedding pipeline:
   ```bash
   python run_embedding_pipeline.py
   ```
3. New images are rendered to `output/rendered_pages/` and upserted to `output/chroma_db/`.

#### B. Running Retrieval Sanity Checks (CLI)
```bash
python query_test.py "How do I configure safety zones?"
```

#### C. Launching the Interactive Web Chatbot (Flask)
```bash
python app.py
```
Open `http://localhost:5000` in your web browser.

