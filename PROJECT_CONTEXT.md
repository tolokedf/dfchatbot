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
│ • gemini-3.5-flash-lite QA │ └───────────────────────┘
└───────────────────────┘
```

### Key Advantages
1. **Preserves Visual Semantics:** Accurately indexes complex layouts, schematics, UI screenshots, tables, wiring diagrams, and flowcharts that break typical OCR extractors.
2. **Persistent Vector Database (ChromaDB):** Replaces flat `.npy`/`.jsonl` arrays with indexed HNSW vector storage, native metadata querying, and unified multi-PDF aggregation.
3. **Native Outline-Aware Filtering & Expansion:** Uses document outline hierarchy (TOC) to filter out non-informative front-matter (covers, copyright, TOCs) and expands retrieved seed pages across neighboring pages strictly bounded within the same chapter.
4. **Single-Hop Embedding:** Eliminates text extraction artifacts, caption hallucination, and multi-model pipeline latency.
### 5. Multi-Mode Embedding Engine & Differential Sync
- **Option 1: Embed Required Only (`mode="required"`, Default/Recommended)**:
  - Smart differential sync:
    1. Scans ChromaDB for any unindexed / lost page IDs (`{stem}_page_XXX`).
    2. Detects if the source PDF was modified/edited after its last embedding timestamp or file size changed.
    3. Renders only the necessary images and embeds **only the specific missing or modified pages** into ChromaDB.
    4. Fast, highly resilient, and avoids burning Gemini multimodal API rate limits.
- **Option 2: Re-embed All Pages (`mode="all"`, Full Overwrite)**:
  - Forces a complete re-rendering and re-embedding of all pages from page 1 to $N$, completely updating and overwriting existing vector records in ChromaDB.
- **Page & Metadata Inspector (`#pageManagerModal`)**:
  - View all high-res page PNGs alongside current metadata tags and ChromaDB indexed states.
  - Delete unneeded or irrelevant pages directly from ChromaDB.
  - Add new custom key-value metatags or delete old metatags.
  - Sync & embed individual pages or batch-selected pages without having to re-embed the whole document.
- **Resilient Per-Page Retry & Backoff**: Catches HTTP 429 quota exhaustion with exponential backoff ($5.0\text{s} \times 1.5$ factor up to $60\text{s} + \text{jitter}$) up to 20 retries per page.
- **Interactive UI**: Real-time SSE streaming progress bar, live ETA, page counters, and a green "Done" button.

### 6. User Authentication, Chat Tabs & Conversational Memory
- **Dedicated User Database Folder (`User database/`)**:
  - `User database/users_and_chats.db`: SQLite database storing user credentials, chat tabs, and multi-turn message histories.
  - `User database/profile_pictures/`: Stores user-uploaded profile picture avatars (`avatar_u<id>_<timestamp>.<ext>`).
  - `User database/uploaded_attachments/`: Stores user-uploaded chat photos and PDF attachments (`att_u<id>_<timestamp>_<name>`).
- **User Authentication (`auth_and_chat_db.py`)**:
  - Validates `username` (Name) and `password` with strict no-spaces rules.
  - Registration includes re-enter password confirmation.
  - Passwords hashed with `werkzeug.security`.
  - **Admin Console & User Login Credentials**:
    - **ID**: `df`
    - **Password**: `df`
  - Profile picture upload support (`POST /api/user/profile-picture` and `GET /api/user/profile-picture/<filename>`).
- **Per-User Chat Tabs & Memory Isolation**:
  - Each user has independent chat tabs stored in SQLite (`User database/users_and_chats.db`).
  - Users can create new tabs (`+ New Tab`) or delete unwanted tabs (`🗑️`).
  - **Conversational Memory**: Chatbot conditioning includes prior turns from that specific tab to seamlessly understand follow-up questions.
- **Conversational Fast Path (Zero-Vector Latency Optimization)**:
  - Greetings, pleasantries, small talk, and general capability queries ("Hi", "How are you", "What can you do", "I have a problem, what can I do") bypass vector embedding and ChromaDB retrieval.
  - Instantly answered directly by Gemini with warm introduction and dynamic listing of available manuals (< 1.5s latency).
- **Multimodal User Attachments (Photos & PDFs - Max 5 Files)**:
  - Users can attach up to 5 images (PNG, JPG, WEBP) or PDF documents directly to any chat question.
  - User-uploaded photos and PDF pages are loaded/rendered as PIL Images and passed directly into `gemini-3.5-flash-lite`'s multimodal vision context alongside retrieved manual pages.
  - Chat transcript bubbles render attached photo thumbnails with click-to-enlarge lightbox and interactive PDF badges.
- **Compact Top-K Results Pill (`#topKModal`)**:
  - Clean compact pill button underneath assistant response (`Top-K Result (N pages) - View more`).
  - Click opens candidate page inspector modal without cluttering the chat stream.
- **Interactive In-Text Citation Buttons (`/pdf-viewer`)**:
  - Assistant attaches section/row citations in `[Manual Name, p.XX]` format.
  - Clicking any citation opens `/pdf-viewer?file=...&page=XX` in a new browser tab directly jumped to the cited page.

---

## 2. Codebase Architecture & File Map

```
/home/tinonn/df_rag_project/
├── config.py                       # Global configuration, hyperparameters, instructions, ChromaDB & User DB paths
├── run_embedding_pipeline.py       # Batch pipeline: TOC parsing, page rendering, ChromaDB upsert
├── query_test.py                   # CLI diagnostic tool to query ChromaDB and rank pages
├── auth_and_chat_db.py             # User authentication, chat tabs & message history SQLite manager
├── pipeline_service.py             # Re-embedding, page deletion, metadata editor, SSE progress
├── app.py                          # Flask web app, fast-path intent classifier, multimodal uploads & REST APIs
├── templates/
│   ├── index.html                  # Responsive Chat UI (Tabs, Conversational Memory, Profile Pic, Attachments, Top-K, Citations)
│   ├── admin.html                  # Password-protected Admin Console (Auth gate ID: df / Pass: df, PDF manager)
│   └── pdf_viewer.html             # Dedicated PDF page citation viewer
├── requirements.txt                # Python package dependencies (google-genai, PyMuPDF, chromadb, flask, etc.)
├── .env                            # Environment secrets (GEMINI_API_KEY, ADMIN_ID, ADMIN_PASSWORD)
├── User database/                  # Dedicated directory for user credentials, tabs, history & avatars
│   ├── users_and_chats.db          # Persistent SQLite database (users, chat_tabs, chat_messages)
│   ├── profile_pictures/           # User-uploaded avatar images
│   └── uploaded_attachments/       # User-uploaded chat photos & PDF files
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
| [`config.py`](file:///home/tinonn/df_rag_project/config.py) | Defines system paths, collection name, rendering specs, model parameters, asymmetric embedding instructions, `ADMIN_PASSWORD` (default: `"df"`), and session `SECRET_KEY`. |
| [`run_embedding_pipeline.py`](file:///home/tinonn/df_rag_project/run_embedding_pipeline.py) | Iterates over `source/*.pdf`, checks `pipeline_state.json` to skip unchanged files, extracts native document outline (TOC) with PyMuPDF, renders pages to PNG, calls the Gemini embedder, and upserts vectors + metadata into ChromaDB. |
| [`embedders/gemini_multimodal_embedder.py`](file:///home/tinonn/df_rag_project/embedders/gemini_multimodal_embedder.py) | Interacts with `google.genai.Client`. Provides `embed_page_image()` with resilient exponential backoff retry for rate limits and `embed_query_text()` for search queries. |
| [`pipeline_service.py`](file:///home/tinonn/df_rag_project/pipeline_service.py) | Core pipeline engine for outline extraction, rendering, incremental per-page ChromaDB upserts, and real-time SSE progress streaming with dynamic ETA calculation. |
| [`app.py`](file:///home/tinonn/df_rag_project/app.py) | Flask web application serving HTML interfaces and REST API endpoints (`/api/status`, `/api/chat`, `/api/admin/*`, `/api/admin/embed/stream`). Features session-based password authentication (`@admin_required`). |
| [`templates/index.html`](file:///home/tinonn/df_rag_project/templates/index.html) | Mobile & tablet friendly chat interface featuring off-canvas slide drawer navigation, real-time index status, chat history, Markdown formatting, citation cards with similarity scores, and a high-resolution touch image lightbox. |
| [`templates/admin.html`](file:///home/tinonn/df_rag_project/templates/admin.html) | Password-protected admin dashboard featuring login lock gate, live loading bar with real-time ETA, total/remaining page metrics, dual table/card views for mobile/desktop, PDF uploader/manager, and settings editor. |
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
`app.py` enforces strict context adherence on `gemini-3.5-flash-lite`:
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

