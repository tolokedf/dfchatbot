# Project Context & Technical Reference: DF Chatbot (Multimodal PDF RAG)

> **Purpose:** This document serves as the persistent cross-session architectural, operational, and development reference for the `dfchatbot` codebase.

---

## ⚠️ Cross-Session Development Rules & Core Principles

### 1. Strict Scope Adherence
> **MANDATORY RULE:** **Do not add additional feature/button/text that did not mention in instruction.**
- **Zero Unsolicited Additions:** Never add unrequested buttons, icons, links, menu items, placeholder cards, badges, or UI text unless explicitly specified by the user.
- **Precise Implementations:** Implement only the exact requested behavior or fix. Avoid adding "nice-to-have" auxiliary features or side-effects that bloat the UI or alter workflow without request.
- **Maintain Existing Logic & Clean State:** When fixing bugs, preserve existing working features and adhere to established patterns rather than refactoring unrelated modules.

---

## 1. Executive Summary

`dfchatbot` is a **Vision-First Multimodal Retrieval-Augmented Generation (RAG)** system designed specifically for complex technical documentation (robotics manuals, hardware deployment handbooks, and software user guides such as NavWiz and DFleet).

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
   │   (data/output/chroma_db/ - 3072 dims)    │
   └─────────────────────┬─────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌───────────────────────┐     ┌───────────────────────┐
│   Flask App (src/app) │     │  CLI Diagnostic Tool  │
│ • Tailwind UI / API   │     │  (scripts/query_test) │
│ • ChromaDB Retrieval  │     │  • ChromaDB Query     │
│ • Chapter Expansion   │     │  • Cosine Similarity  │
│ • gemini-3.5-flash-lite│    └───────────────────────┘
└───────────────────────┘
```

### Key Advantages
1. **Preserves Visual Semantics:** Accurately indexes complex layouts, schematics, UI screenshots, tables, wiring diagrams, and flowcharts that break typical OCR extractors.
2. **Persistent Vector Database (ChromaDB):** Replaces flat `.npy`/`.jsonl` arrays with indexed HNSW vector storage, native metadata querying, and unified multi-PDF aggregation.
3. **Native Outline-Aware Filtering & Expansion:** Uses document outline hierarchy (TOC) to filter out non-informative front-matter (covers, copyright, TOCs) and expands retrieved seed pages across neighboring pages strictly bounded within the same chapter.
4. **Single-Hop Embedding:** Eliminates text extraction artifacts, caption hallucination, and multi-model pipeline latency.

---

## 2. Key System Features & Operational Mechanics

### 2.1 Multi-Mode Embedding Engine & Differential Sync
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

### 2.2 User Authentication, Chat Tabs & Session Lifecycle
- **Isolated User Storage (`data/user_storage/`)**:
  - `data/user_storage/users_and_chats.db`: SQLite database storing user credentials, chat tabs, and multi-turn message histories.
  - `data/user_storage/profile_pictures/`: Stores user-uploaded profile picture avatars (`avatar_u<id>_<timestamp>.<ext>`).
  - `data/user_storage/uploaded_attachments/`: Stores user-uploaded chat photos and PDF attachments (`att_u<id>_<timestamp>_<name>`).
- **User Authentication (`src/auth_and_chat_db.py`)**:
  - Validates `username` (Name) and `password` with strict no-spaces rules.
  - Registration includes re-enter password confirmation. Passwords hashed with `werkzeug.security`.
  - **Admin Credentials**: ID: `df` / Password: `df`.
  - **Guest Access Option (`POST /api/auth/guest`)**:
    - Users can click *"Continue as Guest"* on the login modal to immediately ask questions and upload photos without creating an account.
    - Guest mode operates in a local session with full manual access.
- **Per-User Chat Tabs & Clean Deletion Lifecycle**:
  - Each user has independent chat tabs stored in SQLite (`chat_tabs` & `chat_messages` tables).
  - Users can create new tabs (`+ New Tab`) or delete unwanted tabs (`🗑️`).
  - **Deleting Tabs / Clearing All Chats**:
    - Backend `auth_and_chat_db.list_user_tabs(user_id)` ensures logged-in users always have at least one tab (auto-creates a single `"New Chat"` if count is 0).
    - When deleting the last active tab, the client cleanly consumes the returned single-tab list without executing duplicate creation requests.
- **Conversational Memory & Fast Path**:
  - Chat conditioning includes prior turns from that specific tab to seamlessly handle follow-up questions.
  - Greetings and capability inquiries ("Hi", "How are you", "What can you do") bypass vector retrieval and are answered directly via fast-path prompt (< 1.5s latency).
- **Target Manual Filtering**:
  - Quick Manual dropdown placed directly above the chat input box and in the sidebar with automatic bidirectional synchronization.
- **Multimodal User Attachments (Photos & PDFs - Max 5 Files)**:
  - Users can attach up to 5 images (PNG, JPG, WEBP) or PDF documents directly to any chat question.
  - User-uploaded attachments are passed directly into `gemini-3.5-flash-lite`'s multimodal vision context alongside retrieved manual pages.
  - Chat transcript bubbles render attached photo thumbnails with click-to-enlarge lightbox.

### 2.3 Admin Console (`templates/admin.html`)
- **Navigation & Header**:
  - Minimalist top navigation with status indicators, "Switch to Chat" button, and "Lock" button.
  - Text spans are enclosed to prevent extraneous flexbox spacing gaps between words.
- **User Management & Chat Inspector**:
  - Inspect user accounts, total logins (`login_count`), avatars, roles, join dates, and timestamps.
  - Permanent account deletion with confirmation (`DELETE /api/admin/users/<id>`) protecting primary `df` admin.
  - Inspect all user chat tabs, questions, attached files, assistant answers, and Top-K candidate sources (`#userChatsModal`).
- **Export Transcripts**:
  - Generate full user conversation transcripts as downloadable HTML or PDF reports (`src/report_exporter.py`).

---

## 2. Codebase Architecture & File Map

```
dfchatbot/
├── src/
│   ├── config.py                   # Global configuration, data paths, instructions
│   ├── app.py                      # Flask web application & REST API endpoints
│   ├── auth_and_chat_db.py         # User authentication, chat tabs & SQLite manager
│   ├── pipeline_service.py         # Incremental sync, page deletion, metadata editor
│   ├── report_exporter.py          # PDF conversation report generator
│   └── embedders/
│       ├── __init__.py
│       └── gemini_multimodal_embedder.py # Gemini 2 Embedding wrapper
├── templates/
│   ├── index.html                  # Responsive Chat UI
│   ├── admin.html                  # Password-protected Admin Console
│   └── pdf_viewer.html             # Dedicated PDF page citation viewer
├── static/                         # Static assets and branding logos
├── scripts/                        # CLI diagnostic & server utilities
├── deploy/                         # Windows deployment scripts
├── data/                           # Runtime Data & Storage (IGNORED BY GIT)
│   ├── source_docs/                # Source PDF manuals to index
│   ├── output/                     # ChromaDB vector store, page renderings & metadata
│   └── user_storage/               # SQLite user database, avatars & chat attachments
└── requirements.txt                # Python package dependencies
```

### Detailed Component Roles

| File | Primary Role & Responsibilities |
|---|---|
| [`src/config.py`](file:///home/tinonn/DF_application/dfchatbot/src/config.py) | Defines system paths, ChromaDB collection name, rendering specs, model parameters, asymmetric embedding instructions, `ADMIN_PASSWORD` (default: `"df"`), and session `SECRET_KEY`. |
| [`src/app.py`](file:///home/tinonn/DF_application/dfchatbot/src/app.py) | Flask web application serving HTML interfaces and REST API endpoints (`/api/status`, `/api/chat`, `/api/admin/*`, `/api/chat/tabs/*`). Enforces `@user_required` and `@admin_required` decorators. |
| [`src/auth_and_chat_db.py`](file:///home/tinonn/DF_application/dfchatbot/src/auth_and_chat_db.py) | SQLite manager handling users, passwords, sessions, chat tabs, messages, attachments, and profile picture avatar metadata. |
| [`src/pipeline_service.py`](file:///home/tinonn/DF_application/dfchatbot/src/pipeline_service.py) | Core pipeline engine for outline extraction, rendering, incremental per-page ChromaDB upserts, and real-time SSE progress streaming with dynamic ETA calculation. |
| [`src/embedders/gemini_multimodal_embedder.py`](file:///home/tinonn/DF_application/dfchatbot/src/embedders/gemini_multimodal_embedder.py) | Interacts with `google.genai.Client`. Provides `embed_page_image()` with exponential backoff retry for rate limits and `embed_query_text()` for search queries. |
| [`src/report_exporter.py`](file:///home/tinonn/DF_application/dfchatbot/src/report_exporter.py) | Generates structured HTML and PDF transcripts of chat sessions for admin inspection. |
| [`templates/index.html`](file:///home/tinonn/DF_application/dfchatbot/templates/index.html) | Responsive chat UI featuring tab management, file attachments, Quick Manual filtering, markdown formatting, lightbox previews, and citations. |
| [`templates/admin.html`](file:///home/tinonn/DF_application/dfchatbot/templates/admin.html) | Admin console with login lock, live progress bar, metadata/page manager, user accounts & chat inspector table. |
| [`templates/pdf_viewer.html`](file:///home/tinonn/DF_application/dfchatbot/templates/pdf_viewer.html) | Citation viewer for navigating directly to cited pages in technical manuals. |

---

## 4. ChromaDB Schema & Data Structures

### 4.1 Collection Configuration
- **Collection Name:** `pdf_pages`
- **Metric:** Cosine similarity (`metadata={"hnsw:space": "cosine"}`)
- **Persistence Path:** `data/output/chroma_db/`

### 4.2 Record Structure in ChromaDB
| Field | Type | Description / Example |
|---|---|---|
| **ID** | `str` | Format: `{pdf_stem}_page_{page_num:03d}` (e.g. `DFleet 4.0 User Manual_page_008`) |
| **Embedding** | `list[float]` | 3072-dimensional vector from `gemini-embedding-2` |
| **Document** | `str` | Text summary: `DFleet 4.0 User Manual - Page 008 \| Chapter: Preface \| Section: Section 1: Introduction` |
| **Metadata** | `dict` | Attributes: `pdf_stem`, `page_image`, `page_number`, `chapter`, `section`, `subsection`, `is_front_matter`, `model`, `dimensions`, `elapsed_seconds` |

---

## 5. Retrieval & Generation Pipeline Mechanics

### Asymmetric Retrieval Instructions
`gemini-embedding-2` uses instruction strings to bridge query and document representation spaces:
- **Document instruction:** `"This is a page from a technical robotics software user manual. It may contain prose, tables, UI screenshots, or diagrams with embedded text. Represent its full content for retrieval by a user's question about the manual."`
- **Query instruction:** `"Represent this question for retrieving the most relevant manual page."`

### Front-Matter Filtering & Chapter-Bounded Neighbor Expansion
1. **Filtering**: Non-content pages (cover, copyright, TOC) marked `"is_front_matter": true` are filtered out during ChromaDB queries.
2. **Neighbor Expansion**: For top seed pages, adjacent pages ($\pm 3$) within the same chapter are included to supply full procedural context to `gemini-3.5-flash-lite`.

### Grounding & Hallucination Guardrail Prompt
Enforces strict context adherence on `gemini-3.5-flash-lite`:
```text
You are an expert technical assistant for DF Automation / technical manuals.
Answer the user's question accurately using only the provided image pages and attachments as context.

STRICT RULE: If the user's question is gibberish, meaningless text, completely unrelated to
technical manuals, or cannot be answered by the provided context, respond EXACTLY with:
"I am not sure about that."
```

---

## 6. Environment & Operational Runbook

### 6.1 Environment Prerequisites
- Python 3.10+ (Virtual Environment at `.venv`)
- `GEMINI_API_KEY` configured in `.env` (automatically loaded and refreshed by `src/config.py`)

### 6.2 Common Workflows

#### A. Ingesting New or Updated PDFs (CLI)
```bash
/home/tinonn/DF_application/dfchatbot/.venv/bin/python scripts/run_embedding_pipeline.py
```

#### B. Running Retrieval Sanity Checks (CLI)
```bash
/home/tinonn/DF_application/dfchatbot/.venv/bin/python scripts/query_test.py "How do I configure safety zones?"
```

#### C. Launching the Server
```bash
/home/tinonn/DF_application/dfchatbot/.venv/bin/python scripts/run_server.py
```
Open `http://localhost:5000` in your web browser.

