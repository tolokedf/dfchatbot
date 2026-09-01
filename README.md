# NavWiz & DFleet Multimodal RAG Assistant

A **Vision-First Multimodal Retrieval-Augmented Generation (RAG)** system designed specifically for complex technical documentation (robotics manuals, hardware deployment handbooks, and software guides).

Instead of traditional OCR/text extraction pipelines, this system renders PDF pages directly into high-resolution images and embeds them into a shared vector space with Google's **`gemini-embedding-2`** model, persisting embeddings and document topology into **ChromaDB** with outline-aware chapter boundaries.

---

## Key Features

- **Direct Multimodal Embeddings (3072 dims):** Preserves complex diagrams, wiring schematics, UI screenshots, tables, and sensor layout flowcharts.
- **Dedicated User Database Folder (`User database/`):**
  - Persistent SQLite database (`User database/users_and_chats.db`) tracking accounts, sessions, and multi-turn message history.
  - Avatar storage (`User database/profile_pictures/`) supporting user-uploaded profile pictures with real-time UI previews.
- **User Authentication & Multi-Tab Experience:**
  - Strict no-space validation for Names and Passwords, with confirm password match validation.
  - Independent **Chat Tabs** (`+ New Tab`, delete tab `🗑️`, auto-topic naming).
  - **Conversational Memory:** Isolated per tab, allowing context-aware follow-up queries.
- **Compact Top-K Results Pill (`#topKModal`):**
  - Replaces raw chat clutter with a sleek pill button (`Top-K Result (N pages) - View more`).
  - Expandable modal displaying candidate page thumbnails, similarity scores, and metadata.
- **In-Text & Section Citation Buttons (`/pdf-viewer`):**
  - Auto-generated citations `[Manual Name, p.XX]` rendered as interactive clickable buttons.
  - Opens the dedicated **PDF Page Viewer** in a new browser tab directly at that page.
- **Admin Management Console (`/admin`):**
  - Secured with **ID: `df`** and **Password: `df`**.
  - **Option 1: Embed Required Only (Smart Sync):** Differential page scanning that only embeds missing or edited pages.
  - **Option 2: Re-embed All Pages:** Full overwrite of all pages.
  - **Page & Metadata Inspector (`Pages & Tags`):** Visual inspection of page PNGs, ChromaDB page deletion, custom metadata tag editor, and single/batch page embedding.
  - Live SSE progress bar with total/remaining pages, completion percentage, and real-time ETA.
  - `.env` API Key, Admin credentials, and Model manager (`gemini-3.5-flash-lite`, `gemini-3.5-flash`).

---

## 1. Quickstart & Setup

### Prerequisites
- Python 3.10+
- A Google Gemini API Key

### Installation

1. **Clone repository and set up virtual environment:**
```bash
git clone <repository_url>
cd df_rag_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Configure environment variables in `.env`:**
```env
GEMINI_API_KEY=your_gemini_api_key_here
ADMIN_ID=df
ADMIN_PASSWORD=df
GEMINI_QA_MODEL=gemini-3.5-flash-lite
```

---

## 2. Ingesting Source PDFs

Place your technical PDF manuals into the `source/` directory:
- `source/NavWiz 4.0 User Manual 1.0.pdf`
- `source/DFleet 4.0 User Manual.pdf`
- `source/Copy of Field Deployment Handbook.pdf`

Run the batch embedding pipeline via CLI:
```bash
python run_embedding_pipeline.py
```
*(Or use the Admin Console at `http://localhost:5000/admin` to upload and embed interactively).*

---

## 3. Running the Web Application

Launch the Flask server:
```bash
python app.py
```
Open your browser at **`http://localhost:5000`**:

- **Chat Interface (`/`)**: Multi-tab chatbot with conversational memory, avatar uploads, compact Top-K results, and direct page citation links.
- **Admin Console (`/admin`)**: Log in with **ID: `df`** and **Password: `df`** to manage manuals, inspect/edit metadata tags, delete specific pages, or trigger differential sync embeddings.

---

## 4. Architecture & Directory Structure

```
df_rag_project/
├── config.py                       # Global configuration, hyperparameters, instructions, ChromaDB & User DB paths
├── run_embedding_pipeline.py       # Batch pipeline: TOC parsing, page rendering, ChromaDB upsert
├── query_test.py                   # CLI diagnostic tool to query ChromaDB and rank pages
├── auth_and_chat_db.py             # User authentication, chat tabs & message history SQLite manager
├── pipeline_service.py             # Re-embedding, page deletion, metadata editor, SSE progress
├── app.py                          # Flask web application, REST APIs, profile picture upload & Admin session auth
├── templates/
│   ├── index.html                  # Responsive Chat UI (Tabs, Conversational Memory, Profile Pic, Top-K, Citations)
│   ├── admin.html                  # Password-protected Admin Console (Auth gate ID: df / Pass: df, PDF manager)
│   └── pdf_viewer.html             # Dedicated PDF page citation viewer
├── requirements.txt                # Python package dependencies
├── .env                            # Environment secrets (GEMINI_API_KEY, ADMIN_ID, ADMIN_PASSWORD)
├── User database/                  # Dedicated directory for user credentials, tabs, history & avatars
│   ├── users_and_chats.db          # Persistent SQLite database (users, chat_tabs, chat_messages)
│   └── profile_pictures/           # User-uploaded avatar images
├── source/                         # Input directory for source PDF documents
└── output/                         # Generated artifacts and vector index
    ├── chroma_db/                  # Persistent ChromaDB vector database (SQLite + HNSW index)
    ├── pipeline_state.json         # State tracking for incremental processing
    └── rendered_pages/             # Cached PNG renderings of all PDF pages
```

---

## 5. Security & Authentication Reference

| Role / Gate | Credentials | Capabilities |
| :--- | :--- | :--- |
| **Admin Console (`/admin`)** | **ID**: `df`<br>**Password**: `df` | Upload/delete PDFs, trigger embeddings (Required only vs All), inspect page PNGs, delete pages from vector store, edit metadata tags, update API keys. |
| **Standard User (`/`)** | Created via Register tab (No spaces allowed) | Independent multi-tab chat sessions, conversational follow-up memory, profile picture upload & persistence. |
