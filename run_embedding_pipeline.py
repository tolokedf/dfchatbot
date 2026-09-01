"""
CLI Entry point to batch-embed all PDFs in source/ into ChromaDB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import config
import pipeline_service


def main() -> None:
    pdfs_info = pipeline_service.get_all_pdfs_status()
    if not pdfs_info:
        print(f"No PDFs found in {config.SOURCE_DIR}")
        return

    print(f"Found {len(pdfs_info)} PDF(s) in {config.SOURCE_DIR}:\n")
    for p in pdfs_info:
        print(f"  • {p['filename']} ({p['size_formatted']}, {p['total_pdf_pages']} pages) -> Status: {p['status'].upper()}")

    print()

    for p in pdfs_info:
        pdf_path = config.SOURCE_DIR / p["filename"]
        if p["status"] == "embedded":
            print(f"⏭️  Skipping '{p['filename']}' (Already embedded with {p['embedded_pages_count']} pages)")
            continue

        print(f"⚙️  Embedding '{p['filename']}'...")
        try:
            res = pipeline_service.process_and_embed_pdf(pdf_path, log_fn=lambda msg: print(f"    {msg}"))
            print(f"✅ Finished '{p['filename']}': {res['pages_embedded']} pages indexed.\n")
        except Exception as e:
            print(f"❌ Error processing '{p['filename']}': {e}\n")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()