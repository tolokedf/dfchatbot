"""
Service for parsing and chunking Excel (.xlsx) workbooks for ChromaDB indexing.
Preserves table headers (sticky headers), strips empty grid rows,
extracts cell fill colors into semantic markdown tags, and generates
pinpoint structured chunks.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

try:
    import openpyxl
    from openpyxl.worksheet.worksheet import Worksheet
except ImportError:
    openpyxl = None

logger = logging.getLogger(__name__)

# Semantic mapping of known hex fill colors to readable status tags
HEX_COLOR_MAP = {
    "FF00FF00": "[🟢 Active/Tested]",
    "FF7FD5A3": "[🟢 Can Recover]",
    "FFA8D08D": "[🟢 Pass/OK]",
    "FFB6D7A8": "[🟢 OK]",
    "FFC5E0B3": "[🟢 Mechanical Assembly]",
    "FFD9EAD3": "[🟢 Verified]",
    "FFFF0000": "[🔴 Abort/Critical]",
    "FFEA9999": "[🔴 Cannot Recover]",
    "FFF4CCCC": "[🔴 Attention]",
    "FFFF9900": "[🟠 Discrepancy/Notice]",
    "FFF7CBAC": "[🟠 Teaching Task]",
    "FFF9CB9C": "[🟠 Long Orientation]",
    "FFFBE5D5": "[🟠 Notice]",
    "FFFFFF00": "[🟡 Warning]",
    "FFFFE599": "[🟡 Pending/Program Flow]",
    "FFFEE599": "[🟡 Pending/Program Flow]",
    "FFFFD966": "[🟡 Header/Attention]",
    "FFBDD7EE": "[🔵 Test Task]",
    "FF8EAADB": "[🔵 Task Template]",
    "FF9CC3E5": "[🔵 Reference]",
    "FFD9E2F3": "[🔵 Component]",
    "FFB4A7D6": "[🟣 Purple]",
    "FFECAAEB": "[🟣 Electrical Wiring]",
    "FFDBDBDB": "[⚪ Mapping Task]",
    "FFD8D8D8": "[⚪ Configured]",
    "FFBFBFBF": "[⚪ Neutral]",
    "FFBF9000": "[🟤 Troubleshoot Task]",
    "FFFCE5CD": "[⚪ Short Orientation]",
}


def sanitize_str(val: any) -> str:
    if val is None:
        return ""
    text = str(val).replace("\xa0", " ").strip()
    # Normalize multiline cells for markdown table compatibility
    text = re.sub(r"\r?\n+", " ", text)
    # Escape pipe characters so markdown table columns do not break
    return text.replace("|", "/")


def get_cell_color_tag(cell) -> str:
    """Extracts background fill color and maps to semantic tag if known."""
    if not cell or not cell.fill or not cell.fill.start_color:
        return ""
    sc = cell.fill.start_color
    rgb_val = str(getattr(sc, "rgb", ""))
    if rgb_val and rgb_val not in ["00000000", "FFFFFFFF", "None"]:
        rgb_upper = rgb_val.upper()
        if rgb_upper in HEX_COLOR_MAP:
            return HEX_COLOR_MAP[rgb_upper]
        return f"[🎨 #{rgb_upper[-6:]}]"
    return ""


def slugify(text: str) -> str:
    clean = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", clean)


def parse_xlsx_to_chunks(xlsx_path: Path, max_rows_per_chunk: int = 25) -> list[dict]:
    """
    Parses an Excel workbook into high-precision, semantic Markdown chunks
    enriched with cell fill colors and metadata.
    """
    if openpyxl is None:
        raise ImportError("openpyxl is not installed. Please run: pip install openpyxl")

    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    chunks: list[dict] = []
    file_stem = xlsx_path.stem
    source_filename = xlsx_path.name

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        
        # 1. Determine active data bounds (ignore trailing empty grid rows)
        last_text_row = 0
        last_text_col = 0
        for r in range(1, sheet.max_row + 1):
            for c in range(1, sheet.max_column + 1):
                val = sheet.cell(row=r, column=c).value
                if val is not None and str(val).strip():
                    if r > last_text_row:
                        last_text_row = r
                    if c > last_text_col:
                        last_text_col = c

        if last_text_row == 0 or last_text_col == 0:
            logger.info(f"Skipping empty sheet: {sheet_name}")
            continue

        # 2. Extract active rows and resolve cell colors
        raw_rows = []
        for r in range(1, last_text_row + 1):
            row_items = []
            has_content = False
            for c in range(1, last_text_col + 1):
                cell = sheet.cell(row=r, column=c)
                val = sanitize_str(cell.value)
                color_tag = get_cell_color_tag(cell)
                
                if val and color_tag:
                    formatted_cell = f"{val} {color_tag}"
                    has_content = True
                elif val:
                    formatted_cell = val
                    has_content = True
                elif color_tag:
                    formatted_cell = color_tag
                    has_content = True
                else:
                    formatted_cell = ""
                row_items.append(formatted_cell)

            if has_content:
                raw_rows.append((r, row_items))

        if not raw_rows:
            continue

        is_front_matter = sheet_name.lower() in ["update", "changelog", "readme", "version", "index"]
        sheet_slug = slugify(sheet_name)

        # 3. Handle Special Sheet: 'TM Error' (Generate structured error cards + table)
        if sheet_name.lower() in ["tm error", "tmerror", "error", "errors"]:
            # Find header
            header_row_idx = 0
            for idx, (r_num, items) in enumerate(raw_rows):
                if any("register" in item.lower() for item in items):
                    header_row_idx = idx
                    break
            
            headers = [h if h else f"Col{i+1}" for i, h in enumerate(raw_rows[header_row_idx][1])]
            data_rows = raw_rows[header_row_idx + 1:]

            chunk_batch = []
            current_chunk_idx = 1
            start_r = data_rows[0][0] if data_rows else 1

            for r_num, items in data_rows:
                padded = items + [""] * max(0, len(headers) - len(items))
                chunk_batch.append((r_num, padded[:len(headers)]))

                if len(chunk_batch) >= 12:
                    end_r = chunk_batch[-1][0]
                    card_lines = [
                        f"[DOCUMENT: {source_filename} | SHEET: {sheet_name} | ERROR CODES & RECOVERY | ROWS: {start_r}-{end_r}]\n",
                        "| " + " | ".join(headers) + " |",
                        "| " + " | ".join(["---"] * len(headers)) + " |"
                    ]
                    for cr, citems in chunk_batch:
                        card_lines.append("| " + " | ".join(citems) + " |")

                    chunk_text = "\n".join(card_lines)
                    chunks.append({
                        "chunk_id": f"{file_stem}_{sheet_slug}_chunk_{current_chunk_idx:03d}",
                        "source_file": source_filename,
                        "file_stem": file_stem,
                        "sheet_name": sheet_name,
                        "section": "Error Codes & Recovery Procedures",
                        "row_start": start_r,
                        "row_end": end_r,
                        "text": chunk_text,
                        "doc_type": "xlsx",
                        "is_front_matter": is_front_matter
                    })
                    current_chunk_idx += 1
                    chunk_batch = []
                    start_r = r_num + 1

            if chunk_batch:
                end_r = chunk_batch[-1][0]
                card_lines = [
                    f"[DOCUMENT: {source_filename} | SHEET: {sheet_name} | ERROR CODES & RECOVERY | ROWS: {start_r}-{end_r}]\n",
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |"
                ]
                for cr, citems in chunk_batch:
                    card_lines.append("| " + " | ".join(citems) + " |")

                chunk_text = "\n".join(card_lines)
                chunks.append({
                    "chunk_id": f"{file_stem}_{sheet_slug}_chunk_{current_chunk_idx:03d}",
                    "source_file": source_filename,
                    "file_stem": file_stem,
                    "sheet_name": sheet_name,
                    "section": "Error Codes & Recovery Procedures",
                    "row_start": start_r,
                    "row_end": end_r,
                    "text": chunk_text,
                    "doc_type": "xlsx",
                    "is_front_matter": is_front_matter
                })
            continue

        # 4. Standard Sheets: Detect Headers & Chunk with Sticky Headers
        header_row_idx = 0
        for idx, (r_num, items) in enumerate(raw_rows[:5]):
            non_empty_cnt = sum(1 for x in items if x.strip())
            if non_empty_cnt >= 2:
                header_row_idx = idx
                break

        headers = [h if h else f"Col{i+1}" for i, h in enumerate(raw_rows[header_row_idx][1])]
        data_rows = raw_rows[header_row_idx + 1:] if header_row_idx + 1 < len(raw_rows) else [raw_rows[header_row_idx]]

        # If total active rows is small (<= 40), keep the entire sheet in a single intact chunk
        if len(data_rows) <= 40:
            start_r = raw_rows[0][0]
            end_r = raw_rows[-1][0]
            table_lines = [
                f"[DOCUMENT: {source_filename} | SHEET: {sheet_name} | FULL SPECIFICATION MATRIX | ROWS: {start_r}-{end_r}]\n",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |"
            ]
            for r_num, items in data_rows:
                padded = items + [""] * max(0, len(headers) - len(items))
                table_lines.append("| " + " | ".join(padded[:len(headers)]) + " |")

            chunk_text = "\n".join(table_lines)
            chunks.append({
                "chunk_id": f"{file_stem}_{sheet_slug}_chunk_001",
                "source_file": source_filename,
                "file_stem": file_stem,
                "sheet_name": sheet_name,
                "section": sheet_name,
                "row_start": start_r,
                "row_end": end_r,
                "text": chunk_text,
                "doc_type": "xlsx",
                "is_front_matter": is_front_matter
            })
        else:
            # Larger sheets: Split into groups of max_rows_per_chunk with sticky headers
            current_chunk_idx = 1
            for i in range(0, len(data_rows), max_rows_per_chunk):
                batch = data_rows[i : i + max_rows_per_chunk]
                start_r = batch[0][0]
                end_r = batch[-1][0]

                table_lines = [
                    f"[DOCUMENT: {source_filename} | SHEET: {sheet_name} | SECTION: {sheet_name} (Part {current_chunk_idx}) | ROWS: {start_r}-{end_r}]\n",
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |"
                ]
                for r_num, items in batch:
                    padded = items + [""] * max(0, len(headers) - len(items))
                    table_lines.append("| " + " | ".join(padded[:len(headers)]) + " |")

                chunk_text = "\n".join(table_lines)
                chunks.append({
                    "chunk_id": f"{file_stem}_{sheet_slug}_chunk_{current_chunk_idx:03d}",
                    "source_file": source_filename,
                    "file_stem": file_stem,
                    "sheet_name": sheet_name,
                    "section": f"{sheet_name} Part {current_chunk_idx}",
                    "row_start": start_r,
                    "row_end": end_r,
                    "text": chunk_text,
                    "doc_type": "xlsx",
                    "is_front_matter": is_front_matter
                })
                current_chunk_idx += 1

    wb.close()
    return chunks
