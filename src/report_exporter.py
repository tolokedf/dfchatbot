"""
Report Exporter Module for DF Chatbot Multimodal RAG.
Generates comprehensive, beautifully formatted reports of user chat histories (all tabs or single tab)
in both standalone HTML and professional PDF formats.
"""

from __future__ import annotations

import io
import html
import re
import zipfile
from datetime import datetime
from typing import Optional, Dict, Any, List

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw exact total page counts and running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#64748b'))
        
        # Running Top Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 808, 'DF Chatbot — DF Automation User Chat History Report')
            self.drawRightString(559, 808, 'Confidential')
            self.setStrokeColor(colors.HexColor('#cbd5e1'))
            self.setLineWidth(0.5)
            self.line(36, 802, 559, 802)

        # Running Bottom Footer (all pages)
        self.setStrokeColor(colors.HexColor('#cbd5e1'))
        self.setLineWidth(0.5)
        self.line(36, 40, 559, 40)
        
        self.drawString(36, 28, 'DF Automation & Robotics • Multimodal RAG Assistant')
        page_str = f'Page {self._pageNumber} of {page_count}'
        self.drawRightString(559, 28, page_str)
        self.restoreState()


def clean_markdown_to_html(text: str) -> str:
    """Converts basic markdown formatting into clean HTML for web report."""
    if not text:
        return ""
    
    # Escape raw HTML first
    escaped = html.escape(text)
    
    # Bold **text** or __text__
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'__(.*?)__', r'<strong>\1</strong>', escaped)
    
    # Italic *text* or _text_
    escaped = re.sub(r'\*(.*?)\*', r'<em>\1</em>', escaped)
    escaped = re.sub(r'_(.*?)_', r'<em>\1</em>', escaped)
    
    # Inline code `text`
    escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
    
    # Format in-text citations [Manual, p.XX]
    citation_regex = r'\[(.*?)(?:,\s*p\.?|\s*-\s*page)\s*(\d+)\]'
    escaped = re.sub(
        citation_regex,
        r'<span class="citation-badge">📖 \1, p.\2</span>',
        escaped
    )
    
    # Convert line breaks to paragraphs and bullet points
    lines = escaped.split('\n')
    out_lines = []
    in_list = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('* ') or stripped.startswith('- ') or stripped.startswith('• '):
            if not in_list:
                out_lines.append('<ul>')
                in_list = True
            content = stripped[2:].strip()
            out_lines.append(f'<li>{content}</li>')
        elif re.match(r'^\d+\.\s+', stripped):
            if not in_list:
                out_lines.append('<ol>')
                in_list = True
            content = re.sub(r'^\d+\.\s+', '', stripped).strip()
            out_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                out_lines.append('</ul>')
                in_list = False
            if stripped:
                out_lines.append(f'<p>{line}</p>')
            else:
                out_lines.append('<div class="spacer"></div>')
                
    if in_list:
        out_lines.append('</ul>')
        
    return '\n'.join(out_lines)


def format_text_for_reportlab(text: str) -> str:
    """Sanitizes and prepares markdown text for ReportLab Paragraph XML parser."""
    if not text:
        return ""
    
    # Basic escaping
    s = html.escape(text)
    
    # Convert bold
    s = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'__(.*?)__', r'<b>\1</b>', s)
    
    # Convert italic
    s = re.sub(r'\*(.*?)\*', r'<i>\1</i>', s)
    s = re.sub(r'_(.*?)_', r'<i>\1</i>', s)
    
    # Convert inline code
    s = re.sub(r'`([^`]+)`', r'<font name="Courier" color="#00787c"><b>\1</b></font>', s)
    
    # Format in-text citations
    citation_regex = r'\[(.*?)(?:,\s*p\.?|\s*-\s*page)\s*(\d+)\]'
    s = re.sub(
        citation_regex,
        r'<font color="#00787c"><b>[\1, p.\2]</b></font>',
        s
    )
    
    # Convert newlines to breaks
    s = s.replace('\n', '<br/>')
    return s


def generate_html_report(user_data: dict, tab_id: Optional[str] = None) -> str:
    """
    Generates a high-quality standalone HTML report of the user's chat history.
    Can export all tabs or a single tab if tab_id is specified.
    """
    username = user_data.get("username", "Unknown User")
    user_id = user_data.get("id", 0)
    role = user_data.get("role", "user").capitalize()
    login_count = user_data.get("login_count", 0)
    created_at = user_data.get("created_at", "--")
    last_login_at = user_data.get("last_login_at") or "--"
    all_tabs = user_data.get("tabs", [])
    
    # Filter tabs if single tab is requested
    target_tab = None
    if tab_id:
        filtered_tabs = [t for t in all_tabs if str(t.get("id")) == str(tab_id)]
        if filtered_tabs:
            target_tab = filtered_tabs[0]
            selected_tabs = [target_tab]
        else:
            selected_tabs = all_tabs
    else:
        selected_tabs = all_tabs
        
    scope_title = f"Single Tab: &ldquo;{html.escape(target_tab.get('title', 'Chat Tab'))}&rdquo;" if target_tab else f"All Chat Tabs ({len(selected_tabs)} Sessions)"
    total_messages_in_report = sum(t.get("message_count", len(t.get("messages", []))) for t in selected_tabs)
    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build Tabs HTML
    tabs_html_list = []
    for tab_idx, tab in enumerate(selected_tabs, 1):
        tab_title = tab.get("title", "Untitled Tab")
        tab_created = tab.get("created_at", "--")
        messages = tab.get("messages", [])
        
        messages_html_list = []
        if not messages:
            messages_html_list.append("""
                <div class="empty-tab-notice">
                    <em>No messages recorded in this chat session.</em>
                </div>
            """)
        else:
            for msg in messages:
                is_user = msg.get("role") == "user"
                role_label = "USER" if is_user else "DF CHATBOT"
                timestamp = msg.get("timestamp") or "--"
                content_raw = msg.get("content") or ""
                formatted_body = clean_markdown_to_html(content_raw)
                
                # Attachments if user message
                attachments_html = ""
                attachments = msg.get("attachments", [])
                if attachments:
                    att_items = []
                    for att in attachments:
                        att_name = html.escape(att.get("name", "Attachment"))
                        att_type = att.get("type", "image")
                        if att_type == "pdf":
                            att_items.append(f'<span class="att-badge pdf">📄 {att_name}</span>')
                        else:
                            att_items.append(f'<span class="att-badge img">🖼️ {att_name}</span>')
                    attachments_html = f'<div class="attachments-row"><strong>Attached Files:</strong> {" ".join(att_items)}</div>'

                # Citations if assistant message
                citations_html = ""
                citations = msg.get("citations", [])
                if citations:
                    cit_items = []
                    for c in citations:
                        c_manual = html.escape(str(c.get("manual", "Manual")))
                        c_page = c.get("page_number", "")
                        cit_items.append(f'<span class="citation-pill">📖 {c_manual}, p.{c_page}</span>')
                    citations_html = f'<div class="citations-row"><strong>Verified In-Text Citations:</strong> {" ".join(cit_items)}</div>'

                # Top-K seeds if assistant message
                top_k_html = ""
                top_k = msg.get("top_k", [])
                if top_k:
                    top_k_items = []
                    for k in top_k[:5]:
                        k_stem = html.escape(str(k.get("pdf_stem", "Manual")))
                        k_page = k.get("page_number", "")
                        k_sim = k.get("similarity", 0.0)
                        top_k_items.append(f'<span class="seed-pill">{k_stem} (p.{k_page}, score: {k_sim:.3f})</span>')
                    top_k_html = f'<div class="topk-row"><strong>Top Matched Seed Pages:</strong> {" • ".join(top_k_items)}</div>'

                msg_card_class = "msg-card user-msg" if is_user else "msg-card assistant-msg"
                role_icon = "👤" if is_user else "🤖"

                messages_html_list.append(f"""
                <div class="{msg_card_class}">
                    <div class="msg-header">
                        <span class="role-badge">{role_icon} {role_label}</span>
                        <span class="msg-time">{timestamp}</span>
                    </div>
                    {attachments_html}
                    <div class="msg-body">
                        {formatted_body}
                    </div>
                    {citations_html}
                    {top_k_html}
                </div>
                """)

        messages_rendered = "\n".join(messages_html_list)

        tabs_html_list.append(f"""
        <section class="tab-section">
            <div class="tab-header">
                <div class="tab-title-wrap">
                    <span class="tab-index-badge">Session #{tab_idx}</span>
                    <h3 class="tab-title">{html.escape(tab_title)}</h3>
                </div>
                <div class="tab-meta">
                    <span>Created: {tab_created}</span>
                    <span class="pill-count">{len(messages)} Messages</span>
                </div>
            </div>
            <div class="tab-messages-stream">
                {messages_rendered}
            </div>
        </section>
        """)

    tabs_rendered = "\n".join(tabs_html_list)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DF Chatbot History Report — {html.escape(username)}</title>
    <style>
        :root {{
            --df-primary: #00b5b8;
            --df-primary-dark: #00787c;
            --df-primary-light: #e0f7f7;
            --slate-900: #0f172a;
            --slate-800: #1e293b;
            --slate-700: #334155;
            --slate-600: #475569;
            --slate-200: #e2e8f0;
            --slate-100: #f1f5f9;
            --slate-50: #f8fafc;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--slate-100);
            color: var(--slate-800);
            line-height: 1.6;
            padding: 24px;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
            border: 1px solid var(--slate-200);
            overflow: hidden;
        }}

        /* Header Bar */
        .report-header {{
            background: linear-gradient(135deg, #0b4e51 0%, #00787c 50%, #00b5b8 100%);
            color: #ffffff;
            padding: 28px 32px;
            position: relative;
        }}

        .header-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .brand-title {{
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .brand-badge {{
            background: rgba(255, 255, 255, 0.2);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        .report-subtitle {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 6px;
        }}

        .header-actions {{
            display: flex;
            gap: 10px;
        }}

        .btn-action {{
            background: #ffffff;
            color: var(--df-primary-dark);
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            text-decoration: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .btn-action:hover {{
            background: var(--df-primary-light);
            transform: translateY(-1px);
        }}

        /* Metadata Grid */
        .meta-card {{
            background: var(--slate-50);
            border-bottom: 1px solid var(--slate-200);
            padding: 20px 32px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }}

        .meta-item {{
            display: flex;
            flex-direction: column;
        }}

        .meta-label {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--slate-600);
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }}

        .meta-val {{
            font-size: 14px;
            font-weight: 600;
            color: var(--slate-900);
        }}

        /* Content Area */
        .report-content {{
            padding: 32px;
        }}

        .tab-section {{
            margin-bottom: 36px;
            border: 1px solid var(--slate-200);
            border-radius: 12px;
            overflow: hidden;
            background: #ffffff;
        }}

        .tab-header {{
            background: #f8fafc;
            border-bottom: 1px solid var(--slate-200);
            padding: 14px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .tab-title-wrap {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .tab-index-badge {{
            background: var(--df-primary-light);
            color: var(--df-primary-dark);
            border: 1px solid #a8eeec;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            font-family: monospace;
        }}

        .tab-title {{
            font-size: 15px;
            font-weight: 700;
            color: var(--slate-900);
        }}

        .tab-meta {{
            font-size: 12px;
            color: var(--slate-600);
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .pill-count {{
            background: #e2e8f0;
            color: #334155;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}

        .tab-messages-stream {{
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            background: #fcfdfe;
        }}

        .msg-card {{
            border-radius: 10px;
            padding: 16px 18px;
            font-size: 13px;
            line-height: 1.6;
        }}

        .user-msg {{
            background: #eef9f9;
            border: 1px solid #bceceb;
            border-left: 4px solid var(--df-primary);
            margin-left: 20px;
        }}

        .assistant-msg {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #00787c;
            margin-right: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }}

        .msg-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(0,0,0,0.06);
        }}

        .role-badge {{
            font-weight: 700;
            font-size: 11px;
            letter-spacing: 0.5px;
            color: var(--slate-700);
        }}

        .msg-time {{
            font-size: 11px;
            color: var(--slate-600);
            font-family: monospace;
        }}

        .msg-body p {{
            margin-bottom: 8px;
        }}

        .msg-body p:last-child {{
            margin-bottom: 0;
        }}

        .msg-body code {{
            background: #f1f5f9;
            color: var(--df-primary-dark);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
            font-family: monospace;
        }}

        .msg-body ul, .msg-body ol {{
            margin-left: 20px;
            margin-bottom: 8px;
        }}

        .msg-body li {{
            margin-bottom: 4px;
        }}

        .citation-badge {{
            display: inline-block;
            background: var(--df-primary-light);
            color: var(--df-primary-dark);
            border: 1px solid #a8eeec;
            padding: 1px 6px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 11px;
            margin: 0 2px;
        }}

        .attachments-row, .citations-row, .topk-row {{
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px dashed var(--slate-200);
            font-size: 11px;
            color: var(--slate-600);
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
        }}

        .att-badge {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }}

        .citation-pill {{
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
            font-family: monospace;
        }}

        .seed-pill {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 1px 5px;
            border-radius: 4px;
            font-size: 10px;
            font-family: monospace;
        }}

        .empty-tab-notice {{
            padding: 24px;
            text-align: center;
            color: var(--slate-600);
            font-size: 13px;
        }}

        /* Footer */
        .report-footer {{
            background: var(--slate-50);
            border-top: 1px solid var(--slate-200);
            padding: 18px 32px;
            text-align: center;
            font-size: 11px;
            color: var(--slate-600);
        }}

        /* Print Media Styles */
        @media print {{
            body {{
                background: #ffffff;
                padding: 0;
            }}
            .no-print {{
                display: none !important;
            }}
            .container {{
                box-shadow: none;
                border: none;
                border-radius: 0;
                max-width: 100%;
            }}
            .tab-section {{
                page-break-inside: avoid;
                margin-bottom: 24px;
            }}
            .msg-card {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>

    <div class="container">
        <!-- Header -->
        <header class="report-header">
            <div class="header-top">
                <div class="brand-title">
                    <span>DF Chatbot</span>
                    <span class="brand-badge">Audit Report</span>
                </div>
                <div class="header-actions no-print">
                    <button onclick="window.print()" class="btn-action">
                        🖨️ Print / Save as PDF
                    </button>
                    <button onclick="window.close()" class="btn-action">
                        ✕ Close Window
                    </button>
                </div>
            </div>
            <div class="report-subtitle">
                Multimodal Retrieval-Augmented Generation & User Conversation History
            </div>
            <div style="font-size: 12px; opacity: 0.85;">
                Report Scope: <strong>{scope_title}</strong>
            </div>
        </header>

        <!-- Metadata Overview -->
        <section class="meta-card">
            <div class="meta-item">
                <span class="meta-label">Username</span>
                <span class="meta-val">{html.escape(username)} (ID: #{user_id})</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Account Role</span>
                <span class="meta-val">{role}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Times of Login</span>
                <span class="meta-val">{login_count} Logins</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Registration Date</span>
                <span class="meta-val">{created_at}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Last Login</span>
                <span class="meta-val">{last_login_at}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Report Generation Time</span>
                <span class="meta-val">{generated_time}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Sessions Included</span>
                <span class="meta-val">{len(selected_tabs)} Chat Tab(s)</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Total Messages</span>
                <span class="meta-val">{total_messages_in_report} Messages</span>
            </div>
        </section>

        <!-- Main Body Content -->
        <main class="report-content">
            {tabs_rendered}
        </main>

        <!-- Footer -->
        <footer class="report-footer">
            <p><strong>DF Automation & Robotics</strong> • DF Chatbot Multimodal Technical Assistant</p>
            <p style="margin-top: 4px;">This conversation transcript report was automatically compiled from the secure user database.</p>
        </footer>
    </div>

</body>
</html>
"""
    return html_template


def generate_pdf_report(user_data: dict, tab_id: Optional[str] = None) -> bytes:
    """
    Generates a structured, professional PDF report of the user's chat history using ReportLab.
    Returns the binary content (bytes) of the PDF.
    """
    username = user_data.get("username", "Unknown User")
    user_id = user_data.get("id", 0)
    role = user_data.get("role", "user").capitalize()
    login_count = user_data.get("login_count", 0)
    created_at = user_data.get("created_at", "--")
    last_login_at = user_data.get("last_login_at") or "--"
    all_tabs = user_data.get("tabs", [])
    
    # Filter tabs if single tab requested
    target_tab = None
    if tab_id:
        filtered_tabs = [t for t in all_tabs if str(t.get("id")) == str(tab_id)]
        if filtered_tabs:
            target_tab = filtered_tabs[0]
            selected_tabs = [target_tab]
        else:
            selected_tabs = all_tabs
    else:
        selected_tabs = all_tabs
        
    scope_str = f"Single Tab: \"{target_tab.get('title', 'Chat Tab')}\"" if target_tab else f"All Chat Tabs ({len(selected_tabs)} Sessions)"
    total_messages = sum(t.get("message_count", len(t.get("messages", []))) for t in selected_tabs)
    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#00787c')
    )

    subtitle_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#475569')
    )

    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#475569')
    )

    meta_val_style = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0f172a')
    )

    tab_heading_style = ParagraphStyle(
        'TabHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#00787c')
    )

    tab_meta_style = ParagraphStyle(
        'TabMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#64748b')
    )

    user_header_style = ParagraphStyle(
        'UserHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#065f63')
    )

    assistant_header_style = ParagraphStyle(
        'AssistantHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#00787c')
    )

    body_style = ParagraphStyle(
        'MsgBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )

    citations_style = ParagraphStyle(
        'CitationsText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#00787c')
    )

    story = []

    # 1. Document Title Banner
    story.append(Paragraph('DF Chatbot — User Chat History Report', title_style))
    story.append(Paragraph(f'Multimodal Technical RAG Audit • Generated: {generated_time}', subtitle_style))
    story.append(Spacer(1, 8))

    # 2. Metadata Summary Table
    meta_data = [
        [
            Paragraph('Username:', meta_label_style),
            Paragraph(f'<b>{html.escape(username)}</b> (ID: #{user_id})', meta_val_style),
            Paragraph('Account Role:', meta_label_style),
            Paragraph(role, meta_val_style),
        ],
        [
            Paragraph('Registration Date:', meta_label_style),
            Paragraph(str(created_at), meta_val_style),
            Paragraph('Times of Login:', meta_label_style),
            Paragraph(f'{login_count} Logins', meta_val_style),
        ],
        [
            Paragraph('Report Scope:', meta_label_style),
            Paragraph(scope_str, meta_val_style),
            Paragraph('Total Messages:', meta_label_style),
            Paragraph(f'{total_messages} Messages in {len(selected_tabs)} Tab(s)', meta_val_style),
        ]
    ]

    meta_table = Table(meta_data, colWidths=[90, 170, 90, 173])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 3. Render Each Tab & Messages
    for tab_idx, tab in enumerate(selected_tabs, 1):
        tab_title = tab.get("title", "Untitled Tab")
        tab_created = tab.get("created_at", "--")
        messages = tab.get("messages", [])

        # Tab Section Header Banner
        tab_header_table = Table([
            [
                Paragraph(f'Session #{tab_idx}: <b>{html.escape(tab_title)}</b>', tab_heading_style),
                Paragraph(f'Created: {tab_created} | {len(messages)} msg(s)', tab_meta_style)
            ]
        ], colWidths=[360, 163])
        tab_header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e0f7f7')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#a8eeec')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ]))

        story.append(Spacer(1, 6))
        story.append(tab_header_table)
        story.append(Spacer(1, 6))

        if not messages:
            story.append(Paragraph('<i>No messages recorded in this chat session.</i>', subtitle_style))
            story.append(Spacer(1, 8))
            continue

        for msg in messages:
            is_user = msg.get("role") == "user"
            timestamp = msg.get("timestamp") or "--"
            content_raw = msg.get("content") or ""
            content_formatted = format_text_for_reportlab(content_raw)

            flowables_in_box = []

            if is_user:
                header_text = f'USER • {timestamp}'
                flowables_in_box.append(Paragraph(header_text, user_header_style))
                
                # User attachments
                attachments = msg.get("attachments", [])
                if attachments:
                    att_names = [f"[{a.get('type', 'img').upper()}] {html.escape(a.get('name', 'file'))}" for a in attachments]
                    att_str = f"<b>Attached Files:</b> {', '.join(att_names)}"
                    flowables_in_box.append(Spacer(1, 2))
                    flowables_in_box.append(Paragraph(att_str, citations_style))

                flowables_in_box.append(Spacer(1, 3))
                flowables_in_box.append(Paragraph(content_formatted, body_style))

                # User box table
                user_box = Table([[flowables_in_box]], colWidths=[523])
                user_box.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#edfcfb')),
                    ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#6ee0dd')),
                    ('LINELEFT', (0, 0), (-1, -1), 3, colors.HexColor('#00b5b8')),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ]))
                story.append(user_box)
            else:
                header_text = f'DF CHATBOT • {timestamp}'
                flowables_in_box.append(Paragraph(header_text, assistant_header_style))
                flowables_in_box.append(Spacer(1, 3))
                flowables_in_box.append(Paragraph(content_formatted, body_style))

                # Citations
                citations = msg.get("citations", [])
                if citations:
                    cit_strs = [f"[{html.escape(str(c.get('manual', 'Manual')))}, p.{c.get('page_number', '')}]" for c in citations]
                    cit_line = f"<b>Verified Citations:</b> {', '.join(cit_strs)}"
                    flowables_in_box.append(Spacer(1, 3))
                    flowables_in_box.append(Paragraph(cit_line, citations_style))

                # Top-K summary
                top_k = msg.get("top_k", [])
                if top_k:
                    top_k_names = [f"{k.get('pdf_stem', 'Manual')} (p.{k.get('page_number', '')})" for k in top_k[:4]]
                    top_k_line = f"<font color='#64748b'><b>Candidate Sources:</b> {', '.join(top_k_names)}</font>"
                    flowables_in_box.append(Spacer(1, 2))
                    flowables_in_box.append(Paragraph(top_k_line, citations_style))

                assistant_box = Table([[flowables_in_box]], colWidths=[523])
                assistant_box.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                    ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
                    ('LINELEFT', (0, 0), (-1, -1), 3, colors.HexColor('#00787c')),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ]))
                story.append(assistant_box)

            story.append(Spacer(1, 6))

    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()


def generate_multi_user_html_report(users_data_list: List[dict]) -> str:
    """
    Generates a unified, comprehensive standalone HTML report for multiple users' chat histories.
    Includes an executive summary dashboard, user directory, and individual user sections.
    """
    total_users = len(users_data_list)
    total_tabs = sum(len(u.get("tabs", [])) for u in users_data_list)
    total_messages = sum(sum(len(t.get("messages", [])) for t in u.get("tabs", [])) for u in users_data_list)
    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build User Directory Table of Contents
    toc_rows_html = []
    for idx, u in enumerate(users_data_list, 1):
        username = html.escape(u.get("username", "Unknown"))
        user_id = u.get("id", 0)
        role = html.escape(u.get("role", "user").capitalize())
        u_tabs = u.get("tabs", [])
        u_msgs = sum(len(t.get("messages", [])) for t in u_tabs)
        logins = u.get("login_count", 0)
        reg_date = u.get("created_at", "--")

        toc_rows_html.append(f"""
        <tr>
            <td style="padding: 10px 14px; font-weight: 700; color: #00787c;">#{idx}</td>
            <td style="padding: 10px 14px;"><a href="#user-sec-{user_id}" style="color: #0b57d0; font-weight: 700; text-decoration: none;">{username} (ID: #{user_id})</a></td>
            <td style="padding: 10px 14px;"><span style="background: #e0f7f7; color: #00787c; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 700;">{role}</span></td>
            <td style="padding: 10px 14px; font-family: monospace;">{logins} logins</td>
            <td style="padding: 10px 14px; font-weight: 600;">{len(u_tabs)} tabs / {u_msgs} msgs</td>
            <td style="padding: 10px 14px; font-size: 11px; color: #64748b;">{reg_date}</td>
        </tr>
        """)
    toc_rendered = "\n".join(toc_rows_html)

    # Build Each User's Full Section
    user_sections_html = []
    for idx, u in enumerate(users_data_list, 1):
        username = html.escape(u.get("username", "Unknown"))
        user_id = u.get("id", 0)
        role = html.escape(u.get("role", "user").capitalize())
        login_count = u.get("login_count", 0)
        created_at = u.get("created_at", "--")
        last_login_at = u.get("last_login_at") or "--"
        u_tabs = u.get("tabs", [])
        u_total_msgs = sum(len(t.get("messages", [])) for t in u_tabs)

        tabs_html_list = []
        if not u_tabs:
            tabs_html_list.append("""
            <div class="empty-tab-notice" style="background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 10px; margin: 12px 0;">
                <em>This user account has no active chat sessions.</em>
            </div>
            """)
        else:
            for tab_idx, tab in enumerate(u_tabs, 1):
                tab_title = tab.get("title", "Untitled Tab")
                tab_created = tab.get("created_at", "--")
                messages = tab.get("messages", [])

                messages_html_list = []
                if not messages:
                    messages_html_list.append("""
                    <div class="empty-tab-notice">
                        <em>No messages recorded in this chat session.</em>
                    </div>
                    """)
                else:
                    for msg in messages:
                        is_user = msg.get("role") == "user"
                        role_label = "USER" if is_user else "DF CHATBOT"
                        timestamp = msg.get("timestamp") or "--"
                        content_raw = msg.get("content") or ""
                        formatted_body = clean_markdown_to_html(content_raw)

                        attachments_html = ""
                        attachments = msg.get("attachments", [])
                        if attachments:
                            att_items = []
                            for att in attachments:
                                att_name = html.escape(att.get("name", "Attachment"))
                                att_type = att.get("type", "image")
                                if att_type == "pdf":
                                    att_items.append(f'<span class="att-badge pdf">📄 {att_name}</span>')
                                else:
                                    att_items.append(f'<span class="att-badge img">🖼️ {att_name}</span>')
                            attachments_html = f'<div class="attachments-row"><strong>Attached Files:</strong> {" ".join(att_items)}</div>'

                        citations_html = ""
                        citations = msg.get("citations", [])
                        if citations:
                            cit_items = []
                            for c in citations:
                                c_manual = html.escape(str(c.get("manual", "Manual")))
                                c_page = c.get("page_number", "")
                                cit_items.append(f'<span class="citation-pill">📖 {c_manual}, p.{c_page}</span>')
                            citations_html = f'<div class="citations-row"><strong>Verified In-Text Citations:</strong> {" ".join(cit_items)}</div>'

                        top_k_html = ""
                        top_k = msg.get("top_k", [])
                        if top_k:
                            top_k_items = []
                            for k in top_k[:4]:
                                k_stem = html.escape(str(k.get("pdf_stem", "Manual")))
                                k_page = k.get("page_number", "")
                                top_k_items.append(f'<span class="seed-pill">{k_stem} (p.{k_page})</span>')
                            top_k_html = f'<div class="topk-row"><strong>Top Matched Seed Pages:</strong> {" • ".join(top_k_items)}</div>'

                        msg_card_class = "msg-card user-msg" if is_user else "msg-card assistant-msg"
                        role_icon = "👤" if is_user else "🤖"

                        messages_html_list.append(f"""
                        <div class="{msg_card_class}">
                            <div class="msg-header">
                                <span class="role-badge">{role_icon} {role_label}</span>
                                <span class="msg-time">{timestamp}</span>
                            </div>
                            {attachments_html}
                            <div class="msg-body">
                                {formatted_body}
                            </div>
                            {citations_html}
                            {top_k_html}
                        </div>
                        """)

                messages_rendered = "\n".join(messages_html_list)

                tabs_html_list.append(f"""
                <section class="tab-section">
                    <div class="tab-header">
                        <div class="tab-title-wrap">
                            <span class="tab-index-badge">Session #{tab_idx}</span>
                            <h3 class="tab-title">{html.escape(tab_title)}</h3>
                        </div>
                        <div class="tab-meta">
                            <span>Created: {tab_created}</span>
                            <span class="pill-count">{len(messages)} Messages</span>
                        </div>
                    </div>
                    <div class="tab-messages-stream">
                        {messages_rendered}
                    </div>
                </section>
                """)

        tabs_rendered = "\n".join(tabs_html_list)

        user_sections_html.append(f"""
        <div id="user-sec-{user_id}" style="margin-top: 40px; padding-top: 24px; border-top: 3px solid #00b5b8;">
            <!-- User Header Card -->
            <div style="background: linear-gradient(135deg, #0b4e51 0%, #00787c 100%); color: #ffffff; padding: 20px 24px; border-radius: 12px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="background: rgba(255,255,255,0.2); padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 700;">USER #{idx} OF {total_users}</span>
                        <h2 style="font-size: 20px; font-weight: 800;">{username} (ID: #{user_id})</h2>
                        <span style="background: #e0f7f7; color: #00787c; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700;">{role}</span>
                    </div>
                    <p style="font-size: 12px; opacity: 0.9; margin-top: 4px;">Registered: {created_at} • Last Active: {last_login_at} • Logins: {login_count}</p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 18px; font-weight: 800; color: #a8eeec;">{len(u_tabs)} Tabs</span>
                    <span style="font-size: 12px; opacity: 0.9; display: block;">{u_total_msgs} Total Messages</span>
                </div>
            </div>

            <!-- User Tabs Stream -->
            {tabs_rendered}
        </div>
        """)

    all_users_rendered = "\n".join(user_sections_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DF Chatbot — Multi-User Chat History Audit Report ({total_users} Users)</title>
    <style>
        :root {{
            --df-primary: #00b5b8;
            --df-primary-dark: #00787c;
            --df-primary-light: #e0f7f7;
            --slate-900: #0f172a;
            --slate-800: #1e293b;
            --slate-700: #334155;
            --slate-600: #475569;
            --slate-200: #e2e8f0;
            --slate-100: #f1f5f9;
            --slate-50: #f8fafc;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--slate-100);
            color: var(--slate-800);
            line-height: 1.6;
            padding: 24px;
        }}

        .container {{
            max-width: 1050px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
            border: 1px solid var(--slate-200);
            overflow: hidden;
        }}

        /* Header Bar */
        .report-header {{
            background: linear-gradient(135deg, #0b4e51 0%, #00787c 50%, #00b5b8 100%);
            color: #ffffff;
            padding: 28px 32px;
            position: relative;
        }}

        .header-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .brand-title {{
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .brand-badge {{
            background: rgba(255, 255, 255, 0.2);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        .report-subtitle {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 6px;
        }}

        .header-actions {{
            display: flex;
            gap: 10px;
        }}

        .btn-action {{
            background: #ffffff;
            color: var(--df-primary-dark);
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            text-decoration: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .btn-action:hover {{
            background: var(--df-primary-light);
            transform: translateY(-1px);
        }}

        /* Metadata Grid */
        .meta-card {{
            background: var(--slate-50);
            border-bottom: 1px solid var(--slate-200);
            padding: 20px 32px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }}

        .meta-item {{
            display: flex;
            flex-direction: column;
        }}

        .meta-label {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--slate-600);
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }}

        .meta-val {{
            font-size: 14px;
            font-weight: 600;
            color: var(--slate-900);
        }}

        /* Content Area */
        .report-content {{
            padding: 32px;
        }}

        .tab-section {{
            margin-bottom: 24px;
            border: 1px solid var(--slate-200);
            border-radius: 12px;
            overflow: hidden;
            background: #ffffff;
        }}

        .tab-header {{
            background: #f8fafc;
            border-bottom: 1px solid var(--slate-200);
            padding: 12px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .tab-title-wrap {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .tab-index-badge {{
            background: var(--df-primary-light);
            color: var(--df-primary-dark);
            border: 1px solid #a8eeec;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
        }}

        .tab-title {{
            font-size: 14px;
            font-weight: 700;
            color: var(--slate-900);
        }}

        .tab-meta {{
            font-size: 11px;
            color: var(--slate-600);
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .pill-count {{
            background: var(--slate-200);
            color: var(--slate-700);
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 600;
        }}

        .tab-messages-stream {{
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            background: #fcfdfe;
        }}

        .msg-card {{
            border-radius: 10px;
            padding: 14px 16px;
            font-size: 13px;
            line-height: 1.6;
        }}

        .user-msg {{
            background: #eef9f9;
            border: 1px solid #bceceb;
            border-left: 4px solid var(--df-primary);
            margin-left: 16px;
        }}

        .assistant-msg {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #00787c;
            margin-right: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }}

        .msg-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(0,0,0,0.06);
        }}

        .role-badge {{
            font-weight: 700;
            font-size: 11px;
            letter-spacing: 0.5px;
            color: var(--slate-700);
        }}

        .msg-time {{
            font-size: 11px;
            color: var(--slate-600);
            font-family: monospace;
        }}

        .msg-body p {{
            margin-bottom: 6px;
        }}

        .msg-body p:last-child {{
            margin-bottom: 0;
        }}

        .citation-badge {{
            display: inline-block;
            background: var(--df-primary-light);
            color: var(--df-primary-dark);
            border: 1px solid #a8eeec;
            padding: 1px 6px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 11px;
            margin: 0 2px;
        }}

        .attachments-row, .citations-row, .topk-row {{
            margin-top: 8px;
            padding-top: 6px;
            border-top: 1px dashed var(--slate-200);
            font-size: 11px;
            color: var(--slate-600);
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
        }}

        .att-badge {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }}

        .citation-pill {{
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
            font-family: monospace;
        }}

        .seed-pill {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 1px 5px;
            border-radius: 4px;
            font-size: 10px;
            font-family: monospace;
        }}

        .empty-tab-notice {{
            padding: 20px;
            text-align: center;
            color: var(--slate-600);
            font-size: 13px;
        }}

        .report-footer {{
            background: var(--slate-50);
            border-top: 1px solid var(--slate-200);
            padding: 18px 32px;
            text-align: center;
            font-size: 11px;
            color: var(--slate-600);
        }}

        @media print {{
            body {{
                background: #ffffff;
                padding: 0;
            }}
            .no-print {{
                display: none !important;
            }}
            .container {{
                box-shadow: none;
                border: none;
                border-radius: 0;
                max-width: 100%;
            }}
            .tab-section {{
                page-break-inside: avoid;
                margin-bottom: 20px;
            }}
        }}
    </style>
</head>
<body>

    <div class="container">
        <!-- Header -->
        <header class="report-header">
            <div class="header-top">
                <div class="brand-title">
                    <span>DF Chatbot</span>
                    <span class="brand-badge">Multi-User Audit</span>
                </div>
                <div class="header-actions no-print">
                    <button onclick="window.print()" class="btn-action">
                        🖨️ Print / Save as PDF
                    </button>
                    <button onclick="window.close()" class="btn-action">
                        ✕ Close Window
                    </button>
                </div>
            </div>
            <div class="report-subtitle">
                Comprehensive Multi-User Conversation History & Technical RAG Audit
            </div>
            <div style="font-size: 12px; opacity: 0.85;">
                Total Selected Users: <strong>{total_users}</strong> • Total Sessions: <strong>{total_tabs}</strong> • Total Messages: <strong>{total_messages}</strong>
            </div>
        </header>

        <!-- Executive Metadata Overview -->
        <section class="meta-card">
            <div class="meta-item">
                <span class="meta-label">Total Users</span>
                <span class="meta-val">{total_users} Accounts</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Total Chat Sessions</span>
                <span class="meta-val">{total_tabs} Tabs</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Total Messages</span>
                <span class="meta-val">{total_messages} Messages</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Generated Timestamp</span>
                <span class="meta-val">{generated_time}</span>
            </div>
        </section>

        <!-- Report Content -->
        <main class="report-content">
            <!-- User Table of Contents / Index -->
            <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; margin-bottom: 30px;">
                <div style="background: #f8fafc; padding: 12px 18px; border-bottom: 1px solid #cbd5e1; font-weight: 700; font-size: 13px; color: #0f172a;">
                    📋 User Accounts Index ({total_users} Selected)
                </div>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; text-align: left; border-collapse: collapse; font-size: 12px;">
                        <thead style="background: #f1f5f9; color: #475569; font-weight: 700; border-bottom: 1px solid #cbd5e1; text-transform: uppercase; font-size: 10px;">
                            <tr>
                                <th style="padding: 8px 14px;">#</th>
                                <th style="padding: 8px 14px;">User</th>
                                <th style="padding: 8px 14px;">Role</th>
                                <th style="padding: 8px 14px;">Logins</th>
                                <th style="padding: 8px 14px;">Chat Activity</th>
                                <th style="padding: 8px 14px;">Registered</th>
                            </tr>
                        </thead>
                        <tbody style="divide-y divide-slate-200;">
                            {toc_rendered}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Full Users Transcript Deck -->
            {all_users_rendered}
        </main>

        <!-- Footer -->
        <footer class="report-footer">
            <p><strong>DF Automation & Robotics Sdn. Bhd.</strong> • DF Chatbot Autonomous Technical Assistant System</p>
            <p style="margin-top: 4px; color: var(--slate-600);">Generated on {generated_time} • Strictly Confidential Internal Audit Report</p>
        </footer>
    </div>

</body>
</html>
"""


def generate_multi_user_pdf_report(users_data_list: List[dict]) -> bytes:
    """
    Generates a structured, professional combined PDF report of multiple users' chat histories.
    """
    total_users = len(users_data_list)
    total_tabs = sum(len(u.get("tabs", [])) for u in users_data_list)
    total_messages = sum(sum(len(t.get("messages", [])) for t in u.get("tabs", [])) for u in users_data_list)
    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'MainTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=colors.HexColor('#00787c')
    )
    subtitle_style = ParagraphStyle(
        'SubTitle', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=12, textColor=colors.HexColor('#475569')
    )
    meta_label_style = ParagraphStyle(
        'MetaLabel', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor('#475569')
    )
    meta_val_style = ParagraphStyle(
        'MetaVal', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor('#0f172a')
    )
    user_banner_style = ParagraphStyle(
        'UserBanner', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=colors.white
    )
    tab_heading_style = ParagraphStyle(
        'TabHeading', parent=styles['Heading3'], fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=colors.HexColor('#00787c')
    )
    tab_meta_style = ParagraphStyle(
        'TabMeta', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor('#64748b')
    )
    user_header_style = ParagraphStyle(
        'UserHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor('#065f63')
    )
    assistant_header_style = ParagraphStyle(
        'AssistantHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor('#00787c')
    )
    body_style = ParagraphStyle(
        'MsgBody', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=12, textColor=colors.HexColor('#1e293b')
    )
    citations_style = ParagraphStyle(
        'CitationsText', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor('#00787c')
    )

    story = []

    # 1. Document Title Banner
    story.append(Paragraph('DF Chatbot — Multi-User Chat History Audit Report', title_style))
    story.append(Paragraph(f'Consolidated Technical RAG Audit ({total_users} Users) • Generated: {generated_time}', subtitle_style))
    story.append(Spacer(1, 8))

    # 2. Executive Metadata Summary Table
    meta_data = [
        [
            Paragraph('Total Selected Users:', meta_label_style),
            Paragraph(f'<b>{total_users} Users</b>', meta_val_style),
            Paragraph('Total Chat Sessions:', meta_label_style),
            Paragraph(f'<b>{total_tabs} Tabs</b>', meta_val_style),
        ],
        [
            Paragraph('Total Messages:', meta_label_style),
            Paragraph(f'<b>{total_messages} Messages</b>', meta_val_style),
            Paragraph('Generated Timestamp:', meta_label_style),
            Paragraph(generated_time, meta_val_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[120, 140, 120, 143])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 3. Iterate Each User
    for u_idx, u in enumerate(users_data_list, 1):
        username = u.get("username", "Unknown User")
        user_id = u.get("id", 0)
        role = u.get("role", "user").capitalize()
        login_count = u.get("login_count", 0)
        created_at = u.get("created_at", "--")
        u_tabs = u.get("tabs", [])
        u_msgs = sum(len(t.get("messages", [])) for t in u_tabs)

        # User Section Banner
        user_banner_table = Table([
            [
                Paragraph(f'<b>User #{u_idx}: {html.escape(username)}</b> (ID: #{user_id} • {role})', user_banner_style),
                Paragraph(f'Logins: {login_count} | {len(u_tabs)} Tabs ({u_msgs} Msgs)', tab_meta_style)
            ]
        ], colWidths=[360, 163])
        user_banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#00787c')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#0b4e51')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ]))

        story.append(Spacer(1, 8))
        story.append(user_banner_table)
        story.append(Spacer(1, 6))

        if not u_tabs:
            story.append(Paragraph('<i>No chat sessions recorded for this user.</i>', subtitle_style))
            story.append(Spacer(1, 10))
            continue

        for tab_idx, tab in enumerate(u_tabs, 1):
            tab_title = tab.get("title", "Untitled Tab")
            tab_created = tab.get("created_at", "--")
            messages = tab.get("messages", [])

            tab_header_table = Table([
                [
                    Paragraph(f'Session #{tab_idx}: <b>{html.escape(tab_title)}</b>', tab_heading_style),
                    Paragraph(f'Created: {tab_created} | {len(messages)} msg(s)', tab_meta_style)
                ]
            ], colWidths=[360, 163])
            tab_header_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e0f7f7')),
                ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#a8eeec')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ]))

            story.append(tab_header_table)
            story.append(Spacer(1, 4))

            if not messages:
                story.append(Paragraph('<i>No messages in this chat session.</i>', subtitle_style))
                story.append(Spacer(1, 6))
                continue

            for msg in messages:
                is_user = msg.get("role") == "user"
                timestamp = msg.get("timestamp") or "--"
                content_raw = msg.get("content") or ""
                content_formatted = format_text_for_reportlab(content_raw)

                flowables_in_box = []

                if is_user:
                    header_text = f'USER • {timestamp}'
                    flowables_in_box.append(Paragraph(header_text, user_header_style))

                    attachments = msg.get("attachments", [])
                    if attachments:
                        att_names = [f"[{a.get('type', 'img').upper()}] {html.escape(a.get('name', 'file'))}" for a in attachments]
                        att_str = f"<b>Attached Files:</b> {', '.join(att_names)}"
                        flowables_in_box.append(Spacer(1, 2))
                        flowables_in_box.append(Paragraph(att_str, citations_style))

                    flowables_in_box.append(Spacer(1, 2))
                    flowables_in_box.append(Paragraph(content_formatted, body_style))

                    user_box = Table([[flowables_in_box]], colWidths=[523])
                    user_box.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#edfcfb')),
                        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#6ee0dd')),
                        ('LINELEFT', (0, 0), (-1, -1), 3, colors.HexColor('#00b5b8')),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ('LEFTPADDING', (0, 0), (-1, -1), 7),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                    ]))
                    story.append(user_box)
                else:
                    header_text = f'DF CHATBOT • {timestamp}'
                    flowables_in_box.append(Paragraph(header_text, assistant_header_style))
                    flowables_in_box.append(Spacer(1, 2))
                    flowables_in_box.append(Paragraph(content_formatted, body_style))

                    citations = msg.get("citations", [])
                    if citations:
                        cit_strs = [f"[{html.escape(str(c.get('manual', 'Manual')))}, p.{c.get('page_number', '')}]" for c in citations]
                        cit_line = f"<b>Verified Citations:</b> {', '.join(cit_strs)}"
                        flowables_in_box.append(Spacer(1, 2))
                        flowables_in_box.append(Paragraph(cit_line, citations_style))

                    assistant_box = Table([[flowables_in_box]], colWidths=[523])
                    assistant_box.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
                        ('LINELEFT', (0, 0), (-1, -1), 3, colors.HexColor('#00787c')),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ('LEFTPADDING', (0, 0), (-1, -1), 7),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                    ]))
                    story.append(assistant_box)

                story.append(Spacer(1, 4))

        story.append(Spacer(1, 10))

    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()


def generate_multi_user_zip(users_data_list: List[dict], format: str = "pdf") -> bytes:
    """
    Generates an in-memory ZIP file containing individual PDF or HTML reports for each user.
    """
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        summary_lines = [
            "DF CHATBOT — BATCH USER CHAT HISTORY EXPORT SUMMARY",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Exported Users: {len(users_data_list)}",
            "-" * 60
        ]

        for u in users_data_list:
            user_id = u.get("id", 0)
            username = u.get("username", "user")
            username_safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', username)
            tabs_count = len(u.get("tabs", []))
            msgs_count = sum(len(t.get("messages", [])) for t in u.get("tabs", []))

            summary_lines.append(f"User #{user_id} ({username}): {tabs_count} tabs, {msgs_count} messages")

            if format == "pdf":
                file_bytes = generate_pdf_report(u)
                file_name = f"df_chatbot_report_{username_safe}_id{user_id}.pdf"
            else:
                html_str = generate_html_report(u)
                file_bytes = html_str.encode("utf-8")
                file_name = f"df_chatbot_report_{username_safe}_id{user_id}.html"

            zf.writestr(file_name, file_bytes)

        zf.writestr("AUDIT_SUMMARY.txt", "\n".join(summary_lines))

    return zip_buf.getvalue()

