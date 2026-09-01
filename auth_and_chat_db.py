"""
Database management module for User Authentication, Chat Tabs, and Chat Message History with Conversational Memory.
Uses SQLite stored persistently in config.OUTPUT_DIR / "users_and_chats.db".
"""

from __future__ import annotations

import sqlite3
import json
import uuid
import logging
from typing import Optional, List, Dict, Any
from werkzeug.security import generate_password_hash, check_password_hash
import config

logger = logging.getLogger(__name__)
DB_PATH = config.USER_DB_PATH


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Initializes the database schema if not already present in the 'User database' folder."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Users Table (with profile_pic support)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                profile_pic TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Gracefully add profile_pic column if table was previously created without it
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT DEFAULT '';")
        except sqlite3.OperationalError:
            pass

        # 2. Chat Tabs / Sessions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_tabs (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # 3. Chat Messages Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tab_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations_json TEXT DEFAULT '[]',
                top_k_json TEXT DEFAULT '[]',
                expanded_count INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(tab_id) REFERENCES chat_tabs(id) ON DELETE CASCADE
            );
        """)
        
        # Seed default admin user 'df' (ID: df, Password: df)
        cursor.execute("SELECT id FROM users WHERE username = 'df'")
        if not cursor.fetchone():
            df_admin_hash = generate_password_hash("df")
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, profile_pic) VALUES (?, ?, ?, ?)",
                ("df", df_admin_hash, "admin", "")
            )
            logger.info("Initialized default admin user 'df' (password: 'df') in User database.")

        # Also seed 'admin' alias
        cursor.execute("SELECT id FROM users WHERE username = 'admin'")
        if not cursor.fetchone():
            default_admin_hash = generate_password_hash(config.ADMIN_PASSWORD)
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, profile_pic) VALUES (?, ?, ?, ?)",
                ("admin", default_admin_hash, "admin", "")
            )
        
        conn.commit()


# ============================================================================
# User Authentication Helpers
# ============================================================================

def validate_no_spaces(val: str, field_name: str = "Field"):
    if not val or not isinstance(val, str):
        raise ValueError(f"{field_name} cannot be empty.")
    if " " in val or "\t" in val or "\n" in val:
        raise ValueError(f"{field_name} cannot contain spaces or whitespace characters.")
    if len(val) < 2:
        raise ValueError(f"{field_name} must be at least 2 characters long.")


def register_user(username: str, password: str, confirm_password: str) -> dict:
    """Registers a new user account with no-space validation and password confirmation."""
    username = (username or "").strip()
    password = password or ""
    confirm_password = confirm_password or ""

    validate_no_spaces(username, "Username / Name")
    validate_no_spaces(password, "Password")

    if password != confirm_password:
        raise ValueError("Passwords do not match. Please re-enter to confirm.")

    if len(password) < 2:
        raise ValueError("Password must be at least 2 characters long.")

    pw_hash = generate_password_hash(password)
    role = "admin" if username.lower() in ["df", "admin"] else "user"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, profile_pic) VALUES (?, ?, ?, ?)",
                (username, pw_hash, role, "")
            )
            user_id = cursor.lastrowid
            
            # Create a default first chat tab for the user
            tab_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO chat_tabs (id, user_id, title) VALUES (?, ?, ?)",
                (tab_id, user_id, "New Chat")
            )
            conn.commit()

            return {
                "id": user_id,
                "username": username,
                "role": role,
                "profile_pic": "",
                "default_tab_id": tab_id
            }
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists. Please choose another username.")


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticates a user by username (Name) and password. Supports ID: df / password: df."""
    username = (username or "").strip()
    password = password or ""

    validate_no_spaces(username, "Username / Name")
    validate_no_spaces(password, "Password")

    # If logging in as admin (ID 'df' with password 'df' or ADMIN_PASSWORD)
    if username.lower() in ["df", "admin"] and (password == "df" or config.verify_admin_password(password)):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username.lower(),))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "username": row["username"],
                    "role": "admin",
                    "profile_pic": row["profile_pic"] or ""
                }
            else:
                pw_hash = generate_password_hash(password)
                cursor.execute(
                    "INSERT INTO users (username, password_hash, role, profile_pic) VALUES (?, ?, 'admin', '')",
                    (username.lower(), pw_hash)
                )
                conn.commit()
                return {
                    "id": cursor.lastrowid,
                    "username": username.lower(),
                    "role": "admin",
                    "profile_pic": ""
                }

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            return None

        if check_password_hash(row["password_hash"], password):
            return {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "profile_pic": row["profile_pic"] or ""
            }
        return None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, profile_pic, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def update_user_profile_picture(user_id: int, profile_pic_filename: str) -> Optional[dict]:
    """Updates user profile picture filename in the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET profile_pic = ? WHERE id = ?", (profile_pic_filename, user_id))
        conn.commit()
        cursor.execute("SELECT id, username, role, profile_pic, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


# ============================================================================
# Chat Tabs / Sessions Helpers
# ============================================================================

def list_user_tabs(user_id: int) -> List[dict]:
    """Returns all tabs for a specific user, ordered by most recently updated."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.id, t.title, t.created_at, t.updated_at,
                   COUNT(m.id) as message_count
            FROM chat_tabs t
            LEFT JOIN chat_messages m ON t.id = m.tab_id
            WHERE t.user_id = ?
            GROUP BY t.id
            ORDER BY t.updated_at DESC, t.created_at DESC
        """, (user_id,))
        rows = cursor.fetchall()
        tabs = [dict(r) for r in rows]

        # If user has 0 tabs, auto-create one
        if not tabs:
            new_tab = create_tab(user_id, "New Chat")
            tabs = [new_tab]

        return tabs


def create_tab(user_id: int, title: str = "New Chat") -> dict:
    tab_id = str(uuid.uuid4())
    title = (title or "New Chat").strip()[:60]
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_tabs (id, user_id, title) VALUES (?, ?, ?)",
            (tab_id, user_id, title)
        )
        conn.commit()
        return {
            "id": tab_id,
            "title": title,
            "message_count": 0
        }


def delete_tab(tab_id: str, user_id: int) -> bool:
    """Deletes a chat tab and all its messages. Verifies user ownership."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_tabs WHERE id = ? AND user_id = ?", (tab_id, user_id))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted


def update_tab_title(tab_id: str, title: str) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chat_tabs SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (title[:60], tab_id)
        )
        conn.commit()


def touch_tab(tab_id: str) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE chat_tabs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (tab_id,))
        conn.commit()


# ============================================================================
# Chat Messages & Conversational Memory Helpers
# ============================================================================

def get_tab_messages(tab_id: str, user_id: int) -> List[dict]:
    """Retrieves full message history for a tab, ensuring user ownership."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Verify ownership
        cursor.execute("SELECT id FROM chat_tabs WHERE id = ? AND user_id = ?", (tab_id, user_id))
        if not cursor.fetchone():
            return []

        cursor.execute("""
            SELECT id, tab_id, role, content, citations_json, top_k_json, expanded_count, timestamp
            FROM chat_messages
            WHERE tab_id = ?
            ORDER BY id ASC
        """, (tab_id,))
        rows = cursor.fetchall()
        
        results = []
        for r in rows:
            item = dict(r)
            try:
                item["citations"] = json.loads(item.get("citations_json") or "[]")
            except Exception:
                item["citations"] = []
            try:
                item["top_k"] = json.loads(item.get("top_k_json") or "[]")
            except Exception:
                item["top_k"] = []
            results.append(item)
        return results


def add_chat_message(
    tab_id: str,
    role: str,
    content: str,
    citations: Optional[List[dict]] = None,
    top_k: Optional[List[dict]] = None,
    expanded_count: int = 0
) -> int:
    """Appends a user or assistant message to the tab history and updates tab timestamp."""
    citations_str = json.dumps(citations or [])
    top_k_str = json.dumps(top_k or [])
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_messages (tab_id, role, content, citations_json, top_k_json, expanded_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (tab_id, role, content, citations_str, top_k_str, expanded_count))
        msg_id = cursor.lastrowid
        cursor.execute("UPDATE chat_tabs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (tab_id,))
        conn.commit()
        return msg_id


def get_tab_conversation_memory(tab_id: str, max_turns: int = 6) -> List[Dict[str, str]]:
    """
    Returns recent multi-turn question & answer pairs from this specific tab history
    formatted for conversational memory conditioning.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content
            FROM chat_messages
            WHERE tab_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (tab_id, max_turns * 2))
        rows = cursor.fetchall()
        
        # Reverse to chronological order (oldest to newest)
        chronological = list(reversed([dict(r) for r in rows]))
        return chronological


# Automatically initialize DB tables on import
init_db()
