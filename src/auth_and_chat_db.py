"""
Database management module for User Authentication, Chat Tabs, and Chat Message History with Conversational Memory.
Uses SQLite stored persistently in config.USER_DB_PATH ("data/user_storage/users_and_chats.db").
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
    """Initializes the database schema if not already present in 'data/user_storage'."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Users Table (with profile_pic, login_count, and last_login_at support)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                profile_pic TEXT DEFAULT '',
                login_count INTEGER DEFAULT 0,
                last_login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Gracefully add profile_pic, login_count, last_login_at columns if table was created earlier
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT DEFAULT '';")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN login_count INTEGER DEFAULT 0;")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN last_login_at TIMESTAMP;")
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
                attachments_json TEXT DEFAULT '[]',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(tab_id) REFERENCES chat_tabs(id) ON DELETE CASCADE
            );
        """)

        # Gracefully add attachments_json column if table was previously created without it
        try:
            cursor.execute("ALTER TABLE chat_messages ADD COLUMN attachments_json TEXT DEFAULT '[]';")
        except sqlite3.OperationalError:
            pass
        
        # Seed default admin user 'df' (ID: df, Password: df)
        cursor.execute("SELECT id FROM users WHERE username = 'df'")
        if not cursor.fetchone():
            df_admin_hash = generate_password_hash("df")
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, profile_pic, login_count) VALUES (?, ?, ?, ?, ?)",
                ("df", df_admin_hash, "admin", "", 1)
            )
            logger.info("Initialized default admin user 'df' (password: 'df') in User database.")

        # Ensure no legacy 'admin' user alias exists and only 'df' has admin role
        cursor.execute("DELETE FROM users WHERE username = 'admin'")
        cursor.execute("UPDATE users SET role = 'user' WHERE username != 'df'")
        cursor.execute("UPDATE users SET role = 'admin' WHERE username = 'df'")
        
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

    if username.lower() == "df":
        raise ValueError("Username 'df' is reserved for the system administrator.")

    if password != confirm_password:
        raise ValueError("Passwords do not match. Please re-enter to confirm.")

    if len(password) < 2:
        raise ValueError("Password must be at least 2 characters long.")

    pw_hash = generate_password_hash(password)
    role = "user"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, profile_pic, login_count, last_login_at) VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)",
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
                "login_count": 1,
                "default_tab_id": tab_id
            }
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists. Please choose another username.")


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticates a user by username (Name) and password. Tracks login counts and timestamps."""
    username = (username or "").strip()
    password = password or ""

    validate_no_spaces(username, "Username / Name")
    validate_no_spaces(password, "Password")

    # If logging in as admin (ID 'df' with password 'df')
    if username.lower() == "df":
        if password == "df" or config.verify_admin_password(password):
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE username = 'df'")
                row = cursor.fetchone()
                if row:
                    user_id = row["id"]
                    new_count = (row["login_count"] or 0) + 1
                    cursor.execute("UPDATE users SET login_count = ?, last_login_at = CURRENT_TIMESTAMP WHERE id = ?", (new_count, user_id))
                    conn.commit()
                    return {
                        "id": user_id,
                        "username": "df",
                        "role": "admin",
                        "profile_pic": row["profile_pic"] or "",
                        "login_count": new_count
                    }
                else:
                    pw_hash = generate_password_hash("df")
                    cursor.execute(
                        "INSERT INTO users (username, password_hash, role, profile_pic, login_count, last_login_at) VALUES ('df', ?, 'admin', '', 1, CURRENT_TIMESTAMP)",
                        (pw_hash,)
                    )
                    conn.commit()
                    return {
                        "id": cursor.lastrowid,
                        "username": "df",
                        "role": "admin",
                        "profile_pic": "",
                        "login_count": 1
                    }
        return None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            return None

        if check_password_hash(row["password_hash"], password):
            user_id = row["id"]
            new_count = (row["login_count"] or 0) + 1
            cursor.execute("UPDATE users SET login_count = ?, last_login_at = CURRENT_TIMESTAMP WHERE id = ?", (new_count, user_id))
            conn.commit()
            return {
                "id": user_id,
                "username": row["username"],
                "role": row["role"],
                "profile_pic": row["profile_pic"] or "",
                "login_count": new_count
            }
        return None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, profile_pic, login_count, last_login_at, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_user_by_username(username: str) -> Optional[dict]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, profile_pic, login_count, last_login_at, created_at FROM users WHERE username = ?", (username.lower(),))
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
        cursor.execute("SELECT id, username, role, profile_pic, login_count, last_login_at, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def delete_user(user_id: int) -> bool:
    """Deletes a user account and cascades deletion of all their chat tabs and message history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, profile_pic FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            return False

        # Protect primary admin 'df' from deletion
        if row["username"].lower() == "df":
            raise ValueError("The administrator account ('df') cannot be deleted.")

        # Clean up profile picture file if exists
        if row["profile_pic"]:
            avatar_path = config.USER_AVATAR_DIR / row["profile_pic"]
            if avatar_path.exists():
                try:
                    avatar_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove avatar file {avatar_path}: {e}")

        # Delete user
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount > 0


# ============================================================================
# Admin Analytics & User Chat History Inspection Helpers
# ============================================================================

def list_all_users_with_stats() -> List[dict]:
    """Returns all users with login counts, tab counts, message counts, and timestamps for Admin Console."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                u.id, 
                u.username, 
                u.role, 
                u.profile_pic, 
                u.login_count, 
                u.created_at, 
                u.last_login_at,
                COUNT(DISTINCT t.id) as total_tabs,
                COUNT(m.id) as total_messages,
                MAX(m.timestamp) as last_message_at
            FROM users u
            LEFT JOIN chat_tabs t ON u.id = t.user_id
            LEFT JOIN chat_messages m ON t.id = m.tab_id
            GROUP BY u.id
            ORDER BY u.id ASC
        """)
        rows = cursor.fetchall()
        users = []
        for r in rows:
            item = dict(r)
            item["login_count"] = item.get("login_count") or 0
            users.append(item)
        return users


def get_user_full_chat_history(user_id: int) -> Optional[dict]:
    """Returns detailed chat tabs and complete message history for a specific user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, role, profile_pic, login_count, created_at, last_login_at
            FROM users 
            WHERE id = ?
        """, (user_id,))
        user_row = cursor.fetchone()
        if not user_row:
            return None
        
        user_data = dict(user_row)
        user_data["login_count"] = user_data.get("login_count") or 0

        cursor.execute("""
            SELECT id, title, created_at, updated_at
            FROM chat_tabs
            WHERE user_id = ?
            ORDER BY updated_at DESC, created_at DESC
        """, (user_id,))
        tabs_rows = cursor.fetchall()
        
        tabs = []
        for t in tabs_rows:
            tab_item = dict(t)
            cursor.execute("""
                SELECT id, tab_id, role, content, citations_json, top_k_json, expanded_count, attachments_json, timestamp
                FROM chat_messages
                WHERE tab_id = ?
                ORDER BY id ASC
            """, (tab_item["id"],))
            msg_rows = cursor.fetchall()
            messages = []
            for m in msg_rows:
                m_dict = dict(m)
                try:
                    m_dict["citations"] = json.loads(m_dict.get("citations_json") or "[]")
                except Exception:
                    m_dict["citations"] = []
                try:
                    m_dict["top_k"] = json.loads(m_dict.get("top_k_json") or "[]")
                except Exception:
                    m_dict["top_k"] = []
                try:
                    m_dict["attachments"] = json.loads(m_dict.get("attachments_json") or "[]")
                except Exception:
                    m_dict["attachments"] = []
                messages.append(m_dict)
            tab_item["messages"] = messages
            tab_item["message_count"] = len(messages)
            tabs.append(tab_item)

        user_data["tabs"] = tabs
        user_data["total_tabs"] = len(tabs)
        user_data["total_messages"] = sum(t["message_count"] for t in tabs)
        return user_data


def get_multiple_users_full_chat_history(user_ids: Optional[List[int]] = None) -> List[dict]:
    """
    Returns full chat history and tabs for multiple specified users (or all users if user_ids is None).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_ids:
            placeholders = ",".join("?" for _ in user_ids)
            cursor.execute(f"SELECT id FROM users WHERE id IN ({placeholders}) ORDER BY id ASC", tuple(user_ids))
        else:
            cursor.execute("SELECT id FROM users ORDER BY id ASC")
        rows = cursor.fetchall()
        
    results = []
    for r in rows:
        u_data = get_user_full_chat_history(r["id"])
        if u_data:
            results.append(u_data)
    return results


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
            SELECT id, tab_id, role, content, citations_json, top_k_json, expanded_count, attachments_json, timestamp
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
            try:
                item["attachments"] = json.loads(item.get("attachments_json") or "[]")
            except Exception:
                item["attachments"] = []
            results.append(item)
        return results


def add_chat_message(
    tab_id: str,
    role: str,
    content: str,
    citations: Optional[List[dict]] = None,
    top_k: Optional[List[dict]] = None,
    expanded_count: int = 0,
    attachments: Optional[List[dict]] = None,
    user_id: Optional[int] = None
) -> int:
    """Appends a user or assistant message to the tab history and updates tab timestamp."""
    if not tab_id or tab_id == "guest-tab":
        return 0

    citations_str = json.dumps(citations or [])
    top_k_str = json.dumps(top_k or [])
    attachments_str = json.dumps(attachments or [])
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Verify if tab exists; if not, ensure it's safely created for valid user
        cursor.execute("SELECT id FROM chat_tabs WHERE id = ?", (tab_id,))
        if not cursor.fetchone():
            if user_id:
                # Check user existence
                cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
                user_row = cursor.fetchone()
                if not user_row:
                    cursor.execute("SELECT id FROM users WHERE username = 'df'")
                    df_row = cursor.fetchone()
                    effective_uid = df_row["id"] if df_row else 1
                else:
                    effective_uid = user_id
                
                try:
                    cursor.execute(
                        "INSERT INTO chat_tabs (id, user_id, title) VALUES (?, ?, ?)",
                        (tab_id, effective_uid, "New Chat")
                    )
                except Exception as e:
                    logger.warning(f"Could not auto-create tab {tab_id}: {e}")
                    return 0
            else:
                logger.warning(f"Skipping message persistence: Tab {tab_id} does not exist in DB.")
                return 0

        cursor.execute("""
            INSERT INTO chat_messages (tab_id, role, content, citations_json, top_k_json, expanded_count, attachments_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (tab_id, role, content, citations_str, top_k_str, expanded_count, attachments_str))
        msg_id = cursor.lastrowid
        cursor.execute("UPDATE chat_tabs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (tab_id,))
        conn.commit()
        return msg_id


def get_tab_conversation_memory(tab_id: str, max_turns: int = 6) -> List[Dict[str, str]]:
    """
    Returns recent multi-turn question & answer pairs from this specific tab history
    formatted for conversational memory conditioning.
    """
    if not tab_id or tab_id == "guest-tab":
        return []

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
