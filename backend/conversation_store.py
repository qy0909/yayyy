import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


MAX_SUMMARY_LINES = 8
SUMMARY_KEEP_RECENT_MESSAGES = 6


class ConversationStore:
    """SQLite-backed conversation persistence with rolling summaries."""

    def __init__(self, db_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = db_path or os.path.join(base_dir, "conversations.db")
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_message_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                """
            )
            
            # Migration: Add metadata column if it doesn't exist (for existing databases)
            try:
                connection.execute("SELECT metadata FROM messages LIMIT 1")
            except sqlite3.OperationalError:
                connection.execute("ALTER TABLE messages ADD COLUMN metadata TEXT DEFAULT '{}'")
                connection.commit()

    def create_conversation(self, title: Optional[str] = None) -> Dict[str, Any]:
        conversation_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        safe_title = (title or "New conversation").strip() or "New conversation"

        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, safe_title[:120], now, now),
            )

        return self.get_conversation(conversation_id)

    def list_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, summary, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        import json
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT id, title, summary, summary_message_count, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise KeyError(f"Conversation not found: {conversation_id}")

            messages = connection.execute(
                "SELECT role, text, metadata, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()

        payload = dict(conversation)
        payload["messages"] = []
        
        for message in messages:
            msg_dict = dict(message)
            metadata_str = msg_dict.get("metadata")
            
            # Parse metadata JSON if it exists
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                    # Merge metadata fields into the message
                    for key, value in metadata.items():
                        msg_dict[key] = value
                except (json.JSONDecodeError, TypeError) as e:
                    pass
            
            # Remove the raw metadata field since it's merged into the message
            msg_dict.pop("metadata", None)
            payload["messages"].append(msg_dict)
            
        return payload

    def get_recent_messages(self, conversation_id: str, limit: int = SUMMARY_KEEP_RECENT_MESSAGES) -> List[Dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, text
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()

        return [dict(row) for row in reversed(rows)]

    def append_message(self, conversation_id: str, role: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        import json
        message_text = (text or "").strip()
        if not message_text:
            return

        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})
        
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO messages (conversation_id, role, text, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, message_text, metadata_json, now),
            )
            conversation = connection.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise KeyError(f"Conversation not found: {conversation_id}")

            title = conversation["title"]
            if title == "New conversation" and role == "user":
                title = self._derive_title(message_text)

            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conversation_id),
            )

    def refresh_summary(self, conversation_id: str) -> Dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        messages = conversation["messages"]
        summary_cutoff = max(0, len(messages) - SUMMARY_KEEP_RECENT_MESSAGES)

        if summary_cutoff <= 0:
            summary = ""
            summary_message_count = 0
        else:
            summary = self._build_summary(messages[:summary_cutoff])
            summary_message_count = summary_cutoff

        now = datetime.now().isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE conversations SET summary = ?, summary_message_count = ?, updated_at = ? WHERE id = ?",
                (summary, summary_message_count, now, conversation_id),
            )

        return self.get_conversation(conversation_id)

    def _derive_title(self, text: str) -> str:
        title = " ".join(text.split())
        title = title[:60].rstrip(" .,!?:;")
        return title or "New conversation"

    def _build_summary(self, messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""

        summary_lines = []
        start_index = max(0, len(messages) - MAX_SUMMARY_LINES)
        for message in messages[start_index:]:
            role = (message.get("role") or "user").lower()
            prefix = "User" if role == "user" else "Assistant"
            text = " ".join((message.get("text") or "").split())
            if not text:
                continue
            summary_lines.append(f"- {prefix}: {text[:220]}")

        return "\n".join(summary_lines)