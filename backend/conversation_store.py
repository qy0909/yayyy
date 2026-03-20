import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from supabase import create_client, Client


MAX_SUMMARY_LINES = 8
SUMMARY_KEEP_RECENT_MESSAGES = 6


class ConversationStore:
    """Supabase-backed conversation persistence with rolling summaries."""

    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        self.supabase: Client = create_client(supabase_url, supabase_key)

    def create_conversation(self, session_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        conversation_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        safe_title = (title or "New conversation").strip() or "New conversation"

        data = {
            "id": conversation_id,
            "session_id": session_id,
            "title": safe_title[:120],
            "summary": "",
            "created_at": now,
            "updated_at": now
        }
        
        self.supabase.table("conversations").insert(data).execute()
        return self.get_conversation(session_id, conversation_id)

    def list_conversations(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = (
            self.supabase.table("conversations")
            .select("id, title, summary, created_at, updated_at")
            .eq("session_id", session_id)
            .order("updated_at", desc=True)
        )
        if limit is not None:
            query = query.limit(limit)
            
        response = query.execute()
        return response.data or []

    def delete_conversation(self, session_id: str, conversation_id: str) -> None:
        # messages delete cascaded by Supabase FK relation
        self.supabase.table("conversations").delete().eq("id", conversation_id).eq("session_id", session_id).execute()

    def get_conversation(self, session_id: str, conversation_id: str) -> Dict[str, Any]:
        conv_response = self.supabase.table("conversations").select("*").eq("id", conversation_id).eq("session_id", session_id).execute()
        if not conv_response.data:
            raise KeyError(f"Conversation not found: {conversation_id}")

        conversation = conv_response.data[0]
        
        msg_response = self.supabase.table("messages").select("role, text, metadata, created_at").eq("conversation_id", conversation_id).order("id", desc=False).execute()
        messages = msg_response.data or []

        payload = dict(conversation)
        payload["messages"] = []
        
        for message in messages:
            msg_dict = dict(message)
            metadata = msg_dict.pop("metadata", {})
            
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            
            if metadata:
                for key, value in metadata.items():
                    msg_dict[key] = value
                    
            payload["messages"].append(msg_dict)
            
        return payload

    def get_recent_messages(self, conversation_id: str, limit: int = SUMMARY_KEEP_RECENT_MESSAGES) -> List[Dict[str, str]]:
        response = self.supabase.table("messages").select("role, text").eq("conversation_id", conversation_id).order("id", desc=True).limit(limit).execute()
        rows = response.data or []
        return list(reversed(rows))

    def append_message(self, session_id: str, conversation_id: str, role: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        message_text = (text or "").strip()
        if not message_text:
            return

        now = datetime.now().isoformat()
        
        # Verify conversation belongs to user
        conv_resp = self.supabase.table("conversations").select("title").eq("id", conversation_id).eq("session_id", session_id).execute()
        if not conv_resp.data:
            raise KeyError(f"Conversation not found: {conversation_id}")

        msg_data = {
            "conversation_id": conversation_id,
            "role": role,
            "text": message_text,
            "metadata": metadata or {},
            "created_at": now
        }
        self.supabase.table("messages").insert(msg_data).execute()
        
        title = conv_resp.data[0]["title"]
        if title == "New conversation" and role == "user":
            title = self._derive_title(message_text)

        self.supabase.table("conversations").update({
            "title": title,
            "updated_at": now
        }).eq("id", conversation_id).execute()

    def refresh_summary(self, session_id: str, conversation_id: str) -> Dict[str, Any]:
        conversation = self.get_conversation(session_id, conversation_id)
        messages = conversation["messages"]
        summary_cutoff = max(0, len(messages) - SUMMARY_KEEP_RECENT_MESSAGES)

        if summary_cutoff <= 0:
            summary = ""
            summary_message_count = 0
        else:
            summary = self._build_summary(messages[:summary_cutoff])
            summary_message_count = summary_cutoff

        now = datetime.now().isoformat()
        self.supabase.table("conversations").update({
            "summary": summary,
            "summary_message_count": summary_message_count,
            "updated_at": now
        }).eq("id", conversation_id).eq("session_id", session_id).execute()

        return self.get_conversation(session_id, conversation_id)

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