"""
CHAT SERVICE MODULE
====================

This sevice owns all chat session and conversation logic . It is used by the 
/chat and /chat/realtime endpoints. Designed for single-user use: one server
has one chatservice and one in-memory session store; the user can have many
session (each identified by session_id).

RESPONSIBILITIES :
    -  get_or_create_session(session_id): Return existing session or create new one.
       If the user sends a session_id that was used before (e.g. before a refresh),
       We try to load it from disk so the conversation continues.
    -  add message / get_chat_history: keep messages in memory per session.
    -  format_history_for_llm: turn the messages list into (user, assistant) paies
       and trim to MAX_CHAT_HISTORY_TURNS so we don't overflow the prompt.
    -  process_message / process_message_realtime: Add user message, call groq (or
       RealtimeGroq), add assistant reply, return reply. 
    -  save _chat_session : Write session to database/chats_data/*.json so it persits
       and can be loaded on next sttartup ( and userby the vector store for retrieval)
      """

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
import uuid

from config import CHATS_DATA_DIR, MAX_CHAT_HISTORY_TURNS
from app.models import ChatMessage, ChatHistory
from app.services.realtime_service import GroqService
from app.services.realtime_service import RealtimeGroqService


logger = logging.getLogger("J.A.R.V.I.S")


# ================================================================================
# CHAT SERVICE CLASS
# ================================================================================

class ChatService:
    """
    Manages chat sessions: in-memory message lists, load/save to disk, and
    calling groq (or Realtime) to get replies. All state for action sessions 
    is in self.sessions; saving to disk is done aftereach message so
    conversations survive restarts.
    """

    def __init__(self, groq_service: GroqService, realtime_service: RealtimeGroqService = None):
        """store references to groq and Realtime services; keep sessions in memory."""
        self.groq_service = groq_service
        self.realtime_service = realtime_service
        # Map: session_id -> list of ChatMessage (user and assistant messages in order).
        self.sessions: Dict[str, List[ChatMessage]] = {}

    # --------------------------------------------------------------------------------
    # SESSION LOAD / VALIDATION / GET-OR-CREATE
    # --------------------------------------------------------------------------------

    def load_session_from_disk(self, session_id: str) -> bool:
        """
        Load a session from database/chats_data/{ if a file for this session_id exists.

        File name is chat_{session_id}.json where safe session_id has dashes/spaces removed. 
        On success we put the messages into self.sessions[session_id] so later requests use them.
        Returns True if loaded, False if file missing or unreadable.
        """
        # Sanitize ID for use in filename (no dashes or spaces).
        safe_session_id = session_id.replace("-", "").replace(" ", "")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename

        if not filepath.exists():
            return False
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_dict = json.load(f)
            # convert stored dicts back to ChatMessage objects.
            messages = [
                ChatMessage(role=msg.get("role"), content=msg.get("content"))
                for msg in chat_dict.get("messages", [])
            ]
            self.sessions[session_id] = messages
            return True
        except Exception as e:
            logger.warning("Failed to load session %s from disk: %s", session_id, e)
            return False
        
    def validate_session_id(self, session_id: str) -> bool:
        """
        Return True if session_id is safe to use (non-empty, no path traversal, length <= 255).
        Used to reject malicious or invalid IDs before we use them in file paths.       
        """
        if not session_id or not session_id.strip():
            return False 
        # Block path traversal and path separators.
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            return False
        if len(session_id) > 255:
            return False
        return True
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """
        Return a session_ID and ensure that session exists in memory.
        
        - If session_id is none: create a new session with a new UUID and return it.
        - If session_id is provided: validate it, if it's in self.sessions return it;
          else try to load from disk; if not found, create new session with this ID.
        Raises ValueError if session_id is invalid (empty, path traversal, or too long).
        """
        if not session_id:
            new_session_id = str(uuid.uuid4())
            self.sessions[new_session_id] = []
            return new_session_id
        
        if not self.validate_session_id(session_id):
            raise ValueError(
                f"Invalid session_id format: {session_id}. session ID must be non-empty,"
                "not contain path traversal characters, and be at most 255 characters."
            )

        if session_id in self.sessions:
            return session_id

        if self.load_session_from_disk(session_id):
            return session_id
        # New session with this ID (e.g. client sent an ID that was never saved.)
        self.sessions[session_id] = []
        return session_id


    # --------------------------------------------------------------------------------
    # MESSAGES AND HISTORY FORMATTING
    # --------------------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str):
        """Append a message (user or assistant) to the session's message list. Create session if missing.""" 
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(ChatMessage(role=role, content=content))

    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """Return the list of messages for this session (chronological). Empty list if session unknown."""
        return self.sessions.get(session_id, [])
    
    def format_history_for_llm(self, session_id: str, exclude_last: bool = False) -> List[ChatMessage]:
        """
        Bild a list of (user_text, assistant_text) pairs for the LLM prompt.

        We only include complete pairs and cap at MAX_CHAT_HISTORY_TURNS (e.g. 20)
        so the prompt doesn not grow unbound. If excluded_last is True we drop the 
        last message (the current user message that we are about to reply to).
        """
        messages = self.get_chat_history(session_id)
        history = []
        # If excluded_last, we skip the last message (the current user message we are about to reply to).
        messages_to_process = messages[:-1] if exclude_last and messages else messages
        i = 0
        while i < len(messages_to_process) - 1:
            user_msg = messages_to_process[i]
            ai_msg = messages_to_process[i + 1]
            if user_msg.role == "user" and ai_msg.role == "assistant":
                history.append((user_msg.content, ai_msg.content))
                i += 2
            else:
                i += 1
            # Keep only the most recent turns so the prompt doesn not exceed token limits.
            if len(history) > MAX_CHAT_HISTORY_TURNS:
                history = history[-MAX_CHAT_HISTORY_TURNS:]
            return history
        
        # --------------------------------------------------------------------------------
        # PROCESS MESSAGE (GENERAL AND REALTIME)
        # --------------------------------------------------------------------------------

    def process_message(self, session_id: str, user_message: str) -> str:
        """
        handle one general-chat message: add user message, call Groq (no web search), add reply, return it.
        """
        self.add_message(session_id, "user", user_message)
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        response = self.groq_service.get_response(question=user_message, chat_history=chat_history)
        self.add_message(session_id, "assistant", response)
        return response
    def process_message_realtime(self, session_id: str, user_message: str) -> str:
        """
        Handle one realtime message: add user message, call realtime service (tavily + Groq), add reply, return it.
        uses the same session as process_message so history is shared. Raises ValueError if realtime_service is NOne.
        """

        if not self.realtime_service:
            raise ValueError("Realtime service is not initialized. Cannot process realtime queries.")
        self.add_message(session_id, "user", user_message)
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        response = self.realtime_service.get_response(question=user_message, chat_history=chat_history)
        self.add_message(session_id, "assistant", response)
        return response
    
    # --------------------------------------------------------------------------------
    # PERSIST SESSION TO DISK
    # --------------------------------------------------------------------------------

    def save_session_to_disk(self, session_id: str):
        """
        write this session's messages to database/chats_data/chat__{safe_id}.json.

        called after each message so the conversation is persisted. the vector store
        is rebuilt on startup from these files, so new  chat are included after restart.
        if the session is missing or empty we do nothing . on write errors we log.
        """
        if session_id not in self.sessions or not self.sessions[session_id]:
            return
        
        messages = self.sessions[session_id]
        safe_session_id = session_id.replace("-", "").replace(" ", "")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename
        chat_dict = {
            "session_id": session_id,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }


        

    def save_chat_session(self, session_id: str):
        """
        Write this session's messages to database/chats_data/chat_{safe_id}.json.

        Called after each message so the conversation is persisted. The vector store
        is rebuilt on startup from these files, so new chats are included after restart.
        If the session is missing or empty we do nothing. On write error we only log.
        """
        if session_id not in self.sessions or not self.sessions[session_id]:
            return

        messages = self.sessions[session_id]
        safe_session_id = session_id.replace("-", "").replace(" ", "_")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename
        chat_dict = {
            "session_id": session_id,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(chat_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save chat session %s to disk: %s", session_id, e)