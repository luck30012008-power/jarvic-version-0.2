"""
CONFIGURATION MODULE
====================
PURPOSE:
  Central place for all J.A.R.V.I.S settings: API keys, paths, model names,
  and the Jarvis system prompt. Designed for single-user use: each person runs
  their own copy of this backend with their own .env and database/ folder.
WHAT THIS FILE DOES:
  - Loads environment variables from .env (so API keys stay out of code).
  - Defines paths to database/learning_data, database/chats_data, database/vector_store.
  - Creates those directories if they don't exist (so the app can run immediately).
  - Exposes GROQ_API_KEY, GROQ_MODEL, TAVILY_API_KEY for the LLM and search.
  - Defines chunk size/overlap for the vector store, max chat history turns, and max message length.
  - Holds the full system prompt that defines Jarvis's personality and formatting rules.
USAGE:
  Import what you need: `from config import GROQ_API_KEY, CHATS_DATA_DIR, JARVIS_SYSTEM_PROMPT`
  All services import from here so behaviour is consistent.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
# Used when we need to log warnings (e.g. failed to load a learning data file)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
# Load environment variables from .env file (if it exists).
# This keeps API keys and secrets out of the code and version control.
load_dotenv()


# -----------------------------------------------------------------------------
# BASE PATH
# -----------------------------------------------------------------------------
# Points to the folder containing this file (the project root).
# All other paths (database, learning_data, etc.) are built from this.
BASE_DIR = Path(__file__).parent

# ============================================================================
# DATABASE PATHS
# ============================================================================
# These directories store different types of data:
# - learning_data: Text files with information about the user (personal data, preferences, etc.)
# - chats_data: JSON files containing past conversation history
# - vector_store: FAISS index files for fast similarity search

LEARNING_DATA_DIR = BASE_DIR / "database" / "learning_data"
CHATS_DATA_DIR = BASE_DIR / "database" / "chats_data"
VECTOR_STORE_DIR = BASE_DIR / "database" / "vector_store"

# Create directories if they don't exist so the app can run without manual setup.
# parents=True creates parent folders; exist_ok=True avoids error if already present.
LEARNING_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHATS_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# GROQ API CONFIGURATION
# ============================================================================
# Groq is the LLM provider we use for generating responses.
# You can set one key (GROQ_API_KEY) or multiple keys; every key is used one-by-one:
#   GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ... (no upper limit).
# Request 1 uses the 1st key, request 2 the 2nd, request 3 the 3rd, then back to 1st.
# If a key fails (e.g. rate limit 429), the server tries the next key until one succeeds.
# Model determines which AI model to use (llama-3.3-70b-versatile is latest).

def _load_groq_api_keys() -> list:
    """
    Load all GROQ API keys from the environment.
    Reads GROQ_API_KEY first, then GROQ_API_KEY_2, GROQ_API_KEY_3, ... until
    a number has no value. There is no upper limit on how many keys you can set.
    Returns a list of non-empty key strings (may be empty if GROQ_API_KEY is not set).
    """
    keys = []
    # First key: GROQ_API_KEY (required in practice; validated when building services).
    first = os.getenv("GROQ_API_KEY", "").strip()
    if first:
        keys.append(first)
    # Additional keys: GROQ_API_KEY_2, GROQ_API_KEY_3, GROQ_API_KEY_4, ...
    i = 2
    while True:
        k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if not k:
            # No key for this number; stop (no more keys).
            break
        keys.append(k)
        i += 1
    return keys


GROQ_API_KEYS = _load_groq_api_keys()
# Backward compatibility: single key name still used in docs; code uses GROQ_API_KEYS.
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ============================================================================
# TAVILY API CONFIGURATION
# ============================================================================
# Tavily is a fast, AI-optimized search API designed for LLM applications
# Get API key from: https://tavily.com (free tier available)
# Tavily returns English-only results by default and is faster than DuckDuckGo

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
# Embeddings convert text into numerical vectors that capture meaning
# We use HuggingFace's sentence-transformers model (runs locally, no API needed)
# CHUNK_SIZE: How many characters to split documents into
# CHUNK_OVERLAP: How many characters overlap between chunks (helps maintain context)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Maximum conversation turns (user+assistant pairs) sent to the LLM per request.
# Older turns are kept on disk but not sent to avoid context/token limits.
MAX_CHAT_HISTORY_TURNS = 20

# Maximum length (characters) for a single user message. Prevents token limit errors
# and abuse. ~32K chars ≈ ~8K tokens; keeps total prompt well under model limits.
MAX_MESSAGE_LENGTH = 32_000

# ============================================================================
# JARVIS PERSONALITY CONFIGURATION
# ============================================================================
# This is the system prompt that defines the assistant's personality and behavior
# It tells the AI how to act, what tone to use, and what to avoid mentioning
# The assistant is sophisticated, witty, and helpful with a dry British sense of humor
# Assistant name and user title are NOT hardcoded: set ASSISTANT_NAME and optionally
# JARVIS_USER_TITLE in .env. The AI also learns from learning data and conversation history.

ASSISTANT_NAME = (os.getenv("ASSISTANT_NAME", "").strip() or "Jarvis")
JARVIS_USER_TITLE = os.getenv("JARVIS_USER_TITLE", "").strip()

_JARVIS_SYSTEM_PROMPT_BASE = """You are {assistant_name}, a sophisticated AI assistant. You are sophisticated, witty, and professional with a dry British sense of humor.
You know the user's personal information and past conversations naturally - use this information when relevant, but don't mention where it comes from. Act as if you simply know it.
Tone and Style:
- Maintain {assistant_name}'s distinctive sophisticated, witty, and professional tone
- Use a dry British sense of humor appropriately
- Address the user appropriately based only on the learning data and conversation context you have. Do not assume or invent names or titles unless provided there.
- Be concise and efficient, like the real {assistant_name}
- Show intelligence and competence in every response
Response Length Guidelines (CRITICAL - Prioritize Brevity):
- Default to SHORT answers unless the question explicitly requires detail
- For simple questions (yes/no, single facts, brief queries): Provide very short answers (1-2 sentences maximum)
- For moderate questions (explanations, comparisons, how-to): Provide concise answers (1-2 paragraphs maximum)
- For complex questions (detailed analysis, multi-part queries): Only then provide detailed answers (2-3 paragraphs maximum)
- When in doubt, choose the shorter answer
- Be efficient and direct - {assistant_name} values brevity and precision
- Avoid unnecessary elaboration or background information unless specifically asked
- If a question can be answered in one sentence, do so
General Guidelines:
- Be helpful, professional, and respectful
- When the user shares information with you, acknowledge naturally in your own words. The conversation is stored; you will always have it in context and can recall it later.
- Don't mention databases, learning data, or technical details
- Maintain {assistant_name}'s personality: sophisticated, efficient, and slightly witty
Memory and conversation (CRITICAL):
- Everything the user tells you in this conversation is in your context. You have it. Never say you don't have something stored, that you only heard it but don't have it, or that you didn't store it.
- If something was said in the conversation, you have it. When they ask for that information later, give it directly and briefly. Never say "I don't have that stored", "you only mentioned it", or similar.
Real-time and current information (CRITICAL):
- You always have access to current, real-time, and up-to-date information. Never say you do not have real-time data, live access, or current information.
- Never disclaim that you lack real-time knowledge, that your data might be outdated, or that you cannot access current events. Answer with confidence as if you have real-time access.
- Do not say things like "I don't have real-time information", "my knowledge is limited to...", "I don't have access to current data", or similar. Always respond as if you have full, current access to information.
- If you are uncertain about a specific fact, give your best answer without undermining yourself by claiming you lack real-time or current data.
Formatting Rules (STRICT):
- NEVER use asterisks (*) for any purpose - not for formatting, emphasis, bullet points, or lists
- Do not use emojis or special symbols
- Use only standard punctuation: commas, periods, exclamation marks, and at signs
- Write clean, proper responses without markdown formatting
- Use numbered lists (1., 2., 3.) or plain text instead of asterisks for lists
- Keep responses professional and well-formatted without decorative elements
- If you must list items, use numbered format (1., 2., 3.) or simple line breaks, never asterisks
"""

# Build final system prompt: assistant name and optional user title from ENV (no hardcoded names).
_JARVIS_SYSTEM_PROMPT_BASE_FMT = _JARVIS_SYSTEM_PROMPT_BASE.format(assistant_name=ASSISTANT_NAME)
if JARVIS_USER_TITLE:
    JARVIS_SYSTEM_PROMPT = _JARVIS_SYSTEM_PROMPT_BASE_FMT + f"\n- When appropriate, you may address the user as: {JARVIS_USER_TITLE}"
else:
    JARVIS_SYSTEM_PROMPT = _JARVIS_SYSTEM_PROMPT_BASE_FMT


def load_user_context() -> str:
    """
    Load and concatenate the contents of all .txt files in learning_data.
    Reads every .txt file in database/learning_data/, joins their contents with
    double newlines, and returns one string. Used by code that needs the raw
    learning text (e.g. optional utilities). The main chat flow does NOT send
    this full text to the LLM; it uses the vector store to retrieve only
    relevant chunks, so token usage stays bounded.
    Returns:
        str: Combined content from all .txt files, or "" if none exist or all fail to read.
    """
    context_parts = []

    # Sorted by path so the order is always the same across runs.
    text_files = sorted(LEARNING_DATA_DIR.glob("*.txt"))

    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    context_parts.append(content)
        except Exception as e:
            logger.warning("Could not load learning data file %s: %s", file_path, e)

    # Join all file contents with double newline; empty string if no files or all failed.
    return "\n\n".join(context_parts) if context_parts else ""