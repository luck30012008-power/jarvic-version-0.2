"""
J.A.R.V.I.S MAIN API
====================

This module defines the FastAPI application and all HTTP endpoints. It is
designed for single-user use: one person runs one server (e.g. python run.py)
and uses it as their personal J.A.R.V.I.S backend. Many people can each run
their own copy of this code on their own machine.

ENDPOINTS:
  GET /                  - Returns API name and list of endpoints.
  GET /health            - Returns status of all services (for monitoring).
  POST /chat             - General chat: pure LLM, no web search. Uses learning data
                           and past chats via vector store retrieval only.
  POST /chat/realtime    - Realtime chat: runs a Tavily web search first, then
                           sends results + context to Groq. Same session as /chat.
  GET /chat/history/{id} - Returns all messages for a session (general + realtime).

SESSION:
  Both /chat and /chat/realtime use the same session_id. If you omit session_id,
  the server generates a UUID and returns it; send it back on the next request
  to continue the conversation. Sessions are saved to disk and survive restarts.

STARTUP:
  On startup, the lifespan function builds the vector store from learning_data/*.txt
  and chats_data/*.json, then creates Groq, Realtime, and Chat services. On shutdown,
  it saves all in-memory sessions to disk.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

from app.models import ChatRequest, ChatResponse

# User-friendly message when Groq rate limit (daily token quota) is exceeded.
RATE_LIMIT_MESSAGE = (
    "You've reached your daily API limit for this assistant. "
    "Your credits will reset in a few hours, or you can upgrade your plan for more. "
    "Please try again later."
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if the exception is a Groq rate limit (429 / tokens per day)."""
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg


from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService
from app.services.chat_service import ChatService
from config import VECTOR_STORE_DIR
from langchain_community.vectorstores import FAISS


# --------------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("J.A.R.V.I.S")


# --------------------------------------------------------------------------
# GLOBAL SERVICE REFERENCES
# --------------------------------------------------------------------------
# Set during startup (lifespan) and used by all route handlers.
# Stored as globals so async endpoints can access the same service instances.
vector_store_service: VectorStoreService = None
groq_service: GroqService = None
realtime_service: RealtimeGroqService = None
chat_service: ChatService = None


def print_title():
    """Print the J.A.R.V.I.S ASCII art title."""
    title = """

   ╔══════════════════════════════════════════════════════════╗
   ║                                                          ║
   ║         ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗          ║
   ║         ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝          ║
   ║         ██║███████║██████╔╝██║   ██║██║███████╗          ║
   ║    ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║          ║
   ║    ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║          ║
   ║     ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝          ║
   ║                                                          ║
   ║          Just A Rather Very Intelligent System           ║
   ║                                                          ║
   ╚══════════════════════════════════════════════════════════╝

    """
    print(title)


# --------------------------------------------------------------------------
# LIFESPAN (STARTUP / SHUTDOWN)
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown.
    
    This function manages the application's lifecycle:
      - STARTUP: Initializes all services in the correct order
        1. VectorStoreService: Creates FAISS index from learning_data and chat_history
        2. GroqService: Sets up general chat AI service
        3. RealtimeGroqService: Sets up realtime chat with Tavily search
        4. ChatService: Manages chat sessions and conversations
      - RUNTIME: Application runs normally
      - SHUTDOWN: Saves all active chat sessions to disk

    The services are initialized in this specific order because:
      - VectorStoreService must be created first (used by GroqService)
      - GroqService must be created before RealtimeGroqService (it inherits from it)
      - ChatService needs both GroqService and RealtimeGroqService

    All services are stored as global variables so they can be accessed by API endpoints.
    """
    global vector_store_service, groq_service, realtime_service, chat_service

    print_title()
    logger.info("-" * 60)
    logger.info("J.A.R.V.I.S - Starting Up...")
    logger.info("-" * 60)

    try:
        # Initialize vector store service
        logger.info("Initializing vector store service...")
        vector_store_service = VectorStoreService()
        vector_store_service.create_vector_store()
        logger.info("Vector store initialized successfully")

        # Initialize Groq service (general chat)
        logger.info("Initializing Groq service (general queries)...")
        groq_service = GroqService(vector_store_service)
        logger.info("Groq service initialized successfully")

        # Initialize Realtime Groq service (with Tavily search)
        logger.info("Initializing Realtime Groq service (with Tavily search)...")
        realtime_service = RealtimeGroqService(vector_store_service)
        logger.info("Realtime Groq service initialized successfully")

        # Initialize chat service
        logger.info("Initializing chat service...")
        chat_service = ChatService(groq_service, realtime_service)
        logger.info("Chat service initialized successfully")

        # Startup complete
        logger.info("-" * 60)
        logger.info("Service Status:")
        logger.info("  - Vector Store: Ready")
        logger.info("  - Groq AI (General): Ready")
        logger.info("  - Groq AI (Realtime): Ready")
        logger.info("  - Chat Service: Ready")
        logger.info("-" * 60)
        logger.info("J.A.R.V.I.S is online and ready!")
        logger.info("API: http://localhost:8005")
        logger.info("Docs: Docs: http://localhost:8005/docs")
        logger.info("-" * 60)

        yield

        # Shutdown: Save active sessions
        logger.info("\nShutting down J.A.R.V.I.S...")
        if chat_service:
            for session_id in list(chat_service.sessions.keys()):
                chat_service.save_chat_session(session_id)
        logger.info("All sessions saved. Goodbye!")

    except Exception as e:
        logger.error(f"Fatal error during startup: {e}", exc_info=True)
        raise


# --------------------------------------------------------------------------
# FASTAPI APP AND CORS
# --------------------------------------------------------------------------
# lifespan runs once at startup (build services) and once at shutdown (save sessions).
app = FastAPI(
    title="J.A.R.V.I.S API",
    description="Just A Rather Very Intelligent System",
    lifespan=lifespan
)

# Allow any origin so a frontend on another port or device can call this API without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# API ENDPOINTS
# ==========================================================================

@app.get("/")
async def root():
    """Return the API name and a short description of each endpoint (for discovery)."""
    return {
        "message": "J.A.R.V.I.S API",
        "endpoints": {
            "/chat": "General chat (pure LLM, no web search)",
            "/chat/realtime": "Realtime chat (with Tavily search)",
            "/chat/history/{session_id}": "Get chat history",
            "/health": "System health check"
        }
    }


@app.get("/health")
async def health():
    """Return 'healthy' and whether each service (vector_store, groq, realtime, chat) is initialized."""
    return {
        "status": "healthy",
        "vector_store": vector_store_service is not None,
        "groq_service": groq_service is not None,
        "realtime_service": realtime_service is not None,
        "chat_service": chat_service is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    General chat endpoint - send a message to J.A.R.V.I.S.
    
    This endpoint uses the general chatbot mode which does NOT perform web searches.
    It's perfect for:
      - Conversational questions
      - Historical information
      - General knowledge queries
      - Questions that don't require current/realtime information

    HOW IT WORKS:
    1. Receives user message and optional session_id
    2. Gets or creates a chat session
    3. Processes message through GroqService (pure LLM, no web search)
    4. Retrieves context from user data files and past conversations
    5. Generates response using Groq AI
    6. Saves session to disk
    7. Returns response and session_id

    SESSION MANAGEMENT:
      - If session_id is NOT provided: Server generates a new UUID (server-managed)
      - If session_id IS provided: Server uses it (loads from disk if exists, creates new if not)
      - Use the SAME session_id with /chat/realtime to seamlessly switch between modes
      - Sessions persist across server restarts (loaded from disk)

    REQUEST BODY:
    {
        "message": "What is Python?",
        "session_id": "optional-session-id"
    }

    RESPONSE:
    {
        "response": "Python is a high-level programming language...",
        "session_id": "session-id-here"
    }
    """
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    try:
        # Get existing session or create a new one (and optionally load from disk).
        session_id = chat_service.get_or_create_session(request.session_id)
        # Process with general chat: no web search; context comes from vector store only.
        response_text = chat_service.process_message(session_id, request.message)
        # Save the session to disk (it will also stay in memory).
        chat_service.save_chat_session(session_id)
        return ChatResponse(response=response_text, session_id=session_id)
    except Exception as e:
        # Invalid session_id (e.g. path traversal ".." or too long).
        logger.warning(f"Invalid session_id: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if _is_rate_limit_error(e):            
            logger.warning(f"rate limit hit: {e}")
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        logger.error(f"Error processing chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
                  

@app.post("/chat/realtime", response_model=ChatResponse)
async def chat_realtime(request: ChatRequest):
    """
    Realtime chat endpoint - send a message to J.A.R.V.I.S with Tavily web search.

    This endpoint uses the realtime chatbot mode which performs web searches via Tavily
    before generating a response. It's perfect for:
    - Current events and news
    - Recent information
    - Questions requiring up-to-date data
    - Anything that needs internet access

    HOW IT WORKS:
    1. Receives user message and optional session_id
    2. Gets or creates a chat session (SAME as /chat endpoint)
    3. Searches Tavily for real-time information (fast, AI-optimized, English-only)
    4. Retrieves context from user_data files and past conversations
    5. Combines search results with context
    6. Generates response using Groq AI with all available information
    7. Saves session to disk
    8. Returns response and session_id

    IMPORTANT: This uses the SAME chat session as /chat endpoint.
    - You can use the same session_id for both endpoints
    - This allows seamless switching between general and realtime modes
    - Conversation history is shared between both modes
    - Example: Ask a general question, then ask a realtime question, then another general question
      - All in the same conversation context
    
      SESSION_MANAGEMENT:
    - Same as /chat endpoint - sessions are shared
    - If session_id is NOT provided: Server generates a new UUID
    - If session_id IS provided: Server uses it (loads from disk if exists)

    REQUEST BODY:
    {
        "message": "What's the latest AI news?",
        "session_id": "optional-session-id"
    }

    RESPONSE:
    {
        "response": "Based on recent search results...",
        "session_id": "session-id-here"
    }

    NOTE: Requires TAVILY_API_KEY to be set in .env file. If not set, realtime mode
    will not be available and will return a 503 error.
    """
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    if not realtime_service:
        raise HTTPException(status_code=503, detail="Realtime service not initialized")

    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        # Realtime: Tavily search first, then Groq with search results + context
        response_text = chat_service.process_realtime_message(session_id, request.message)
        chat_service.save_chat_session(session_id)
        return ChatResponse(response=response_text, session_id=session_id)
    except ValueError as e:
        logger.warning(f"Invalid session_id: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if _is_rate_limit_error(e):
            logger.warning(f"Rate limit hit: {e}")
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        logger.error(f"Error processing realtime chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
    

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Get chat history for a specific session.

    This endpoint retrieves all messages from a chat session, including both
    general and realtime messages since they share the same session.

    HOW IT WORKS:
    1. Receives session_id as URL parameter
    2. Retrieves all messages from that session
    3. Returns messages in chronological order

    RESPONSE:
    {
        "session_id": "session-id",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Good day. How may I assist you?"},
            ...
        ]
    }

    NOTE: If session doesn't exist, returns empty messages array.
    """
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    try:
        # Returns in-memory messages for this session (empty if session not loaded).
        messages = chat_service.get_chat_history(session_id)
        return {
            "session_id": session_id,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }
    except Exception as e:
        logger.error(f"Error retrieving history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


# -------------------------------------------------------------------------------
# STANDALONE RUN (python -m app.main)
# -------------------------------------------------------------------------------
def run():
    """Start the uvicorn server (same as run.py; used if someone does python -m app.main)."""
    uvicorn.run(
        "app.main:app",
        host="127.0.1.1",
        port=8005,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    run()

    