"""
GROQ SERVICE MODULE
===================

This module handles geneal chat: no web search,  only the Groq LLM plus context
from the vector store (learning  data + past chats) Used by chatservice for 
POSt / chat.

MULTIPLE API KEYS (round  and fallback):
-   You can set multiple Groq API keys in.env: GROQ_API_KEY, GROQ_API_KEY_2
    GROQ_API_KEY_3, ... (no limit).
-   Each request uses one key in rotation: 1st request → 1st key, and so on. Even key 
    2nd key, 3rd request → 3rd key, then back to 1st key, and so on. Every key 
    is used one-by-one so usage is spread across keys.
-   The round-robin counter is shared across all instances (GroqService and 
    RealtimeGroqService), so both /chat and /realtime endpoint use the
    same rotation sequence.
-   If the chosen key fails (rate limit 429 or any error), we try the next key,
    then the next, until one succeeds or all have been tried. 
-   All API key uses is logged with masked keys (first 8 and last 4 chars visible) 
    for security and debugging purposes.

FLOW:
1.   get_response(question, chat_history) is called.
2.   We ask the vector store for the top-k chunks most similar to the question (retrieval).
3.   We send a system message: JARVIS_SYSTEM_PROMPT + current time + retrieved context.
4.   We send to Groq using the next key in rotation (or fallback to next key on failure).
5.   We return the assistant's reply.

context is only what we retrieve (no full duump of learning data), so token usage stays bounded. 
"""

from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import logging

from config import GROQ_API_KEYS, GROQ_MODEL, JARVIS_SYSTEM_PROMPT
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information

logger = logging.getLogger("J.A.R.V.I.S")


#================================================================================
# HELPER: ESCAPE CURLY BRACES FOR LANGCHAIN
#================================================================================
# Langchain Prompt templates use {variable_name}. If learning data or chat 
# content contains { or }, the template engine can break. Doubling them 
# makes them literal in the final string. 

def escape_curly_braces(text: str) -> str:
    """
    Double every { and } so langchain does not treat them as template variables.
    Learing data or chat content might contain { or }; without escaping, invoke() can fail.
    """
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Return True if the exception ndicates a Groq rate (e.g. 429, token per day).
    Used for logging: actual fallback tries the next key on any failure when multiple keys exist.
    """ 
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "token per day" in msg


def _mask_api_key(key: str) -> str:
    """
    Mask an API key for safe logging. shows first 8 and last 4 characters, masks the middle.
    Example: gsk_1234567890abcdef -> gsk_1234...cdef
    """
    if not key or len(key) < 12:
        return "***masked***"
    return f"{key[:8]}...{key[-4:]}"
    

    #================================================================================
    # GROQ SERVICE CLASS
    #================================================================================

class GroqService:
    """
    General chat: retrieves context from vector store and calls the Groq LLM. 
    supports multiple API keys: each request uses the next key in rotation (one-by-one),
    and if that key fails, the service tries the next key until one succeeds or all fail. 

    ROUND-ROBIN BEHAVIOR:
    -  Request 1 uses key 0 (first key)
    -  Request 2 uses key 1 (second key)
    -  Request 3 uses key 2 (third key)
    -  After all keys are used, cycles back to key 0
    -  If a key fails (rate limit, error), tries the next key in sequence
    -  All requests share the same round-robin counter (class-level)
    """

    # Class-level counter share across all instances (GroqService and RealtimeGroqService)
    # This ensures that round-robin works across both /chat and /realtime endpoints
    _shared_key_index = 0
    _lock = None  # will be set to threading.Lock if threading is needed (currently single-threaded)

    def __init__(self, vector_store_service: VectorStoreService):
        """
        create one Groq LLM client per API key and store the vector store for retrieval. 
        self.llms[i] corresponds to GROQ_API_KEYS[i]; request N uses key at index (N % len(keys)).
        """
        if not GROQ_API_KEYS:
            raise ValueError(
                "No Groq API keys configured. set GROQ_API_KEY (and optionally GROQ_API_KEY_2, GROQ_API_KEY_3, ...)in "
                )
        # one ChatGroq instance per key; each request will use one of these in rotation.
        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model_name=GROQ_MODEL,
                temperature=0.8,
            )
            for key in GROQ_API_KEYS
        ]
        self.vector_store_service = vector_store_service
        logger.info(f"Initialized GroqService with {len(GROQ_API_KEYS)} API keys(s)")
    
    def _invoke_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: List,
        question: str, 
    ) -> str:
        """
        call the LLM using the next key in roration; on failure, try the next key until one succeeds.
        
        - Round-robin: the request useskey at index (_shared_key_index % n), then we increment 
          _shared_key_index so the next request uses the next key. all instances share the same counter.
        - Fallback: if the chosen key raises (e.g. 429 rate limit), we try next key, then the next,
          until one returns successfully or we have tried all keys.
        Returns response.content. Raises if all keys fail. 
        """
        n = len(self.llms)
        # which key to try first for this request (round-robin: request 1 → key 0, request 2 → key 1, ...).
        # use class-level counter so all instances (groqservice and RealtimeGroqService) share the same rotation.
        start_i = GroqService._shared_key_index % n
        current_key_index = GroqService._shared_key_index
        GroqService._shared_key_index += 1  # next request will use the next key.

        # log which key we 're using (masked for security)
        masked_key = _mask_api_key(GROQ_API_KEYS[start_i])
        logger.info(f"using API key #{start_i + 1}/{n} (round-robin index: {current_key_index}): {masked_key}")

        last_exc = None
        keys_tried = []
        # try each key in order starting from start_i (wrap around with % n).
        for j in range(n):
            i = (start_i + j) % n  # current key index to try
            keys_tried.append(i)
            try:
                # Build chain with this key's LLM and invoke once. 
                chain = prompt | self.llms[i]
                response = chain.invoke({"history": messages, "question": question})
                # Log success if we had to fallback to a different key
                if j > 0:
                    masked_success_key = _mask_api_key(GROQ_API_KEYS[i])
                    logger.info(f"Fallback successful: API key #{i + 1}/{n} succceeded: {masked_success_key}")
                return response.content
            except Exception as e:
                last_exc = e
                masked_failed_key = _mask_api_key(GROQ_API_KEYS[i])
                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i + 1}/{n} rate limited: {masked_failed_key}")
                else:
                    logger.warning(f"API key #{i + 1}/{n} failed: {masked_failed_key} - {str(e)[:100]}")
                    # If we have more than one key, try the next one:otherwise raise immediately.
                    if n > 1:
                        continue
                    raise Exception(f"Error getting response from Groq: {str(e)}") from e
            # All keys were tried and all failed; raise the last expection.
            masked_all_keys = ", ".join([_mask_api_key(GROQ_API_KEYS[i]) for i in keys_tried])
            logger.error(f"All API keys failed. Tried keys : {masked_all_keys}")
            raise Exception(f"Error getting response from Groq: {str(last_exc)}") from last_exc

    def get_response(
            self, 
            question: str,
            chat_history: Optional[List[tuple]] = None,
    ) -> str:
        """
        Return the assistant's reply for this question (general chat, no web search).
        Retrieves context from the vector store, builds the prompt, them calls _invoke_llm
        which uses the next API key in rotation and falls back to other keys on failure.
        """
        try:
            # Get retrievant chunks from learning data and past chats (bounded token usage).
            # If retrival fails (e.g. vector store not ready), use empty context so the LLM still answers.
            context = ""
            try:
                retriever = self.vector_store_service.get_retriever(k=10)
                context_docs = retriever.invoke(question)
                context = "\n".join([doc.page_content for doc in context_docs]) if context_docs else ""
            except Exception as retrieval_err:
                logger.warning("Vector store retrieval failed, using, empty context: %s", retrieval_err)

            # Build system message: personality + current time + retrieved context. 
            time_info = get_time_information()
            system_message = JARVIS_SYSTEM_PROMPT + f"/n/nCurrent time and date:{time_info}"     
            if context:
                system_message += f"\n\nRelevant context from your leatning data and past conversations:\n{escape_curly_braces(context)}"

            #  prompt template: system message, chat history placeholder, current question.
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"), 
                ("human", "{question}"),
            ])
            # convert (user, assistant) pairs to langchain message objects.
            messages = []
            if chat_history:
                for human_msg, ai_msg in chat_history:
                    messages.append(HumanMessage(content=human_msg))
                    messages.append(AIMessage(content=ai_msg))

            # use next key in rotation; on failure, try remaining keys (same as realtime).
            return self._invoke_llm(prompt, messages, question)
        except Exception as e:
            raise Exception(f"Error getting response from Groq: {str(e)}") from e
        
                    

        