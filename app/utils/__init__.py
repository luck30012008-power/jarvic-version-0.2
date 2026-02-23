"""
UTILITIES PACKAGE
=================

Helpers used by the services (no HTTP, no business logic):

   time_info  - get_time_information(): returns a string with current data/time for the LLM prompt.
   retry      - with_retry(fn): calls fn(); on failure retryies with exponential backoff (Groq/Tavily).
"""
