"""
RUN SCRIPT - Start the J.A.R.V.I.S server
=========================================

PURPOSE:
  Single entry point to start the backend. Run this once per user/machine;
  the server then handles all chat and realtime requests for that instance.

WHAT IT DOES:
  - Imports the FastAPI app from app.main.
  - Runs it with uvicorn on host 127.0.1.1 (accept connections from any interface) and port 8005.
  - reload=True means any change to Python files will restart the server (handy for development).

USAGE:
  python run.py

  Then open http://localhost:8005 in the browser, or use the API from another app.
  API docs: http://localhost:8005/docs

NOTE:
  Before running, set GROQ_API_KEY (and optionally TAVILY_API_KEY for realtime search) in .env.
"""

import uvicorn

# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------
# Only run uvicorn when this file is executed directly (python run.py),
# not when it is imported by another module.
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",    # String path to the FastAPI app instance (module:variable).
        host="127.0.1.1",    # Listen on all network interfaces so other devices can connect.
        port=8005,         # HTTP port; change if 8000 is already in use.
        reload=True        # Auto-restart when .py files change (useful during development).
    )

    
    
    
# 