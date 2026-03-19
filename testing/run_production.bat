@echo off
REM Production-optimized startup script for fast responses
REM Run this instead of uvicorn main:app to avoid reload delays

echo.
echo 🚀 Starting Multilingual RAG Bot in PRODUCTION MODE
echo    - Models will be pre-loaded at startup (1-2 minutes)
echo    - Auto-reload is DISABLED for fast first response
echo    - First user query will be instant!
echo.

REM Set environment variables for optimal performance
set PYTHONUNBUFFERED=1
set DEV_MODE=false
set SPEED_MODE=true
set HF_HUB_OFFLINE=false

REM Run the server without reload
python -m uvicorn main:app ^
  --host 0.0.0.0 ^
  --port 8000 ^
  --no-reload ^
  --log-level info
