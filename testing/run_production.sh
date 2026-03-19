#!/bin/bash
# Production-optimized startup script for fast responses
# Run this instead of uvicorn main:app to avoid reload delays

echo "🚀 Starting Multilingual RAG Bot in PRODUCTION MODE"
echo "   - Models will be pre-loaded at startup (1-2 minutes)"
echo "   - Auto-reload is DISABLED for fast first response"
echo "   - First user query will be instant!"
echo ""

# Set environment variables for optimal performance
export PYTHONUNBUFFERED=1
export DEV_MODE=false
export SPEED_MODE=true
export HF_HUB_OFFLINE=false

# Run the server without reload
cd "$(dirname "$0")"
python -m uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --no-reload \
  --log-level info
