"""
FastAPI Server for Multilingual RAG Bot
========================================
Exposes RAG pipeline as REST API for Next.js frontend
"""

import sys
import io
import os
import shutil

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
import uvicorn
from dotenv import load_dotenv
import whisper #voice processing
from conversation_store import ConversationStore

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual RAG Bot API",
    description="RAG-powered multilingual citizen services chatbot",
    version="1.0.0"
)

stt_model = whisper.load_model("base", device="cpu")

# Lazy import RAG Pipeline (loaded on first use to avoid startup issues)
RAGPipeline = None

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline (loads on first use)
rag_pipeline = None
conversation_store = ConversationStore()

def get_rag_pipeline():
    """Lazy load RAG pipeline on first use"""
    global rag_pipeline, RAGPipeline
    if rag_pipeline is None:
        try:
            if RAGPipeline is None:
                from pipeline.RAG import RAGPipeline as RAGPipelineClass
                RAGPipeline = RAGPipelineClass
            print("🚀 Initializing RAG Pipeline...")
            rag_pipeline = RAGPipeline()
            print("✅ RAG Pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize RAG pipeline: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"RAG pipeline initialization failed: {str(e)}"
            )
    return rag_pipeline


@app.on_event("startup")
async def preload_rag_pipeline():
    """Warm up pipeline at server boot to avoid first-request cold start."""
    try:
        get_rag_pipeline()
        print("✅ Startup preload completed")
    except Exception as e:
        # Keep API alive; first request will surface detailed initialization errors.
        print(f"⚠️  Startup preload skipped: {e}")


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for chat queries"""
    query: str
    user_language: Optional[str] = None  # If user wants to specify language
    top_k: Optional[int] = 5
    conversation_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Bagaimana cara membayar cukai pendapatan?",
                "user_language": "ms",
                "top_k": 5,
                "conversation_id": "example-conversation-id",
                "conversation_history": [
                    {"role": "user", "text": "Saya perlukan bantuan kewangan."},
                    {"role": "assistant", "text": "Boleh saya tahu bantuan untuk pendidikan atau perubatan?"}
                ]
            }
        }
    )


class QueryResponse(BaseModel):
    """Response model for chat queries"""
    success: bool
    query: str
    detected_language: str
    answer: str
    sources: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]] = []
    intent: Optional[str] = None
    rag_used: Optional[bool] = None
    processing_time: float
    conversation_id: str
    conversation_title: Optional[str] = None
    conversation_summary: Optional[str] = None
    error: Optional[str] = None
    debug_logs: Optional[List[str]] = None  # Debug logs from RAG pipeline


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = None


class ConversationMessage(BaseModel):
    role: str
    text: str
    created_at: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    title: str
    summary: str
    created_at: str
    updated_at: str
    messages: List[ConversationMessage] = []


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]


class SummarizeRequest(BaseModel):
    """Request model for summarization"""
    text: str
    chunk_size: Optional[int] = 4000


class SummarizeResponse(BaseModel):
    """Response model for summarization"""
    success: bool
    summary: str
    error: Optional[str] = None


class SimplifyRequest(BaseModel):
    """Request model for text simplification"""
    text: str
    reading_level: Optional[str] = '5th-grade'


class SimplifyResponse(BaseModel):
    """Response model for text simplification"""
    success: bool
    simplified_text: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    rag_initialized: bool
    message: str


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return {
        "status": "online",
        "rag_initialized": rag_pipeline is not None,
        "message": "Multilingual RAG Bot API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rag_initialized": rag_pipeline is not None,
        "message": "RAG pipeline will load on first query" if rag_pipeline is None else "RAG pipeline ready"
    }

@app.post("/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):
    try:
        # Save the incoming audio blob to a temporary file
        temp_path = "temp_voice_input.wav"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run Whisper transcription
        result = stt_model.transcribe(temp_path, fp16=False)
        
        print(f"Transcribed: {result['text']}")
        
        return {
            "success": True, 
            "text": result["text"].strip(),
            "detected_lang": result.get("language")
        }
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Main chat endpoint - processes user query through RAG pipeline
    
    Example usage:
    ```
    POST /api/chat
    {
        "query": "How do I pay income tax?",
        "top_k": 5
    }
    ```
    """
    import time
    start_time = time.time()
    
    try:
        # Lazy load RAG pipeline on first request
        pipeline = get_rag_pipeline()
        conversation = None
        try:
            if request.conversation_id:
                conversation = conversation_store.get_conversation(request.conversation_id)
            else:
                conversation = conversation_store.create_conversation()
        except KeyError:
            conversation = conversation_store.create_conversation()

        conversation_id = conversation["id"]
        previous_history = [
            {"role": message["role"], "text": message["text"]}
            for message in conversation.get("messages", [])[-6:]
        ]

        conversation_store.append_message(conversation_id, "user", request.query)

        # When server has no stored history, use client-provided history as fallback
        # (important for step-gate confirmation detection when conversation_id is absent)
        effective_history = previous_history if previous_history else (request.conversation_history or [])

        conversation_store.append_message(conversation_id, "user", request.query)
        
        # Process query through RAG pipeline
        result = pipeline.process_query(
            user_query=request.query,
            user_language_hint=request.user_language,
            conversation_history=effective_history,
            conversation_summary=conversation.get("summary", ""),
        )
        
        processing_time = time.time() - start_time
        
        # Format the answer for the frontend
        answer_text = result.get("answer_text", result.get("answer", []))
        if isinstance(answer_text, list):
            answer_text = "\n".join(answer_text)

        conversation_store.append_message(conversation_id, "assistant", answer_text)
        conversation = conversation_store.refresh_summary(conversation_id)
        
        # Get detected language (RAG returns 'language' field, not 'detected_language')
        detected_lang = result.get("detected_language", result.get("language", "unknown"))
        
        # Get sources from RAG pipeline
        sources = result.get("sources", [])
        
        return {
            "success": result.get("status") == "success",
            "query": request.query,
            "detected_language": detected_lang,
            "answer": answer_text,
            "sources": sources,
            "evidence": result.get("evidence", []),
            "intent": result.get("intent", "task_or_policy"),
            "rag_used": result.get("rag_used", True),
            "processing_time": round(processing_time, 2),
            "conversation_id": conversation_id,
            "conversation_title": conversation.get("title"),
            "conversation_summary": conversation.get("summary", ""),
            "error": None,
            "debug_logs": result.get("debug_logs", [])
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)
        
        # Log error
        print(f"❌ Error processing query: {error_message}")
        
        return {
            "success": False,
            "query": request.query,
            "detected_language": "unknown",
            "answer": "",
            "sources": [],
            "processing_time": round(processing_time, 2),
            "conversation_id": request.conversation_id or "",
            "conversation_title": None,
            "conversation_summary": None,
            "error": error_message,
            "debug_logs": []
        }


@app.get("/api/conversations", response_model=ConversationListResponse)
async def list_conversations():
    conversations = conversation_store.list_conversations(limit=20)
    return {"conversations": [{**conversation, "messages": []} for conversation in conversations]}


@app.post("/api/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationCreateRequest):
    conversation = conversation_store.create_conversation(request.title)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    try:
        conversation = conversation_store.refresh_summary(conversation_id)
        return conversation
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Summarize a long document into bullet points.
    """
    try:
        pipeline = get_rag_pipeline()
        summary = pipeline.summarize_document(
            document_text=request.text,
            chunk_size=request.chunk_size
        )
        return {
            "success": True,
            "summary": summary,
        }
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return {
            "success": False,
            "summary": "",
            "error": str(e)
        }


@app.post("/api/simplify", response_model=SimplifyResponse)
async def simplify(request: SimplifyRequest):
    """
    Simplify a text to a specified reading level.
    """
    try:
        pipeline = get_rag_pipeline()
        simplified_text = pipeline.simplify_text(
            text=request.text,
            reading_level=request.reading_level
        )
        return {
            "success": True,
            "simplified_text": simplified_text,
        }
    except Exception as e:
        print(f"❌ Error during simplification: {e}")
        return {
            "success": False,
            "simplified_text": "",
            "error": str(e)
        }


@app.get("/api/supported-languages")
async def supported_languages():
    """Get list of supported languages"""
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "ms", "name": "Malay"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ta", "name": "Tamil"},
            {"code": "th", "name": "Thai"},
            {"code": "vi", "name": "Vietnamese"},
            {"code": "tl", "name": "Tagalog"},
            {"code": "id", "name": "Indonesian"}
        ]
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    print("=" * 60)
    print(f"  Multilingual RAG Bot API Server")
    print(f"  http://localhost:{port}")
    print(f"  Docs: http://localhost:{port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
