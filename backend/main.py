"""
FastAPI Server for Multilingual RAG Bot
========================================
Exposes RAG pipeline as REST API for Next.js frontend
"""
import sys
import io
import os
import shutil
import threading

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
import uvicorn
from dotenv import load_dotenv
from conversation_store import ConversationStore

try:
    import whisper  # type: ignore
    WHISPER_IMPORT_ERROR = None
except Exception as whisper_error:
    whisper = None
    WHISPER_IMPORT_ERROR = whisper_error

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("⚠️ edge-tts not installed. Highly human-like voice features will fail. Run: pip install edge-tts")

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual RAG Bot API",
    description="RAG-powered multilingual citizen services chatbot",
    version="1.0.0"
)

stt_model = None


def get_stt_model():
    """Lazy-load Whisper model so backend can start even if STT deps are missing."""
    global stt_model
    if stt_model is not None:
        return stt_model

    if whisper is None:
        raise RuntimeError(
            "Whisper STT is not available. Install openai-whisper and ffmpeg, "
            f"then restart backend. Import error: {WHISPER_IMPORT_ERROR}"
        )

    stt_model = whisper.load_model("base", device="cpu")
    return stt_model

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
pipeline_init_lock = threading.Lock()
conversation_store = ConversationStore()
warmup_state = {
    "status": "starting",  # starting | warming | ready | failed
    "error": None,
}

def get_rag_pipeline():
    """Lazy load RAG pipeline on first use"""
    global rag_pipeline, RAGPipeline, warmup_state
    if rag_pipeline is not None:
        return rag_pipeline

    with pipeline_init_lock:
        if rag_pipeline is not None:
            return rag_pipeline

        try:
            if RAGPipeline is None:
                from pipeline.RAG import RAGPipeline as RAGPipelineClass
                RAGPipeline = RAGPipelineClass
            print("🚀 Initializing RAG Pipeline...")
            warmup_state["status"] = "warming"
            warmup_state["error"] = None
            rag_pipeline = RAGPipeline()
            warmup_state["status"] = "ready"
            print("✅ RAG Pipeline initialized successfully")
        except Exception as e:
            warmup_state["status"] = "failed"
            warmup_state["error"] = str(e)
            print(f"❌ Failed to initialize RAG pipeline: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"RAG pipeline initialization failed: {str(e)}"
            )
    return rag_pipeline


@app.on_event("startup")
async def preload_rag_pipeline():
    """Warm up pipeline in the background so startup endpoints stay responsive."""

    def _background_warmup() -> None:
        global warmup_state
        print("\n" + "=" * 70)
        print("🔥 WARMUP PHASE: PRE-LOADING ALL MODELS IN BACKGROUND")
        print("=" * 70)
        try:
            warmup_state["status"] = "warming"
            warmup_state["error"] = None
            print("⏱️  Loading embedding model, translation model, LLM...")
            pipeline = get_rag_pipeline()

            print("\n📊 Running embedding warmup...")
            test_embedding = pipeline.create_query_embedding("test warmup query")
            print(f"   ✓ Embedding warmup successful (dimension: {len(test_embedding)})")

            warmup_state["status"] = "ready"
            print("\n✅ Background warmup completed")
            print("=" * 70 + "\n")
        except Exception as e:
            warmup_state["status"] = "failed"
            warmup_state["error"] = str(e)
            print(f"\n⚠️  Background warmup failed: {e}")
            print("   Pipeline will initialize on first chat request.")
            print("=" * 70 + "\n")

    threading.Thread(target=_background_warmup, daemon=True).start()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for chat queries"""
    query: str
    user_language: Optional[str] = None  # If user wants to specify language
    top_k: Optional[int] = 5
    summary_mode_enabled: Optional[bool] = None
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
    summary_mode: Optional[bool] = None
    summary_mode_reason: Optional[str] = None
    used_citation_tags: Optional[List[str]] = None
    unused_citation_tags: Optional[List[str]] = None
    citation_stats: Optional[Dict[str, Any]] = None
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
    # Metadata fields that can be stored
    sources: Optional[List[Dict[str, Any]]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    detectedLanguage: Optional[str] = None
    intent: Optional[str] = None
    ragUsed: Optional[bool] = None
    summaryMode: Optional[bool] = None
    summaryModeReason: Optional[str] = None
    usedCitationTags: Optional[List[str]] = None
    unusedCitationTags: Optional[List[str]] = None
    citationStats: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    debugLogs: Optional[List[str]] = None
    
    class Config:
        extra = "allow"  # Allow any extra fields to pass through


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
    warmup_status: str
    warmup_error: Optional[str] = None
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
        "warmup_status": warmup_state["status"],
        "warmup_error": warmup_state["error"],
        "message": "Multilingual RAG Bot API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    if warmup_state["status"] == "ready":
        message = "RAG pipeline ready"
    elif warmup_state["status"] == "failed":
        message = "RAG warmup failed; backend will retry on first query"
    else:
        message = "RAG warmup in progress"

    return {
        "status": "healthy",
        "rag_initialized": rag_pipeline is not None,
        "warmup_status": warmup_state["status"],
        "warmup_error": warmup_state["error"],
        "message": message,
    }

@app.post("/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):
    temp_path = "temp_voice_input.wav"
    try:
        # Save the incoming audio blob to a temporary file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run Whisper transcription
        model = get_stt_model()
        result = model.transcribe(temp_path, fp16=False)
        
        print(f"Transcribed: {result['text']}")
        
        return {
            "success": True, 
            "text": result["text"].strip(),
            "detected_lang": result.get("language")
        }
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

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

        # When server has no stored history, use client-provided history as fallback
        # (important for step-gate confirmation detection when conversation_id is absent)
        effective_history = previous_history if previous_history else (request.conversation_history or [])

        conversation_store.append_message(conversation_id, "user", request.query)
        
        # Process query through RAG pipeline
        result = pipeline.process_query(
            user_query=request.query,
            user_language_hint=request.user_language,
            top_k=request.top_k,
            summary_mode_enabled=request.summary_mode_enabled,
            conversation_history=effective_history,
            conversation_summary=conversation.get("summary", ""),
        )
        
        processing_time = time.time() - start_time
        
        # Format the answer for the frontend
        answer_text = result.get("answer_text", result.get("answer", []))
        if isinstance(answer_text, list):
            answer_text = "\n".join(answer_text)

        # Prepare metadata for assistant message
        assistant_metadata = {
            "sources": result.get("sources", []),
            "evidence": result.get("evidence", []),
            "detectedLanguage": result.get("detected_language", result.get("language", "unknown")),
            "intent": result.get("intent", "task_or_policy"),
            "ragUsed": result.get("rag_used", True),
            "summaryMode": result.get("summary_mode", False),
            "summaryModeReason": result.get("summary_mode_reason"),
            "usedCitationTags": result.get("used_citation_tags", []),
            "unusedCitationTags": result.get("unused_citation_tags", []),
            "citationStats": result.get("citation_stats", {}),
            "status": result.get("status", "unknown"),
            "debugLogs": result.get("debug_logs", []),
        }
        
        conversation_store.append_message(conversation_id, "assistant", answer_text, metadata=assistant_metadata)
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
            "summary_mode": result.get("summary_mode", False),
            "summary_mode_reason": result.get("summary_mode_reason"),
            "used_citation_tags": result.get("used_citation_tags", []),
            "unused_citation_tags": result.get("unused_citation_tags", []),
            "citation_stats": result.get("citation_stats", {}),
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


@app.get("/api/source-preview")
async def source_preview(
    source_url: str = Query(..., description="Exact source URL stored in embeddings.source_url"),
    highlight_chunk_index: Optional[int] = Query(None, description="Chunk index to highlight in reconstructed preview"),
    highlight_title: Optional[str] = Query(None, description="Source title/chunk title fallback for highlight resolution"),
):
    """Return reconstructed document preview from chunks stored in Supabase.

    This avoids client-side iframe internet rendering by reconstructing text
    from indexed chunks and letting UI highlight the retrieved chunk.
    """
    pipeline = get_rag_pipeline()

    try:
        response = (
            pipeline.supabase_client
            .table('embeddings')
            .select('title,content,source_url,chunk_index,total_chunks,page_number,page_start,page_end,section,subsection,source_type')
            .eq('source_url', source_url)
            .order('chunk_index', desc=False)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch source chunks: {e}")

    chunks = response.data or []
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for source_url")

    def sort_key(item: Dict[str, Any]) -> tuple:
        idx = item.get('chunk_index')
        if isinstance(idx, int):
            return (0, idx)
        return (1, 10**9)

    chunks = sorted(chunks, key=sort_key)

    resolved_highlight_index = highlight_chunk_index
    resolved_highlight_position: Optional[int] = None

    if resolved_highlight_index is None and highlight_title:
        normalized_target = highlight_title.strip().lower()
        for pos, chunk in enumerate(chunks):
            chunk_title = (chunk.get('title') or '').strip().lower()
            if chunk_title and chunk_title == normalized_target:
                chunk_idx = chunk.get('chunk_index')
                if isinstance(chunk_idx, int):
                    resolved_highlight_index = chunk_idx
                else:
                    resolved_highlight_position = pos
                break

    reconstructed_parts = []
    for position, chunk in enumerate(chunks):
        idx = chunk.get('chunk_index')
        content = (chunk.get('content') or '').strip()
        if not content:
            continue

        meta_segments = []
        if chunk.get('page_number') is not None:
            meta_segments.append(f"Page {chunk.get('page_number')}")
        elif chunk.get('page_start') is not None and chunk.get('page_end') is not None:
            meta_segments.append(f"Pages {chunk.get('page_start')}-{chunk.get('page_end')}")

        if idx is not None:
            meta_segments.append(f"Chunk {idx}")
        else:
            meta_segments.append(f"Chunk {position}")

        section = (chunk.get('section') or '').strip()
        subsection = (chunk.get('subsection') or '').strip()
        if section:
            meta_segments.append(section)
        if subsection and subsection != section:
            meta_segments.append(subsection)

        heading = " | ".join(meta_segments)
        reconstructed_parts.append(f"### {heading}\n\n{content}")

    reconstructed_markdown = "\n\n---\n\n".join(reconstructed_parts)

    return {
        "source_url": source_url,
        "source_title": chunks[0].get('title') or 'Document',
        "source_type": chunks[0].get('source_type') or 'unknown',
        "chunk_count": len(chunks),
        "highlight_chunk_index": resolved_highlight_index,
        "highlight_chunk_position": resolved_highlight_position,
        "chunks": chunks,
        "reconstructed_markdown": reconstructed_markdown,
    }


@app.get("/api/conversations", response_model=ConversationListResponse)
async def list_conversations():
    conversations = conversation_store.list_conversations()
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


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    try:
        conversation_store.delete_conversation(conversation_id)
        return {"success": True, "conversation_id": conversation_id}
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


@app.get("/api/tts")
async def text_to_speech(text: str, lang: Optional[str] = None):
    """
    Generate highly human-like TTS using Microsoft Edge Neural voices.
    """
    if not EDGE_TTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="edge-tts package is not installed. Run: pip install edge-tts")
        
    # Detect the actual language of the text to prevent mismatched accents
    try:
        from langdetect import detect
        base_lang = detect(text).split('-')[0]
        
        # Fix Malay vs Indonesian confusion: 
        # If it's detected as either, trust the 'lang' parameter from the frontend, 
        # or default to 'ms' (Malay) to prevent Indonesian voice for Malay text.
        if base_lang in ['id', 'ms']:
            hint = (lang or '').lower().split('-')[0]
            base_lang = hint if hint in ['ms', 'id'] else 'ms'
    except Exception:
        # Fallback to provided lang or English if detection fails
        base_lang = (lang or 'en').lower().split('-')[0]

    # Select best neural voices for ASEAN languages
    voice_map = {
        'en': 'en-US-AriaNeural',
        'ms': 'ms-MY-OsmanNeural',
        'zh': 'zh-CN-XiaoxiaoNeural',
        'ta': 'ta-IN-PallaviNeural',
        'th': 'th-TH-PremwadeeNeural',
        'vi': 'vi-VN-HoaiMyNeural',
        'tl': 'fil-PH-BlessicaNeural',
        'id': 'id-ID-GadisNeural'
    }
    
    # Default to English if language code not found
    voice = voice_map.get(base_lang, 'en-US-AriaNeural')
    
    try:
        async def audio_stream():
            communicate = edge_tts.Communicate(text, voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
            
        return StreamingResponse(audio_stream(), media_type="audio/mpeg")
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    # Fix multiprocessing spawn issues in Python 3.13+ on Windows
    import multiprocessing
    if sys.platform == 'win32':
        try:
            # Use 'spawn' method for Windows (safest for libraries like sentence-transformers)
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set in this process
    
    # Enable faster model loading at startup
    os.environ.setdefault('HF_HUB_OFFLINE', 'false')  # Use cached models when available
    
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    # Check if running in DEV or PROD mode
    dev_mode = os.getenv("DEV_MODE", "true").lower() in {"1", "true", "yes"}
    
    print("\n" + "=" * 60)
    print(f"  Multilingual RAG Bot API Server")
    print(f"  http://localhost:{port}")
    print(f"  Docs: http://localhost:{port}/docs")
    print(f"  Mode: {'🔧 DEVELOPMENT (reload enabled)' if dev_mode else '🚀 PRODUCTION (no reload)'}")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=dev_mode,  # Only reload in dev mode to avoid cold starts in production
        log_level="info"
    )
