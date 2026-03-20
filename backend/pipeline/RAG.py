"""
Complete RAG Pipeline for Multilingual Bot
===========================================
This pipeline handles user queries through:
1. Language detection
2. Query embedding
3. Vector search in Supabase
4. LLM-based summarization and translation
5. Formatted response with sources
"""

import os
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import threading
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)

# ============================================================================
# 🔧 CONFIGURABLE: CORE LIBRARIES AND MODELS
# ============================================================================
from sentence_transformers import SentenceTransformer
from langdetect import detect, detect_langs
from openai import OpenAI
from supabase import create_client, Client
import spacy
from .kshot_library import KSHOT_LIBRARY

# 🔧 HACKATHON: Southeast Asian LLM Support (Hugging Face Transformers)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not installed. Install with: pip install transformers torch")

# 🔧 CLOUD: Groq Support
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq not installed. Install with: pip install groq")

# 🔧 CLOUD: Hugging Face Inference API
try:
    from huggingface_hub import InferenceClient
    HF_INFERENCE_AVAILABLE = True
except ImportError:
    HF_INFERENCE_AVAILABLE = False
    print("⚠️  HF Hub not installed. Install with: pip install huggingface_hub")

# 🔧 TRANSLATION: NLLB-200 for ASEAN Dialect Translation
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
    NLLB_AVAILABLE = True
except ImportError:
    NLLB_AVAILABLE = False
    print("⚠️  Translation models not available. Install with: pip install transformers sentencepiece")

# ============================================================================
# 🔧 CONFIGURABLE: MODEL SELECTION
# Change these models based on your requirements
# ============================================================================

# Embedding Model - Used for converting text to vectors
# Options: 'BAAI/bge-m3', 'distiluse-base-multilingual-cased-v2',
#          'all-MiniLM-L6-v2', 'paraphrase-multilingual-mpnet-base-v2'
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'

# Embedding provider
# Options: 'hf_inference' (cloud), 'local' (SentenceTransformer)
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'hf_inference').lower()

# SpaCy Language Models
# Options: 'en_core_web_sm', 'en_core_web_md', 'en_core_web_lg' (English)
#          'xx_ent_wiki_sm' (Multilingual)
SPACY_MODEL_EN = 'en_core_web_sm'
SPACY_MODEL_MULTILINGUAL = 'xx_ent_wiki_sm'

# ============================================================================
# 🔧 HACKATHON: LLM Provider Selection
# Options: 'hf_inference', 'groq', 'huggingface', 'openai'
# ============================================================================
# Enable speed-first defaults for lower latency in production demos
SPEED_MODE = os.getenv('SPEED_MODE', 'true').lower() in {'1', 'true', 'yes'}

# Default to HF Inference now that OpenRouter/Gemini are removed
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'hf_inference').lower()  # Change based on your preference

# Enable automatic fallback if primary provider fails
ENABLE_FALLBACK = os.getenv('ENABLE_FALLBACK', 'true').lower() in {'1', 'true', 'yes'}
FALLBACK_ORDER = ['groq', 'openai', 'hf_inference', 'huggingface']  # Try in this order

# OpenAI-compatible LLM Model Selection
# Examples: 'openai/gpt-oss-120b', 'gpt-4o', 'gpt-4.1-mini'
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME', 'openai/gpt-oss-120b')

# ============================================================================
# 🔧 HACKATHON: Southeast Asian LLM Models (Hugging Face)
# Recommended for multilingual support (Tagalog, Malay, Thai, Vietnamese)
# ============================================================================
HUGGINGFACE_MODEL_OPTIONS = {
    # 🇸🇬 SEA-LION v4 - Best overall SEA language support
    'sealion': 'aisingapore/Apertus-SEA-LION-v4-8B-IT',
    
    # 🇸🇬 Sailor 2 - Good multilingual chat (lighter)
    'sailor': 'sail/Sailor2-8B-Chat',
    
    # 🇲🇾 ILMU - Malaysian-focused (Malay + English)
    'ilmu': 'malaysian-open-llm/ILMU-7B',
    
    # 🇹🇭 Typhoon - Thai + English
    'typhoon': 'scb10x/typhoon-7b-instruct',
    
    # 🇻🇳 VinaLLaMA - Vietnamese
    'vinallama': 'vilm/vinallama-7b-chat',
}

# Select which SEA model to use (change this!)
SEA_MODEL_CHOICE = 'sealion'  # Options: 'sealion', 'sailor', 'ilmu', 'typhoon', 'vinallama'
HUGGINGFACE_MODEL_NAME = HUGGINGFACE_MODEL_OPTIONS[SEA_MODEL_CHOICE]

# Use 4-bit quantization to reduce memory (8GB RAM → 4GB RAM)
USE_QUANTIZATION = True  # Set False if you have >16GB RAM

# ============================================================================
# 🔧 CLOUD: Model Selection for Cloud Providers
# ============================================================================

# Groq Models (all FREE)
# 'llama-3.3-70b-versatile' - Newest, best quality
# 'llama-3.1-70b-versatile' - Stable, good performance
# 'mixtral-8x7b-32768' - Alternative, large context
GROQ_MODEL = 'llama-3.3-70b-versatile'

# Hugging Face Inference API Models
# Can use any public model on HF Hub
HF_INFERENCE_MODEL = 'aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct'

# ============================================================================
# 🔧 CONFIGURABLE: API KEYS AND DATABASE CONNECTION
# ============================================================================

# Set your API keys here or via environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'YOUR_OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
SUPABASE_URL = os.getenv('SUPABASE_URL', 'YOUR_SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'YOUR_SUPABASE_KEY')

# 🔧 CLOUD: API Keys for Free Cloud Providers
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'YOUR_GROQ_API_KEY')  # Get at: https://console.groq.com
HF_TOKEN = os.getenv('HF_TOKEN', 'YOUR_HF_TOKEN')  # Get at: https://huggingface.co/settings/tokens

# Hugging Face API resilience settings
HF_EMBEDDING_MAX_RETRIES = int(os.getenv('HF_EMBEDDING_MAX_RETRIES', '1' if SPEED_MODE else '3'))
HF_LLM_MAX_RETRIES = int(os.getenv('HF_LLM_MAX_RETRIES', '2' if SPEED_MODE else '2'))
HF_RETRY_DELAY_SECONDS = float(os.getenv('HF_RETRY_DELAY_SECONDS', '0.5' if SPEED_MODE else '1.5'))
HF_EMBEDDING_WARMUP = os.getenv('HF_EMBEDDING_WARMUP', 'false').lower() in {'1', 'true', 'yes'}
HF_EMBEDDING_TIMEOUT_SECONDS = float(os.getenv('HF_EMBEDDING_TIMEOUT_SECONDS', '4.0' if SPEED_MODE else '10.0'))
HF_EMBEDDING_MAX_LATENCY_SECONDS = float(os.getenv('HF_EMBEDDING_MAX_LATENCY_SECONDS', '3.5' if SPEED_MODE else '8.0'))
LOCAL_EMBEDDING_STANDBY = os.getenv('LOCAL_EMBEDDING_STANDBY', 'true' if SPEED_MODE else 'false').lower() in {'1', 'true', 'yes'}

# ============================================================================
# 🔧 CONFIGURABLE: VECTOR SEARCH PARAMETERS
# ============================================================================

# Number of similar documents to retrieve
# Increased from 3 to 6 for better context breadth
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '6'))

# Pull more than top-k from Supabase, then rerank/filter in Python.
# Increased from 4 to 10 for wider candidate pool (~240 candidates before rerank)
VECTOR_CANDIDATE_MULTIPLIER = int(os.getenv('VECTOR_CANDIDATE_MULTIPLIER', '10'))
# Increased from 10 to 40 for minimum candidate threshold
VECTOR_MIN_CANDIDATES = int(os.getenv('VECTOR_MIN_CANDIDATES', '40'))
# Hard ceiling to prevent unbounded retrieval latency on broad queries.
VECTOR_MAX_CANDIDATES = int(os.getenv('VECTOR_MAX_CANDIDATES', '120'))

# Quality filters to avoid tiny chunks like single-word headings.
MIN_RETRIEVED_CHUNK_CHARS = int(os.getenv('MIN_RETRIEVED_CHUNK_CHARS', '60'))
MIN_RETRIEVED_CHUNK_WORDS = int(os.getenv('MIN_RETRIEVED_CHUNK_WORDS', '8'))
MIN_RETRIEVED_QUERY_OVERLAP = int(os.getenv('MIN_RETRIEVED_QUERY_OVERLAP', '1'))

# Supabase table name for embeddings
EMBEDDINGS_TABLE_NAME = 'embeddings'

# Supabase table name for labour office directory lookup
LABOUR_OFFICES_TABLE_NAME = os.getenv('LABOUR_OFFICES_TABLE_NAME', 'labour_offices')

# Supabase table name for community-submitted local phrases/slang.
SLANG_DICTIONARY_TABLE_NAME = os.getenv('SLANG_DICTIONARY_TABLE', 'community_dictionary')

# Query expansion settings for approved local phrase entries.
ENABLE_SLANG_QUERY_EXPANSION = os.getenv('ENABLE_SLANG_QUERY_EXPANSION', 'true').lower() in {'1', 'true', 'yes'}
SLANG_TERMS_CACHE_TTL_SECONDS = int(os.getenv('SLANG_TERMS_CACHE_TTL_SECONDS', '300'))
SLANG_MAX_TERMS_PER_QUERY = int(os.getenv('SLANG_MAX_TERMS_PER_QUERY', '5'))

# Similarity threshold (0-1, higher = more strict)
SIMILARITY_THRESHOLD = 0.15

# ============================================================================
# 🔧 CONFIGURABLE: LLM PROMPT SETTINGS
# ============================================================================

# Number of bullet points in summary
NUM_BULLET_POINTS = '3-5'

# Response style for generated answers
# Options: 'narrative', 'bullet', 'mixed'
ANSWER_STYLE = os.getenv('ANSWER_STYLE', 'narrative').lower()

# Controls how much reasoning/detail the final answer should include.
# Options: 'concise', 'balanced', 'detailed'
RESPONSE_DEPTH = os.getenv('RESPONSE_DEPTH', 'balanced').lower()

# Set true to always end with a follow-up question.
FORCE_FOLLOW_UP_QUESTION = os.getenv('FORCE_FOLLOW_UP_QUESTION', 'true').lower() in {'1', 'true', 'yes'}

# Controls follow-up behavior:
# - adaptive: choose follow-up based on response type
# - fixed: keep legacy "next step" follow-up
# - off: disable automatic follow-up
FOLLOW_UP_STYLE = os.getenv('FOLLOW_UP_STYLE', 'adaptive').strip().lower()

# Step-gating can add one extra turn; keep it off by default for direct guidance.
ENABLE_STEP_GATE = os.getenv('ENABLE_STEP_GATE', 'false').lower() in {'1', 'true', 'yes'}

# Reading level for simplification
READING_LEVEL = '5th-grade'

# Enable LLM to auto-detect and respond in user's language (k-shot prompting)
ENABLE_LANGUAGE_AUTO_DETECTION = os.getenv('ENABLE_LANGUAGE_AUTO_DETECTION', 'true').lower() in {'1', 'true', 'yes'}

# K-shot examples for low-resource languages/dialects (comma-separated language codes)
# Examples: 'ms,tl,th,vi' to enable k-shot examples for Malay, Tagalog, Thai, Vietnamese


LOW_RESOURCE_LANGUAGE_EXAMPLES = os.getenv('LOW_RESOURCE_LANGUAGE_EXAMPLES', 'ms,tl,th').lower().split(',')

# K-shot prompt budget controls
KSHOT_MAX_EXAMPLES_PER_LANGUAGE = int(os.getenv('KSHOT_MAX_EXAMPLES_PER_LANGUAGE', '4'))
KSHOT_MAX_TOTAL_CHARS = int(os.getenv('KSHOT_MAX_TOTAL_CHARS', '5000'))

# Reduce chunk size before sending retrieved context to the LLM
CHUNK_SUMMARY_MAX_CHARS = 700 if SPEED_MODE else 900
CHUNK_SUMMARY_MAX_SENTENCES = 3 if SPEED_MODE else 4
TOTAL_CONTEXT_MAX_CHARS = 2200 if SPEED_MODE else 3200

# Auto summary mode: switch to actionable 3-5 bullets when retrieved context is long/complex.
# DISABLED by default to preserve inline citations. Only explicit user intent ("summarize", "ringkasan") triggers summary.
# This ensures detailed citations remain the default, and summary is opt-in, not automatic.
AUTO_ACTION_BULLET_SUMMARY = os.getenv('AUTO_ACTION_BULLET_SUMMARY', 'false').lower() in {'1', 'true', 'yes'}
AUTO_SUMMARY_LONG_CONTEXT_CHARS = int(os.getenv('AUTO_SUMMARY_LONG_CONTEXT_CHARS', '1400'))
AUTO_SUMMARY_LONG_CHUNK_COUNT = int(os.getenv('AUTO_SUMMARY_LONG_CHUNK_COUNT', '4'))
AUTO_SUMMARY_COMPLEX_SOURCE_COUNT = int(os.getenv('AUTO_SUMMARY_COMPLEX_SOURCE_COUNT', '3'))
AUTO_SUMMARY_COMPLEX_CONTEXT_CHARS = int(os.getenv('AUTO_SUMMARY_COMPLEX_CONTEXT_CHARS', '900'))

# Recursive summarization controls for chat summary-mode responses.
RECURSIVE_SUMMARY_CHUNK_SIZE = int(os.getenv('RECURSIVE_SUMMARY_CHUNK_SIZE', '1200'))
RECURSIVE_SUMMARY_INPUT_MAX_CHARS = int(os.getenv('RECURSIVE_SUMMARY_INPUT_MAX_CHARS', '14000'))

# Maximum tokens for LLM response
MAX_TOKENS = 550 if SPEED_MODE else 1000

# Temperature for LLM (0-2, lower = more focused)
LLM_TEMPERATURE = 0.7

# ============================================================================
# 🔧 CONFIGURABLE: TRANSLATION SETTINGS (ASEAN Dialect Support)
# ============================================================================

# Enable dialect-aware translation
ENABLE_TRANSLATION = True

# Translation model - NLLB-200 supports 200+ languages including ASEAN dialects
# Options: 'facebook/nllb-200-distilled-600M' (faster, 2.4GB)
#          'facebook/nllb-200-1.3B' (better quality, 5GB)
#          'facebook/nllb-200-3.3B' (best quality, 13GB - not recommended for demo)
TRANSLATION_MODEL_NAME = 'facebook/nllb-200-distilled-600M'

# Pivot languages for RAG retrieval (your embeddings are in these languages)
PIVOT_LANGUAGES = ['eng_Latn', 'zsm_Latn']  # English and Malay

# If enabled, queries in embedding-supported non-pivot languages are sent
# directly to vector search instead of being translated to pivot first.
DIRECT_RETRIEVAL_FOR_SUPPORTED_LANGS = os.getenv(
    'DIRECT_RETRIEVAL_FOR_SUPPORTED_LANGS',
    'true'
).lower() in {'1', 'true', 'yes'}

# NLLB Language Code Mapping for ASEAN Countries
# Format: 'langdetect_code' -> 'nllb_flores_code'
ASEAN_LANGUAGE_MAP = {
    # English
    'en': 'eng_Latn',
    
    # Malay/Malaysian/Indonesian
    'ms': 'zsm_Latn',      # Standard Malay
    'id': 'ind_Latn',      # Indonesian
    
    # Tagalog/Filipino
    'tl': 'tgl_Latn',      # Tagalog
    'fil': 'tgl_Latn',     # Filipino (use Tagalog)
    
    # Thai
    'th': 'tha_Thai',
    
    # Vietnamese
    'vi': 'vie_Latn',
    
    # Burmese/Myanmar
    'my': 'mya_Mymr',
    
    # Khmer/Cambodian
    'km': 'khm_Khmr',
    
    # Lao
    'lo': 'lao_Laoo',
    
    # Chinese variants
    'zh-cn': 'zho_Hans',   # Simplified Chinese
    'zh-tw': 'zho_Hant',   # Traditional Chinese
    'zh': 'zho_Hans',      # Default to Simplified
    
    # Tamil (Singapore/Malaysia)
    'ta': 'tam_Taml',
    
    # Javanese (Indonesia)
    'jv': 'jav_Latn',
    
    # Sundanese (Indonesia)
    'su': 'sun_Latn',
    
    # Cebuano (Philippines)
    'ceb': 'ceb_Latn',
    
    # Shan (Myanmar)
    'shn': 'shn_Mymr',
    
    # Minangkabau (Indonesia)
    'min': 'min_Latn',
    
    # Acehnese (Indonesia)
    'ace': 'ace_Latn',
    
    # Banjar (Indonesia)
    'bjn': 'bjn_Latn',
}

# Fallback behavior for unsupported dialects
TRANSLATION_FALLBACK = 'zsm_Latn'  # Default to Malay if dialect unsupported


class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) Pipeline
    """
    
    def __init__(self):
        """Initialize all models and connections"""
        print("🚀 Initializing RAG Pipeline...")
        
        # Debug logging for frontend
        self.debug_logs = []
        
        # Step 0: Initialize embedding provider
        self.embedding_provider = EMBEDDING_PROVIDER
        self.embedding_model = None
        self.embedding_client = None
        self.embedding_dim = None
        self.embedding_fallback_active = False
        self._embedding_local_lock = threading.Lock()

        print(f"📦 Initializing embeddings ({self.embedding_provider}): {EMBEDDING_MODEL_NAME}")
        if self.embedding_provider == 'hf_inference':
            if not HF_INFERENCE_AVAILABLE:
                print("⚠️  HF embedding provider selected but huggingface_hub is not installed")
                print("   Falling back to local SentenceTransformer embeddings")
                self.embedding_provider = 'local'
            elif not HF_TOKEN or HF_TOKEN == 'YOUR_HF_TOKEN':
                print("⚠️  HF embedding provider selected but HF_TOKEN is not configured")
                print("   Falling back to local SentenceTransformer embeddings")
                self.embedding_provider = 'local'
            else:
                try:
                    self.embedding_client = InferenceClient(
                        model=EMBEDDING_MODEL_NAME,
                        token=HF_TOKEN,
                        timeout=HF_EMBEDDING_TIMEOUT_SECONDS,
                    )
                    if HF_EMBEDDING_WARMUP:
                        warmup_vec = self._get_hf_embedding_with_retry("Embedding warmup test")
                        self.embedding_dim = len(warmup_vec)
                        print(f"   ✓ HF embeddings initialized (dimension: {self.embedding_dim})")
                    else:
                        print("   ✓ HF embeddings client initialized (warmup disabled)")
                except Exception as e:
                    print(f"   ⚠️  HF embedding initialization failed: {e}")
                    print("   ℹ️  Falling back to local SentenceTransformer embeddings")
                    self.embedding_provider = 'local'

        if self.embedding_provider == 'hf_inference' and LOCAL_EMBEDDING_STANDBY:
            def _warm_local_standby() -> None:
                try:
                    print("   🔥 Preloading local embedding standby in background...")
                    self._ensure_local_embedding_model()
                    print("   ✓ Local embedding standby ready")
                except Exception as e:
                    print(f"   ⚠️  Local embedding standby preload skipped: {e}")

            threading.Thread(target=_warm_local_standby, daemon=True).start()

        if self.embedding_provider == 'local':
            print(f"📦 Loading local embedding model: {EMBEDDING_MODEL_NAME}")
            try:
                # Prevent multiprocessing spawn issues in Python 3.13+ on Windows
                import torch
                torch.set_num_threads(2)
            except Exception as e:
                print(f"   ℹ️  Could not configure torch threading: {e}")
            
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"   ✓ Local embedding dimension: {self.embedding_dim}")
            
            # Warmup embedding model to cache in memory (startup warmup)
            try:
                print(f"   🔥 Warming up embedding model...")
                _ = self.embedding_model.encode("warmup test query")
                print(f"   ✓ Embedding model warmup complete")
            except Exception as e:
                print(f"   ⚠️  Warmup failed but model is ready: {e}")
        
        # Initialize SpaCy models
        print(f"📦 Loading SpaCy models...")
        try:
            self.nlp_en = spacy.load(SPACY_MODEL_EN)
            self.nlp_multi = spacy.load(SPACY_MODEL_MULTILINGUAL)
            # xx_ent_wiki_sm has no parser/sentencizer by default — add one so doc.sents works
            if not self.nlp_multi.has_pipe("sentencizer"):
                self.nlp_multi.add_pipe("sentencizer")
        except OSError as e:
            print(f"⚠️  SpaCy model not found: {e}")
            print("Run: python -m spacy download en_core_web_sm")
            print("Run: python -m spacy download xx_ent_wiki_sm")
            raise
        
        # Initialize Translation Model (NLLB-200 via Hugging Face Inference API)
        self.translator = None
        self.translation_enabled = False
        if ENABLE_TRANSLATION:
            if not HF_INFERENCE_AVAILABLE:
                print(f"⚠️  Translation enabled but huggingface_hub not available")
                print(f"   Install with: pip install huggingface_hub")
            elif not HF_TOKEN or HF_TOKEN == 'YOUR_HF_TOKEN':
                print(f"⚠️  Translation enabled but HF_TOKEN is not configured")
                print(f"   Get a token at: https://huggingface.co/settings/tokens")
            else:
                print(f"🌐 Initializing NLLB-200 translation via HF Inference API: {TRANSLATION_MODEL_NAME}")
                try:
                    # Use cloud-hosted NLLB model instead of downloading locally
                    self.translation_client = InferenceClient(
                        model=TRANSLATION_MODEL_NAME,
                        token=HF_TOKEN,
                    )
                    print(f"   ✓ Translation via HF Inference initialized "
                          f"(Supports {len(ASEAN_LANGUAGE_MAP)} ASEAN languages/dialects)")
                    self.translation_enabled = True
                except Exception as e:
                    print(f"   ⚠️  Translation client initialization failed: {e}")
                    print(f"   ℹ️  Continuing without translation support")
        
        # Initialize LLM based on provider
        self.llm_provider = LLM_PROVIDER
        self.active_llm_provider = None
        self._init_llm_provider(LLM_PROVIDER)
    
    def _get_llm_model_name(self, provider: str) -> str:
        """Return the configured model name for a provider."""
        model_map = {
            'groq': GROQ_MODEL,
            'hf_inference': HF_INFERENCE_MODEL,
            'openai': OPENAI_MODEL_NAME,
            'huggingface': HUGGINGFACE_MODEL_NAME,
        }
        return model_map.get(provider, 'unknown-model')

    def _describe_active_llm(self, provider: Optional[str] = None) -> str:
        """Build a readable label for the currently selected LLM."""
        resolved_provider = provider or self.active_llm_provider or self.llm_provider
        model_name = self._get_llm_model_name(resolved_provider)
        return f"provider={resolved_provider}, model={model_name}"

    def _init_llm_provider(self, provider: str):
        """Initialize the specified LLM provider"""
        self.active_llm_provider = provider

        if provider == 'groq' and GROQ_AVAILABLE:
            print(f"⚡ Initializing Groq: {GROQ_MODEL}")
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            print(f"   ✓ Groq initialized (Ultra-fast inference!)")
            
        elif provider == 'hf_inference' and HF_INFERENCE_AVAILABLE:
            print(f"🤗 Initializing Hugging Face Inference API: {HF_INFERENCE_MODEL}")
            self.hf_inference_client = InferenceClient(token=HF_TOKEN)
            print(f"   ✓ HF Inference initialized (SEA-LION cloud access!)")
            
        elif provider == 'openai':
            print(f"🤖 Initializing OpenAI-compatible client with model: {OPENAI_MODEL_NAME}")
            using_openrouter = (
                OPENROUTER_API_KEY and OPENROUTER_API_KEY != 'YOUR_OPENROUTER_API_KEY'
            )
            if using_openrouter:
                self.openai_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
                print(f"   ✓ OpenAI-compatible client initialized via OpenRouter")
            else:
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                print(f"   ✓ OpenAI client initialized")
            
        elif provider == 'huggingface' and TRANSFORMERS_AVAILABLE:
            print(f"💻 Loading Local SEA LLM: {SEA_MODEL_CHOICE} ({HUGGINGFACE_MODEL_NAME})")
            print(f"   ⚠️  This may take 2-5 minutes on first run (downloading model)...")
            
            if USE_QUANTIZATION:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.llm_model = pipeline(
                    "text-generation",
                    model=HUGGINGFACE_MODEL_NAME,
                    model_kwargs={"quantization_config": quantization_config},
                    device_map="auto"
                )
            else:
                self.llm_model = pipeline(
                    "text-generation",
                    model=HUGGINGFACE_MODEL_NAME,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            print(f"   ✓ Local model loaded successfully!")
            
        else:
            available = []
            if GROQ_AVAILABLE: available.append('groq')
            if HF_INFERENCE_AVAILABLE: available.append('hf_inference')
            if TRANSFORMERS_AVAILABLE: available.append('huggingface')
            available.append('openai')
            
            raise ValueError(
                f"Provider '{provider}' not available or dependencies not installed.\n"
                f"Available providers: {', '.join(available)}\n"
                f"Install missing: pip install groq huggingface_hub"
            )
        
        # Initialize Supabase client
        print(f"🗄️  Connecting to Supabase...")
        self.supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.slang_terms_cache: List[Dict[str, Any]] = []
        self.slang_terms_cache_loaded_at = 0.0
        self.last_query_expansions: List[Dict[str, str]] = []
        print(f"🧠 Active LLM configured: {self._describe_active_llm(provider)}")
        
        print("✅ RAG Pipeline initialized successfully!\n")
    
    # ========================================================================
    # STEP 1: LANGUAGE DETECTION
    # ========================================================================
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language and dialect of the input text
        
        Args:
            text: Input query text
            
        Returns:
            Dictionary with language code and confidence
        """
        print(f"🔍 Step 1: Detecting language for query: '{text[:50]}...'")
        
        try:
            # Special handling for very short words that langdetect struggles with
            common_short_english = {
                'yes', 'no', 'ok', 'okay', 'yep', 'nope', 'yeah', 'nah',
                'sure', 'maybe', 'perhaps', 'likely', 'probably',
                'true', 'false', 'correct', 'wrong', 'right', 'good', 'bad',
                'thanks', 'thank you', 'welcome', 'hi', 'hello', 'hey', 'bye',
                'continue', 'proceed', 'go ahead', 'please', 'ok thanks',
            }
            
            text_normalized = text.strip().lower()
            if text_normalized in common_short_english:
                print(f"   ✓ Detected short English word: '{text_normalized}' (confidence: 1.00)")
                return {
                    'primary_language': 'en',
                    'all_languages': [{'lang': 'en', 'probability': 1.0}],
                    'confidence': 1.0
                }

            # Lightweight lexical disambiguation for closely related languages
            # (especially Javanese vs Indonesian/Malay).
            tokens = set(re.findall(r"[a-zA-Z']+", text_normalized))
            javanese_markers = {
                'aku', 'arep', 'piye', 'ora', 'opo', 'kowe', 'nganti',
                'durung', 'wis', 'saben', 'dina', 'dhuwit', 'bengi',
            }
            indonesian_markers = {
                'saya', 'tidak', 'bagaimana', 'akan', 'lembur', 'gaji', 'kerja',
            }
            malay_markers = {
                'anda', 'boleh', 'pejabat', 'aduan', 'kerajaan', 'syarat',
            }

            jv_hits = len(tokens & javanese_markers)
            id_hits = len(tokens & indonesian_markers)
            ms_hits = len(tokens & malay_markers)

            if jv_hits >= 2 and jv_hits >= id_hits and jv_hits >= ms_hits:
                print(
                    "   ✓ Detected Javanese via lexical markers "
                    f"(jv={jv_hits}, id={id_hits}, ms={ms_hits})"
                )
                return {
                    'primary_language': 'jv',
                    'all_languages': [
                        {'lang': 'jv', 'probability': 0.95},
                        {'lang': 'id', 'probability': 0.04},
                        {'lang': 'ms', 'probability': 0.01},
                    ],
                    'confidence': 0.95
                }
            
            # Get primary language
            primary_lang = detect(text)
            
            # Get all detected languages with probabilities
            lang_probs = detect_langs(text)
            
            result = {
                'primary_language': primary_lang,
                'all_languages': [
                    {'lang': lp.lang, 'probability': lp.prob} 
                    for lp in lang_probs
                ],
                'confidence': lang_probs[0].prob if lang_probs else 0.0
            }
            
            print(f"   ✓ Detected language: {primary_lang} (confidence: {result['confidence']:.2f})")
            return result
            
        except Exception as e:
            print(f"   ⚠️  Language detection failed: {e}")
            return {
                'primary_language': 'en',
                'all_languages': [{'lang': 'en', 'probability': 1.0}],
                'confidence': 0.0
            }
    
    # ========================================================================
    # STEP 1.5: DIALECT-AWARE TRANSLATION (ASEAN Languages)
    # ========================================================================
    
    def _is_embedding_language_supported(self, langdetect_code: str) -> bool:
        """
        Check if the language is reasonably covered by the embedding model.
        
        This roughly follows BGE-M3 / multilingual sentence-transformer coverage:
        - Major languages: English, Malay, Indonesian, Tagalog, Thai, Vietnamese,
          Burmese, Khmer, Lao, Chinese variants, etc.
        - Anything outside this set is treated as a potential low-resource dialect.
        """
        major_langs = {
            # Core pivot / national languages
            'en', 'ms', 'id',
            # ASEAN national languages
            'tl', 'fil', 'th', 'vi', 'my', 'km', 'lo',
            # Chinese variants (Singapore/Malaysia/region)
            'zh', 'zh-cn', 'zh-tw',
            # Some common regional / migrant languages
            'ta', 'jv', 'su', 'ceb', 'shn', 'min', 'ace', 'bjn',
        }
        return langdetect_code in major_langs
    
    def _normalize_language_hint(self, code: Optional[str]) -> Optional[str]:
        """
        Normalize various language codes from the frontend into the compact
        forms we use internally (e.g., 'ms-MY' -> 'ms', 'en-US' -> 'en').
        """
        if not code:
            return None
        
        c = code.strip().lower()
        
        direct_map = {
            'ms-my': 'ms',
            'en-us': 'en',
            'en-gb': 'en',
            'zh-cn': 'zh',
            'zh-hans': 'zh',
            'zh-tw': 'zh',
            'zh-hant': 'zh',
        }
        if c in direct_map:
            return direct_map[c]
        
        if '-' in c:
            base, _region = c.split('-', 1)
            return base
        
        return c
    
    def _detect_script_language_hint(self, text: str) -> Optional[str]:
        """
        Lightweight script-based heuristic to improve language detection accuracy,
        especially when langdetect is noisy.
        
        This never overrides an explicit user hint, but can override langdetect
        when script is very clear (e.g., Chinese, Thai, Khmer).
        """
        for ch in text:
            cp = ord(ch)
            # CJK Unified Ideographs → Chinese (generic)
            if 0x4E00 <= cp <= 0x9FFF:
                return 'zh'
            # Thai
            if 0x0E00 <= cp <= 0x0E7F:
                return 'th'
            # Lao
            if 0x0E80 <= cp <= 0x0EFF:
                return 'lo'
            # Khmer
            if 0x1780 <= cp <= 0x17FF:
                return 'km'
            # Myanmar
            if 0x1000 <= cp <= 0x109F:
                return 'my'
        return None
    
    def _get_nllb_language_code(self, langdetect_code: str) -> str:
        """
        Convert langdetect language code to NLLB-200 FLORES code
        
        Args:
            langdetect_code: Language code from langdetect (e.g., 'en', 'ms', 'tl')
            
        Returns:
            NLLB FLORES code (e.g., 'eng_Latn', 'zsm_Latn', 'tgl_Latn')
        """
        # Look up in our ASEAN language map
        nllb_code = ASEAN_LANGUAGE_MAP.get(langdetect_code, None)
        
        if nllb_code:
            return nllb_code
        
        # Fallback: use Malay as default for unsupported dialects
        self._log_debug(f"[Translation] ⚠️  Language '{langdetect_code}' not in ASEAN map, using fallback: {TRANSLATION_FALLBACK}")
        return TRANSLATION_FALLBACK
    
    def _needs_translation(self, nllb_code: str) -> bool:
        """Check if text needs translation to pivot language (Malay/English)"""
        return nllb_code not in PIVOT_LANGUAGES
    
    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """
        Translate text between ASEAN languages using NLLB-200
        
        Args:
            text: Text to translate
            source_lang: Source language (langdetect code or NLLB code)
            target_lang: Target language (langdetect code or NLLB code)
            
        Returns:
            Dictionary with translated text and metadata
        """
        # If translation is disabled, return original text
        if not self.translation_enabled:
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'translation_performed': False,
                'note': 'Translation disabled or model not available'
            }
        
        try:
            # Convert to NLLB codes if needed
            src_nllb = self._get_nllb_language_code(source_lang) if '_' not in source_lang else source_lang
            tgt_nllb = self._get_nllb_language_code(target_lang) if '_' not in target_lang else target_lang
            
            # Skip translation if source and target are the same
            if src_nllb == tgt_nllb:
                return {
                    'translated_text': text,
                    'source_lang': src_nllb,
                    'target_lang': tgt_nllb,
                    'translation_performed': False,
                    'note': 'Source and target language are the same'
                }
            
            self._log_debug(f"[Translation] 🌐 Translating: {src_nllb} → {tgt_nllb}")
            
            # Use Hugging Face Inference API NLLB-200 translation endpoint
            translated_text = self.translation_client.translation(
                text,
                src_lang=src_nllb,
                tgt_lang=tgt_nllb,
            )
            
            self._log_debug(f"[Translation] ✓ Translation completed")
            
            return {
                'translated_text': translated_text,
                'source_lang': src_nllb,
                'target_lang': tgt_nllb,
                'translation_performed': True,
                'original_text': text
            }
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            self._log_debug(f"[Translation] ⚠️  {error_msg}")
            print(f"   ⚠️  {error_msg}")
            
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'translation_performed': False,
                'error': error_msg,
                'note': 'Using original text due to translation error'
            }
    
    # ========================================================================
    # STEP 2: QUERY EMBEDDING
    # ========================================================================

    def _coerce_embedding_vector(self, raw_embedding: Any) -> List[float]:
        """Normalize embedding outputs from HF/local providers into a flat float list."""
        if hasattr(raw_embedding, 'tolist'):
            raw_embedding = raw_embedding.tolist()

        if isinstance(raw_embedding, dict):
            for key in ('embedding', 'embeddings', 'vector'):
                if key in raw_embedding:
                    return self._coerce_embedding_vector(raw_embedding[key])

        if isinstance(raw_embedding, list):
            if raw_embedding and isinstance(raw_embedding[0], list):
                raw_embedding = raw_embedding[0]
            return [float(x) for x in raw_embedding]

        raise ValueError("Unexpected embedding response format")

    def _is_transient_hf_error(self, error: Exception) -> bool:
        """Return True for retryable network or temporary Hugging Face errors."""
        text = str(error).lower()
        transient_markers = (
            'readtimeout',
            'timed out',
            'timeout',
            'connection aborted',
            'connection reset',
            'temporarily unavailable',
            'service unavailable',
            'bad gateway',
            'gateway timeout',
            '429',
            '502',
            '503',
            '504',
        )
        return any(marker in text for marker in transient_markers)

    def _get_hf_embedding_with_retry(self, text: str) -> List[float]:
        """Get embedding from HF Inference API with retry/backoff for transient failures."""
        if not self.embedding_client:
            raise RuntimeError("HF embedding client not initialized")

        last_error = None
        for attempt in range(1, HF_EMBEDDING_MAX_RETRIES + 1):
            try:
                raw_embedding = self.embedding_client.feature_extraction(text)
                return self._coerce_embedding_vector(raw_embedding)
            except Exception as error:
                last_error = error
                is_retryable = self._is_transient_hf_error(error)
                if not is_retryable or attempt == HF_EMBEDDING_MAX_RETRIES:
                    break

                wait_seconds = HF_RETRY_DELAY_SECONDS * attempt
                msg = (
                    f"[Step 2] ⚠️ HF embedding request failed (attempt {attempt}/"
                    f"{HF_EMBEDDING_MAX_RETRIES}): {type(error).__name__}. "
                    f"Retrying in {wait_seconds:.1f}s"
                )
                self._log_debug(msg)
                print(f"   {msg}")
                time.sleep(wait_seconds)

        raise RuntimeError(
            "Embedding service is temporarily unavailable. "
            "Please retry in a few seconds."
        ) from last_error

    def _ensure_local_embedding_model(self) -> None:
        """Load local embedding model lazily for fallback scenarios."""
        if self.embedding_model is not None:
            return

        with self._embedding_local_lock:
            if self.embedding_model is not None:
                return

            print(f"📦 Loading local fallback embedding model: {EMBEDDING_MODEL_NAME}")
            try:
                import torch
                torch.set_num_threads(2)
            except Exception as e:
                print(f"   ℹ️  Could not configure torch threading: {e}")

            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            local_dim = self.embedding_model.get_sentence_embedding_dimension()
            if self.embedding_dim is None:
                self.embedding_dim = local_dim

            try:
                _ = self.embedding_model.encode("fallback warmup query")
                print("   ✓ Local fallback embedding model ready")
            except Exception as e:
                print(f"   ⚠️  Local fallback warmup skipped: {e}")

    def _get_local_embedding(self, query: str) -> List[float]:
        """Generate embedding using local SentenceTransformer model."""
        self._ensure_local_embedding_model()
        return self.embedding_model.encode(query).tolist()
    
    def create_query_embedding(self, query: str) -> List[float]:
        """
        Convert user query to vector embedding
        
        Args:
            query: User query text
            
        Returns:
            List of floats representing the embedding vector
        """
        print(f"🔢 Step 2: Creating query embedding...")

        if self.embedding_provider == 'hf_inference' and self.embedding_client:
            try:
                started = time.perf_counter()
                embedding = self._get_hf_embedding_with_retry(query)
                elapsed = time.perf_counter() - started

                # If HF is consistently slow, switch to local for subsequent RAG requests.
                if elapsed > HF_EMBEDDING_MAX_LATENCY_SECONDS:
                    slow_msg = (
                        f"[Step 2] ⚠️ HF embedding latency {elapsed:.2f}s exceeded "
                        f"threshold {HF_EMBEDDING_MAX_LATENCY_SECONDS:.2f}s. "
                        "Switching to local embeddings for faster RAG responses."
                    )
                    self._log_debug(slow_msg)
                    print(f"   {slow_msg}")
                    self._ensure_local_embedding_model()
                    self.embedding_fallback_active = True
                    self.embedding_provider = 'local'
            except Exception as hf_error:
                fallback_msg = (
                    f"[Step 2] ⚠️ HF embeddings unavailable ({type(hf_error).__name__}). "
                    "Falling back to local embedding model."
                )
                self._log_debug(fallback_msg)
                print(f"   {fallback_msg}")

                try:
                    embedding = self._get_local_embedding(query)
                    if not self.embedding_fallback_active:
                        self.embedding_fallback_active = True
                        self.embedding_provider = 'local'
                        self._log_debug("[Step 2] ✅ Switched embedding provider to local fallback")
                        print("   ✅ Switched embedding provider to local fallback")
                except Exception as local_error:
                    raise RuntimeError(
                        "Both HF embedding service and local fallback embedding model failed."
                    ) from local_error
        else:
            embedding = self._get_local_embedding(query)

        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
            print(f"   ℹ️  Detected embedding dimension: {self.embedding_dim}")

        print(f"   ✓ Generated embedding of dimension: {len(embedding)}")
        return embedding
    
    # ========================================================================
    # STEP 3: VECTOR SEARCH IN SUPABASE
    # ========================================================================
    
    def _compute_retrieval_budget(self, user_query: str, requested_top_k: Optional[int] = None) -> Dict[str, int]:
        """Compute adaptive retrieval budget so broad queries can retrieve richer context."""
        if isinstance(requested_top_k, int) and requested_top_k > 0:
            base_top_k = requested_top_k
        else:
            base_top_k = TOP_K_RESULTS

        base_top_k = max(3, min(base_top_k, 12))

        normalized_query = (user_query or '').lower().strip()
        word_count = len(re.findall(r"[A-Za-zÀ-ÿ0-9']+", normalized_query))

        broad_query_terms = {
            'all', 'overall', 'complete', 'everything', 'compare', 'difference',
            'options', 'types', 'requirements', 'documents', 'procedure', 'steps',
            'benefits', 'rights', 'policy', 'policies', 'eligibility',
            'keseluruhan', 'semua', 'syarat', 'dokumen', 'langkah', 'prosedur',
            'penuh', 'lengkap', 'semuanya',
        }

        is_broad_query = (
            word_count >= 10
            or any(term in normalized_query for term in broad_query_terms)
        )

        effective_top_k = min(base_top_k + 2, 12) if is_broad_query else base_top_k
        effective_multiplier = VECTOR_CANDIDATE_MULTIPLIER + (2 if is_broad_query else 0)
        effective_min_candidates = VECTOR_MIN_CANDIDATES + (20 if is_broad_query else 0)

        candidate_count = max(effective_top_k * effective_multiplier, effective_min_candidates)
        candidate_count = min(candidate_count, VECTOR_MAX_CANDIDATES)

        return {
            'top_k': effective_top_k,
            'candidate_count': candidate_count,
            'broad_query': is_broad_query,
        }

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        user_query: str = '',
    ) -> List[Dict[str, Any]]:
        """
        Search Supabase vector database for similar documents
        
        Args:
            query_embedding: Vector representation of the query
            
        Returns:
            List of relevant document chunks with metadata
        """
        retrieval_budget = self._compute_retrieval_budget(user_query, top_k)
        candidate_count = retrieval_budget['candidate_count']
        effective_top_k = retrieval_budget['top_k']
        broad_query = retrieval_budget['broad_query']

        print(
            "🔎 Step 3: Searching vector database "
            f"(top {candidate_count} candidates → rerank to {effective_top_k}"
            f"; broad_query={broad_query})..."
        )
        
        try:
            # ================================================================
            # 🔧 CONFIGURABLE: Supabase Vector Search Query
            # Adjust this query based on your Supabase table structure
            # ================================================================
            
            # Example using Supabase's vector similarity search
            # Note: Adjust column names based on your schema
            response = self.supabase_client.rpc(
                'match_documents',  # Your Supabase function name
                {
                    'query_embedding': query_embedding,
                    'match_count': candidate_count,
                    'similarity_threshold': SIMILARITY_THRESHOLD
                }
            ).execute()
            
            results = response.data if response.data else []
            
            # Alternative: Direct table query if you have a different setup
            # response = self.supabase_client.table(EMBEDDINGS_TABLE_NAME) \
            #     .select("*") \
            #     .limit(TOP_K_RESULTS) \
            #     .execute()
            
            print(f"   ✓ Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            print(f"   ⚠️  Vector search failed: {e}")
            print(f"   ℹ️  Make sure you have created the 'match_documents' function in Supabase")
            return []

    def _tokenize_terms(self, text: str) -> set:
        """Tokenize text for light lexical overlap scoring."""
        return {
            token
            for token in re.findall(r"[A-Za-zÀ-ÿ0-9']+", (text or '').lower())
            if len(token) > 2
        }

    def _extract_used_citation_counts(self, answer_text: str) -> Dict[str, int]:
        """Extract [S#] citation tags from answer text and count their usage."""
        citation_counts: Dict[str, int] = {}
        for match in re.findall(r"\[\s*S(\d+)\s*\]", answer_text or '', flags=re.IGNORECASE):
            tag = f"S{int(match)}"
            citation_counts[tag] = citation_counts.get(tag, 0) + 1
        return citation_counts

    def _rerank_and_filter_chunks(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Filter low-information chunks and rerank candidates before top-k selection."""
        if not retrieved_chunks:
            return []

        effective_top_k = max(3, min(top_k or TOP_K_RESULTS, 12))

        query_terms = self._tokenize_terms(user_query)
        scored_chunks = []
        dropped_for_quality = 0

        for chunk in retrieved_chunks:
            text = (chunk.get('content', chunk.get('text', '')) or '').strip()
            words = re.findall(r"[A-Za-zÀ-ÿ0-9']+", text)
            word_count = len(words)
            char_count = len(text)

            content_terms = self._tokenize_terms(text)
            overlap = len(query_terms & content_terms)

            # Hard filter for tiny, contextless chunks unless they strongly match the query.
            if (char_count < MIN_RETRIEVED_CHUNK_CHARS or word_count < MIN_RETRIEVED_CHUNK_WORDS) and overlap < MIN_RETRIEVED_QUERY_OVERLAP:
                dropped_for_quality += 1
                continue

            similarity = float(chunk.get('similarity') or 0.0)
            overlap_score = min(overlap / 4.0, 1.0)
            length_score = min(char_count / 700.0, 1.0)

            # Blend vector similarity with lexical and information-density signals.
            rerank_score = (0.55 * similarity) + (0.30 * overlap_score) + (0.15 * length_score)

            enriched = dict(chunk)
            enriched['rerank_score'] = rerank_score
            enriched['query_overlap'] = overlap
            enriched['content_char_count'] = char_count
            scored_chunks.append(enriched)

        if not scored_chunks:
            # Fail-safe: if filters were too strict, use original results.
            return retrieved_chunks[:effective_top_k]

        scored_chunks.sort(
            key=lambda item: (
                item.get('rerank_score', 0.0),
                item.get('similarity', 0.0),
                item.get('content_char_count', 0),
            ),
            reverse=True,
        )

        self._log_debug(
            f"[Step 3] 🧪 Retrieval rerank: {len(retrieved_chunks)} candidates, "
            f"dropped {dropped_for_quality} low-information chunks, kept {len(scored_chunks)}"
        )

        return scored_chunks[:effective_top_k]
    
    # ========================================================================
    # STEP 4: LLM PROMPT PREPARATION
    # ========================================================================
    
    def _is_process_question(self, user_query: str) -> bool:
        """Detect whether a query is procedural and should keep step-by-step guidance."""
        process_keywords = {
            'how', 'steps', 'apply', 'process', 'cara', 'bagaimana', 'langkah',
            'proses', 'mohon', 'daftar', 'permohonan', 'report', 'complaint',
            'proof', 'evidence', 'summarize', 'what happens', 'file', 'what can i do',
            'what should i do', 'how do i'
        }
        lowered_query = (user_query or '').lower()
        return any(keyword in lowered_query for keyword in process_keywords)

    def _is_affirmative_confirmation(self, user_query: str) -> bool:
        """Return True when user sends a short affirmative reply (e.g., 'yes')."""
        lowered = (user_query or '').strip().lower()
        if not lowered:
            return False

        confirmation_patterns = [
            r'^(yes|yeah|yep|ya|sure|ok|okay|please|go ahead|continue|proceed|tell me|show me)\b',
            r'^(boleh|ya|ok|teruskan|sila|lanjut|iya|baik|boleh terus)\b',
            r'^(oo|sige|pwede|tuloy|ige)\b',
            r'^(c[oó]|v[aâ]ng|d[uư]ợc|ok)\b',
        ]
        return any(re.search(pattern, lowered) for pattern in confirmation_patterns)

    def _detect_contextual_followup_confirmation(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> Optional[Dict[str, str]]:
        """Rewrite bare confirmations into context-aware retrieval queries.
        
        Returns the EXACT follow-up question that was asked, along with type classification.
        """
        if not conversation_history or not self._is_affirmative_confirmation(user_query):
            return None

        last_assistant_index = -1
        for idx in range(len(conversation_history) - 1, -1, -1):
            if (conversation_history[idx].get('role') or '').lower() == 'assistant':
                last_assistant_index = idx
                break

        if last_assistant_index < 0:
            return None

        last_assistant_text = (conversation_history[last_assistant_index].get('text') or '').strip()
        if not last_assistant_text or '?' not in last_assistant_text:
            return None

        previous_user_text = ''
        for idx in range(last_assistant_index - 1, -1, -1):
            if (conversation_history[idx].get('role') or '').lower() == 'user':
                candidate = (conversation_history[idx].get('text') or '').strip()
                if candidate:
                    previous_user_text = candidate
                    break

        if not previous_user_text:
            return None

        # EXTRACT THE EXACT FOLLOW-UP QUESTION (last sentence ending with ?)
        exact_followup_question = ''
        sentences = re.split(r'(?<=[.!?])\s+', last_assistant_text)
        for sent in reversed(sentences):
            sent_stripped = sent.strip()
            if sent_stripped.endswith('?'):
                exact_followup_question = sent_stripped
                break
        
        if not exact_followup_question:
            exact_followup_question = last_assistant_text if last_assistant_text.endswith('?') else f"{last_assistant_text}?"

        # Classify the follow-up type
        assistant_lower = last_assistant_text.lower()
        followup_type = 'fact_details'
        if any(marker in assistant_lower for marker in ['next step', 'step-by-step', 'langkah seterusnya', 'walk you through', 'detailed']):
            followup_type = 'process_steps'
        elif any(marker in assistant_lower for marker in ['checklist', 'required document', 'requirements', 'dokumen', 'syarat', 'eligibility']):
            followup_type = 'eligibility_requirements'
        elif any(marker in assistant_lower for marker in ['safe option', 'if this gets worse', 'retaliation', 'risk', 'amaran', 'danger', 'threat']):
            followup_type = 'warning_or_risk'

        # Build contextual query that includes both original question AND what the follow-up asked for
        if followup_type == 'process_steps':
            contextual_query = f"{previous_user_text}. Provide detailed step-by-step instructions in order."
        elif followup_type == 'eligibility_requirements':
            contextual_query = f"{previous_user_text}. Provide complete eligibility criteria and necessary documents/checklist."
        elif followup_type == 'warning_or_risk':
            contextual_query = f"{previous_user_text}. Provide safer alternatives, warnings, and preventive/protective measures."
        else:
            contextual_query = f"{previous_user_text}. Provide more details, examples, and relevant info."

        return {
            'followup_type': followup_type,
            'contextual_query': contextual_query,
            'exact_followup_question': exact_followup_question,  # NEW: The exact question asked
            'previous_user_question': previous_user_text,  # NEW: Original user question
        }

    def _classify_followup_type(self, user_query: str, answer_text: str, allow_step_format: bool) -> str:
        """Classify answer style to choose a tailored follow-up question."""
        lowered_query = (user_query or '').lower()
        lowered_answer = (answer_text or '').lower()

        if allow_step_format or self._is_process_question(user_query):
            return 'process_steps'

        if any(token in lowered_query for token in ['eligible', 'eligibility', 'requirement', 'requirements', 'document', 'dokumen', 'syarat', 'layak']):
            return 'eligibility_requirements'

        if any(token in lowered_query for token in ['danger', 'threat', 'retaliation', 'risk', 'abuse', 'takut', 'bahaya']) or any(
            token in lowered_answer for token in ['urgent', 'immediately', 'report', 'hotline', 'danger', 'risk']
        ):
            return 'warning_or_risk'

        return 'fact_details'

    def _get_followup_question(self, language_code: str, followup_type: str) -> str:
        """Return language-aware follow-up question using adaptive or fixed style."""
        fixed_templates = {
            'en': 'Would you like help with the next step?',
            'ms': 'Adakah anda memerlukan bantuan untuk langkah seterusnya?',
            'id': 'Apakah anda memerlukan bantuan untuk langkah berikutnya?',
            'tl': 'Kailangan mo ba ng tulong para sa susunod na hakbang?',
            'jv': 'Apa kowe perlu bantuan kanggo langkah sabanjure?',
            'th': 'คุณต้องการความช่วยเหลือสำหรับขั้นตอนถัดไปหรือไม่?',
            'vi': 'Bạn có cần giúp đỡ cho bước tiếp theo không?',
            'ilo': 'Kayat mo kadi ti tulong para iti sumaruno a langkah?',
        }

        adaptive_templates = {
            'fact_details': {
                'en': 'Would you like more details, related policy points, or practical examples for this?',
                'ms': 'Adakah anda mahu butiran lanjut, poin dasar berkaitan, atau contoh praktikal untuk perkara ini?',
                'id': 'Apakah Anda ingin detail tambahan, poin kebijakan terkait, atau contoh praktis untuk ini?',
                'tl': 'Gusto mo ba ng mas detalyeng paliwanag, kaugnay na patakaran, o praktikal na halimbawa tungkol dito?',
                'th': 'คุณต้องการรายละเอียดเพิ่มเติม ประเด็นนโยบายที่เกี่ยวข้อง หรือ ตัวอย่างที่ใช้ได้จริงไหม?',
                'vi': 'Bạn có muốn thêm chi tiết, các điểm chính sách liên quan, hoặc ví dụ thực tế cho nội dung này không?',
                'ilo': 'Kayat mo kadi ti ad-adu a detalye, mayannurot a patakaran, wenno praktikal a ehemplo maipapan iti daytoy?',
            },
            'process_steps': {
                'en': 'Would you like me to walk you through the detailed next steps one by one?',
                'ms': 'Adakah anda mahu saya bimbing langkah seterusnya satu demi satu dengan lebih terperinci?',
                'id': 'Apakah Anda ingin saya pandu langkah berikutnya satu per satu secara lebih rinci?',
                'tl': 'Gusto mo ba na gabayan kita sa susunod na mga hakbang nang paisa-isa?',
                'th': 'ต้องการให้ฉันพาไล่ทีละขั้นตอนแบบละเอียดไหม?',
                'vi': 'Bạn có muốn tôi hướng dẫn chi tiết từng bước tiếp theo không?',
                'ilo': 'Kayat mo kadi nga iturong ka iti sumaruno a detalyado a langkah, maysa-maysa?',
            },
            'eligibility_requirements': {
                'en': 'Would you like a full eligibility and document checklist tailored to your case?',
                'ms': 'Adakah anda mahu senarai semak penuh kelayakan dan dokumen yang disesuaikan dengan kes anda?',
                'id': 'Apakah Anda ingin daftar kelayakan dan checklist dokumen lengkap yang disesuaikan dengan kasus Anda?',
                'tl': 'Gusto mo ba ng kumpletong checklist ng kwalipikasyon at dokumento na akma sa sitwasyon mo?',
                'th': 'ต้องการเช็กลิสต์คุณสมบัติและเอกสารแบบครบถ้วนที่ตรงกับกรณีของคุณไหม?',
                'vi': 'Bạn có muốn danh sách điều kiện và giấy tờ đầy đủ, phù hợp với trường hợp của bạn không?',
                'ilo': 'Kayat mo kadi ti kumpleto a checklist ti kwalipikasion ken dokumento a maiyannatop iti kasasaad mo?',
            },
            'warning_or_risk': {
                'en': 'Would you like safer options and what to do if this situation gets worse?',
                'ms': 'Adakah anda mahu pilihan yang lebih selamat dan langkah jika keadaan ini menjadi lebih buruk?',
                'id': 'Apakah Anda ingin opsi yang lebih aman dan langkah jika situasi ini memburuk?',
                'tl': 'Gusto mo ba ng mas ligtas na mga opsyon at mga hakbang kung lumala pa ang sitwasyon?',
                'th': 'ต้องการตัวเลือกที่ปลอดภัยกว่าและสิ่งที่ควรทำหากสถานการณ์แย่ลงไหม?',
                'vi': 'Bạn có muốn các lựa chọn an toàn hơn và cần làm gì nếu tình huống xấu đi không?',
                'ilo': 'Kayat mo kadi dagiti natalged nga opsion ken aramiden no lumala ti kasasaad?',
            },
        }

        lang_code = language_code if language_code in fixed_templates else 'en'
        if FOLLOW_UP_STYLE == 'fixed':
            return fixed_templates.get(lang_code, fixed_templates['en'])
        if FOLLOW_UP_STYLE == 'adaptive':
            templates = adaptive_templates.get(followup_type, adaptive_templates['fact_details'])
            return templates.get(lang_code, templates['en'])
        return ''

    def _is_summary_request(self, user_query: str) -> bool:
        """Detect explicit user intent to receive a summary output."""
        lowered_query = (user_query or '').lower().strip()
        if not lowered_query:
            return False

        normalized_query = unicodedata.normalize('NFKD', lowered_query)
        normalized_query = ''.join(ch for ch in normalized_query if not unicodedata.combining(ch))
        normalized_query = re.sub(r'\s+', ' ', normalized_query).strip()

        summary_keywords = {
            'summarize', 'summary', 'tl;dr', 'tldr',
            'ringkaskan', 'ringkasan', 'rumuskan', 'rumusan',
            'buat ringkas', 'pendekkan',
            'ringkas', 'rangkum', 'rangkuman', 'simpulkan', 'kesimpulan',
            'buod', 'ibuod', 'lagumin',
            'tom tat',
            'sruob',
        }
        if any(keyword in normalized_query for keyword in summary_keywords):
            return True

        # Support short imperative phrasing such as "in short" / "short version".
        return bool(re.search(r'\b(in short|short version|briefly|in brief)\b', normalized_query))

    def classify_query_intent(self, query: str) -> str:
        """
        Classify whether the query needs RAG retrieval or is a general/conversational message.

        Returns:
            'general'        -- greetings, thanks, chit-chat: skip vector search
            'task_or_policy' -- procedure / eligibility / document questions: full RAG
        """
        stripped = (query or '').strip()
        word_count = len(stripped.split())
        lowered = stripped.lower()

        # Fast-path: known greeting / filler patterns
        GENERAL_PATTERNS = [
            r'^(hi|hello|hey|helo|hai|howdy|yo)\b',
            r'^(thank(s| you)|terima kasih|thanks?)\b',
            r'^(ok|okay|alright|got it|understood|i see|i understand)\b',
            r'^(bye|goodbye|selamat tinggal|sampai jumpa)\b',
            r'^(good morning|good afternoon|good evening|selamat pagi|selamat tengahari|selamat petang)\b',
            r'^(yes|no|yep|nope|yeah|ya|tidak)\s*$',
            r'^(who are you|what are you|what can you do|are you a bot|are you human)\b',
            r'^(test|testing)\s*$',
        ]
        for pattern in GENERAL_PATTERNS:
            if re.search(pattern, lowered):
                return 'general'

        # Task / policy indicator keywords — any match forces full RAG
        TASK_INDICATORS = {
            'how', 'what', 'when', 'where', 'who', 'which', 'why',
            'apply', 'register', 'submit', 'pay', 'claim', 'check',
            'eligible', 'eligibility', 'document', 'requirement', 'deadline',
            'form', 'fee', 'cost', 'amount', 'income', 'salary', 'age',
            'cara', 'bagaimana', 'apa', 'bila', 'di', 'mana', 'siapa',
            'mohon', 'daftar', 'bayar', 'semak', 'layak', 'dokumen', 'borang',
            'process', 'procedure', 'step', 'langkah', 'proses',
            'benefit', 'allowance', 'assistance', 'support', 'scheme', 'grant', 'aid',
            'bantuan', 'elaun', 'skim', 'faedah', 'subsidi',
            'policy', 'rule', 'regulation', 'law', 'act',
            'dasar', 'peraturan', 'undang', 'syarat',
            'hospital', 'clinic', 'school', 'university', 'college',
            'tax', 'cukai', 'permit', 'license', 'lesen',
        }
        query_tokens = set(re.findall(r'\b\w+\b', lowered))
        if query_tokens & TASK_INDICATORS:
            return 'task_or_policy'

        # Short message with no task indicator → treat as general
        if word_count <= 6:
            return 'general'

        return 'task_or_policy'

    def _detect_step_confirmation(self, user_query: str, conversation_history: Optional[List[Dict[str, str]]]) -> bool:
        """
        Return True when the user is confirming a previous step-gate offer.
        Checks if the last assistant message offered detailed steps AND the current
        user message is an affirmative.
        """
        if not conversation_history:
            return False

        GATE_MARKER = '[STEP_GATE]'
        # Last assistant turn
        last_assistant = None
        for turn in reversed(conversation_history):
            if (turn.get('role') or '').lower() == 'assistant':
                last_assistant = turn.get('text', '')
                break

        if not last_assistant or GATE_MARKER not in last_assistant:
            return False

        CONFIRM_PATTERNS = [
            r'\b(yes|yeah|yep|ya|sure|ok|okay|please|continue|go on|proceed|tell me|show me|i want|yes please|go ahead)\b',
            r'\b(bolton|boleh|teruskan|lanjutkan|mahu|nak|sila|cerita|bagitahu)\b',
        ]
        lowered = (user_query or '').lower()
        return any(re.search(p, lowered) for p in CONFIRM_PATTERNS)

    def _detect_labour_office_followup_confirmation(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> bool:
        """
        Return True when the user confirms a previous assistant offer to find
        labour office contact/location details.
        """
        if not conversation_history:
            return False

        # Find latest assistant turn.
        last_assistant = ''
        for turn in reversed(conversation_history):
            if (turn.get('role') or '').lower() == 'assistant':
                last_assistant = (turn.get('text') or '').lower()
                break

        if not last_assistant:
            return False

        office_question_markers = [
            'labour office', 'labor office', 'nearest labour office',
            'closest labour office', 'office in your city',
            'contact details for the office', 'find the closest office',
            'pejabat buruh', 'jabatan tenaga kerja',
        ]
        asked_about_office = any(marker in last_assistant for marker in office_question_markers)
        if not asked_about_office:
            return False

        confirmation_patterns = [
            r'^(yes|yeah|yep|ya|sure|ok|okay|please|go ahead|continue|proceed)\b',
            r'^(boleh|ya|ok|teruskan|sila|lanjut|iya)\b',
        ]
        lowered_query = (user_query or '').strip().lower()
        return any(re.search(pattern, lowered_query) for pattern in confirmation_patterns)

    def _should_lookup_labour_office(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> bool:
        """Determine whether this turn should fetch labour office contacts."""
        lowered = (user_query or '').lower()
        direct_lookup_markers = [
            'labour office', 'labor office', 'jabatan tenaga kerja', 'pejabat buruh',
            'closest office', 'nearest office', 'office contact', 'contact details',
            'alamat pejabat', 'nombor pejabat', 'find office', 'where is the office',
        ]
        if any(marker in lowered for marker in direct_lookup_markers):
            return True

        return self._detect_labour_office_followup_confirmation(user_query, conversation_history)

    def _extract_location_hint(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> Optional[str]:
        """
        Extract a likely city/region from current query and recent user turns.
        """
        text_pool = [user_query or '']
        for turn in (conversation_history or [])[-6:]:
            if (turn.get('role') or '').lower() == 'user':
                text_pool.append(turn.get('text') or '')

        merged = ' '.join(text_pool).lower()

        location_aliases = {
            'kuala lumpur': 'kuala lumpur',
            'kl': 'kuala lumpur',
            'selangor': 'selangor',
            'shah alam': 'shah alam',
            'johor': 'johor',
            'johor bahru': 'johor bahru',
            'penang': 'penang',
            'pulau pinang': 'penang',
            'perak': 'perak',
            'ipoh': 'ipoh',
            'kedah': 'kedah',
            'melaka': 'melaka',
            'negeri sembilan': 'negeri sembilan',
            'pahang': 'pahang',
            'terengganu': 'terengganu',
            'kelantan': 'kelantan',
            'perlis': 'perlis',
            'sabah': 'sabah',
            'sarawak': 'sarawak',
            'putrajaya': 'putrajaya',
        }

        for alias, canonical in location_aliases.items():
            if re.search(rf'\b{re.escape(alias)}\b', merged):
                return canonical

        return None

    def lookup_labour_offices(self, location_hint: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Lookup labour offices from Supabase table.

        Strategy:
        - If location is known, search city/state/region fields with case-insensitive match.
        - If no location is found, return national offices first.
        """
        try:
            table = self.supabase_client.table(LABOUR_OFFICES_TABLE_NAME)

            if location_hint:
                # 1) Exact-ish location match across common location columns.
                location = location_hint.strip()
                query = (
                    table
                    .select('*')
                    .eq('is_active', True)
                    .or_(
                        f"city.ilike.%{location}%,state_region.ilike.%{location}%,"
                        f"district.ilike.%{location}%,country_code.ilike.%{location}%"
                    )
                    .limit(limit)
                )
                response = query.execute()
                data = response.data or []
                if data:
                    return data

            # 2) Fallback: national helpline/office entries.
            response = (
                table
                .select('*')
                .eq('is_active', True)
                .eq('is_national', True)
                .limit(limit)
                .execute()
            )
            data = response.data or []
            if data:
                return data

            # 3) Last fallback: any active offices.
            response = (
                table
                .select('*')
                .eq('is_active', True)
                .limit(limit)
                .execute()
            )
            return response.data or []
        except Exception as error:
            self._log_debug(f"[Office Lookup] ⚠️ Failed lookup from '{LABOUR_OFFICES_TABLE_NAME}': {error}")
            return []

    def _format_office_chunk(self, office: Dict[str, Any]) -> Dict[str, Any]:
        """Convert labour office row into RAG chunk format."""
        name = office.get('office_name', 'Labour Office')
        city = office.get('city', '')
        state_region = office.get('state_region', '')
        address = office.get('address', 'Address not available')
        phone = office.get('phone', 'Phone not available')
        email = office.get('email', 'Email not available')
        website = office.get('website', '')
        open_hours = office.get('open_hours', 'Opening hours not available')
        country_code = office.get('country_code', 'MY')

        location_label = ', '.join([part for part in [city, state_region] if part]).strip(', ')
        if not location_label:
            location_label = country_code

        content = (
            f"Official labour office contact for {location_label}: {name}. "
            f"Address: {address}. Phone: {phone}. Email: {email}. "
            f"Opening hours: {open_hours}."
        )

        return {
            'title': f"Labour Office Directory: {name}",
            'content': content,
            'source_url': website,
            'url': website,
            'language': 'en',
            'similarity': 1.0,
        }

    def _prepare_general_prompt(
        self,
        user_query: str,
        target_language: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: str = "",
    ) -> str:
        """Build a lightweight conversational prompt when no RAG retrieval is needed."""
        history_lines = []
        for turn in (conversation_history or [])[-4:]:
            role = (turn.get('role') or 'user').strip().lower()
            text = re.sub(r'\s+', ' ', (turn.get('text') or '')).strip()
            if role not in {'user', 'assistant'} or not text:
                continue
            speaker = 'User' if role == 'user' else 'Assistant'
            history_lines.append(f"- {speaker}: {text[:200]}")
        conversation_context = '\n'.join(history_lines) if history_lines else 'No prior conversation.'
        summary_text = (conversation_summary or '').strip() or 'No earlier summary.'

        if ENABLE_LANGUAGE_AUTO_DETECTION:
            return f"""You are CivicGuide, a warm and helpful public-service assistant.

User Message (auto-detected): \"{user_query}\"

Earlier Conversation Summary:
{summary_text}

Recent Conversation:
{conversation_context}

Instructions:
1. Detect the language of the user message automatically.
2. Respond in the SAME language as the user message.
3. For greetings or thanks, reply warmly in 1-2 sentences.
4. If the user seems to have a real question or need, provide one short practical guidance step first, then ask one clarifying question.
5. Do not invent government policies or procedures.
6. Keep your reply short (2-4 sentences maximum) and action-oriented.
7. Do not start with stock phrases like \"Here's a helpful response\" or \"Certainly!\".
8. If unsure of the detected language, respond in the language that seems most natural for the user's input.
"""
        else:
            return f"""You are CivicGuide, a warm and helpful public-service assistant.

User message (in {target_language}): \"{user_query}\"

Earlier Conversation Summary:
{summary_text}

Recent Conversation:
{conversation_context}

Instructions:
1. Respond in natural, friendly language in {target_language}.
2. For greetings or thanks, reply warmly in 1-2 sentences.
3. If the user seems to have a real question or need, provide one short practical guidance step first, then ask one clarifying question.
4. Do not invent government policies or procedures.
5. Keep your reply short (2-4 sentences maximum) and action-oriented.
6. Do not start with stock phrases like \"Here's a helpful response\" or \"Certainly!\".
"""

    def _build_multilingual_kshot_examples(self, user_query: str = "") -> str:
        """
        Build k-shot examples for low-resource languages to help LLM
        detect language and respond in the same language/dialect.
        
        These examples teach the LLM how to:
        1. Auto-detect the language from user input
        2. Respond in the same language
        3. Handle dialects and low-resource languages
        """
        if not ENABLE_LANGUAGE_AUTO_DETECTION or not LOW_RESOURCE_LANGUAGE_EXAMPLES:
            return ""
        
        # Centralized k-shot examples are maintained in pipeline/kshot_library.py
        kshot_library = KSHOT_LIBRARY
        
        # Build query-aware k-shot string from configured languages
        query_tokens = {
            token
            for token in re.findall(r"[A-Za-zÀ-ÿ0-9']+", (user_query or '').lower())
            if len(token) > 2
        }

        kshot_parts = []
        running_chars = 0
        for lang_code in LOW_RESOURCE_LANGUAGE_EXAMPLES:
            lang_code = lang_code.strip().lower()
            if lang_code in kshot_library:
                examples = kshot_library[lang_code]['examples']

                # Rank examples by lexical overlap with the current query.
                scored_examples = []
                for idx, example in enumerate(examples):
                    ex_tokens = {
                        token
                        for token in re.findall(r"[A-Za-zÀ-ÿ0-9']+", example['user'].lower())
                        if len(token) > 2
                    }
                    overlap = len(query_tokens & ex_tokens) if query_tokens else 0
                    scored_examples.append((overlap, idx, example))

                # Highest overlap first, then keep deterministic ordering.
                scored_examples.sort(key=lambda item: (-item[0], item[1]))
                selected_examples = [
                    item[2] for item in scored_examples[:max(1, KSHOT_MAX_EXAMPLES_PER_LANGUAGE)]
                ]

                language_header = f"Language: {lang_code.upper()}"
                if running_chars + len(language_header) > KSHOT_MAX_TOTAL_CHARS:
                    break
                kshot_parts.append(language_header)
                running_chars += len(language_header)

                for i, example in enumerate(selected_examples, 1):
                    lines = [
                        f"  Example {i}:",
                        f"    Input: \"{example['user']}\"",
                    ]
                    if 'detected_lang' in example:
                        lines.append(f"    Detected Language: {example['detected_lang']}")
                    lines.append(f"    Response: \"{example['assistant']}\"")

                    block = "\n".join(lines)
                    projected = running_chars + len(block)
                    if projected > KSHOT_MAX_TOTAL_CHARS:
                        break

                    kshot_parts.extend(lines)
                    running_chars = projected
        
        if kshot_parts:
            return "\n\nLanguage-Specific K-Shot Examples (learn to detect and respond in user's language):\n" + "\n".join(kshot_parts)
        return ""

    def prepare_llm_prompt(
        self, 
        user_query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        target_language: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: str = "",
        use_step_gate: bool = False,
        compressed_chunks: Optional[List[Dict[str, Any]]] = None,
        answer_style_override: Optional[str] = None,
        recursive_context_summary: str = "",
        compact_citation_mode: bool = False,
        is_affirmative_followup: bool = False,
        original_user_query: Optional[str] = None,
        exact_followup_question: Optional[str] = None,  # NEW: The exact question to answer
    ) -> str:
        """
        Prepare the prompt for the LLM with retrieved context
        
        Args:
            user_query: User query (or synthesized contextual query if follow-up confirmation)
            retrieved_chunks: Relevant document chunks from vector search
            target_language: Language to translate the answer to
            conversation_history: Recent conversation turns for follow-up context
            conversation_summary: Rolling summary of older conversation context
            is_affirmative_followup: True if user said "yes" and query was rewritten from context
            original_user_query: Original "yes"/confirmation if is_affirmative_followup=True
            exact_followup_question: The EXACT follow-up question user is answering "yes" to
            
        Returns:
            Formatted prompt string
        """
        print(f"📝 Step 4: Preparing LLM prompt...")
        effective_answer_style = (answer_style_override or ANSWER_STYLE).lower()
        compressed_chunks = compressed_chunks or self._compress_retrieved_chunks(user_query, retrieved_chunks)
        
        # Format retrieved documents with citation tags
        context_docs = []
        for idx, chunk in enumerate(compressed_chunks, 1):
            # Adjust field names based on your Supabase schema
            text = chunk.get('summary', chunk.get('content', chunk.get('text', '')))
            source = chunk.get('source_url', chunk.get('url', 'N/A'))
            title = chunk.get('title', 'Document')
            lang = chunk.get('language', 'unknown')
            
            context_docs.append(
                f"{idx}. [S{idx}] [{title}] ({lang})\n"
                f"   Content: {text}\n"
                f"   Source: {source}\n"
            )
        
        context_text = "\n".join(context_docs)
        recursive_summary_text = (recursive_context_summary or '').strip()
        recursive_summary_section = (
            f"\nRecursive Context Summary (from recursive condensation):\n{recursive_summary_text}\n"
            if recursive_summary_text
            else ""
        )
        history_lines = []
        for turn in (conversation_history or [])[-6:]:
            role = (turn.get('role') or 'user').strip().lower()
            text = re.sub(r'\s+', ' ', (turn.get('text') or '')).strip()
            if role not in {'user', 'assistant'} or not text:
                continue
            speaker = 'User' if role == 'user' else 'Assistant'
            history_lines.append(f"- {speaker}: {text[:300]}")

        conversation_context = "\n".join(history_lines) if history_lines else "No prior conversation."
        summary_text = (conversation_summary or "").strip() or "No earlier summary."
        
        # Decide if the user is asking a process/how-to question.
        is_process_question = self._is_process_question(user_query)
        lowered_query = (user_query or '').lower()
        labor_keywords = {
            'overtime', 'salary', 'wage', 'unpaid', 'employer', 'boss', 'passport',
            'work permit', 'labour', 'labor', 'complaint', 'report', 'evidence', 'proof'
        }
        labor_mode = any(keyword in lowered_query for keyword in labor_keywords)

        if effective_answer_style == 'bullet':
            response_format = (
                f"Write {NUM_BULLET_POINTS} concise bullet points only. "
                "Each point must be practical and easy to follow."
            )
        elif use_step_gate:
            # Step-gating: give a concise overview first, then offer detailed steps
            response_format = (
                "Give a brief overview in 2-3 sentences answering the core question. "
                "Then end with exactly this sentence: "
                "\"Would you like me to walk you through the detailed step-by-step process? [STEP_GATE]\""
            )
        elif is_process_question:
            response_format = (
                "Use a short introduction in plain language, then provide 3-5 steps using numbered points or bullet points. "
                "End with one gentle follow-up question that helps the user continue."
            )
        else:
            if RESPONSE_DEPTH == 'concise':
                response_format = (
                    "Default to natural paragraph style, not bullet-heavy formatting. "
                    "Give a direct answer first in 1-2 sentences, then one concrete next action. "
                    "Keep total length to 3-4 sentences."
                )
            elif RESPONSE_DEPTH == 'detailed':
                response_format = (
                    "Default to natural paragraph style, not bullet-heavy formatting. "
                    "Structure your answer into short paragraphs covering: "
                    "(1) direct answer, "
                    "(2) why this applies to the user using source-backed facts, "
                    "(3) what the user should do now, "
                    "(4) what documents or checks to prepare if relevant."
                )
            else:
                response_format = (
                    "Default to natural paragraph style, not bullet-heavy formatting. "
                    "Use this structure: (1) direct answer in 1-2 sentences, "
                    "(2) short explanation of why this applies using source-backed details, "
                    "(3) what the user should do next."
                )

        list_format_guardrail = ""
        if effective_answer_style == 'narrative' and not is_process_question:
            list_format_guardrail = (
                "Do NOT use bullet points, numbered lists, markdown list markers, or list-style formatting. "
                "Write in short, plain paragraphs suitable for low-literacy users."
            )

        labor_mode_guardrail = ""
        if labor_mode:
            labor_mode_guardrail = """
Labour-rights guidance requirements:
- Use practical worker-safety guidance in simple language.
- Explicitly mention worker rights when fear, threats, or retaliation is discussed.
- If overtime is discussed, explain that work beyond normal hours is overtime and should be paid at a higher rate.
- If overtime is discussed, also advise the user to keep records of working hours.
- If unpaid salary is discussed, include: collect proof, contact Labour Department, submit a complaint.
- If unpaid salary is discussed, explicitly state employers must pay wages on time.
- If passport retention is discussed, state workers should keep personal documents and can seek help from authorities/support centers.
- If retaliation fear is discussed, reassure the user and mention workers should not be punished for reporting violations.
- If asked about evidence, include all of these examples explicitly: work schedule, employer messages, salary records, and photos or notes.
- If asked what happens after complaint, mention investigation, request for evidence, and possible order to pay owed wages.
- When location/help is asked, mention Labour Department and also mention migrant worker support organizations or help centers.
- If the question asks "what can I do" or asks for a summary of next actions, give a clear checklist with numbered steps.
- For a summary of next actions, include this checklist content: record hours/unpaid wages, keep proof/messages, contact Labour Department, submit complaint, seek migrant worker support organization/NGO help.
"""

        few_shot_examples = ""
        if is_process_question:
            few_shot_examples = """
    Few-shot examples (style guide only, do not copy facts):
    Example A (procedural):
    User: "How do I report unpaid salary?"
    Assistant: "You can report this in 4 steps. [S1]\n1) Collect proof of unpaid wages, such as payslips or messages. [S1]\n2) Write down dates and amounts owed. [S2]\n3) Contact the Labour Department and submit a complaint form. [S2]\n4) Keep your case reference number and follow up. [S2]\nWould you like a checklist you can bring to the office?"

    Example B (procedural):
    User: "What proof should I prepare?"
    Assistant: "Bring any evidence you have: work schedule, employer messages, salary records, and notes of hours worked. [S1] If one item is missing, submit what you have and explain clearly. [S2] Would you like me to format this as a one-page checklist?"
    """
        else:
            few_shot_examples = """
    Few-shot examples (style guide only, do not copy facts):
    Example A (general inquiry):
    User: "I am scared and do not know where to start."
    Assistant: "You can start by writing down what happened and keeping any messages or documents you already have. [S1] After that, contact the nearest Labour Department counter for guidance on your next step. [S2] Would you like help finding the closest office?"

    Example B (general inquiry):
    User: "My boss keeps my passport."
    Assistant: "Your passport is your personal document and you should keep it with you. [S1] Ask for it back politely and seek help from the Labour Department or a migrant support center if needed. [S2] Do you want a simple script you can use when asking for it back?"
    """

        # Build multilingual k-shot examples if auto-detection is enabled
        multilingual_kshot = self._build_multilingual_kshot_examples(user_query=user_query)
        
        # Determine if we're using language auto-detection
        language_instruction = ""
        followup_context_instruction = ""
        if is_affirmative_followup and exact_followup_question:
            followup_context_instruction = f"""
🎯 FOLLOW-UP CONFIRMATION CONTEXT:
The assistant previously asked: "{exact_followup_question}"
The user responded: "{original_user_query}" (confirming YES)

⚠️ CRITICAL INSTRUCTION:
Your response MUST directly and FULLY answer the question above.
Do NOT provide a brief or partial answer.
Do NOT answer a different question.
FOCUS on addressing what was specifically asked: {exact_followup_question}
"""
        elif is_affirmative_followup:
            followup_context_instruction = f"""
Follow-up Confirmation Context:
- The user has confirmed their interest by saying "{original_user_query}" (yes/ok/continue)
- The following query was AUTOMATICALLY SYNTHESIZED from the previous conversation to capture the user's intent:
- Query: "{user_query}"
- Respond directly to this synthesized query using the retrieved context.
- Make sure your answer DIRECTLY ADDRESSES what the user confirmed they wanted (based on the previous assistant's follow-up offer)."""
        
        if ENABLE_LANGUAGE_AUTO_DETECTION:
            language_instruction = "User Query (auto-detected): \"" + user_query + "\""
            language_context = """
Language Detection & Response Requirements:
- Detect the language of the user query automatically (it may be in English, Malay, Tagalog, Thai, Vietnamese, or other ASEAN languages).
- Respond in the SAME language as the detected user input.
- Reference the multilingual examples to guide your language-specific responses.
- If you are uncertain about the language, respond in the language that seems most natural based on the user's input."""
        else:
            language_instruction = "User Query (in " + target_language + "): \"" + user_query + "\""
            language_context = f"6. Respond in natural language in {target_language}."

        prompt = f"""You are CivicGuide, an inclusive public-service assistant.

Mission:
- Help every citizen understand and access government support.
- Reduce information barriers for users with low literacy or limited digital skills.

{language_instruction}
{followup_context_instruction}

Earlier Conversation Summary:
{summary_text}

Recent Conversation:
{conversation_context}

Retrieved Official Context:
{context_text}
{recursive_summary_section}

{few_shot_examples}
{multilingual_kshot}

{labor_mode_guardrail}

Instructions:
1. Use retrieved context as evidence. Do not copy large chunks verbatim.
2. Use plain language at approximately {READING_LEVEL} reading level. Keep sentences short and clear.
3. Be respectful, calm, and practical. Avoid legalistic or bureaucratic tone.
4. If context is incomplete or uncertain, say so clearly and provide the safest next step.
5. Do not invent eligibility rules, deadlines, required documents, or links.
{language_context}
7. Do not include raw source URLs in the answer body because sources are shown separately in the UI.
8. {response_format}
9. {list_format_guardrail}
10. Start with the answer directly. Avoid stock lead-ins such as "Here's a helpful response", "Here's a simplified explanation", "Here's a guide", or similar filler.
11. Avoid sounding like a template. Do not add decorative titles or headings unless they are needed to explain steps clearly.
12. If the user is asking a follow-up question, use the recent conversation to resolve references like "it", "that", or "the deadline" before answering.
{("13. THIS IS AN AFFIRMATIVE FOLLOW-UP: The user previously said \"" + original_user_query + "\" to answer \"" + exact_followup_question + "\". Provide a comprehensive, detailed answer to THAT SPECIFIC QUESTION. Do NOT give a brief answer or answer a different question." if (is_affirmative_followup and exact_followup_question) else "13. " + ("For summary-style outputs, do not add citation tags to every sentence. Add at most one citation tag per bullet (for example [S1]) or provide one short source line at the end." if compact_citation_mode else "When you use information from a source, add an inline citation tag like [S1] or [S2] immediately after the relevant sentence. Use the source numbers provided in the Retrieved Official Context above."))}
14. When answering follow-up confirmations, prioritize ACCURACY and COMPLETENESS over brevity.
15. If Recursive Context Summary is provided, use it to stay concise, but verify details against Retrieved Official Context before finalizing.
16. Think through the answer internally, but output only the final answer without showing hidden reasoning.

Before finalizing, check:
- Can a 12-year-old understand this answer?
- Does this answer include a clear next action?
- Is the answer concise without unnecessary list formatting?

Do not simply repeat document snippets. Synthesize them into one helpful answer.
"""
        
        print(f"   ✓ Prompt prepared ({len(prompt)} characters)")
        return prompt

    def _expand_chunks_by_source_url(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand retrieval set by loading all chunks for each matched source_url."""
        if not retrieved_chunks:
            return []

        source_order: List[str] = []
        source_best_scores: Dict[str, Dict[str, float]] = {}

        for chunk in retrieved_chunks:
            source_url = (chunk.get('source_url') or chunk.get('url') or '').strip()
            if not source_url:
                continue

            if source_url not in source_order:
                source_order.append(source_url)

            rerank_score = float(chunk.get('rerank_score') or 0.0)
            similarity = float(chunk.get('similarity') or 0.0)
            previous = source_best_scores.get(source_url, {'rerank_score': 0.0, 'similarity': 0.0})
            source_best_scores[source_url] = {
                'rerank_score': max(previous.get('rerank_score', 0.0), rerank_score),
                'similarity': max(previous.get('similarity', 0.0), similarity),
            }

        if not source_order:
            return retrieved_chunks

        expanded_chunks: List[Dict[str, Any]] = []
        for source_url in source_order:
            try:
                response = (
                    self.supabase_client
                    .table('embeddings')
                    .select('title,content,source_url,chunk_index,total_chunks,page_number,page_start,page_end,section,subsection,source_type,language')
                    .eq('source_url', source_url)
                    .order('chunk_index', desc=False)
                    .execute()
                )
                source_rows = response.data or []
            except Exception as error:
                self._log_debug(f"[Step 4] ⚠️ Failed expanding source chunks for {source_url}: {error}")
                source_rows = []

            if not source_rows:
                # Fallback to original retrieved chunks from this source.
                fallback_rows = [
                    chunk for chunk in retrieved_chunks
                    if (chunk.get('source_url') or chunk.get('url') or '').strip() == source_url
                ]
                expanded_chunks.extend(fallback_rows)
                continue

            score_meta = source_best_scores.get(source_url, {'rerank_score': 0.0, 'similarity': 0.0})
            for row in source_rows:
                enriched_row = dict(row)
                enriched_row['url'] = source_url
                enriched_row['rerank_score'] = score_meta.get('rerank_score', 0.0)
                enriched_row['similarity'] = score_meta.get('similarity', 0.0)
                expanded_chunks.append(enriched_row)

        self._log_debug(
            f"[Step 4] 📚 Expanded retrieval from {len(retrieved_chunks)} matched chunks to "
            f"{len(expanded_chunks)} chunks across {len(source_order)} source URL(s)"
        )
        return expanded_chunks or retrieved_chunks

    def _should_auto_action_bullet_summary(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        compressed_chunks: List[Dict[str, Any]],
    ) -> bool:
        """Decide whether response should be forced to 3-5 actionable bullets."""
        if not AUTO_ACTION_BULLET_SUMMARY:
            return False

        if not compressed_chunks:
            return False

        total_context_chars = sum(
            len((chunk.get('summary') or chunk.get('content') or chunk.get('text') or '').strip())
            for chunk in compressed_chunks
        )
        chunk_count = len(compressed_chunks)

        source_keys = set()
        for chunk in retrieved_chunks:
            source_key = (chunk.get('source_url') or chunk.get('url') or chunk.get('title') or '').strip().lower()
            if source_key:
                source_keys.add(source_key)
        source_count = len(source_keys)

        long_context = (
            total_context_chars >= AUTO_SUMMARY_LONG_CONTEXT_CHARS
            or chunk_count >= AUTO_SUMMARY_LONG_CHUNK_COUNT
        )
        complex_context = (
            source_count >= AUTO_SUMMARY_COMPLEX_SOURCE_COUNT
            and total_context_chars >= AUTO_SUMMARY_COMPLEX_CONTEXT_CHARS
        )

        if long_context or complex_context:
            self._log_debug(
                "[Step 4] 🧭 Auto-summary mode ON "
                f"(chars={total_context_chars}, chunks={chunk_count}, sources={source_count})"
            )
            return True

        self._log_debug(
            "[Step 4] 🧭 Auto-summary mode OFF "
            f"(chars={total_context_chars}, chunks={chunk_count}, sources={source_count})"
        )
        return False

    def _compress_retrieved_chunks(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Shrink retrieved chunks before prompt construction to fit the LLM context window."""
        compressed_chunks = []
        total_chars = 0
        original_total_chars = 0

        for chunk in retrieved_chunks:
            text = chunk.get('content', chunk.get('text', '')) or ''
            original_total_chars += len(text)

            summary = self._summarize_chunk_text(user_query, text)

            if total_chars >= TOTAL_CONTEXT_MAX_CHARS:
                break

            remaining_chars = TOTAL_CONTEXT_MAX_CHARS - total_chars
            if len(summary) > remaining_chars:
                summary = summary[:remaining_chars].rsplit(' ', 1)[0].strip()
                if summary:
                    summary += '...'

            if not summary:
                continue

            compressed_chunk = dict(chunk)
            compressed_chunk['summary'] = summary
            compressed_chunks.append(compressed_chunk)
            total_chars += len(summary)

        self._log_debug(
            f"[Step 4] 📉 Context compressed from {original_total_chars} to {total_chars} chars across {len(compressed_chunks)} chunks"
        )
        return compressed_chunks

    def _summarize_chunk_text(self, user_query: str, text: str) -> str:
        """Create a compact extractive summary of a retrieved chunk using query relevance."""
        cleaned_text = re.sub(r'\s+', ' ', (text or '')).strip()
        if not cleaned_text:
            return ''

        if len(cleaned_text) <= CHUNK_SUMMARY_MAX_CHARS:
            return cleaned_text

        query_terms = {
            token
            for token in re.findall(r"[A-Za-zÀ-ÿ0-9']+", user_query.lower())
            if len(token) > 2
        }

        nlp = self.nlp_en if all(ord(ch) < 128 for ch in cleaned_text[:200]) else self.nlp_multi
        try:
            doc = nlp(cleaned_text)
            sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
        except Exception:
            # Model lacks a sentencizer/parser — fall back to regex sentence splitting.
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_text) if s.strip()]

        if not sentences:
            return cleaned_text[:CHUNK_SUMMARY_MAX_CHARS].rsplit(' ', 1)[0].strip() + '...'

        scored_sentences = []
        for index, sentence in enumerate(sentences):
            sentence_terms = {
                token
                for token in re.findall(r"[A-Za-zÀ-ÿ0-9']+", sentence.lower())
                if len(token) > 2
            }
            overlap_score = len(query_terms & sentence_terms)
            position_bonus = max(0, 2 - min(index, 2))
            length_bonus = min(len(sentence) / 200, 1)
            score = overlap_score * 3 + position_bonus + length_bonus
            scored_sentences.append((score, index, sentence))

        selected = []
        current_chars = 0
        for _, index, sentence in sorted(scored_sentences, key=lambda item: (-item[0], item[1])):
            sentence_length = len(sentence)
            if selected and len(selected) >= CHUNK_SUMMARY_MAX_SENTENCES:
                break
            if current_chars + sentence_length > CHUNK_SUMMARY_MAX_CHARS and selected:
                continue
            selected.append((index, sentence))
            current_chars += sentence_length + 1
            if current_chars >= CHUNK_SUMMARY_MAX_CHARS:
                break

        if not selected:
            summary = cleaned_text[:CHUNK_SUMMARY_MAX_CHARS]
        else:
            summary = ' '.join(sentence for _, sentence in sorted(selected, key=lambda item: item[0]))

        summary = summary.strip()
        if len(summary) < len(cleaned_text):
            summary += '...'
        return summary
    
    # ========================================================================
    # STEP 5: CALL LLM
    # ========================================================================
    
    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM to generate a response with automatic fallback
        
        Args:
            prompt: Prepared prompt with context
            
        Returns:
            LLM-generated response text
        """
        active_llm_label = self._describe_active_llm()
        self._log_debug(f"[Step 5] 🧠 Active LLM for this request: {active_llm_label}")
        print(f"🧠 Active LLM for this request: {active_llm_label}")

        # Try primary provider
        try:
            return self._call_llm_provider(self.llm_provider, prompt)
        except Exception as e:
            error_msg = f"⚠️  Primary provider '{self.llm_provider}' failed: {str(e)}"
            self._log_debug(f"[Step 5] {error_msg}")
            print(f"   {error_msg}")
            
            # Try fallback providers if enabled
            if ENABLE_FALLBACK:
                for fallback_provider in FALLBACK_ORDER:
                    if fallback_provider == self.llm_provider:
                        continue  # Skip already tried provider
                    
                    try:
                        fallback_msg = f"🔄 Trying fallback provider: {fallback_provider}"
                        self._log_debug(f"[Step 5] {fallback_msg}")
                        print(f"   {fallback_msg}")
                        self._init_llm_provider(fallback_provider)
                        fallback_label = self._describe_active_llm(fallback_provider)
                        self._log_debug(f"[Step 5] 🧠 Switched active LLM: {fallback_label}")
                        print(f"🧠 Switched active LLM: {fallback_label}")
                        return self._call_llm_provider(fallback_provider, prompt)
                    except Exception as fallback_error:
                        fallback_error_msg = f"⚠️  Fallback '{fallback_provider}' also failed: {str(fallback_error)}"
                        self._log_debug(f"[Step 5] {fallback_error_msg}")
                        print(f"   {fallback_error_msg}")
                        continue
            
            # All providers failed
            final_error = f"❌ All LLM providers failed. Last error: {str(e)}"
            self._log_debug(f"[Step 5] {final_error}")
            return "Sorry, I couldn't generate a response at this time. Please check your API keys and internet connection."
    
    def _call_llm_provider(self, provider: str, prompt: str) -> str:
        """Route to specific provider's LLM call"""
        if provider == 'groq':
            return self._call_groq_llm(prompt)
        elif provider == 'hf_inference':
            return self._call_hf_inference_llm(prompt)
        elif provider == 'openai':
            return self._call_openai_llm(prompt)
        elif provider == 'huggingface':
            return self._call_huggingface_llm(prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _build_system_message(self) -> str:
        """
        Build a dynamic system message that supports language auto-detection.
        """
        if ENABLE_LANGUAGE_AUTO_DETECTION:
            return """You are CivicGuide, an inclusive public-service assistant.

Your key responsibility: Detect the user's language automatically and respond in that same language.

Supported languages: English, Malay, Tagalog, Thai, Vietnamese, Indonesian, and other ASEAN languages.

Always:
- Detect the input language from context
- Respond in the same language as the user
- Be warm, helpful, and respectful
- Use simple language suitable for low-literacy users
- Provide practical guidance with clear next steps"""
        else:
            return """You are CivicGuide, an inclusive public-service assistant.

Your key responsibility: Help users understand government policies and access support services.

Always:
- Be warm, helpful, and respectful
- Use simple language suitable for low-literacy users
- Provide practical guidance with clear next steps
- Avoid legalistic or bureaucratic language
- Respond in the user's language"""
    
    def _call_groq_llm(self, prompt: str) -> str:
        """Call Groq API (Ultra-fast inference)"""
        print(f"⚡ Step 5: Calling Groq ({GROQ_MODEL})...")
        self._log_debug(f"[Step 5] ⚡ Using Groq: {GROQ_MODEL}")
        
        try:
            # Check if API key is valid
            if not GROQ_API_KEY or GROQ_API_KEY == 'YOUR_GROQ_API_KEY':
                error = "GROQ_API_KEY not configured. Get your key at: https://console.groq.com"
                self._log_debug(f"[Step 5] ❌ {error}")
                raise ValueError(error)
            
            self._log_debug(f"[Step 5] 📡 Sending request to Groq API...")
            
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_message()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            
            answer = response.choices[0].message.content
            self._log_debug(f"[Step 5] ✓ Groq response: {len(answer)} chars, {response.usage.total_tokens} tokens")
            print(f"   ✓ Groq response generated ({len(answer)} characters)")
            print(f"   ℹ️  Speed: Ultra-fast! Tokens: {response.usage.total_tokens}")
            return answer
            
        except Exception as e:
            error_details = f"Groq API failed: {type(e).__name__}: {str(e)}"
            self._log_debug(f"[Step 5] ❌ {error_details}")
            print(f"   ⚠️  {error_details}")
            
            if "api_key" in str(e).lower() or "401" in str(e):
                self._log_debug(f"[Step 5] 💡 Fix: Check your GROQ_API_KEY in .env file")
            
            raise


    def _call_hf_inference_llm(self, prompt: str) -> str:
        """Call Hugging Face Inference API (Cloud SEA models)"""
        print(f"🤗 Step 5: Calling HF Inference ({HF_INFERENCE_MODEL.split('/')[-1]})...")
        self._log_debug(f"[Step 5] 🤗 Using HF Inference: {HF_INFERENCE_MODEL}")
        
        try:
            # Check if token is valid
            if not HF_TOKEN or HF_TOKEN == 'YOUR_HF_TOKEN':
                error = "HF_TOKEN not configured. Get your token at: https://huggingface.co/settings/tokens"
                self._log_debug(f"[Step 5] ❌ {error}")
                raise ValueError(error)
            
            # Format for chat models
            messages = [
                {
                    "role": "system",
                    "content": self._build_system_message()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            self._log_debug(f"[Step 5] 📡 Sending request to HF Inference API...")
            
            response = None
            last_error = None
            for attempt in range(1, HF_LLM_MAX_RETRIES + 1):
                try:
                    response = self.hf_inference_client.chat_completion(
                        messages=messages,
                        model=HF_INFERENCE_MODEL,
                        max_tokens=MAX_TOKENS,
                        temperature=LLM_TEMPERATURE,
                    )
                    break
                except Exception as error:
                    last_error = error
                    is_retryable = self._is_transient_hf_error(error)
                    if not is_retryable or attempt == HF_LLM_MAX_RETRIES:
                        raise

                    wait_seconds = HF_RETRY_DELAY_SECONDS * attempt
                    self._log_debug(
                        f"[Step 5] ⚠️ HF LLM request failed (attempt {attempt}/{HF_LLM_MAX_RETRIES}): "
                        f"{type(error).__name__}. Retrying in {wait_seconds:.1f}s"
                    )
                    time.sleep(wait_seconds)

            if response is None and last_error:
                raise last_error
            
            answer = response.choices[0].message.content
            self._log_debug(f"[Step 5] ✓ HF Inference response: {len(answer)} chars")
            print(f"   ✓ HF Inference response generated ({len(answer)} characters)")
            print(f"   ℹ️  Using cloud-hosted SEA-LION model")
            return answer
            
        except Exception as e:
            error_details = f"HF Inference API failed: {type(e).__name__}: {str(e)}"
            self._log_debug(f"[Step 5] ❌ {error_details}")
            print(f"   ⚠️  {error_details}")
            
            # Provide helpful error messages
            if "401" in str(e) or "unauthorized" in str(e).lower():
                self._log_debug(f"[Step 5] 💡 Fix: Check your HF_TOKEN in .env file")
            elif "model" in str(e).lower():
                self._log_debug(f"[Step 5] 💡 Fix: Model {HF_INFERENCE_MODEL} might not support Inference API")
            elif "timeout" in str(e).lower():
                self._log_debug(f"[Step 5] 💡 Fix: Check your internet connection")
            
            raise
    
    def _call_openai_llm(self, prompt: str) -> str:
        """Call OpenAI GPT models"""
        print(f"🤖 Step 5: Calling OpenAI-compatible LLM ({OPENAI_MODEL_NAME})...")
        self._log_debug(f"[Step 5] 🤖 Using OpenAI-compatible model: {OPENAI_MODEL_NAME}")
        
        try:
            has_openai_key = OPENAI_API_KEY and OPENAI_API_KEY != 'YOUR_OPENAI_API_KEY'
            has_openrouter_key = OPENROUTER_API_KEY and OPENROUTER_API_KEY != 'YOUR_OPENROUTER_API_KEY'
            if not has_openai_key and not has_openrouter_key:
                error = (
                    "No OpenAI-compatible API key configured. "
                    "Set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file."
                )
                self._log_debug(f"[Step 5] ❌ {error}")
                raise ValueError(error)

            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {
                        "role": "system", 
                        "content": self._build_system_message()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            answer = response.choices[0].message.content
            self._log_debug(f"[Step 5] ✓ OpenAI-compatible response: {len(answer)} chars")
            print(f"   ✓ LLM response generated ({len(answer)} characters)")
            print(f"   ℹ️  Tokens used: {response.usage.total_tokens}")
            return answer
            
        except Exception as e:
            error_details = f"OpenAI-compatible LLM call failed: {type(e).__name__}: {e}"
            self._log_debug(f"[Step 5] ❌ {error_details}")
            print(f"   ⚠️  {error_details}")
            raise
    
    def _call_huggingface_llm(self, prompt: str) -> str:
        """Call Hugging Face Southeast Asian LLM models"""
        print(f"🤖 Step 5: Calling SEA LLM ({SEA_MODEL_CHOICE})...")
        
        try:
            # Format prompt for chat models
            system_msg = self._build_system_message()
            formatted_prompt = f"""<|system|>
{system_msg}
<|user|>
{prompt}
<|assistant|>
"""
            
            # Generate response
            response = self.llm_model(
                formatted_prompt,
                max_new_tokens=MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.llm_model.tokenizer.eos_token_id
            )
            
            # Extract answer (remove prompt)
            full_text = response[0]['generated_text']
            answer = full_text.split('<|assistant|>')[-1].strip()
            
            print(f"   ✓ SEA LLM response generated ({len(answer)} characters)")
            return answer
            
        except Exception as e:
            print(f"   ⚠️  Hugging Face LLM call failed: {e}")
            return "Sorry, I couldn't generate a response at this time."
    
    # ========================================================================
    # STEP 6: POST-PROCESSING
    # ========================================================================
    
    def post_process_response(
        self,
        llm_response: str,
        language: str,
        user_query: str = "",
        allow_step_format: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        answer_style_override: Optional[str] = None,
        strip_citations: bool = False,
    ) -> Dict[str, Any]:
        """
        Format the LLM response for frontend display
        
        Args:
            llm_response: Raw response from LLM
            language: Detected/target language
            user_query: Original user query for safety-tailored response guards
            allow_step_format: Preserve numbered step layout for process questions
            conversation_history: Recent conversation for context awareness (to avoid redundant questions)
            
        Returns:
            Formatted response dictionary
        """
        print(f"✨ Step 6: Post-processing response...")
        effective_answer_style = (answer_style_override or ANSWER_STYLE).lower()

        def infer_response_language(text: str, fallback_language: str) -> str:
            """
            Infer language from response text for consistent follow-up language.
            This helps when initial detection is noisy (e.g., Ilocano vs Indonesian).
            """
            lowered = (text or '').lower()

            # Quick script checks first.
            for ch in lowered:
                cp = ord(ch)
                if 0x0E00 <= cp <= 0x0E7F:
                    return 'th'

            # Ilocano lexical markers.
            ilocano_markers = {
                'iti', 'dagiti', 'wenno', 'ania', 'ti', 'nga', 'awanan',
                'awan', 'agturong', 'agreklamo', 'masapul', 'aramiden',
            }
            # Indonesian and Malay markers to avoid false positives.
            indonesian_markers = {'anda', 'apakah', 'langkah', 'berikutnya', 'saya', 'tidak'}
            malay_markers = {'adakah', 'seterusnya', 'anda', 'langkah', 'boleh'}

            tokens = set(re.findall(r"[a-zA-Z']+", lowered))
            ilo_hits = len(tokens & ilocano_markers)
            id_hits = len(tokens & indonesian_markers)
            ms_hits = len(tokens & malay_markers)

            if ilo_hits >= 2 and ilo_hits > id_hits and ilo_hits > ms_hits:
                return 'ilo'

            return fallback_language

        def normalize_narrative_text(text: str) -> str:
            """Convert list-like output into paragraph-style narrative text."""
            lines = text.strip().split('\n')
            normalized_lines = []

            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    normalized_lines.append('')
                    continue

                # Strip bullet and numbered list prefixes.
                line = re.sub(r"^[-*•]+\s*", "", line)
                line = re.sub(r"^\d+[\.)]\s*", "", line)
                line = re.sub(r"^[a-zA-Z][\.)]\s*", "", line)
                normalized_lines.append(line)

            # Rebuild paragraphs while preserving intentional blank lines.
            paragraphs = []
            current = []
            for line in normalized_lines:
                if line == '':
                    if current:
                        paragraphs.append(' '.join(current).strip())
                        current = []
                    continue
                current.append(line)

            if current:
                paragraphs.append(' '.join(current).strip())

            return '\n\n'.join(p for p in paragraphs if p)
        
        cleaned_response = llm_response.strip()
        cleaned_response = re.sub(
            r"^(?:here(?:'|’)s|this is)\s+(?:a\s+)?(?:helpful|simple|simplified|clear|concise|quick)?\s*(?:response|explanation|guide|answer|summary)\s*:\s*",
            "",
            cleaned_response,
            flags=re.IGNORECASE,
        )
        cleaned_response = re.sub(
            r"^(?:here(?:'|’)s\s+how\s+to\s+|getting\s+[^\n:]{0,80}:\s*)",
            "",
            cleaned_response,
            flags=re.IGNORECASE,
        )
        if strip_citations:
            cleaned_response = re.sub(r"\[\s*S\d+\s*\]", "", cleaned_response, flags=re.IGNORECASE)
            cleaned_response = re.sub(r"\s{2,}", " ", cleaned_response)
            self._log_debug("[Step 6] 🧹 Summary mode: inline citations removed from answer text")
        paragraphs = [segment.strip() for segment in re.split(r'\n\s*\n', cleaned_response) if segment.strip()]

        if effective_answer_style == 'bullet':
            # Normalize inline bullets/numbered lists into line-separated markdown list items.
            bullet_ready_response = cleaned_response
            bullet_ready_response = re.sub(r"\s*•\s+", "\n- ", bullet_ready_response)
            bullet_ready_response = re.sub(r"\s+[-*]\s+", "\n- ", bullet_ready_response)
            bullet_ready_response = re.sub(r"\s+(\d+[\.)])\s+", r"\n\1 ", bullet_ready_response)

            answer_items = []
            for line in bullet_ready_response.split('\n'):
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*') or line[0].isdigit()):
                    normalized = line.lstrip('•-*0123456789. ').strip()
                    if normalized:
                        answer_items.append(f"- {normalized}")
            if not answer_items:
                answer_items = [cleaned_response]
            answer_text = '\n'.join(answer_items)
        else:
            if allow_step_format:
                # Keep numbered instructions for procedural questions, but remove bullet markers.
                step_lines = []
                for raw_line in cleaned_response.split('\n'):
                    line = raw_line.strip()
                    if not line:
                        step_lines.append('')
                        continue
                    line = re.sub(r"^[-*•]+\s*", "", line)
                    step_lines.append(line)
                narrative_text = '\n'.join(step_lines).strip()
            else:
                narrative_text = normalize_narrative_text(cleaned_response)

            if not narrative_text:
                if not paragraphs:
                    paragraphs = [cleaned_response]
                narrative_text = '\n\n'.join(paragraphs)
            answer_text = narrative_text
            answer_items = [answer_text]
        
        query_lower = (user_query or "").lower()

        # Ensure users who are missing documents always get a safe alternative path.
        if any(keyword in query_lower for keyword in ["salary slip", "missing", "don't have", "do not have", "without"]):
            has_alternative_hint = any(
                phrase in answer_text.lower()
                for phrase in ["alternative", "instead", "can still apply", "other document", "supporting document"]
            )
            if not has_alternative_hint:
                answer_text = (
                    f"{answer_text}\n\n"
                    "If you do not have one document, ask the agency whether they can accept an alternative document such as a letter, bank statement, or other supporting document instead."
                )

        # Ensure users with low digital confidence get an offline option.
        if any(keyword in query_lower for keyword in ["online form", "don't know how", "do not know how", "not good with", "digital"]):
            has_offline_hint = any(
                phrase in answer_text.lower()
                for phrase in ["visit", "counter", "centre", "center", "walk-in", "office", "service kiosk", "utc"]
            )
            if not has_offline_hint:
                answer_text = (
                    f"{answer_text}\n\n"
                    "If online forms are difficult, visit the nearest government service counter for walk-in help."
                )

        # Align follow-up language with the actual generated answer language.
        resolved_language = infer_response_language(answer_text, language)

        # Language-specific follow-up questions (avoid asking if already a question at the end)
        self._should_add_followup = False
        if FORCE_FOLLOW_UP_QUESTION and FOLLOW_UP_STYLE != 'off' and '[STEP_GATE]' not in answer_text:
            # Check if response already ends with a question
            answer_stripped = answer_text.strip()
            already_has_question = re.search(r"[?]\s*$", answer_stripped)
            
            # Check if user just answered affirmatively to a prior yes/no question
            # (to avoid asking the same thing again)
            is_user_confirming = False
            if conversation_history:
                # Look for recent yes/no question from assistant + user confirming with yes/yeah/ok
                last_assistant_msg = None
                for turn in reversed(conversation_history):
                    if (turn.get('role') or '').lower() == 'assistant':
                        last_assistant_msg = turn.get('text', '')
                        break
                
                if last_assistant_msg and re.search(r"[?]\s*$", last_assistant_msg.strip()):
                    # Last assistant message was a question; check if current user query is confirmation
                    user_query_lower = (user_query or '').lower().strip()
                    confirmation_patterns = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'please', 'go ahead']
                    is_user_confirming = any(user_query_lower.startswith(p) for p in confirmation_patterns)
            
            if not already_has_question and not is_user_confirming:
                followup_type = self._classify_followup_type(user_query, answer_text, allow_step_format)
                followup_text = self._get_followup_question(resolved_language, followup_type)
                if followup_text:
                    answer_text = f"{answer_text.rstrip()}\n\n{followup_text}"
                    self._should_add_followup = True
                    self._log_debug(
                        f"[Step 6] ❓ Added follow-up (style={FOLLOW_UP_STYLE}, type={followup_type}, lang={resolved_language})"
                    )

        if effective_answer_style != 'bullet':
            answer_items = [answer_text]

        result = {
            'answer': answer_items,
            'answer_text': answer_text,
            'language': resolved_language,
            'timestamp': datetime.now().isoformat(),
            'raw_response': llm_response
        }
        
        print(f"   ✓ Response formatted for style '{effective_answer_style}'")
        return result

    # ========================================================================
    # STEP 8: RECURSIVE SUMMARIZATION
    # ========================================================================

    def summarize_document(self, document_text: str, chunk_size: int = 4000) -> str:
        """
        Recursively summarize a long document into 3-5 bullet points.

        Args:
            document_text: The long text of the document to summarize.
            chunk_size: The approximate size of each chunk in characters.

        Returns:
            A string containing the final summary in bullet points.
        """
        self._log_debug(f"[Summarize] 🔄 Starting recursive summarization for document of {len(document_text)} chars.")

        # 1. Chunk the document
        chunks = self._chunk_text(document_text, chunk_size)
        self._log_debug(f"[Summarize] 📄 Split document into {len(chunks)} chunks.")

        # 2. Recursively summarize
        summary = self._recursive_summarize_chunks(chunks)
        self._log_debug(f"[Summarize] ✨ Final condensed summary: {len(summary)} chars.")

        # 3. Final formatting
        final_prompt = f"""Based on the following summary, generate a final list of 3-5 actionable bullet points.
The points should be concise and easy to understand.

Summary:
{summary}

Final bullet points:
"""
        final_summary = self.call_llm(final_prompt)
        self._log_debug(f"[Summarize] ✅ Final bullet points generated.")
        return final_summary

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Splits text into chunks of a specified size without breaking sentences."""
        cleaned = (text or '').strip()
        if not cleaned:
            return []

        if len(cleaned) <= chunk_size:
            return [cleaned]

        nlp = self.nlp_en if all(ord(ch) < 128 for ch in text[:200]) else self.nlp_multi
        try:
            doc = nlp(cleaned)
            sentences = [s.text for s in doc.sents]
        except Exception:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if s.strip()]

        def split_oversized_sentence(sentence: str) -> List[str]:
            sentence = sentence.strip()
            if len(sentence) <= chunk_size:
                return [sentence]

            words = sentence.split()
            if not words:
                return [sentence[:chunk_size]]

            parts: List[str] = []
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip() if current else word
                if len(candidate) > chunk_size and current:
                    parts.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                parts.append(current)
            return parts

        chunks = []
        current_chunk = ""
        for sentence in sentences:
            for part in split_oversized_sentence(sentence):
                if not part:
                    continue
                candidate = f"{current_chunk} {part}".strip() if current_chunk else part
                if len(candidate) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = part
                else:
                    current_chunk = candidate
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _recursive_summarize_chunks(self, chunks: List[str]) -> str:
        """Helper for the recursive summarization logic."""
        if len(chunks) == 1:
            # Base case: summarize the final chunk
            prompt = f"Summarize the following text concisely:\n\n{chunks[0]}"
            summary = self.call_llm(prompt)
            return summary

        # Summarize each chunk individually
        summaries = []
        for i, chunk in enumerate(chunks):
            self._log_debug(f"[Summarize] summarizing chunk {i+1}/{len(chunks)}")
            prompt = f"Summarize the following text concisely:\n\n{chunk}"
            summary = self.call_llm(prompt)
            summaries.append(summary)

        # Recursive step: summarize the summaries
        concatenated_summaries = "\n\n".join(summaries)
        new_chunks = self._chunk_text(concatenated_summaries, 4000)
        return self._recursive_summarize_chunks(new_chunks)

    # ========================================================================
    # STEP 9: LEXICAL SIMPLIFICATION
    # ========================================================================

    def simplify_text(self, text: str, reading_level: str = '5th-grade') -> str:
        """
        Simplifies complex jargon in a text to a specified reading level.

        Args:
            text: The text to simplify.
            reading_level: The target reading level (e.g., '5th-grade', 'simple').

        Returns:
            The simplified text.
        """
        self._log_debug(f"[Simplify] 🔬 Simplifying text to {reading_level} level.")

        prompt = f"""Please simplify the following text to a {reading_level} reading level.
Replace complex legal, medical, or technical jargon with simple, everyday language.
Do not change the meaning of the text.

Original text:
{text}

Simplified text:
"""
        simplified_text = self.call_llm(prompt)
        self._log_debug(f"[Simplify] ✅ Text simplified.")
        return simplified_text

    def _normalize_phrase_for_match(self, text: str) -> str:
        normalized = (text or '').lower().strip()
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _get_approved_slang_terms(self) -> List[Dict[str, Any]]:
        if not ENABLE_SLANG_QUERY_EXPANSION:
            return []

        now = time.time()
        cache_age = now - self.slang_terms_cache_loaded_at
        if self.slang_terms_cache and cache_age < SLANG_TERMS_CACHE_TTL_SECONDS:
            return self.slang_terms_cache

        try:
            response = (
                self.supabase_client
                .table(SLANG_DICTIONARY_TABLE_NAME)
                .select('phrase,normalized_form,meaning,dialect,language_code,is_active,status')
                .eq('status', 'approved')
                .eq('is_active', True)
                .limit(1000)
                .execute()
            )
            rows = response.data or []
            self.slang_terms_cache = rows
            self.slang_terms_cache_loaded_at = now
            return rows
        except Exception as exc:
            self._log_debug(
                f"[Step 1.6] ⚠️ Slang lookup unavailable ({SLANG_DICTIONARY_TABLE_NAME}): {exc}"
            )
            self.slang_terms_cache = []
            self.slang_terms_cache_loaded_at = now
            return []

    def _expand_query_with_slang(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        if not ENABLE_SLANG_QUERY_EXPANSION:
            return query, []

        normalized_query = self._normalize_phrase_for_match(query)
        if not normalized_query:
            return query, []

        approved_terms = self._get_approved_slang_terms()
        if not approved_terms:
            return query, []

        expansions: List[Dict[str, str]] = []
        expansion_terms: List[str] = []

        for row in approved_terms:
            phrase = str(row.get('phrase') or '').strip()
            if not phrase:
                continue

            normalized_phrase = self._normalize_phrase_for_match(phrase)
            if not normalized_phrase:
                continue

            # Match complete phrase first to reduce accidental partial hits.
            phrase_pattern = rf'(^|\s){re.escape(normalized_phrase)}(\s|$)'
            if not re.search(phrase_pattern, normalized_query):
                continue

            normalized_form = str(row.get('normalized_form') or '').strip()
            meaning = str(row.get('meaning') or '').strip()
            expansion = normalized_form or meaning
            if not expansion:
                continue

            expansions.append({
                'phrase': phrase,
                'expansion': expansion,
                'dialect': str(row.get('dialect') or ''),
                'language_code': str(row.get('language_code') or ''),
            })
            expansion_terms.append(expansion)

            if len(expansions) >= SLANG_MAX_TERMS_PER_QUERY:
                break

        deduped_terms: List[str] = []
        seen = set()
        for term in expansion_terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped_terms.append(term)

        if not deduped_terms:
            return query, []

        expanded_query = f"{query.strip()} {' '.join(deduped_terms)}".strip()
        return expanded_query, expansions
    
    # ========================================================================
    # STEP 7: MAIN PIPELINE EXECUTION
    # ========================================================================
    
    def _log_debug(self, message: str):
        """Add debug message to logs (for frontend display)"""
        self.debug_logs.append(message)
        print(message)  # Also print to terminal
    
    def process_query(
        self,
        user_query: str,
        user_language_hint: Optional[str] = None,
        top_k: Optional[int] = None,
        summary_mode_enabled: Optional[bool] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: str = "",
    ) -> Dict[str, Any]:
        """
        Main pipeline: Process a user query end-to-end with dialect-aware translation
        
        Workflow:
        1. Detect user's language/dialect (using user_language_hint if provided)
        2. Translate query to pivot language (Malay/English) if needed
        3. Create embedding from (translated) query
        4. RAG retrieval
        5. LLM generates answer
        6. Translate answer back to user's dialect
        7. Return response with sources
        
        Args:
            user_query: User's input query in any ASEAN dialect
            user_language_hint: Optional language code from frontend (e.g., 'ms', 'en', 'zh')
            top_k: Optional retrieval width override from frontend
            summary_mode_enabled: Optional hard toggle from UI.
                True = force summary mode.
                False = force normal mode.
                None = infer from user query intent.
            conversation_history: Recent conversation turns for short-term memory
            conversation_summary: Rolling summary of earlier turns
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        # Clear debug logs for this query
        self.debug_logs = []
        self.last_query_expansions = []
        
        self._log_debug("\n" + "="*70)
        self._log_debug(f"🎯 Processing Query: '{user_query}'")
        self._log_debug(f"🧠 Conversation turns provided: {len(conversation_history or [])}")
        self._log_debug(f"🗒️ Conversation summary length: {len(conversation_summary or '')}")
        self._log_debug("="*70 + "\n")
        
        try:
            # Step 0: Classify query intent — skip RAG for greetings/chit-chat
            self._log_debug("[Step 0] 🧭 Classifying query intent...")
            intent = self.classify_query_intent(user_query)
            self._log_debug(f"[Step 0] ✓ Intent: {intent}")
            retrieval_query_override = None

            # Override: a step-gate confirmation must always run full RAG
            if intent == 'general' and self._detect_step_confirmation(user_query, conversation_history):
                intent = 'task_or_policy'
                self._log_debug("[Step 0] 🔄 Step-gate confirmation detected — overriding intent to task_or_policy")

            # Override: if user confirms office-finding follow-up (e.g., "yes"),
            # run full pipeline so we can fetch actual labour office contacts.
            if intent == 'general' and self._detect_labour_office_followup_confirmation(user_query, conversation_history):
                intent = 'task_or_policy'
                self._log_debug("[Step 0] 🔄 Labour-office follow-up confirmation detected — overriding intent to task_or_policy")

            contextual_followup = self._detect_contextual_followup_confirmation(user_query, conversation_history)
            is_affirmative_followup = False
            exact_followup_question = None
            if intent == 'general' and contextual_followup:
                intent = 'task_or_policy'
                retrieval_query_override = contextual_followup.get('contextual_query')
                is_affirmative_followup = True
                exact_followup_question = contextual_followup.get('exact_followup_question')
                self._log_debug(
                    "[Step 0] 🔄 Contextual follow-up confirmation DETECTED (user said 'YES')"
                )
                self._log_debug(
                    f"[Step 0] ❓ Exact Follow-up Question: {exact_followup_question}"
                )
                self._log_debug(
                    f"[Step 0] 📋 Follow-up Type: {contextual_followup.get('followup_type')}"
                )
                self._log_debug(
                    f"[Step 0] 🔍 Retrieval Query: {contextual_followup.get('contextual_query')[:100]}..."
                )
                self._log_debug(
                    "[Step 0] ⚙️ Overriding intent to task_or_policy for full RAG pipeline"
                )

            # Step 1: Detect language/dialect
            self._log_debug("[Step 1] 🔍 Detecting language...")
            
            # 1a. Use explicit user language hint if provided (highest priority)
            hint_normalized = self._normalize_language_hint(user_language_hint) if user_language_hint else None
            if hint_normalized:
                detected_language = hint_normalized
                self._log_debug(f"[Step 1] ℹ️ Using user-provided language hint: {detected_language}")
            else:
                # 1b. Try script-based heuristic
                script_hint = self._detect_script_language_hint(user_query)
                detected_language = script_hint if script_hint else None
                
                # 1c. Fallback to langdetect
                try:
                    lang_info = self.detect_language(user_query)
                    primary_lang = lang_info['primary_language']
                    self._log_debug(f"[Step 1] ℹ️ Langdetect primary: {primary_lang}")
                    
                    # If script-based hint exists and disagrees, prefer script hint for clear scripts
                    if detected_language and detected_language != primary_lang:
                        self._log_debug(
                            f"[Step 1] ⚖️ Script hint '{detected_language}' vs langdetect '{primary_lang}' "
                            f"→ preferring script hint"
                        )
                    else:
                        detected_language = primary_lang
                except Exception as e:
                    self._log_debug(f"[Step 1] ⚠️ Langdetect failed, defaulting to English: {e}")
                    if not detected_language:
                        detected_language = 'en'
            user_nllb_code = self._get_nllb_language_code(detected_language)
            self._log_debug(f"[Step 1] ✓ Detected: {detected_language} ({user_nllb_code})")

            # ----------------------------------------------------------------
            # FAST PATH: general/conversational query — no RAG needed
            # ----------------------------------------------------------------
            if intent == 'general':
                self._log_debug("[Fast Path] 💬 General query — skipping vector search")
                if ENABLE_LANGUAGE_AUTO_DETECTION:
                    target_lang_for_llm = detected_language
                else:
                    target_lang_for_llm = 'zsm_Latn' if user_nllb_code != 'eng_Latn' else 'eng_Latn'
                general_prompt = self._prepare_general_prompt(
                    user_query,
                    target_language=target_lang_for_llm,
                    conversation_history=conversation_history,
                    conversation_summary=conversation_summary,
                )
                llm_response = self.call_llm(general_prompt)
                final_response = self.post_process_response(
                    llm_response, detected_language,
                    user_query=user_query, allow_step_format=False,
                    conversation_history=conversation_history
                )
                # Translate back if needed
                if (
                    self.translation_enabled
                    and self._needs_translation(user_nllb_code)
                    and not ENABLE_LANGUAGE_AUTO_DETECTION
                ):
                    translated_segments = []
                    for segment in final_response['answer']:
                        text_only = segment.lstrip('• -*').strip()
                        trans_result = self.translate_text(
                            text=text_only, source_lang='zsm_Latn', target_lang=user_nllb_code
                        )
                        translated_segments.append(
                            trans_result['translated_text'] if trans_result['translation_performed'] else segment
                        )
                    final_response['answer_text'] = '\n\n'.join(translated_segments)
                    final_response['answer'] = [final_response['answer_text']]
                final_response['status'] = 'success'
                final_response['intent'] = 'general'
                final_response['rag_used'] = False
                final_response['evidence'] = []
                final_response['query_expansions'] = self.last_query_expansions
                final_response['detected_language'] = (
                    final_response.get('language', detected_language)
                    if ENABLE_LANGUAGE_AUTO_DETECTION
                    else detected_language
                )
                final_response['user_language_code'] = user_nllb_code
                final_response['sources'] = []
                final_response['retrieved_chunks_count'] = 0
                final_response['debug_logs'] = self.debug_logs
                return final_response

            # Check coverage: embeddings + translation
            embedding_supported = self._is_embedding_language_supported(detected_language)
            translation_supported = user_nllb_code in ASEAN_LANGUAGE_MAP.values()
            
            # True low-resource dialect: neither embeddings nor NLLB can reasonably handle it
            if not embedding_supported and (not self.translation_enabled or not translation_supported):
                self._log_debug("[Step 1] ⚠️ Detected potential low-resource dialect "
                                "(no embedding or translation coverage)")
                
                fallback_msg = (
                    "Sorry, your dialect is currently not supported. "
                    "Please enter your question in Malay or English."
                )
                if ANSWER_STYLE == 'bullet':
                    fallback_msg = f"- {fallback_msg}"
                
                return {
                    'answer': [fallback_msg],
                    'language': detected_language,
                    'user_language_code': user_nllb_code,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'low_resource_dialect',
                    'query_expansions': self.last_query_expansions,
                    'debug_logs': self.debug_logs,
                    'translation_note': (
                        "Low-resource dialect: neither embedding model nor NLLB-200 "
                        "cover this language reliably."
                    ),
                }
            
            # Step 1.5: Choose retrieval language (direct vs pivot translation)
            query_for_retrieval = retrieval_query_override or user_query
            translation_note = None
            should_translate_query = False

            # Prefer direct retrieval for non-pivot languages when embeddings support them.
            use_direct_retrieval = (
                DIRECT_RETRIEVAL_FOR_SUPPORTED_LANGS
                and embedding_supported
                and user_nllb_code not in PIVOT_LANGUAGES
            )
            
            if use_direct_retrieval:
                self._log_debug(
                    "[Step 1.5] ✅ Using direct retrieval for embedding-supported "
                    f"language ({detected_language}/{user_nllb_code})"
                )
            elif self.translation_enabled and self._needs_translation(user_nllb_code):
                should_translate_query = True
                self._log_debug(f"[Step 1.5] 🌐 Translating query to pivot language...")
                
                # Choose pivot language (prefer Malay for ASEAN, English as fallback)
                pivot_lang = 'zsm_Latn'  # Malay
                
                translation_result = self.translate_text(
                    text=user_query,
                    source_lang=user_nllb_code,
                    target_lang=pivot_lang
                )
                
                if translation_result['translation_performed']:
                    query_for_retrieval = translation_result['translated_text']
                    self._log_debug(f"[Step 1.5] ✓ Query translated: '{query_for_retrieval[:50]}...'")
                else:
                    self._log_debug(f"[Step 1.5] ⚠️  Translation skipped, using original query")
            else:
                if self.translation_enabled:
                    self._log_debug(f"[Step 1.5] ℹ️  Query already in pivot language, no translation needed")
                else:
                    self._log_debug(f"[Step 1.5] ⚠️  Translation disabled")
                    # Add disclaimer if translation is needed but disabled
                    if user_nllb_code not in PIVOT_LANGUAGES:
                        translation_note = f"⚠️ Note: Translation unavailable. Showing results in Malay/English."

            # Step 1.6: Expand slang/local phrases using approved community dictionary.
            expanded_query_for_retrieval, matched_expansions = self._expand_query_with_slang(query_for_retrieval)
            if matched_expansions:
                self.last_query_expansions = matched_expansions
                query_for_retrieval = expanded_query_for_retrieval
                self._log_debug(
                    "[Step 1.6] 🧩 Query expanded using community dictionary: "
                    + ", ".join(f"{item['phrase']}→{item['expansion']}" for item in matched_expansions)
                )
            else:
                self._log_debug("[Step 1.6] ℹ️ No community slang expansions matched")
            
            # Step 2: Create query embedding (from translated query)
            self._log_debug("[Step 2] 📊 Creating query embedding...")
            query_embedding = self.create_query_embedding(query_for_retrieval)
            self._log_debug(f"[Step 2] ✓ Embedding created (dim: {len(query_embedding)})")
            
            # Step 3: Vector search
            self._log_debug("[Step 3] 🔎 Searching vector database...")
            retrieved_chunks = self.vector_search(
                query_embedding,
                top_k=top_k,
                user_query=query_for_retrieval,
            )
            retrieved_chunks = self._rerank_and_filter_chunks(
                query_for_retrieval,
                retrieved_chunks,
                top_k=top_k,
            )

            # Optional augmentation: if user asks to find nearest labour office (or confirms "yes"
            # to that question), fetch directory contacts from Supabase and prepend them.
            if self._should_lookup_labour_office(user_query, conversation_history):
                location_hint = self._extract_location_hint(user_query, conversation_history)
                if location_hint:
                    self._log_debug(f"[Step 3] 📍 Labour office lookup enabled (location hint: {location_hint})")
                else:
                    self._log_debug("[Step 3] 📍 Labour office lookup enabled (no location hint; using national fallback)")

                office_rows = self.lookup_labour_offices(location_hint=location_hint, limit=3)
                office_chunks = [self._format_office_chunk(row) for row in office_rows]
                if office_chunks:
                    retrieved_chunks = office_chunks + (retrieved_chunks or [])
                    self._log_debug(f"[Step 3] ✓ Added {len(office_chunks)} labour office contact result(s)")
                else:
                    self._log_debug("[Step 3] ⚠️ Labour office lookup returned no active entries")
            
            if not retrieved_chunks:
                self._log_debug("[Step 3] ⚠️ No relevant documents found")
                no_results_msg = "• No relevant information found for your query. Please try rephrasing."
                if ANSWER_STYLE == 'narrative':
                    no_results_msg = "No relevant information found for your query. Please try rephrasing."
                
                # Translate "no results" message back to user's language if needed
                if self.translation_enabled and self._needs_translation(user_nllb_code):
                    trans_result = self.translate_text(
                        text=no_results_msg,
                        source_lang='eng_Latn',
                        target_lang=user_nllb_code
                    )
                    if trans_result['translation_performed']:
                        translated_text = trans_result['translated_text'].lstrip('• ').strip()
                        if ANSWER_STYLE == 'bullet':
                            no_results_msg = f"- {translated_text}"
                        else:
                            no_results_msg = translated_text
                
                return {
                    'answer': [no_results_msg],
                    'language': detected_language,
                    'user_language_code': user_nllb_code,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'no_results',
                    'query_expansions': self.last_query_expansions,
                    'debug_logs': self.debug_logs,
                    'translation_note': translation_note
                }
            
            self._log_debug(f"[Step 3] ✓ Found {len(retrieved_chunks)} relevant chunks")

            prompt_chunks = retrieved_chunks
            compressed_chunks = self._compress_retrieved_chunks(query_for_retrieval, retrieved_chunks)
            effective_answer_style = ANSWER_STYLE
            inferred_summary_requested = self._is_summary_request(user_query)
            summary_requested = inferred_summary_requested
            auto_summary_triggered = False

            if summary_mode_enabled is True:
                summary_requested = True
                self._log_debug("[Step 4] 🧭 Summary mode forced ON by user toggle")
            elif summary_mode_enabled is False:
                summary_requested = False
                self._log_debug("[Step 4] 🧭 Summary mode forced OFF by user toggle")

            if summary_requested:
                effective_answer_style = 'bullet'
                if summary_mode_enabled is True:
                    self._log_debug("[Step 4] 🧭 User toggle enabled summary; forcing bullet summary mode")
                elif inferred_summary_requested:
                    self._log_debug("[Step 4] 🧭 User requested summary; forcing bullet summary mode")
            elif summary_mode_enabled is None and self._should_auto_action_bullet_summary(user_query, retrieved_chunks, compressed_chunks):
                effective_answer_style = 'bullet'
                auto_summary_triggered = True
            self._log_debug(f"[Step 4] ✍️ Effective response style: {effective_answer_style}")

            summary_mode_active = summary_requested or auto_summary_triggered
            summary_mode_reason = None
            if summary_mode_enabled is True and summary_mode_active:
                summary_mode_reason = 'user_toggle_on'
            elif summary_mode_enabled is False:
                summary_mode_reason = 'user_toggle_off'
            elif summary_requested:
                summary_mode_reason = 'user_requested'
            elif auto_summary_triggered:
                summary_mode_reason = 'auto_long_context'

            recursive_context_summary = ""
            summary_context_chunks = retrieved_chunks
            if summary_requested:
                summary_context_chunks = self._expand_chunks_by_source_url(retrieved_chunks)
                prompt_chunks = summary_context_chunks
                compressed_chunks = self._compress_retrieved_chunks(query_for_retrieval, summary_context_chunks)
                recursive_source_blocks = []
                for idx, chunk in enumerate(summary_context_chunks, 1):
                    # Use richer chunk text here so recursive summarization has enough depth.
                    chunk_text = (chunk.get('content') or chunk.get('text') or chunk.get('summary') or '').strip()
                    if chunk_text:
                        recursive_source_blocks.append(f"[S{idx}] {chunk_text}")

                recursive_input_text = "\n\n".join(recursive_source_blocks).strip()
                if len(recursive_input_text) > RECURSIVE_SUMMARY_INPUT_MAX_CHARS:
                    recursive_input_text = recursive_input_text[:RECURSIVE_SUMMARY_INPUT_MAX_CHARS]

                # Fallback: if expanded chunks exist but recursive text is empty, use compressed summaries.
                if not recursive_input_text:
                    recursive_input_text = "\n\n".join(
                        (chunk.get('summary') or chunk.get('content') or chunk.get('text') or '').strip()
                        for chunk in compressed_chunks
                        if (chunk.get('summary') or chunk.get('content') or chunk.get('text') or '').strip()
                    )
                    if len(recursive_input_text) > RECURSIVE_SUMMARY_INPUT_MAX_CHARS:
                        recursive_input_text = recursive_input_text[:RECURSIVE_SUMMARY_INPUT_MAX_CHARS]

                if recursive_input_text:
                    self._log_debug(
                        "[Step 4] 🔁 Running recursive summarization for summary-mode response "
                        f"(chars={len(recursive_input_text)}, chunk_size={RECURSIVE_SUMMARY_CHUNK_SIZE}, "
                        f"expanded_chunks={len(summary_context_chunks)})"
                    )
                    recursive_context_summary = self.summarize_document(
                        recursive_input_text,
                        chunk_size=RECURSIVE_SUMMARY_CHUNK_SIZE,
                    )
                else:
                    self._log_debug("[Step 4] ⚠️ Skipping recursive summarization (empty context after compression)")
            elif effective_answer_style == 'bullet':
                self._log_debug("[Step 4] ⏭️ Recursive summarization disabled (no explicit summary request)")

            # Build evidence array from retrieved chunks
            evidence = []
            for idx, chunk in enumerate(retrieved_chunks, 1):
                original_text = (
                    chunk.get('content', chunk.get('text', '')) or ''
                ).strip()
                evidence.append({
                    'citation_tag': f'S{idx}',
                    'source_name': chunk.get('title', f'Source {idx}'),
                    'source_url': chunk.get('source_url', chunk.get('url', '')),
                    'original_excerpt': original_text[:400],
                    'rerank_score': chunk.get('rerank_score', None),
                    'similarity': chunk.get('similarity', None),
                })

            # Step 4: Prepare prompt (use pivot language for LLM)
            self._log_debug("[Step 4] 📝 Preparing LLM prompt...")
            is_process_question = self._is_process_question(retrieval_query_override or user_query)

            # Step-gating is optional; default is direct step-by-step response.
            step_confirmed = self._detect_step_confirmation(user_query, conversation_history)
            use_step_gate = ENABLE_STEP_GATE and is_process_question and not step_confirmed

            prompt = self.prepare_llm_prompt(
                query_for_retrieval,  # Use translated query (or contextual query if "yes" detected)
                prompt_chunks,
                detected_language if ENABLE_LANGUAGE_AUTO_DETECTION
                else ('zsm_Latn' if user_nllb_code != 'eng_Latn' else 'eng_Latn'),
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                use_step_gate=use_step_gate,
                compressed_chunks=compressed_chunks,
                answer_style_override=effective_answer_style,
                recursive_context_summary=recursive_context_summary,
                compact_citation_mode=(effective_answer_style == 'bullet'),
                is_affirmative_followup=is_affirmative_followup,
                original_user_query=user_query if is_affirmative_followup else None,
                exact_followup_question=exact_followup_question,  # NEW: Pass exact question
            )
            if is_affirmative_followup:
                self._log_debug(f"[Step 4] ✅ LLM will respond to: '{exact_followup_question}'")
                self._log_debug(f"[Step 4] ✓ Prompt prepared with affirmative followup context (length: {len(prompt)})")
            else:
                self._log_debug(f"[Step 4] ✓ Prompt prepared (length: {len(prompt)})")
            self._log_debug(f"[Step 4] ✓ Prompt prepared (length: {len(prompt)})")
            
            # Step 5: Call LLM
            self._log_debug(f"[Step 5] 🤖 Calling LLM ({self.llm_provider})...")
            llm_response = self.call_llm(prompt)
            self._log_debug(f"[Step 5] ✓ LLM response received (length: {len(llm_response)})")
            
            # Step 6: Post-process
            self._log_debug("[Step 6] 🔧 Post-processing response...")
            final_response = self.post_process_response(
                llm_response,
                detected_language,
                user_query=user_query,
                allow_step_format=is_process_question,
                conversation_history=conversation_history,
                answer_style_override=effective_answer_style,
                strip_citations=summary_mode_active,
            )
            
            # Step 6.5: Translate answer back to user's dialect
            if (
                self.translation_enabled
                and self._needs_translation(user_nllb_code)
                and not ENABLE_LANGUAGE_AUTO_DETECTION
            ):
                self._log_debug(f"[Step 6.5] 🌐 Translating answer back to user's language ({user_nllb_code})...")
                
                translated_segments = []
                for segment in final_response['answer']:
                    text_only = segment.lstrip('• -*').strip()
                    
                    trans_result = self.translate_text(
                        text=text_only,
                        source_lang='zsm_Latn',  # From Malay
                        target_lang=user_nllb_code
                    )
                    
                    if trans_result['translation_performed']:
                        translated_segments.append(trans_result['translated_text'])
                    else:
                        translated_segments.append(segment)
                
                if effective_answer_style == 'bullet':
                    final_response['answer'] = [
                        segment if re.match(r"^\s*[-*•]\s+", segment) else f"- {segment.strip()}"
                        for segment in translated_segments
                    ]
                    final_response['answer_text'] = '\n'.join(final_response['answer'])
                else:
                    final_response['answer_text'] = '\n\n'.join(translated_segments)
                    final_response['answer'] = [final_response['answer_text']]

                final_response['answer_original_language'] = llm_response  # Keep original for reference
                self._log_debug(f"[Step 6.5] ✓ Answer translated to user's language")
            else:
                if not self.translation_enabled and user_nllb_code not in PIVOT_LANGUAGES:
                    # Add note that we're showing in Malay/English
                    note = translation_note or "⚠️ Showing results in Malay/English (translation unavailable)"
                    if effective_answer_style == 'bullet':
                        final_response['answer'].insert(0, note)
                        final_response['answer_text'] = '\n'.join(final_response['answer'])
                    else:
                        final_response['answer_text'] = f"{note}\n\n{final_response['answer_text']}"
                        final_response['answer'] = [final_response['answer_text']]
            
            # Add metadata
            final_response['status'] = 'success'
            final_response['retrieved_chunks_count'] = len(retrieved_chunks)
            final_response['detected_language'] = (
                final_response.get('language', detected_language)
                if ENABLE_LANGUAGE_AUTO_DETECTION
                else detected_language
            )
            final_response['user_language_code'] = user_nllb_code
            final_response['query_translated'] = should_translate_query and query_for_retrieval != user_query
            final_response['answer_translated'] = self.translation_enabled and self._needs_translation(user_nllb_code)
            final_response['intent'] = 'task_or_policy'
            final_response['rag_used'] = True
            final_response['evidence'] = evidence
            final_response['summary_mode'] = summary_mode_active
            final_response['summary_mode_reason'] = summary_mode_reason
            final_response['query_expansions'] = self.last_query_expansions

            citation_source_text = final_response.get('raw_response') or final_response.get('answer_text', '')
            citation_counts = self._extract_used_citation_counts(citation_source_text)
            used_tags = {
                item['citation_tag']
                for item in evidence
                if citation_counts.get(item['citation_tag'], 0) > 0
            }

            for item in evidence:
                usage_count = citation_counts.get(item['citation_tag'], 0)
                item['cited_in_answer'] = usage_count > 0
                item['citation_usage_count'] = usage_count

            ordered_tags = [item['citation_tag'] for item in evidence]
            used_ordered = [tag for tag in ordered_tags if tag in used_tags]
            unused_ordered = [tag for tag in ordered_tags if tag not in used_tags]
            coverage = (len(used_ordered) / len(ordered_tags)) if ordered_tags else 0.0

            final_response['used_citation_tags'] = used_ordered
            final_response['unused_citation_tags'] = unused_ordered
            final_response['citation_stats'] = {
                'used_count': len(used_ordered),
                'unused_count': len(unused_ordered),
                'retrieved_count': len(ordered_tags),
                'coverage': round(coverage, 3),
                'tracking_source': 'raw_response' if final_response.get('raw_response') else 'answer_text',
            }

            self._log_debug(
                "[Step 6] 🔗 Citation usage: "
                f"used={len(used_ordered)}, unused={len(unused_ordered)}, retrieved={len(ordered_tags)}"
            )

            # Add sources for frontend display with all required fields
            formatted_sources = []
            for chunk in retrieved_chunks:
                formatted_source = {
                    'title': chunk.get('title', 'Document'),
                    'content': chunk.get('summary') or chunk.get('content') or chunk.get('text', ''),
                    'summary': chunk.get('summary') or chunk.get('content') or chunk.get('text', ''),
                    'source_url': chunk.get('source_url') or chunk.get('url', ''),
                    'url': chunk.get('source_url') or chunk.get('url', ''),
                    'rerank_score': chunk.get('rerank_score'),
                    'similarity': chunk.get('similarity'),
                    'language': chunk.get('language', 'unknown'),
                    'chunk_index': chunk.get('chunk_index'),
                    'total_chunks': chunk.get('total_chunks'),
                    'page_number': chunk.get('page_number'),
                    'page_start': chunk.get('page_start'),
                    'page_end': chunk.get('page_end'),
                    'section': chunk.get('section'),
                    'subsection': chunk.get('subsection'),
                    'source_type': chunk.get('source_type'),
                }
                formatted_sources.append(formatted_source)
            final_response['sources'] = formatted_sources if formatted_sources else []
            
            # Add debug logs to response
            final_response['debug_logs'] = self.debug_logs
            
            self._log_debug("\n" + "="*70)
            self._log_debug("✅ Query processing completed successfully!")
            self._log_debug("="*70 + "\n")
            
            return final_response
            
        except Exception as e:
            self._log_debug(f"\n❌ Error processing query: {e}")
            import traceback
            self._log_debug(f"Traceback: {traceback.format_exc()}")

            error_text = str(e)
            if self._is_transient_hf_error(e):
                user_message = (
                    "The AI service timed out while processing your request. "
                    "Please try again in a few seconds."
                )
            else:
                user_message = f"Error: {error_text}"
            
            return {
                'answer': [user_message],
                'language': 'en',
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'query_expansions': self.last_query_expansions,
                'debug_logs': self.debug_logs
            }


# ============================================================================
# STEP 8: OPTIONAL ENHANCEMENTS
# ============================================================================

class RAGPipelineEnhanced(RAGPipeline):
    """
    Enhanced RAG Pipeline with optional features:
    - Voice output (TTS)
    - Query logging
    - Fallback handling
    """
    
    def __init__(self, enable_logging: bool = True):
        super().__init__()
        self.enable_logging = enable_logging
        self.query_log = []
    
    def log_query(self, query: str, response: Dict[str, Any]):
        """Log query-answer pairs for demo/analysis"""
        if self.enable_logging:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'language': response.get('language', 'unknown')
            }
            self.query_log.append(log_entry)
    
    def save_logs(self, filepath: str = 'query_logs.json'):
        """Save logs to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.query_log, f, ensure_ascii=False, indent=2)
        print(f"📊 Logs saved to {filepath}")
    
    def process_query_with_logging(self, user_query: str) -> Dict[str, Any]:
        """Process query and log the interaction"""
        response = self.process_query(user_query)
        self.log_query(user_query, response)
        return response


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize the pipeline
    rag = RAGPipeline()
    
    # Example query
    example_query = "Paano ako makakakuha ng healthcare subsidy sa Penang?"
    
    # Process the query
    result = rag.process_query(example_query)
    
    # Print result
    print("\n📋 FINAL RESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
